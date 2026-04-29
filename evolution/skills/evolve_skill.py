"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.config import EvolutionConfig, get_hermes_agent_path

# Surface our own package's logs alongside DSPy's. Without this the
# BudgetAwareProposer per-call observability log (commit 1 of PR #5)
# stays invisible because Python's `logging` module defaults to WARNING
# on un-configured root and our package logger never gets reached.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.stats import paired_bootstrap
from evolution.core.fitness import LLMJudge, make_skill_fitness_metric
from evolution.core.constraints import ConstraintValidator
from evolution.skills.budget_aware_proposer import BudgetAwareProposer
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)
from evolution.skills.knee_point import select_knee_point, CandidatePick

console = Console()


_BUDGET_BY_ITERATIONS = {1: "light", 2: "medium", 3: "heavy"}


# Quality-gate presets bundle the three curve parameters into named
# operating modes so users don't need to set each independently. Tuned
# from PR data points; "default" is calibrated against PR #5 obsidian
# (+24.2% growth, ~+0.07 expected improvement).
_QUALITY_GATE_PRESETS: dict[str, dict[str, float]] = {
    "strict": {
        # Tighter free threshold + steeper curve + lower abs ceiling.
        # Use when shipping cost is dominated by inference per-token.
        "growth_free_threshold": 0.10,
        "growth_quality_slope": 0.50,
        "max_absolute_chars": 3000,
    },
    "default": {
        "growth_free_threshold": 0.20,
        "growth_quality_slope": 0.30,
        "max_absolute_chars": 5000,
    },
    "lenient": {
        # Looser everywhere — useful early in evolution iteration when
        # the LM judge metric or holdout dataset is noisy and we want
        # to surface genuinely-improved candidates that strict would reject.
        "growth_free_threshold": 0.30,
        "growth_quality_slope": 0.20,
        "max_absolute_chars": 8000,
    },
    "off": {
        # Effectively disable the gate by setting the free threshold
        # higher than any realistic growth. Static checks (size,
        # non_empty, structure) still apply.
        "growth_free_threshold": 100.0,
        "growth_quality_slope": 0.0,
        "max_absolute_chars": 100_000,
    },
}


def _write_gate_decision(output_dir: Path, decision: dict[str, Any]) -> Path:
    """Persist the deploy-gate's structured decision for future calibration.

    Each run writes one of these regardless of outcome (deploy or reject).
    Recalibrating the curve is then `jq -s '...' output/*/*/gate_decision.json`
    rather than parsing free-form failure notes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gate_decision.json"
    path.write_text(json.dumps(decision, indent=2))
    return path


def _knee_point_payload(knee_pick: Optional[CandidatePick]) -> dict[str, Any]:
    """Serialize a CandidatePick (or its absence) for gate_decision.json.

    `applied: false` lands when MIPROv2 fallback fired (no detailed_results)
    so a calibration script can distinguish "knee-point ran" from "knee-point
    skipped" without checking field presence.
    """
    if knee_pick is None:
        return {"applied": False, "reason": "no_detailed_results"}
    return {
        "applied": True,
        "fallback": knee_pick.fallback,
        "epsilon": knee_pick.epsilon,
        "band_size": knee_pick.band_size,
        "picked_idx": knee_pick.picked_idx,
        "picked_val_score": knee_pick.val_score,
        "picked_val_rank_in_band": knee_pick.val_rank_in_band,
        "picked_body_chars": knee_pick.body_chars,
        "gepa_default_idx": knee_pick.gepa_default_idx,
        "gepa_default_body_chars": knee_pick.gepa_default_body_chars,
        "band_roster": knee_pick.band_roster,
    }


def _holdout_evaluate_with_metric(module, holdout_examples, metric, lm) -> tuple[float, list[float]]:
    """Score `module` on the holdout via dspy.Evaluate.

    The GEPA-shaped metric takes 5 positional args; dspy.Evaluate calls
    metric(example, prediction). Wrap it.

    Returns (mean_score, per_example_scores). Per-example scores feed
    the bootstrap CI in the deploy gate.
    """
    def two_arg_metric(example, prediction, *_args, **_kwargs):
        result = metric(example, prediction)
        return float(getattr(result, "score", result))

    evaluator = dspy.Evaluate(
        devset=holdout_examples,
        metric=two_arg_metric,
        num_threads=4,
        provide_traceback=True,
        max_errors=len(holdout_examples) * 100,
    )
    with dspy.context(lm=lm):
        result = evaluator(module)
    # dspy.Evaluate returns EvaluationResult(score=mean*100, results=[(ex, pred, score), ...]).
    # Per-example scores are in devset order (evaluate.py:179: zip(devset, results, strict=False)).
    mean = float(result.score) / 100.0
    per_example = [float(s) for _, _, s in result.results]
    return mean, per_example


def _resolve_budget(iterations: int, budget: Optional[str]) -> str:
    """Pick the GEPA budget. Explicit `budget` always wins.

    `iterations` is the legacy CLI knob: only the values 1/2/3 carry a
    meaningful mapping; anything else collapses to "light". Callers
    should prefer `--budget` and treat `--iterations` as deprecated.
    """
    if budget is not None:
        return budget
    return _BUDGET_BY_ITERATIONS.get(iterations, "light")


def _default_gepa_runner(
    *,
    baseline_module: SkillModule,
    trainset: list,
    valset: list,
    metric,
    gepa_budget: str,
    optimizer_model: str,
    seed: int,
    instruction_proposer=None,
    reflection_model: Optional[str] = None,
):
    # Resolve reflection LM: explicit reflection_model wins; otherwise
    # fall back to the eval optimizer_model. DSPy's GEPA docstring
    # recommends gpt-5 with max_tokens=32000; reasoning models also
    # require max_tokens >= 16000 (DSPy raises ValueError otherwise).
    reflection_lm_model = reflection_model or optimizer_model
    optimizer = dspy.GEPA(
        metric=metric,
        auto=gepa_budget,
        # cache=False on the reflection LM: at temperature=1.0 the disk
        # cache would replay stale mutations across runs and shrink the
        # diversity of proposed candidates.
        reflection_lm=dspy.LM(
            reflection_lm_model,
            temperature=1.0,
            max_tokens=32000,
            cache=False,
        ),
        seed=seed,
        # track_stats=True attaches DspyGEPAResult to optimized_module's
        # .detailed_results, exposing all candidates (not just the best),
        # the Pareto front mapping, and per-example val_subscores. Doesn't
        # affect this PR's deploy decision but unlocks future work
        # (knee-point selection, champion-challenger registry) without
        # re-architecture.
        track_stats=True,
        # When set, GEPA calls our proposer instead of its default
        # InstructionProposalSignature path (dspy/teleprompt/gepa/
        # gepa_utils.py:112-118). The proposer runs inside the
        # reflection_lm context — the reflection_lm above still drives it.
        instruction_proposer=instruction_proposer,
    )
    return optimizer.compile(baseline_module, trainset=trainset, valset=valset)


def _default_mipro_runner(
    *,
    baseline_module: SkillModule,
    trainset: list,
    metric,
    seed: int,
):
    # MIPROv2 expects a metric returning a float; our GEPA-shaped metric
    # returns dspy.Prediction(score, feedback). Unwrap .score for MIPROv2's
    # score aggregation; pass-through if the metric already returns float.
    def float_metric(*args, **kwargs):
        result = metric(*args, **kwargs)
        return float(getattr(result, "score", result))

    optimizer = dspy.MIPROv2(
        metric=float_metric,
        auto="light",
        init_temperature=0.5,
        seed=seed,
    )
    return optimizer.compile(baseline_module, trainset=trainset)


def _print_fallback_banner(exc: Exception, failure_log_path: Optional[Path]) -> None:
    tb = traceback.format_exc()
    if failure_log_path is not None:
        failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        failure_log_path.write_text(f"{type(exc).__name__}: {exc}\n\n{tb}")
        location_line = f"Full traceback: {failure_log_path}"
    else:
        location_line = "Re-run with --no-fallback to surface GEPA's traceback."

    console.print(Panel(
        f"[bold]GEPA failed:[/bold] {type(exc).__name__}: {exc}\n\n"
        f"Falling back to MIPROv2.\n"
        f"{location_line}",
        title="[bold yellow]GEPA fallback[/bold yellow]",
        border_style="red",
    ))


def _build_optimizer_and_compile(
    *,
    baseline_module: SkillModule,
    trainset: list,
    valset: list,
    metric,
    gepa_budget: str,
    optimizer_model: str,
    seed: int,
    no_fallback: bool,
    failure_log_path: Optional[Path] = None,
    instruction_proposer=None,
    reflection_model: Optional[str] = None,
    _gepa_runner=_default_gepa_runner,
    _mipro_runner=_default_mipro_runner,
):
    """Run GEPA; fall back to MIPROv2 on failure unless `no_fallback`.

    Returns `(optimized_module, optimizer_name)`. ImportError from the
    MIPROv2 path (raised lazily inside MIPROv2.compile when optuna is
    missing) is re-raised with the GEPA failure preserved as `__cause__`
    so the user keeps both diagnostics.
    """
    try:
        optimized = _gepa_runner(
            baseline_module=baseline_module,
            trainset=trainset,
            valset=valset,
            metric=metric,
            gepa_budget=gepa_budget,
            optimizer_model=optimizer_model,
            seed=seed,
            instruction_proposer=instruction_proposer,
            reflection_model=reflection_model,
        )
        return optimized, "GEPA"
    except Exception as gepa_exc:
        if no_fallback:
            raise
        _print_fallback_banner(gepa_exc, failure_log_path)
        try:
            optimized = _mipro_runner(
                baseline_module=baseline_module,
                trainset=trainset,
                metric=metric,
                seed=seed,
            )
            return optimized, "MIPROv2"
        except ImportError as ie:
            console.print(
                "[red]✗ MIPROv2 fallback requires the [miprov2] extra. "
                "Install with: pip install hermes-agent-self-evolution[miprov2][/red]"
            )
            raise ie from gepa_exc


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
    seed: int = 42,
    budget: Optional[str] = None,
    no_fallback: bool = False,
    reflection_model: Optional[str] = None,
    length_penalty_weight: float = 0.0,
    quality_gate: str = "default",
    growth_free_threshold: Optional[float] = None,
    growth_quality_slope: Optional[float] = None,
    max_absolute_chars: Optional[int] = None,
    bootstrap_confidence: Optional[float] = None,
    bootstrap_n_resamples: Optional[int] = None,
    knee_point_epsilon: Optional[float] = None,
):
    """Main evolution function — orchestrates the full optimization loop."""

    # Resolve quality-gate preset; explicit overrides win.
    preset = _QUALITY_GATE_PRESETS[quality_gate]
    resolved_free = growth_free_threshold if growth_free_threshold is not None else preset["growth_free_threshold"]
    resolved_slope = growth_quality_slope if growth_quality_slope is not None else preset["growth_quality_slope"]
    resolved_abs = max_absolute_chars if max_absolute_chars is not None else preset["max_absolute_chars"]

    config_kwargs = dict(
        iterations=iterations,
        optimizer_model=optimizer_model,
        reflection_model=reflection_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        run_pytest=run_tests,
        seed=seed,
        length_penalty_weight=length_penalty_weight,
        growth_free_threshold=resolved_free,
        growth_quality_slope=resolved_slope,
        max_absolute_chars=int(resolved_abs),
    )
    if bootstrap_confidence is not None:
        config_kwargs["bootstrap_confidence"] = bootstrap_confidence
    if bootstrap_n_resamples is not None:
        config_kwargs["bootstrap_n_resamples"] = bootstrap_n_resamples
    config = EvolutionConfig(**config_kwargs)
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n")

    skill_path = find_skill(skill_name, config.hermes_agent_path)
    if not skill_path:
        console.print(f"[red]✗ Skill '{skill_name}' not found in {config.hermes_agent_path / 'skills'}[/red]")
        sys.exit(1)

    skill = load_skill(skill_path)
    console.print(f"  Loaded: {skill_path.relative_to(config.hermes_agent_path)}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    if dry_run:
        resolved_budget = _resolve_budget(iterations, budget)
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization (budget={resolved_budget})")
        console.print(f"  Would validate constraints and create PR")
        return

    # ── 2. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

    if eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path), seed=config.seed)
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
    elif eval_source == "sessiondb":
        save_path = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill["raw"],
            sources=["claude-code", "copilot", "hermes"],
            output_path=save_path,
            model=eval_model,
            seed=config.seed,
        )
        if not dataset.all_examples:
            console.print("[red]✗ No relevant examples found from session history[/red]")
            sys.exit(1)
        console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
    elif eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(
            artifact_text=skill["raw"],
            artifact_type="skill",
        )
        # Save for reuse
        save_path = Path("datasets") / "skills" / skill_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
    elif dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        console.print("[red]✗ Specify --dataset-path or use --eval-source synthetic[/red]")
        sys.exit(1)

    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # Refuse to run if the holdout would be too small to gate on. The
    # quality-gated growth check consumes the holdout improvement delta;
    # a 1-2 example holdout has stdev ~0.2 and would make the gate's
    # decisions essentially noise. Raise eval_dataset_size or
    # holdout_ratio rather than override min_holdout_size.
    if len(dataset.holdout) < config.min_holdout_size:
        console.print(
            f"[red]✗ Holdout has only {len(dataset.holdout)} examples; need ≥{config.min_holdout_size} "
            f"to gate on improvement signal. Increase eval_dataset_size or holdout_ratio.[/red]"
        )
        sys.exit(1)

    # ── 3. Validate constraints on baseline ─────────────────────────────
    # Static checks only — there's no quality signal for the baseline
    # because there's nothing to compare it against. The growth-with-
    # quality gate runs on the evolved skill once the holdout has
    # produced an improvement signal (§9 below).
    console.print(f"\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    baseline_constraints = validator.validate_static(skill["raw"], "skill")
    all_pass = True
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[yellow]⚠ Baseline skill has constraint violations — proceeding anyway[/yellow]")

    # ── 4. Set up DSPy + GEPA optimizer ─────────────────────────────────
    gepa_budget = _resolve_budget(iterations, budget)
    console.print(f"\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA (budget={gepa_budget})")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    # Configure DSPy
    lm = dspy.LM(eval_model)
    # warn_on_type_mismatch gates DSPy's typeguard validation of
    # InputField values against their declared annotations. Several
    # signatures in this codebase pass empty/None into `str` inputs
    # (e.g. assistant_response in RelevanceFilter when an event has no
    # response yet) — fine, but spams a warning per call.
    dspy.configure(lm=lm, warn_on_type_mismatch=False)

    # Create the baseline skill module
    baseline_module = SkillModule(skill["body"])

    # Build the LLM-as-judge metric. The judge is shared by GEPA's
    # per-iteration scoring and the holdout eval below; constructing it
    # once means DSPy's LM cache lines up across both surfaces.
    # Pass baseline + growth budget so the metric can append a [BUDGET]
    # line to feedback when GEPA hands us pred_trace, teaching the
    # reflection LM to prefer concise instructions.
    # The metric's [BUDGET] feedback line targets growth_free_threshold —
    # the zone where the deploy gate doesn't require quality justification.
    # The optimizer learns to land there; growth above only deploys if
    # the holdout improvement justifies it (validate_growth_with_quality).
    judge = LLMJudge(config)
    metric = make_skill_fitness_metric(
        judge,
        baseline_skill_text=skill["body"],
        max_growth=config.growth_free_threshold,
    )

    # Prepare DSPy examples
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization (budget={gepa_budget})...[/bold cyan]\n")

    start_time = time.time()
    failure_log_path = Path("output") / skill_name / "gepa_failure.log"

    # Inject a custom instruction proposer that bakes the size budget into
    # the reflection prompt the LM sees on every iteration. DSPy's
    # gepa_kwargs={"reflection_prompt_template": ...} would be the simpler
    # path, but gepa.api rejects it whenever the adapter (DspyAdapter
    # always) provides its own propose_new_texts (gepa/api.py:317-321).
    # This is the documented DSPy-side extension point instead.
    # Same target as the metric: the no-quality-justification zone.
    proposer = BudgetAwareProposer(
        baseline_chars=len(skill["body"]),
        max_growth=config.growth_free_threshold,
    )

    optimized_module, optimizer_name = _build_optimizer_and_compile(
        baseline_module=baseline_module,
        trainset=trainset,
        valset=valset,
        metric=metric,
        gepa_budget=gepa_budget,
        optimizer_model=optimizer_model,
        seed=config.seed,
        no_fallback=no_fallback,
        failure_log_path=failure_log_path,
        instruction_proposer=proposer,
        reflection_model=config.reflection_model,
    )

    elapsed = time.time() - start_time
    console.print(f"\n  {optimizer_name} optimization completed in {elapsed:.1f}s")

    # ── 5b. Knee-point Pareto selection ─────────────────────────────────
    # GEPA's default = "best by aggregate valset score." With small valsets
    # (N≤10) that overfits aggressively. Scan candidates within ε=1/n_val of
    # best valset and pick the most parsimonious one. Skip cleanly when
    # MIPROv2 fallback fired (no detailed_results attribute).
    knee_pick: Optional[CandidatePick] = None
    if hasattr(optimized_module, "detailed_results"):
        details = optimized_module.detailed_results
        knee_pick = select_knee_point(
            candidates=details.candidates,
            val_aggregate_scores=details.val_aggregate_scores,
            n_val=len(valset),
            static_validator=lambda txt: validator.validate_static(
                reassemble_skill(skill["frontmatter"], txt), "skill",
            ),
            gepa_default_idx=details.best_idx,
            epsilon=knee_point_epsilon,
        )
        # Fresh SkillModule(knee_text) — never mutate the predictor in place.
        # Avoids carrying ChainOfThought state (e.g. demos) from the
        # GEPA-default module; the picked candidate's instruction text is
        # the only thing we want.
        optimized_module = SkillModule(knee_pick.skill_text)
        console.print(
            f"\n[bold]Knee-point selection[/bold]: picked candidate "
            f"{knee_pick.picked_idx} (val={knee_pick.val_score:.3f}, "
            f"rank {knee_pick.val_rank_in_band} of {knee_pick.band_size} "
            f"in band, {knee_pick.body_chars} chars vs GEPA default "
            f"{knee_pick.gepa_default_body_chars} chars; ε={knee_pick.epsilon:.3f}; "
            f"fallback={knee_pick.fallback})"
        )

    # ── 6. Extract evolved skill text ───────────────────────────────────
    # The optimized module's instructions contain the evolved skill text
    evolved_body = optimized_module.skill_text
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # Per-run output dir (created here so failure paths can also write
    # to a timestamped subdir rather than clobbering evolved_FAILED.md).
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 7. Static constraints (size, structure, non-empty) ──────────────
    # Fail-fast on broken artifacts before spending judge-call budget on
    # the holdout. Growth-with-quality is checked after the holdout in §9.
    console.print(f"\n[bold]Validating evolved skill (static checks)[/bold]")
    static_constraints = validator.validate_static(evolved_full, "skill")
    static_pass = True
    for c in static_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            static_pass = False

    if not static_pass:
        console.print("[red]✗ Evolved skill FAILED static constraints — not deploying[/red]")
        failed_path = output_dir / "evolved_FAILED.md"
        failed_path.write_text(evolved_full)
        _write_gate_decision(output_dir, {
            "schema_version": "3",
            "decision": "reject",
            "reason": "static_constraint_failure",
            "failed_constraints": [c.constraint_name for c in static_constraints if not c.passed],
            "messages": [c.message for c in static_constraints if not c.passed],
            "knee_point": _knee_point_payload(knee_pick),
        })
        console.print(f"  Saved failed variant to {failed_path}")
        return

    # ── 8. Evaluate on holdout set ──────────────────────────────────────
    console.print(
        f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]"
    )
    console.print(
        "  [dim]Holdout uses the same LLM-as-judge metric as GEPA — expect ~"
        f"{2 * len(dataset.holdout)} judge calls.[/dim]"
    )

    holdout_examples = dataset.to_dspy_examples("holdout")
    avg_baseline, baseline_per_example = _holdout_evaluate_with_metric(
        baseline_module, holdout_examples, metric, lm,
    )
    avg_evolved, evolved_per_example = _holdout_evaluate_with_metric(
        optimized_module, holdout_examples, metric, lm,
    )
    improvement = avg_evolved - avg_baseline

    # ── 9. Growth-with-quality gate (paired bootstrap on holdout) ──────
    console.print(f"\n[bold]Validating growth against holdout improvement[/bold]")
    bootstrap = paired_bootstrap(
        baseline_per_example,
        evolved_per_example,
        confidence=config.bootstrap_confidence,
        n_resamples=config.bootstrap_n_resamples,
        seed=config.seed,
    )
    growth_constraints = validator.validate_growth_with_quality(
        evolved_full, skill["raw"], bootstrap,
    )
    growth_pass = True
    for c in growth_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            growth_pass = False

    growth_pct = (len(evolved_full) - len(skill["raw"])) / max(1, len(skill["raw"]))
    required_improvement = max(
        0.0,
        config.growth_quality_slope * (growth_pct - config.growth_free_threshold),
    )
    decision_payload = {
        "schema_version": "3",
        "decision": "deploy" if growth_pass else "reject",
        "reason": "passed" if growth_pass else "growth_quality_gate",
        "decision_rule_used": "no_regression_only" if required_improvement == 0.0 else "dual_check",
        "growth_pct": growth_pct,
        "required_improvement": required_improvement,
        "baseline_chars": len(skill["raw"]),
        "evolved_chars": len(evolved_full),
        "absolute_char_ceiling": config.max_absolute_chars,
        "growth_free_threshold": config.growth_free_threshold,
        "growth_quality_slope": config.growth_quality_slope,
        "baseline_per_example": baseline_per_example,
        "evolved_per_example": evolved_per_example,
        "avg_baseline": avg_baseline,
        "avg_evolved": avg_evolved,
        "bootstrap": bootstrap,
        "failed_constraints": [c.constraint_name for c in growth_constraints if not c.passed],
        "messages": [c.message for c in growth_constraints if not c.passed],
        "knee_point": _knee_point_payload(knee_pick),
    }
    gate_path = _write_gate_decision(output_dir, decision_payload)
    console.print(f"  [dim]Gate decision logged to {gate_path}[/dim]")

    if not growth_pass:
        console.print("[red]✗ Evolved skill REJECTED by quality gate — not deploying[/red]")
        failed_path = output_dir / "evolved_FAILED.md"
        failed_path.write_text(evolved_full)
        console.print(f"  Saved failed variant to {failed_path}")
        return

    # ── 9. Report results ───────────────────────────────────────────────
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Holdout Score",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    table.add_row(
        "Skill Size",
        f"{len(skill['body']):,} chars",
        f"{len(evolved_body):,} chars",
        f"{len(evolved_body) - len(skill['body']):+,} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")

    console.print()
    console.print(table)

    # ── 10. Save output ─────────────────────────────────────────────────
    # output_dir + timestamp were created earlier (before §7) so failure
    # paths can also write evolved_FAILED.md + gate_decision.json into
    # the same per-run subdir. Reuse them here.

    # Save evolved skill
    (output_dir / "evolved_skill.md").write_text(evolved_full)

    # Save baseline for comparison
    (output_dir / "baseline_skill.md").write_text(skill["raw"])

    # Save metrics
    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if improvement > 0:
        console.print(f"\n[bold green]✓ Evolution improved skill by {improvement:+.3f} ({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]")
        console.print(f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not improve skill (change: {improvement:+.3f})[/yellow]")
        console.print("  Try: more iterations, better eval dataset, or different optimizer model")


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option(
    "--iterations",
    default=10,
    help="DEPRECATED. Maps 1→light, 2→medium, 3→heavy GEPA budget; any other value falls through to light. Prefer --budget.",
)
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden", "sessiondb"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option(
    "--optimizer-model",
    default="openai/gpt-4.1",
    help="Default LM bound to dspy.configure (eval LM). Reflection LM is "
    "controlled separately by --reflection-model.",
)
@click.option(
    "--reflection-model",
    default="openai/gpt-5-mini",
    help="Model for the GEPA reflection LM (the LM the instruction proposer "
    "calls). DSPy's GEPA docstring recommends gpt-5-class reasoning models; "
    "Decagon's production blog reports gpt-4o-mini failed completely here. "
    "Reasoning models require max_tokens >= 16000 (we set 32000).",
)
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--seed", default=42, type=int, help="RNG seed for dataset shuffles and DSPy optimizer")
@click.option(
    "--budget",
    default=None,
    type=click.Choice(["light", "medium", "heavy"]),
    help="GEPA optimization budget. Overrides --iterations mapping.",
)
@click.option(
    "--no-fallback",
    is_flag=True,
    help="Re-raise GEPA failures instead of falling back to MIPROv2 (for debugging)",
)
@click.option(
    "--length-penalty-weight",
    default=0.0,
    type=float,
    help="Forward-wired for an upcoming custom-DspyAdapter PR that adds a "
    "score-side λ-penalty for instruction length. Currently a no-op.",
)
@click.option(
    "--quality-gate",
    default="default",
    type=click.Choice(["strict", "default", "lenient", "off"]),
    help="Preset for the deploy gate's growth-vs-improvement curve. "
    "strict=(0.10/0.50/3000), default=(0.20/0.30/5000), "
    "lenient=(0.30/0.20/8000), off=disabled (static checks only).",
)
@click.option(
    "--growth-free-threshold",
    default=None,
    type=float,
    help="Advanced: override the preset's growth_free_threshold (growth "
    "below which no improvement justification is required).",
)
@click.option(
    "--growth-quality-slope",
    default=None,
    type=float,
    help="Advanced: override the preset's growth_quality_slope (linear "
    "rate at which required holdout improvement scales with growth above "
    "the free threshold).",
)
@click.option(
    "--max-absolute-chars",
    default=None,
    type=int,
    help="Advanced: override the preset's max_absolute_chars (hard char "
    "ceiling on the evolved artifact, independent of growth %).",
)
@click.option(
    "--bootstrap-confidence",
    default=None,
    type=float,
    help="Advanced: confidence level for the paired-bootstrap CI on the "
    "holdout improvement (default 0.90).",
)
@click.option(
    "--bootstrap-resamples",
    default=None,
    type=int,
    help="Advanced: number of bootstrap resamples (default 2000).",
)
@click.option(
    "--knee-point-epsilon",
    default=None,
    type=float,
    help="Advanced: ε tolerance for knee-point Pareto selection. Default = "
    "1/n_val (one valset example's worth of disagreement). Override only when "
    "you have a calibrated reason — random tightening narrows the band and "
    "biases selection back toward the GEPA default.",
)
def main(skill, iterations, eval_source, dataset_path, optimizer_model, reflection_model,
         eval_model, hermes_repo, run_tests, dry_run, seed, budget, no_fallback,
         length_penalty_weight, quality_gate, growth_free_threshold,
         growth_quality_slope, max_absolute_chars, bootstrap_confidence,
         bootstrap_resamples, knee_point_epsilon):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization."""
    evolve(
        skill_name=skill,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        reflection_model=reflection_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        run_tests=run_tests,
        dry_run=dry_run,
        seed=seed,
        budget=budget,
        no_fallback=no_fallback,
        length_penalty_weight=length_penalty_weight,
        quality_gate=quality_gate,
        growth_free_threshold=growth_free_threshold,
        growth_quality_slope=growth_quality_slope,
        max_absolute_chars=max_absolute_chars,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_n_resamples=bootstrap_resamples,
        knee_point_epsilon=knee_point_epsilon,
    )


if __name__ == "__main__":
    main()
