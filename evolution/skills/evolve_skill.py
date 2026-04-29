"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import json
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import LLMJudge, make_skill_fitness_metric
from evolution.core.constraints import ConstraintValidator
from evolution.skills.budget_aware_proposer import BudgetAwareProposer
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()


_BUDGET_BY_ITERATIONS = {1: "light", 2: "medium", 3: "heavy"}


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
):
    optimizer = dspy.GEPA(
        metric=metric,
        auto=gepa_budget,
        # cache=False on the reflection LM: at temperature=1.0 the disk
        # cache would replay stale mutations across runs and shrink the
        # diversity of proposed candidates.
        reflection_lm=dspy.LM(
            optimizer_model,
            temperature=1.0,
            max_tokens=4000,
            cache=False,
        ),
        seed=seed,
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
):
    """Main evolution function — orchestrates the full optimization loop."""

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        run_pytest=run_tests,
        seed=seed,
    )
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

    # ── 3. Validate constraints on baseline ─────────────────────────────
    # Validate the full file (frontmatter + body) so the structure check
    # has the markers it needs and the growth comparison stays symmetric
    # with the evolved-side validation below.
    console.print(f"\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    baseline_constraints = validator.validate_all(skill["raw"], "skill")
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
    judge = LLMJudge(config)
    metric = make_skill_fitness_metric(
        judge,
        baseline_skill_text=skill["body"],
        max_growth=config.max_prompt_growth,
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
    proposer = BudgetAwareProposer(
        baseline_chars=len(skill["body"]),
        max_growth=config.max_prompt_growth,
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
    )

    elapsed = time.time() - start_time
    console.print(f"\n  {optimizer_name} optimization completed in {elapsed:.1f}s")

    # ── 6. Extract evolved skill text ───────────────────────────────────
    # The optimized module's instructions contain the evolved skill text
    evolved_body = optimized_module.skill_text
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 7. Validate evolved skill ───────────────────────────────────────
    console.print(f"\n[bold]Validating evolved skill[/bold]")
    evolved_constraints = validator.validate_all(evolved_full, "skill", baseline_text=skill["raw"])
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved skill FAILED constraints — not deploying[/red]")
        # Still save for inspection
        output_path = Path("output") / skill_name / "evolved_FAILED.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_full)
        console.print(f"  Saved failed variant to {output_path}")
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

    baseline_scores = []
    evolved_scores = []
    for ex in holdout_examples:
        # Metric returns dspy.Prediction(score, feedback); aggregate on .score.
        with dspy.context(lm=lm):
            baseline_pred = baseline_module(task_input=ex.task_input)
            baseline_scores.append(float(metric(ex, baseline_pred).score))

            evolved_pred = optimized_module(task_input=ex.task_input)
            evolved_scores.append(float(metric(ex, evolved_pred).score))

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

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
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
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
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model,
         hermes_repo, run_tests, dry_run, seed, budget, no_fallback):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization."""
    evolve(
        skill_name=skill,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        run_tests=run_tests,
        dry_run=dry_run,
        seed=seed,
        budget=budget,
        no_fallback=no_fallback,
    )


if __name__ == "__main__":
    main()
