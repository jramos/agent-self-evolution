"""Configuration and hermes-agent repo discovery."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvolutionConfig:
    """Configuration for a self-evolution optimization run."""

    # hermes-agent repo path
    hermes_agent_path: Path = field(default_factory=lambda: get_hermes_agent_path())

    # Optimization parameters
    iterations: int = 10
    population_size: int = 5

    # LLM configuration
    optimizer_model: str = "openai/gpt-4.1"  # Default for the eval LM bound to dspy.configure
    # Reflection LM (the model GEPA's instruction proposer calls). Defaults
    # to None → falls back to optimizer_model. Separated because reflection
    # is the only place where model strength meaningfully changes outcomes
    # (Decagon's blog: gpt-4o-mini "failed completely" as reflection LM).
    # CLI default in evolve_skill.py overrides this to gpt-5-mini.
    reflection_model: Optional[str] = None
    eval_model: str = "openai/gpt-4.1-mini"  # Model for LLM-as-judge scoring
    judge_model: str = "openai/gpt-4.1"  # Model for dataset generation
    # Forward-wired for an upcoming custom-DspyAdapter PR that adds a
    # score-side λ-penalty for instruction length. Currently unread by
    # the metric; non-zero values are ignored until that PR lands.
    length_penalty_weight: float = 0.0

    # Constraints
    max_skill_size: int = 15_000  # 15KB hard ceiling on absolute size (deployment-cost backstop)
    max_tool_desc_size: int = 500  # chars
    max_param_desc_size: int = 200  # chars
    # Legacy field consulted only by the back-compat ConstraintValidator.validate_all()
    # path that lacks holdout_improvement context. The primary deploy gate is now
    # the continuous quality-gated curve (growth_free_threshold + growth_quality_slope
    # below) consumed by validate_growth_with_quality(...).
    max_prompt_growth: float = 0.30
    # Continuous quality-gated growth curve: required holdout improvement to
    # justify any growth above growth_free_threshold scales linearly with slope.
    #     min_improvement(growth) = max(0, slope * (growth - growth_free_threshold))
    # Defaults (growth_free=0.20, slope=0.30) are calibrated against PR #5's
    # obsidian run (+24.2% growth, ~+0.07 expected holdout improvement); see the
    # full doc block above growth_free_threshold for rationale and recalibration
    # procedure once we have ≥10 deployed skills' (growth, improvement) pairs.
    # Continuous (vs. tiered step function) chosen because the prompt-optimization
    # literature has no precedent for tiered deploy gates and tiers create
    # boundary-gaming pathology (grow to 49.9% to dodge the next tier).
    growth_free_threshold: float = 0.20
    growth_quality_slope: float = 0.30
    # Hard char ceiling on the evolved artifact, applied independently of
    # the growth curve. Escape hatch for short baselines that legitimately
    # need expansion (a 200-char baseline that grows to 1500 to be useful
    # is +650% growth, well past any sane curve, but only 1500 chars
    # absolute — fine to deploy).
    max_absolute_chars: int = 5000

    # Eval dataset
    eval_dataset_size: int = 20  # Total examples to generate
    train_ratio: float = 0.5
    val_ratio: float = 0.25
    # Bumped 0.25 → 0.40 in PR #6 to mitigate sampling noise on the holdout
    # improvement delta consumed by the quality gate. With eval_dataset_size=20
    # this gives 8 holdout examples (was 5); judge-call cost ~+$0.005/run.
    holdout_ratio: float = 0.40
    # Refuse to gate on a holdout smaller than this — the improvement delta
    # is too noisy to trust for deploy decisions. Raise eval_dataset_size or
    # holdout_ratio rather than override this.
    min_holdout_size: int = 6

    # Benchmark gating
    run_pytest: bool = True
    run_tblite: bool = False  # Expensive — opt-in
    tblite_regression_threshold: float = 0.02  # Max 2% regression allowed

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    create_pr: bool = True

    # Seeds dataset shuffles and is forwarded as GEPA/MIPROv2 seed=.
    seed: int = 42


def get_hermes_agent_path() -> Path:
    """Discover the hermes-agent repo path.

    Priority:
    1. HERMES_AGENT_REPO env var
    2. ~/.hermes/hermes-agent (standard install location)
    3. ../hermes-agent (sibling directory)
    """
    env_path = os.getenv("HERMES_AGENT_REPO")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    home_path = Path.home() / ".hermes" / "hermes-agent"
    if home_path.exists():
        return home_path

    sibling_path = Path(__file__).parent.parent.parent / "hermes-agent"
    if sibling_path.exists():
        return sibling_path

    raise FileNotFoundError(
        "Cannot find hermes-agent repo. Set HERMES_AGENT_REPO env var "
        "or ensure it exists at ~/.hermes/hermes-agent"
    )
