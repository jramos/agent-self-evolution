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
    max_skill_size: int = 15_000  # absolute deployment-cost backstop (deploy-time)
    max_tool_desc_size: int = 500
    max_param_desc_size: int = 200
    # Continuous quality-gated growth curve consumed by
    # validate_growth_with_quality. Required holdout improvement scales
    # linearly with growth above growth_free_threshold:
    #     required(growth) = max(0, slope * (growth - free))
    # Pass requires both mean improvement ≥ required AND bootstrap lower
    # bound > 0 (no-regression).
    growth_free_threshold: float = 0.20
    growth_quality_slope: float = 0.30
    # Hard absolute char ceiling on the evolved artifact; independent of
    # growth %. Escape hatch for short baselines that legitimately need
    # expansion — a 200-char baseline growing to 1500 is +650% but only
    # 1500 chars absolute.
    max_absolute_chars: int = 5000
    # Bootstrap CI on the per-example holdout improvement vector.
    # Method = basic (reverse percentile), the literature-recommended
    # choice when N is small. BCa is the upgrade path once we routinely
    # see N≥20 holdouts. Lower bound = (1-confidence)/2 percentile of
    # the bootstrap distribution.
    bootstrap_confidence: float = 0.90
    bootstrap_n_resamples: int = 2000

    # Eval dataset
    # Bumped 30→60 + val 0.25→0.40 after the bigger-valset spike on
    # `obsidian` produced the framework's first deploy: bootstrap mean
    # +0.018, knee-point band size 8 (was 2 at N=30). Below N_val≈18 the
    # bootstrap CI swamps real lift and every evolution rejects regardless
    # of the optimizer's actual quality. After builder normalizes ratios
    # summing to 1.40: train ≈ 21, val ≈ 17, holdout ≈ 22. Knee-point ε
    # automatically tightens to 1/17 ≈ 0.059.
    eval_dataset_size: int = 60
    train_ratio: float = 0.5
    val_ratio: float = 0.40
    holdout_ratio: float = 0.50
    # Refuse to gate on a holdout smaller than this — the bootstrap lower
    # bound has very low resolution at tiny N. Raise eval_dataset_size or
    # holdout_ratio rather than override this.
    min_holdout_size: int = 10

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
