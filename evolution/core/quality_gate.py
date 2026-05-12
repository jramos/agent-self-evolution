"""Artifact-agnostic quality-gate helpers extracted from ``evolve_skill``.

These helpers — preset table, proposer-mode resolution, and gate-decision
persistence — were never skill-specific. Hosting them in ``evolution.core``
lets the upcoming tool description pipeline reuse the deploy-gate scaffolding
without depending on the skill pipeline.
"""

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from evolution.core.lm_timing_callback import COST_LEDGER, CostCeilingExceeded
from evolution.skills.budget_aware_proposer import ProposerMode

_console = Console()


# `default` is calibrated against the obsidian deploy (+24.2% growth,
# ~+0.07 expected improvement). `off` disables the slope/ceiling checks
# but still enforces bootstrap.mean ≥ 0 — see deprecation warning when
# users select it. `non-inferiority` is the recommended preset for
# compression-focused runs: it ships variants statistically not-worse-
# than-baseline by more than ``inferiority_tolerance``.
#
# Type widens to ``Any`` because ``gate_mode`` is a string and
# ``inferiority_tolerance`` is a float — no longer a uniform float dict.
QUALITY_GATE_PRESETS: dict[str, dict[str, Any]] = {
    "strict": {
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
        "growth_free_threshold": 0.30,
        "growth_quality_slope": 0.20,
        "max_absolute_chars": 8000,
    },
    "off": {
        "growth_free_threshold": 100.0,
        "growth_quality_slope": 0.0,
        "max_absolute_chars": 100_000,
    },
    "non-inferiority": {
        "growth_free_threshold": 100.0,
        "growth_quality_slope": 0.0,
        "max_absolute_chars": 100_000,
        "gate_mode": "non_inferiority",
        # See reports/calibration_findings.md for the tolerance sweep.
        "inferiority_tolerance": 0.05,
    },
}


def write_gate_decision(output_dir: Path, decision: dict[str, Any]) -> Path:
    """Persist the deploy-gate's structured decision for future calibration.

    Each run writes one of these regardless of outcome (deploy or reject).
    Recalibrating the curve is then `jq -s '...' output/*/*/gate_decision.json`
    rather than parsing free-form failure notes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gate_decision.json"
    path.write_text(json.dumps(decision, indent=2))
    return path


def write_cost_ceiling_abort(
    exc: CostCeilingExceeded,
    *,
    output_dir: Path,
    run_inputs: dict[str, Any],
    extra_fields: dict[str, Any] | None = None,
) -> Path:
    """Write a ``decision="aborted"`` gate_decision for a cost-ceiling trip.

    Both ``evolve_skill`` and ``evolve_tool`` catch ``CostCeilingExceeded``
    at their top level and call this helper. The console message is emitted
    here so both paths render the same way.

    ``extra_fields`` lets the tool path carry ``artifact_type`` /
    ``target_tool`` so downstream calibration scripts can group abort
    rates by surface; the skill path passes ``None``.
    """
    cost_summary = COST_LEDGER.summary()
    _console.print(
        f"\n[bold red]✗ Aborting: cost ${exc.total_usd:.4f} exceeded "
        f"ceiling ${exc.ceiling_usd:.4f}[/bold red]"
    )
    payload: dict[str, Any] = {
        "schema_version": "4",
        "decision": "aborted",
        "reason": "cost_ceiling_exceeded",
        "cost_ceiling_usd": exc.ceiling_usd,
        "cost_at_abort_usd": exc.total_usd,
        "cost_summary": cost_summary,
        "run_inputs": run_inputs,
    }
    if extra_fields:
        payload.update(extra_fields)
    return write_gate_decision(output_dir, payload)


def resolve_proposer_mode(fitness_profile: str) -> ProposerMode:
    """Map the user's fitness profile to a proposer mode.

    Each profile selects its own template: `growth` swings the proposer toward
    additions, `compression` toward cuts, and `balanced` toward direction-agnostic
    revisions. Unknown values fall back to compression-mode defensively.
    """
    if fitness_profile == "growth":
        return "growth"
    if fitness_profile == "balanced":
        return "balanced"
    return "compression"  # compression profile, plus defensive fallback for unknown
