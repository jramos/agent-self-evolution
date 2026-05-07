"""Study B — pick knee-point ε.

Replays each Study C run's band against ε ∈ {0.5, 1, 2, 3}/n_val:
simulate the val-best pick, score the picked candidate's val→holdout
transfer error, and simulate the gate's deploy/reject decision.

Pick ε that minimizes mean transfer error across runs, subject to
deploy-rejection rate within ±10% of the rate observed at ε=1/n_val.
Tiebreaker: smaller ε (closer to GEPA's own default → lower variance).

Inputs (per Study C run):
    output/<skill>/<ts>/gate_decision.json    — baseline scores, growth params
    output/<skill>/<ts>/band_holdout.json     — candidate val/holdout scores

Output:
    reports/study_b_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from evolution.core.stats import paired_bootstrap

EPSILON_MULTIPLIERS = (0.5, 1.0, 2.0, 3.0)
DEPLOY_RATE_TOLERANCE = 0.10  # ±10% of the 1/n_val rate
BASELINE_MULTIPLIER = 1.0


def _val_best_pick(candidates: list[dict[str, Any]], epsilon: float) -> dict[str, Any]:
    """Replicate `select_knee_point` strategy="val-best" inline.

    Band = candidates with val_score >= max - ε. Within the band:
    highest val_score wins, smallest body_chars as tiebreak.
    """
    best_val = max(c["val_score"] for c in candidates)
    band = [c for c in candidates if c["val_score"] >= best_val - epsilon]
    band.sort(key=lambda c: (-c["val_score"], c["body_chars"], c["idx"]))
    return band[0]


def _simulate_gate(
    *,
    baseline_per_example: list[float],
    candidate: dict[str, Any],
    baseline_chars: int,
    growth_free: float,
    growth_slope: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Mirror the no_regression / dual_check gate from constraints.py.

    Skips non_inferiority and inferiority_tolerance — Study C runs use
    --quality-gate lenient, which is no_regression-shaped.
    """
    body_chars = candidate["body_chars"]
    growth_pct = (body_chars - baseline_chars) / baseline_chars if baseline_chars else 0.0
    required = max(0.0, growth_slope * (growth_pct - growth_free))
    bs = paired_bootstrap(
        baseline_per_example,
        candidate["holdout_per_example"],
        seed=bootstrap_seed,
    )
    if required > 0.0:
        deploy = bs["mean"] >= required and bs["lower_bound"] > 0.0
        rule = "dual_check"
    else:
        deploy = bs["mean"] >= 0.0
        rule = "no_regression_only"
    return {
        "rule": rule,
        "growth_pct": growth_pct,
        "required_improvement": required,
        "bootstrap_mean": bs["mean"],
        "bootstrap_lower": bs["lower_bound"],
        "deploy": deploy,
    }


def _evaluate_epsilon(
    runs: list[tuple[str, str, dict[str, Any], dict[str, Any]]],
    multiplier: float,
) -> dict[str, Any]:
    """For ε = multiplier / n_val, simulate the pick + gate per run."""
    transfer_errors: list[float] = []
    deploys = 0
    per_run: list[dict[str, Any]] = []
    for skill, ts, gate, band in runs:
        n_val = band.get("n_val") or len(band["candidates"])
        epsilon = multiplier / max(1, n_val)
        picked = _val_best_pick(band["candidates"], epsilon)
        transfer_error = abs(picked["val_score"] - picked["holdout_score"])
        transfer_errors.append(transfer_error)
        gate_result = _simulate_gate(
            baseline_per_example=gate["baseline_per_example"],
            candidate=picked,
            baseline_chars=gate["baseline_chars"],
            growth_free=gate.get("growth_free_threshold", 0.20),
            growth_slope=gate.get("growth_quality_slope", 0.30),
            bootstrap_seed=hash((skill, ts)) & 0xFFFFFFFF,
        )
        if gate_result["deploy"]:
            deploys += 1
        per_run.append({
            "skill": skill, "timestamp": ts,
            "epsilon": epsilon, "picked_idx": picked["idx"],
            "transfer_error": transfer_error,
            "deploy": gate_result["deploy"],
            "growth_pct": gate_result["growth_pct"],
            "rule": gate_result["rule"],
        })
    n = len(runs)
    return {
        "multiplier": multiplier,
        "mean_transfer_error": statistics.mean(transfer_errors) if transfer_errors else 0.0,
        "median_transfer_error": statistics.median(transfer_errors) if transfer_errors else 0.0,
        "deploy_rate": deploys / n if n else 0.0,
        "n_runs": n,
        "per_run": per_run,
    }


def _pick_optimal(
    grid: list[dict[str, Any]],
    tolerance: float,
) -> dict[str, Any] | None:
    baseline = next((r for r in grid if r["multiplier"] == BASELINE_MULTIPLIER), None)
    if baseline is None:
        return None
    target_rate = baseline["deploy_rate"]
    eligible = [
        r for r in grid
        if abs(r["deploy_rate"] - target_rate) <= tolerance
    ]
    eligible.sort(key=lambda r: (r["mean_transfer_error"], r["multiplier"]))
    return eligible[0] if eligible else None


def _load_runs(output_root: Path):
    """Yield (skill, ts, gate_dict, band_dict) for every run with both
    files present and the band file containing a non-empty candidates
    list."""
    for skill_dir in sorted(output_root.iterdir()):
        if not skill_dir.is_dir():
            continue
        for run_dir in sorted(skill_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            gate_path = run_dir / "gate_decision.json"
            band_path = run_dir / "band_holdout.json"
            if not (gate_path.exists() and band_path.exists()):
                continue
            try:
                gate = json.loads(gate_path.read_text())
                band = json.loads(band_path.read_text())
            except json.JSONDecodeError:
                continue
            if not band.get("candidates"):
                continue
            if not gate.get("baseline_per_example"):
                continue
            yield skill_dir.name, run_dir.name, gate, band


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--deploy-rate-tolerance", type=float, default=DEPLOY_RATE_TOLERANCE,
        help="±this fraction of the 1/n_val deploy rate is acceptable",
    )
    args = parser.parse_args()

    runs = list(_load_runs(Path(args.output_root)))
    if not runs:
        print(
            f"✗ No runs with both gate_decision.json AND band_holdout.json found "
            f"under {args.output_root}/. Study B requires runs launched with "
            f"--evaluate-band-on-holdout."
        )
        return
    print(f"  Loaded {len(runs)} run(s) with band_holdout.json")

    grid = [_evaluate_epsilon(runs, m) for m in EPSILON_MULTIPLIERS]
    optimal = _pick_optimal(grid, args.deploy_rate_tolerance)

    payload = {
        "n_runs_analyzed": len(runs),
        "skills": sorted({r[0] for r in runs}),
        "epsilon_multipliers": list(EPSILON_MULTIPLIERS),
        "deploy_rate_tolerance": args.deploy_rate_tolerance,
        "grid": grid,
        "picked_multiplier": optimal["multiplier"] if optimal else None,
    }
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "study_b_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  Wrote {out_path}")

    if optimal:
        print(
            f"\n  Picked: ε* = {optimal['multiplier']}/n_val "
            f"(mean transfer error {optimal['mean_transfer_error']:.4f}, "
            f"deploy rate {optimal['deploy_rate']:.1%} vs baseline "
            f"{grid[1]['deploy_rate']:.1%})"
        )
        print("  Export for run_campaign.sh:")
        print(f"    # ε* is computed per-run as multiplier / n_val:")
        print(f"    # for n_val ≈ X, ε* ≈ {optimal['multiplier']:.1f}/X")
        print(f"    export EPSILON_MULTIPLIER={optimal['multiplier']}")
    else:
        print("\n  ✗ No ε satisfies the deploy-rate tolerance. Either widen the "
              "tolerance or accept that no ε grid point is robust on this corpus.")


if __name__ == "__main__":
    main()
