"""Study C — pick (growth_free_threshold*, growth_quality_slope*).

For each growth run, reserve 30% of the recorded paired holdout
(minimum 10 examples) as ground truth — does the variant *generalize*?
The other 70% is what the gate would have actually seen during the
run; we bootstrap that to predict the gate's deploy/reject decision
under each (free, slope) candidate pair.

Pick the pair maximizing Youden's J = TPR − FPR. With n_runs ≤ 8 the
J distribution is coarse, so:
  - Two-tier tiebreaker: smallest `free` first (parsimony, doc-
    specified), then largest `slope` at that `free` (bias toward
    conservative gate when `free` ties).
  - Report ALL tied pairs in the JSON output for human review.
  - 1000-iter paired bootstrap CI on J. If the lower bound < 0 we
    can't reject random performance — verdict is INSUFFICIENT_DATA
    and (free, slope) is dropped from the campaign per the doc's
    partial-outcome plan.

Inputs:
    output/<skill>/<ts>/gate_decision.json — baseline_per_example,
    evolved_per_example, baseline_chars (for growth_pct).

Outputs:
    reports/study_c_results.json
    reports/study_c_roc.png   (when --plot)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evolution.core.stats import paired_bootstrap

FREE_GRID = (0.10, 0.15, 0.20, 0.25, 0.30)
SLOPE_GRID = (0.15, 0.20, 0.30, 0.40, 0.50)
RESERVATION_FRACTION = 0.30
MIN_RESERVATION = 10
MIN_GATE_HOLDOUT = 10  # leftover after reservation must be ≥ this
J_BOOTSTRAP_ITERATIONS = 1000
J_BOOTSTRAP_CONFIDENCE = 0.90
DEFAULT_GROWTH_RUN_FILTER = ("lenient",)  # Study C uses --quality-gate lenient


def _split_paired(
    baseline: list[float],
    evolved: list[float],
    seed: int,
    fraction: float,
    minimum: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Reserve `max(minimum, ceil(fraction × n))` paired examples by
    deterministic seed. Returns (gate_b, gate_e, reserved_b, reserved_e)."""
    n = len(baseline)
    reservation_size = max(minimum, math.ceil(fraction * n))
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    reserved_idx = sorted(indices[:reservation_size].tolist())
    gate_idx = sorted(indices[reservation_size:].tolist())
    return (
        [baseline[i] for i in gate_idx],
        [evolved[i] for i in gate_idx],
        [baseline[i] for i in reserved_idx],
        [evolved[i] for i in reserved_idx],
    )


def _predict_deploy(
    *,
    gate_baseline: list[float],
    gate_evolved: list[float],
    growth_pct: float,
    free: float,
    slope: float,
    seed: int,
) -> bool:
    """`(bootstrap.mean ≥ slope·(growth−free)) ∧ (bootstrap.lower > 0)`
    matches `_check_growth_with_quality_gate`'s dual_check branch.
    When `required` is 0 we fall back to `mean ≥ 0` (no_regression)."""
    bs = paired_bootstrap(gate_baseline, gate_evolved, seed=seed)
    required = max(0.0, slope * (growth_pct - free))
    if required > 0.0:
        return bs["mean"] >= required and bs["lower_bound"] > 0.0
    return bs["mean"] >= 0.0


def _confusion(predictions: list[bool], ground_truth: list[bool]) -> dict[str, Any]:
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
    pos = tp + fn
    neg = fp + tn
    tpr = tp / pos if pos > 0 else 0.0
    fpr = fp / neg if neg > 0 else 0.0
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "tpr": tpr, "fpr": fpr, "j": tpr - fpr,
    }


def _evaluate_pair(
    runs: list[dict[str, Any]],
    free: float,
    slope: float,
) -> dict[str, Any]:
    predictions: list[bool] = []
    ground_truth: list[bool] = []
    for run in runs:
        deployed = _predict_deploy(
            gate_baseline=run["gate_baseline"],
            gate_evolved=run["gate_evolved"],
            growth_pct=run["growth_pct"],
            free=free,
            slope=slope,
            seed=run["seed"],
        )
        predictions.append(deployed)
        ground_truth.append(run["generalized"])
    cm = _confusion(predictions, ground_truth)
    return {"free": free, "slope": slope, **cm}


def _bootstrap_j(
    runs: list[dict[str, Any]],
    free: float,
    slope: float,
    n_iter: int,
    confidence: float,
) -> dict[str, float]:
    """Resample run indices with replacement, recompute J each iter."""
    rng = np.random.default_rng(seed=hash((free, slope)) & 0xFFFFFFFF)
    n = len(runs)
    js: list[float] = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        sub = [runs[i] for i in idx]
        result = _evaluate_pair(sub, free, slope)
        js.append(result["j"])
    alpha = (1.0 - confidence) / 2.0
    return {
        "j_mean": float(np.mean(js)),
        "j_lower": float(np.quantile(js, alpha)),
        "j_upper": float(np.quantile(js, 1.0 - alpha)),
        "confidence": confidence,
    }


def _pick_with_tiebreakers(grid: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    max_j = max(r["j"] for r in grid)
    tied = [r for r in grid if r["j"] == max_j]
    # Tier 1: smallest free. Tier 2: largest slope at that free.
    tied.sort(key=lambda r: (r["free"], -r["slope"]))
    return tied[0], tied


def _load_growth_runs(output_root: Path, since: str | None) -> list[dict[str, Any]]:
    """Walk Study C runs (gate_decision.json with growth_pct > 0).

    Pulls baseline_per_example, evolved_per_example, baseline_chars,
    evolved_chars from gate_decision.json. Skips runs whose holdout is
    too small for the reservation + gate-bootstrap split.
    """
    runs: list[dict[str, Any]] = []
    for skill_dir in sorted(output_root.iterdir()):
        if not skill_dir.is_dir():
            continue
        for run_dir in sorted(skill_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if since and run_dir.name < since:
                continue
            gate_path = run_dir / "gate_decision.json"
            if not gate_path.exists():
                continue
            try:
                gate = json.loads(gate_path.read_text())
            except json.JSONDecodeError:
                continue
            growth_pct = gate.get("growth_pct")
            baseline = gate.get("baseline_per_example") or []
            evolved = gate.get("evolved_per_example") or []
            if growth_pct is None or growth_pct <= 0:
                continue
            if len(baseline) != len(evolved) or len(baseline) == 0:
                continue
            n = len(baseline)
            reservation_size = max(MIN_RESERVATION, math.ceil(RESERVATION_FRACTION * n))
            if n - reservation_size < MIN_GATE_HOLDOUT:
                print(
                    f"  ⚠ {skill_dir.name}/{run_dir.name}: n_holdout={n} too small "
                    f"for {reservation_size} reservation + {MIN_GATE_HOLDOUT} gate floor — skipped"
                )
                continue
            seed = hash((skill_dir.name, run_dir.name)) & 0xFFFFFFFF
            gate_b, gate_e, res_b, res_e = _split_paired(
                baseline, evolved, seed, RESERVATION_FRACTION, MIN_RESERVATION,
            )
            generalized = (sum(res_e) / len(res_e)) > (sum(res_b) / len(res_b))
            runs.append({
                "skill": skill_dir.name,
                "timestamp": run_dir.name,
                "growth_pct": growth_pct,
                "n_holdout": n,
                "reservation_size": reservation_size,
                "gate_baseline": gate_b,
                "gate_evolved": gate_e,
                "generalized": generalized,
                "reserved_baseline_mean": sum(res_b) / len(res_b),
                "reserved_evolved_mean": sum(res_e) / len(res_e),
                "seed": seed,
            })
    return runs


def _plot_roc(grid: list[dict[str, Any]], picked: dict[str, Any], out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot. Install with: "
              "uv pip install -e '.[analysis]'")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter([r["fpr"] for r in grid], [r["tpr"] for r in grid], s=20, alpha=0.6, label="grid points")
    ax.scatter([picked["fpr"]], [picked["tpr"]], s=120, marker="*", color="red",
               label=f"picked free={picked['free']}, slope={picked['slope']}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="random")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("FPR (deploy when did NOT generalize)")
    ax.set_ylabel("TPR (deploy when DID generalize)")
    ax.set_title("Study C — ROC over (free, slope) grid")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--since", default=None,
        help="Only consider runs whose timestamp dir >= this prefix",
    )
    parser.add_argument("--plot", action="store_true", help="Emit study_c_roc.png")
    args = parser.parse_args()

    runs = _load_growth_runs(Path(args.output_root), args.since)
    if not runs:
        print(f"✗ No growth runs (growth_pct > 0) under {args.output_root}/. "
              "Study C requires runs from Stage 5 of the campaign.")
        return
    print(f"  Loaded {len(runs)} growth run(s)")
    n_generalized = sum(1 for r in runs if r["generalized"])
    print(f"  Ground truth: {n_generalized}/{len(runs)} generalized "
          f"(reserved-holdout evolved mean > baseline mean)")

    grid = [_evaluate_pair(runs, free, slope) for free in FREE_GRID for slope in SLOPE_GRID]
    picked, tied = _pick_with_tiebreakers(grid)
    j_ci = _bootstrap_j(runs, picked["free"], picked["slope"],
                        J_BOOTSTRAP_ITERATIONS, J_BOOTSTRAP_CONFIDENCE)

    verdict = "INSUFFICIENT_DATA" if j_ci["j_lower"] < 0 else "ACCEPT"

    payload = {
        "n_runs_analyzed": len(runs),
        "skills": sorted({r["skill"] for r in runs}),
        "ground_truth_positive": n_generalized,
        "ground_truth_negative": len(runs) - n_generalized,
        "reservation_fraction": RESERVATION_FRACTION,
        "min_reservation": MIN_RESERVATION,
        "free_grid": list(FREE_GRID),
        "slope_grid": list(SLOPE_GRID),
        "grid_results": grid,
        "picked": picked,
        "tied_with_picked": tied,
        "j_bootstrap": j_ci,
        "verdict": verdict,
    }
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "study_c_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  Wrote {out_path}")

    print(
        f"\n  Picked: free* = {picked['free']}, slope* = {picked['slope']} "
        f"(J = {picked['j']:.3f}, "
        f"TPR = {picked['tpr']:.2f}, FPR = {picked['fpr']:.2f})"
    )
    print(f"  Bootstrap CI on J at {int(j_ci['confidence']*100)}%: "
          f"[{j_ci['j_lower']:.3f}, {j_ci['j_upper']:.3f}]  →  {verdict}")
    if len(tied) > 1:
        print(f"  ({len(tied)} pairs tied at J = {picked['j']:.3f}; "
              "smallest-free / largest-slope tiebreaker applied)")
    print()
    if verdict == "ACCEPT":
        print("  Export for run_campaign.sh:")
        print(f"    export FREE_STAR={picked['free']}")
        print(f"    export SLOPE_STAR={picked['slope']}")
    else:
        print("  Verdict INSUFFICIENT_DATA: J's lower CI bound < 0 — cannot reject "
              "random performance. Fall back to current defaults for (free, slope) "
              "and ship N*/ratio*/ε* only.")
        print("    export FREE_STAR=KEEP_CURRENT")

    if args.plot:
        _plot_roc(grid, picked, reports_dir / "study_c_roc.png")


if __name__ == "__main__":
    main()
