"""Study A — pick eval_dataset_size N* and holdout_ratio*.

Re-evaluation only, no GEPA. For each historical run with per-example
holdout scores, subsample at every (N, ratio) grid point, recompute
the paired bootstrap, and pick the smallest (N*, ratio*) pair where
the median CI half-width across runs is ≤ TARGET_HALF_WIDTH.

The picked pair must satisfy `N* × ratio* ≥ 20` so that Stage 6's sub-
holdout reservation (30% of holdout, minimum 10) leaves ≥10 examples
for the bootstrap. Pairs failing this are pre-filtered.

Usage:
    uv run python scripts/analysis/study_a_pick_n.py
    uv run python scripts/analysis/study_a_pick_n.py --output-root output \\
        --target-half-width 0.025 --plot

Inputs:
    output/<skill>/<ts>/gate_decision.json — must contain
    `evolved_per_example` and `baseline_per_example` arrays. Runs without
    these (e.g. static-failure rejects) are skipped.

Outputs:
    reports/study_a_results.json
    reports/study_a_ci_vs_n.png   (when --plot)
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from evolution.core.stats import paired_bootstrap

N_GRID = (50, 100, 150, 250, 400)
RATIO_GRID = (0.36, 0.50, 0.65)
TARGET_HALF_WIDTH = 0.025
MIN_N_HOLDOUT_FOR_C = 20  # n_holdout − reservation(30% min 10) ≥ 10
SUBSAMPLE_SEEDS = (42, 7, 13)  # average over a few subsample seeds to dampen noise


def _load_runs(output_root: Path):
    """Yield (skill, timestamp, baseline_per_example, evolved_per_example)
    for every run with non-empty paired score arrays."""
    for skill_dir in sorted(output_root.iterdir()):
        if not skill_dir.is_dir():
            continue
        for run_dir in sorted(skill_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            gate_path = run_dir / "gate_decision.json"
            if not gate_path.exists():
                continue
            try:
                payload = json.loads(gate_path.read_text())
            except json.JSONDecodeError:
                continue
            baseline = payload.get("baseline_per_example") or []
            evolved = payload.get("evolved_per_example") or []
            if len(baseline) != len(evolved) or len(baseline) == 0:
                continue
            yield skill_dir.name, run_dir.name, baseline, evolved


def _half_width(bootstrap: dict[str, Any]) -> float:
    return (bootstrap["upper_bound"] - bootstrap["lower_bound"]) / 2.0


def _evaluate_pair(
    runs: list[tuple[str, str, list[float], list[float]]],
    n: int,
    ratio: float,
) -> dict[str, Any]:
    """Compute median CI half-width across runs at (n, ratio).

    n_holdout = round(n × ratio). For each run we subsample (with
    replacement, since the historical holdout may be smaller than
    n_holdout) and average the half-width over SUBSAMPLE_SEEDS seeds
    to dampen subsampling noise.
    """
    n_holdout = max(1, round(n * ratio))
    half_widths: list[float] = []
    for _, _, baseline, evolved in runs:
        per_seed: list[float] = []
        for seed in SUBSAMPLE_SEEDS:
            rng = np.random.default_rng(seed)
            available = len(baseline)
            # Sample with replacement when n_holdout > available so the
            # grid extrapolates beyond what the historical run actually
            # measured. The bootstrap CI from a small underlying sample
            # is wider, which is the correct signal — picking a large N
            # against a small underlying sample doesn't tighten the CI.
            replace = n_holdout > available
            indices = rng.choice(available, size=n_holdout, replace=replace)
            sub_baseline = [baseline[i] for i in indices]
            sub_evolved = [evolved[i] for i in indices]
            bs = paired_bootstrap(sub_baseline, sub_evolved, seed=seed)
            per_seed.append(_half_width(bs))
        half_widths.append(statistics.mean(per_seed))
    return {
        "n": n,
        "ratio": ratio,
        "n_holdout": n_holdout,
        "median_half_width": statistics.median(half_widths),
        "mean_half_width": statistics.mean(half_widths),
        "per_run_half_widths": half_widths,
        "satisfies_min_holdout": n * ratio >= MIN_N_HOLDOUT_FOR_C,
    }


def _pick_optimal(
    grid_results: list[dict[str, Any]],
    target: float,
) -> dict[str, Any] | None:
    """Smallest N satisfying the half-width target AND the Study C
    holdout-floor constraint. Tiebreak among same-N: smaller ratio."""
    eligible = [
        r for r in grid_results
        if r["satisfies_min_holdout"] and r["median_half_width"] <= target
    ]
    if not eligible:
        return None
    eligible.sort(key=lambda r: (r["n"], r["ratio"]))
    return eligible[0]


def _plot(grid_results: list[dict[str, Any]], target: float, out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot. Install with: "
              "uv pip install -e '.[analysis]'")
        return
    by_ratio: dict[float, list[tuple[int, float]]] = {}
    for r in grid_results:
        by_ratio.setdefault(r["ratio"], []).append((r["n"], r["median_half_width"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    for ratio in sorted(by_ratio.keys()):
        pts = sorted(by_ratio[ratio])
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker="o",
            label=f"holdout_ratio = {ratio}",
        )
    ax.axhline(y=target, linestyle="--", color="gray", label=f"target = {target}")
    ax.set_xlabel("eval_dataset_size (N)")
    ax.set_ylabel("median bootstrap CI half-width across runs")
    ax.set_title("Study A — CI half-width vs N at three holdout ratios")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--target-half-width", type=float, default=TARGET_HALF_WIDTH)
    parser.add_argument("--plot", action="store_true", help="Emit study_a_ci_vs_n.png")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    runs = list(_load_runs(output_root))
    if not runs:
        print(f"✗ No usable runs found under {output_root}/")
        return
    print(f"  Loaded {len(runs)} run(s) with paired per-example scores")

    grid_results = [
        _evaluate_pair(runs, n=n, ratio=ratio)
        for n in N_GRID
        for ratio in RATIO_GRID
    ]
    optimal = _pick_optimal(grid_results, args.target_half_width)

    payload = {
        "n_runs_analyzed": len(runs),
        "skills": sorted({r[0] for r in runs}),
        "target_half_width": args.target_half_width,
        "min_n_holdout_for_study_c": MIN_N_HOLDOUT_FOR_C,
        "grid": grid_results,
        "picked": optimal,
    }
    out_path = reports_dir / "study_a_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  Wrote {out_path}")

    if optimal:
        print(
            f"\n  Picked: N* = {optimal['n']}, ratio* = {optimal['ratio']} "
            f"(median half-width {optimal['median_half_width']:.4f}, "
            f"n_holdout = {optimal['n_holdout']})"
        )
        print("  Export for run_campaign.sh:")
        print(f"    export N_STAR={optimal['n']}")
        print(f"    export RATIO_STAR={optimal['ratio']}")
    else:
        print(
            f"\n  ✗ No (N, ratio) in the grid satisfies both target half-width "
            f"({args.target_half_width}) and min-holdout-for-Study-C "
            f"({MIN_N_HOLDOUT_FOR_C}). Either expand the grid or relax the target."
        )

    if args.plot:
        _plot(grid_results, args.target_half_width, reports_dir / "study_a_ci_vs_n.png")


if __name__ == "__main__":
    main()
