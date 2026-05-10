"""BAP max_growth calibration: pick a value from the campaign sweep.

Walks Stage 2 runs, groups by the recorded `bap_max_growth` field in
gate_decision.json (the resolved proposer target — see the chore
commit that added it), and reports per-group:

  - n_runs : total runs at this max_growth.
  - deploy_rate : fraction of runs whose gate decision was "deploy".
  - n_deployed_real : count of deploy decisions where the evolved
    artifact differs from the baseline (excludes no-op deploys where
    GEPA's knee fell back to candidate 0).
  - mean_lift_on_real_deploys : mean (avg_evolved − avg_baseline)
    restricted to real deploys.
  - no_op_rate : fraction where evolved_chars == baseline_chars (the
    proposer found nothing better than baseline; the gate accepted
    the no-op).
  - median_growth_pct : median growth_pct across all runs at this
    value (negative = compression direction).

Pick rule: the value that **maximizes mean_lift_on_real_deploys**
subject to **no_op_rate < 0.30**. Tiebreak: closer to current 0.20
wins (parsimony). The 30% threshold at n=4 runs/value means the gate
is "1 no-op or fewer" — flag this in the output so the human knows
the statistical-power floor.

Usage:
    uv run python scripts/analysis/bap_max_growth_pick.py
    uv run python scripts/analysis/bap_max_growth_pick.py \\
        --since 20260510_120000 --no-op-cap 0.30

Output: reports/bap_max_growth_results.json + stdout summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

DEFAULT_NO_OP_CAP = 0.30
CURRENT_DEFAULT = 0.20  # tiebreak target


def _load_runs(output_root: Path, since: str | None):
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
            if "bap_max_growth" not in gate:
                # Pre-bap-decoupling run; not part of this campaign.
                continue
            yield skill_dir.name, run_dir.name, gate


def _summarize(runs: list[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
    n = len(runs)
    if n == 0:
        return {"n_runs": 0}
    deploys = [g for _, _, g in runs if g.get("decision") == "deploy"]
    no_ops = [g for g in deploys if g.get("evolved_chars") == g.get("baseline_chars")]
    real_deploys = [g for g in deploys if g.get("evolved_chars") != g.get("baseline_chars")]
    lifts = [
        float(g.get("avg_evolved", 0.0)) - float(g.get("avg_baseline", 0.0))
        for g in real_deploys
    ]
    growths = [float(g.get("growth_pct", 0.0)) for _, _, g in runs]
    return {
        "n_runs": n,
        "deploy_rate": len(deploys) / n,
        "n_deployed_real": len(real_deploys),
        "no_op_rate": len(no_ops) / n,
        "mean_lift_on_real_deploys": statistics.mean(lifts) if lifts else 0.0,
        "median_growth_pct": statistics.median(growths),
        "runs": [
            {
                "skill": s, "timestamp": t,
                "decision": g.get("decision"),
                "growth_pct": g.get("growth_pct"),
                "lift": float(g.get("avg_evolved", 0.0)) - float(g.get("avg_baseline", 0.0)),
                "no_op": g.get("evolved_chars") == g.get("baseline_chars"),
            }
            for s, t, g in runs
        ],
    }


def _pick(grouped: dict[float, dict[str, Any]], no_op_cap: float) -> dict[str, Any] | None:
    """Pick the max_growth that maximizes mean_lift_on_real_deploys
    subject to no_op_rate < cap. Tiebreak: smaller |value − CURRENT|."""
    eligible = [
        (mg, summary) for mg, summary in grouped.items()
        if summary.get("n_runs", 0) > 0 and summary.get("no_op_rate", 1.0) < no_op_cap
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda x: (
            -x[1]["mean_lift_on_real_deploys"],
            abs(x[0] - CURRENT_DEFAULT),
        ),
    )
    mg, summary = eligible[0]
    return {"max_growth": mg, **summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--since", default=None,
        help="Filter to runs whose timestamp directory >= this prefix",
    )
    parser.add_argument(
        "--no-op-cap", type=float, default=DEFAULT_NO_OP_CAP,
        help="Reject sweep values whose no-op rate ≥ this threshold (default 0.30)",
    )
    args = parser.parse_args()

    runs = list(_load_runs(Path(args.output_root), args.since))
    if not runs:
        print(f"✗ No runs with bap_max_growth recorded under {args.output_root}/")
        return

    by_max_growth: dict[float, list[tuple[str, str, dict[str, Any]]]] = {}
    for skill, ts, gate in runs:
        mg = float(gate["bap_max_growth"])
        by_max_growth.setdefault(mg, []).append((skill, ts, gate))

    grouped = {mg: _summarize(rs) for mg, rs in sorted(by_max_growth.items())}
    picked = _pick(grouped, args.no_op_cap)

    payload = {
        "n_runs_total": len(runs),
        "no_op_cap": args.no_op_cap,
        "current_default": CURRENT_DEFAULT,
        "by_max_growth": {str(mg): s for mg, s in grouped.items()},
        "picked": picked,
    }
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "bap_max_growth_results.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"Loaded {len(runs)} run(s) (since={args.since or 'all-time'})\n")
    print(f"  {'max_growth':>10}  {'n':>3}  {'deploy_rate':>11}  {'n_real':>6}  "
          f"{'no_op_rate':>10}  {'mean_lift':>11}  {'median_growth':>13}")
    print(f"  {'-'*10}  {'-'*3}  {'-'*11}  {'-'*6}  {'-'*10}  {'-'*11}  {'-'*13}")
    for mg in sorted(grouped.keys()):
        s = grouped[mg]
        print(
            f"  {mg:>10.2f}  {s['n_runs']:>3}  {s['deploy_rate']:>11.3f}  "
            f"{s['n_deployed_real']:>6}  {s['no_op_rate']:>10.3f}  "
            f"{s['mean_lift_on_real_deploys']:>+11.4f}  "
            f"{s['median_growth_pct']:>+13.4f}"
        )
    print()
    if picked:
        print(
            f"  Picked: max_growth* = {picked['max_growth']:.2f} "
            f"(mean_lift_on_real_deploys={picked['mean_lift_on_real_deploys']:+.4f}, "
            f"no_op_rate={picked['no_op_rate']:.2f})"
        )
        if picked["max_growth"] == CURRENT_DEFAULT:
            print(f"  Note: pick matches current default ({CURRENT_DEFAULT}) — "
                  "decoupling lands as architectural cleanup; no constant change needed.")
    else:
        print(f"  Picked: NO_CLEAR_PICK — no value satisfies no_op_rate < {args.no_op_cap}.")
        print("  Stage 3 still lands the decoupling at default 0.20 (the architectural "
              "win is independent of the campaign verdict).")
    print()
    print(f"  ⚠ Statistical-power note: at n=4 runs/value with the {args.no_op_cap} no-op")
    print(f"    cap, the eligibility gate is '1 no-op or fewer'. A single noisy run flips")
    print(f"    inclusion. Treat the pick as provisional; Stage 3 is the validation gate.")
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
