"""Study D — compare current vs proposed defaults on held-out skills.

Auto-discovers the 12 Stage 7 runs (3 validation skills × 2 conditions
× 2 seeds), groups by the recorded growth_free_threshold and
growth_quality_slope in each gate_decision.json, then prints one of:

  - ACCEPT_PROPOSED — proposed defaults clear one of the doc's two
    acceptance criteria. Stage 8 ships them.

  - KEEP_CURRENT — both conditions ran but the proposed defaults do
    NOT clear either criterion. Defaults untouched; campaign reports
    a no-improvement outcome at n=12 validation runs.

  - NO_SIGNAL — only one condition's runs are present, or the
    conditions are indistinguishable on (free, slope) (e.g. when
    Study C verdict was INSUFFICIENT_DATA and only N*/ratio*/ε*
    differ — those are not visible in gate_decision.json's
    free/slope fields, so the script can't tell the conditions
    apart by signature alone).

Acceptance criteria (from plan, Stage 7):
  1. Same deploy rate, higher mean lift on deployed → ACCEPT (gate is
     more selective).
  2. Higher deploy rate, unchanged-or-better mean lift on deployed,
     no holdout regressions → ACCEPT (gate is less wasteful).

Inputs:
    output/<skill>/<ts>/gate_decision.json (since campaign start)

Output:
    reports/study_d_results.json + verdict printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

# EvolutionConfig.growth_free_threshold default at campaign start. Not
# imported from config so this script keeps working after Stage 8 lands
# (when the field's default will change).
CURRENT_FREE = 0.20
CURRENT_SLOPE = 0.30

# Acceptance thresholds. "Higher / lower" are not evaluated as raw
# floating-point comparisons because runs have natural variance — we
# require the difference to exceed a small absolute tolerance to
# count as signal.
DEPLOY_RATE_TOLERANCE = 0.10  # ±10pp counts as "same"
LIFT_TOLERANCE = 0.005         # ±0.5pp counts as "unchanged"


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
            if "decision" not in gate:
                continue
            yield skill_dir.name, run_dir.name, gate


def _classify_condition(gate: dict[str, Any]) -> str:
    free = gate.get("growth_free_threshold")
    slope = gate.get("growth_quality_slope")
    if free == CURRENT_FREE and slope == CURRENT_SLOPE:
        return "current"
    return f"proposed(free={free},slope={slope})"


def _summarize(condition_runs: list[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
    n = len(condition_runs)
    deployed = [(s, t, g) for s, t, g in condition_runs if g.get("decision") == "deploy"]
    deploy_rate = len(deployed) / n if n else 0.0

    def lift(g: dict[str, Any]) -> float:
        return float(g.get("avg_evolved", 0.0)) - float(g.get("avg_baseline", 0.0))

    lifts_all = [lift(g) for _, _, g in condition_runs]
    lifts_deployed = [lift(g) for _, _, g in deployed]
    regressions = sum(1 for x in lifts_deployed if x < 0)

    return {
        "n": n,
        "deploy_rate": deploy_rate,
        "n_deployed": len(deployed),
        "mean_lift_all": statistics.mean(lifts_all) if lifts_all else 0.0,
        "mean_lift_deployed": statistics.mean(lifts_deployed) if lifts_deployed else 0.0,
        "median_lift_deployed": statistics.median(lifts_deployed) if lifts_deployed else 0.0,
        "deployed_with_regression": regressions,
        "runs": [
            {
                "skill": s, "timestamp": t,
                "decision": g.get("decision"), "lift": lift(g),
                "growth_pct": g.get("growth_pct"),
            }
            for s, t, g in condition_runs
        ],
    }


def _verdict(current: dict[str, Any], proposed: dict[str, Any]) -> tuple[str, str]:
    """Return (verdict, rationale)."""
    if current["n"] == 0 or proposed["n"] == 0:
        return "NO_SIGNAL", f"current_n={current['n']}, proposed_n={proposed['n']} — need both arms populated"

    deploy_delta = proposed["deploy_rate"] - current["deploy_rate"]
    lift_delta = proposed["mean_lift_deployed"] - current["mean_lift_deployed"]

    same_deploy_rate = abs(deploy_delta) <= DEPLOY_RATE_TOLERANCE
    higher_deploy_rate = deploy_delta > DEPLOY_RATE_TOLERANCE
    higher_lift = lift_delta > LIFT_TOLERANCE
    unchanged_or_better_lift = lift_delta >= -LIFT_TOLERANCE
    no_regressions = proposed["deployed_with_regression"] == 0

    # Criterion 1: same deploy rate, higher mean lift on deployed
    if same_deploy_rate and higher_lift:
        return (
            "ACCEPT_PROPOSED",
            f"criterion 1: deploy rate held within ±{DEPLOY_RATE_TOLERANCE} "
            f"({deploy_delta:+.3f}), mean lift on deployed rose by {lift_delta:+.4f}"
        )

    # Criterion 2: higher deploy rate, unchanged-or-better lift, no regressions
    if higher_deploy_rate and unchanged_or_better_lift and no_regressions:
        return (
            "ACCEPT_PROPOSED",
            f"criterion 2: deploy rate rose by {deploy_delta:+.3f}, lift "
            f"unchanged within ±{LIFT_TOLERANCE} ({lift_delta:+.4f}), "
            "no deployed variant regressed"
        )

    return (
        "KEEP_CURRENT",
        f"deploy rate {deploy_delta:+.3f}, mean-lift-deployed {lift_delta:+.4f}, "
        f"deployed regressions={proposed['deployed_with_regression']} — neither criterion met"
    )


def _print_table(current: dict[str, Any], proposed: dict[str, Any]):
    print()
    print(f"  {'Metric':<32} {'Current':>15} {'Proposed':>15}")
    print(f"  {'-'*32} {'-'*15} {'-'*15}")
    print(f"  {'n':<32} {current['n']:>15} {proposed['n']:>15}")
    print(f"  {'deploy_rate':<32} {current['deploy_rate']:>15.3f} {proposed['deploy_rate']:>15.3f}")
    print(f"  {'n_deployed':<32} {current['n_deployed']:>15} {proposed['n_deployed']:>15}")
    print(f"  {'mean_lift_all':<32} {current['mean_lift_all']:>15.4f} {proposed['mean_lift_all']:>15.4f}")
    print(f"  {'mean_lift_deployed':<32} {current['mean_lift_deployed']:>15.4f} {proposed['mean_lift_deployed']:>15.4f}")
    print(f"  {'deployed_with_regression':<32} {current['deployed_with_regression']:>15} {proposed['deployed_with_regression']:>15}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--since", default=None,
        help="Filter to runs whose timestamp directory >= this prefix",
    )
    args = parser.parse_args()

    runs = list(_load_runs(Path(args.output_root), args.since))
    if not runs:
        print(
            f"✗ No runs under {args.output_root}/. "
            "Study D requires Stage 7 runs from the campaign."
        )
        return

    by_condition: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    for skill, ts, gate in runs:
        cond = _classify_condition(gate)
        by_condition.setdefault(cond, []).append((skill, ts, gate))

    if "current" not in by_condition:
        print("✗ No 'current' condition runs found "
              f"(expected free={CURRENT_FREE}, slope={CURRENT_SLOPE}). "
              "Cannot compute Study D verdict.")
        return

    proposed_keys = [k for k in by_condition if k != "current"]
    if len(proposed_keys) == 0:
        print("✗ Only 'current' condition runs found — Study D needs both arms.")
        return
    if len(proposed_keys) > 1:
        print(f"⚠ Multiple proposed conditions found: {proposed_keys}. "
              "Picking the one with the most runs.")
        proposed_keys.sort(key=lambda k: -len(by_condition[k]))
    proposed_key = proposed_keys[0]

    current = _summarize(by_condition["current"])
    proposed = _summarize(by_condition[proposed_key])
    verdict, rationale = _verdict(current, proposed)

    payload = {
        "campaign_since": args.since,
        "current_signature": {"free": CURRENT_FREE, "slope": CURRENT_SLOPE},
        "proposed_signature": proposed_key,
        "current": current,
        "proposed": proposed,
        "deploy_rate_tolerance": DEPLOY_RATE_TOLERANCE,
        "lift_tolerance": LIFT_TOLERANCE,
        "verdict": verdict,
        "rationale": rationale,
    }
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "study_d_results.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"  Loaded {len(runs)} run(s); current arm n={current['n']}, "
          f"proposed arm n={proposed['n']}")
    _print_table(current, proposed)
    print()
    print(f"  Verdict: {verdict}")
    print(f"  Rationale: {rationale}")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
