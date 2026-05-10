"""Option 1 — gate-rule replay: no_regression vs non_inferiority.

Re-evaluates every gate_decision.json under both rules without re-running
GEPA. Pure stat replay, zero LLM calls. Tests whether the non-inferiority
gate (Decagon-style: accept when bootstrap.lower_bound ≥ -tolerance)
better matches the campaign's compression-direction behavior than the
current `no_regression_only` rule (accept when bootstrap.mean ≥ 0.0).

For runs that hit dual_check (growth_pct > free, required > 0), the
rule is not applicable — non-inferiority is a no_regression alternative,
not a dual_check alternative. Those runs are reported as
"rule_not_applicable" and their decisions don't change.

Output classification per run:
  - same          : both rules give same deploy/reject decision
  - flip_to_deploy: no_regression rejected, non_inferiority would accept
                    (the noise-level rejections we want to recover)
  - flip_to_reject: no_regression accepted, non_inferiority would reject
                    (potential downside — should be rare and only when
                    lower_bound < -tolerance but mean ≥ 0)
  - rule_n/a       : dual_check fired; gate-rule choice doesn't matter
  - static_fail   : static constraint (size ceiling, etc.) blocked the
                    run before the bootstrap rule fired

Usage:
    uv run python scripts/analysis/option1_replay_gate_rule.py
    uv run python scripts/analysis/option1_replay_gate_rule.py \\
        --tolerance 0.02 --since 20260507_000000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
            yield skill_dir.name, run_dir.name, gate


def _replay(gate: dict[str, Any], tolerance: float) -> dict[str, Any]:
    """Replay both gate rules on a single run's recorded bootstrap."""
    decision_rule = gate.get("decision_rule_used", "no_regression_only")
    bootstrap = gate.get("bootstrap") or {}
    mean = bootstrap.get("mean")
    lower = bootstrap.get("lower_bound")
    static_failed = (
        gate.get("reason") == "static_constraint_failure"
        or "absolute_char_ceiling" in (gate.get("failed_constraints") or [])
    )

    actual_decision = gate.get("decision")
    if static_failed:
        return {
            "actual_decision": actual_decision,
            "no_regression_would": "reject",
            "non_inferiority_would": "reject",
            "classification": "static_fail",
            "decision_rule": decision_rule,
            "mean": mean,
            "lower": lower,
        }
    if decision_rule == "dual_check":
        return {
            "actual_decision": actual_decision,
            "no_regression_would": None,
            "non_inferiority_would": None,
            "classification": "rule_n/a",
            "decision_rule": decision_rule,
            "mean": mean,
            "lower": lower,
        }
    if mean is None or lower is None:
        return {
            "actual_decision": actual_decision,
            "no_regression_would": None,
            "non_inferiority_would": None,
            "classification": "missing_bootstrap",
            "decision_rule": decision_rule,
            "mean": mean,
            "lower": lower,
        }

    no_reg = "deploy" if mean >= 0.0 else "reject"
    non_inf = "deploy" if lower >= -tolerance else "reject"

    if no_reg == non_inf:
        classification = "same"
    elif no_reg == "reject" and non_inf == "deploy":
        classification = "flip_to_deploy"
    elif no_reg == "deploy" and non_inf == "reject":
        classification = "flip_to_reject"
    else:
        classification = "unexpected"

    return {
        "actual_decision": actual_decision,
        "no_regression_would": no_reg,
        "non_inferiority_would": non_inf,
        "classification": classification,
        "decision_rule": decision_rule,
        "mean": mean,
        "lower": lower,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--tolerance", type=float, default=0.02,
        help="Non-inferiority tolerance: deploy when lower_bound ≥ -tolerance",
    )
    parser.add_argument(
        "--since", default=None,
        help="Filter to runs whose timestamp >= this prefix",
    )
    args = parser.parse_args()

    runs = list(_load_runs(Path(args.output_root), args.since))
    if not runs:
        print(f"✗ No runs under {args.output_root}/")
        return

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {
        "same": 0, "flip_to_deploy": 0, "flip_to_reject": 0,
        "rule_n/a": 0, "static_fail": 0, "missing_bootstrap": 0,
        "unexpected": 0,
    }
    for skill, ts, gate in runs:
        replay = _replay(gate, args.tolerance)
        row = {"skill": skill, "timestamp": ts, **replay}
        rows.append(row)
        counts[replay["classification"]] = counts.get(replay["classification"], 0) + 1

    print(f"Loaded {len(runs)} runs (since={args.since or 'all-time'}, tolerance=±{args.tolerance})\n")
    print(f"  {'Classification':<22} {'Count':>6}")
    print(f"  {'-'*22} {'-'*6}")
    for cls in ("same", "flip_to_deploy", "flip_to_reject", "rule_n/a", "static_fail", "missing_bootstrap", "unexpected"):
        if counts.get(cls, 0) > 0:
            print(f"  {cls:<22} {counts[cls]:>6}")
    print()

    flips_to_deploy = [r for r in rows if r["classification"] == "flip_to_deploy"]
    flips_to_reject = [r for r in rows if r["classification"] == "flip_to_reject"]

    if flips_to_deploy:
        print("FLIPS TO DEPLOY (rejected by no_regression, would deploy under non-inferiority):")
        print(f"  {'skill/timestamp':<46} {'mean':>8} {'lower':>8}  {'rule':<22}")
        print(f"  {'-'*46} {'-'*8} {'-'*8}  {'-'*22}")
        for r in flips_to_deploy:
            print(f"  {r['skill']+'/'+r['timestamp']:<46} {r['mean']:>+8.4f} {r['lower']:>+8.4f}  {r['decision_rule']:<22}")
        print()

    if flips_to_reject:
        print("FLIPS TO REJECT (deployed by no_regression, would reject under non-inferiority):")
        print(f"  {'skill/timestamp':<46} {'mean':>8} {'lower':>8}  {'rule':<22}")
        print(f"  {'-'*46} {'-'*8} {'-'*8}  {'-'*22}")
        for r in flips_to_reject:
            print(f"  {r['skill']+'/'+r['timestamp']:<46} {r['mean']:>+8.4f} {r['lower']:>+8.4f}  {r['decision_rule']:<22}")
        print()

    payload = {
        "tolerance": args.tolerance,
        "since": args.since,
        "n_runs": len(runs),
        "counts": counts,
        "runs": rows,
    }
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "option1_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")

    # Bottom-line interpretation
    n_applicable = counts["same"] + counts["flip_to_deploy"] + counts["flip_to_reject"]
    if n_applicable == 0:
        print("\n  No runs in the no_regression_only path — non-inferiority comparison "
              "is moot for this corpus.")
        return
    flip_d_pct = counts["flip_to_deploy"] / n_applicable * 100
    flip_r_pct = counts["flip_to_reject"] / n_applicable * 100
    print(f"\n  Among the {n_applicable} runs where the gate-rule choice matters:")
    print(f"    {flip_d_pct:.0f}% would flip rejected → deployed under non-inferiority")
    print(f"    {flip_r_pct:.0f}% would flip deployed → rejected under non-inferiority")
    if counts["flip_to_deploy"] > 0 and counts["flip_to_reject"] == 0:
        print("\n  Recommendation: non-inferiority strictly dominates no-regression on "
              "this corpus. Worth shipping for compression-bias evaluation.")
    elif counts["flip_to_deploy"] == 0 and counts["flip_to_reject"] == 0:
        print("\n  Recommendation: no-regression and non-inferiority agree on every "
              "run. The rule choice doesn't matter on this corpus.")
    elif counts["flip_to_deploy"] == 0 and counts["flip_to_reject"] > 0:
        print("\n  Recommendation: non-inferiority is strictly more conservative than "
              "no-regression at this tolerance — would reject runs the current rule "
              "accepts. Tolerance is set too tight; loosen it or stick with no-regression.")
    else:
        print("\n  Recommendation: trade-off — review the flip-to-reject cases manually "
              "to see whether the new rejections are justified.")


if __name__ == "__main__":
    main()
