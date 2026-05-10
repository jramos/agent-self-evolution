"""Audit knee-point selection across the existing run corpus.

Question this script answers: when knee-point fires (picks a candidate
different from GEPA's default), does it earn its place? Specifically:
  - How often does it fire at all? (pick rate)
  - How many body-chars does it save when it fires? (parsimony win)
  - How much val score does it sacrifice? (val cost)
  - Does the picked candidate generalize better than the default on
    the holdout? (the original justification for knee-point)

The script presents the metrics; whether knee-point earns its place
is a follow-on decision after a human reads the data. No hardcoded
verdict logic — defending arbitrary thresholds in a one-shot analysis
isn't worth the surface.

Usage:
    uv run python scripts/analysis/knee_point_audit.py
    uv run python scripts/analysis/knee_point_audit.py --output-root output \\
        --since 20260507_000000

Inputs:
  - output/<skill>/<ts>/gate_decision.json — required for every run.
    Reads `knee_point` block (picked_idx, gepa_default_idx,
    picked_body_chars, gepa_default_body_chars, picked_val_score,
    band_roster).
  - output/<skill>/<ts>/band_holdout.json — optional; when present,
    the audit also computes holdout-vs-val transfer for picked vs
    default. Without sufficient coverage (≥5 runs AND ≥30%) the
    holdout-transfer block is suppressed with a notice.

Output:
    reports/knee_point_audit.json + a stdout summary table.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

_HOLDOUT_COVERAGE_FLOOR = 5
_HOLDOUT_COVERAGE_FRACTION = 0.30


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
            knee = gate.get("knee_point") or {}
            if not knee.get("applied"):
                continue
            band_path = run_dir / "band_holdout.json"
            band = None
            if band_path.exists():
                try:
                    band = json.loads(band_path.read_text())
                except json.JSONDecodeError:
                    band = None
            yield skill_dir.name, run_dir.name, knee, band


def _best_val_in_band(knee: dict[str, Any]) -> float | None:
    roster = knee.get("band_roster") or []
    if not roster:
        return None
    return max(float(c["val_score"]) for c in roster)


def _holdout_score_for_idx(band: dict[str, Any], idx: int) -> float | None:
    """Pull a candidate's holdout score from band_holdout.json by idx."""
    for c in band.get("candidates") or []:
        if c.get("idx") == idx:
            score = c.get("holdout_score")
            return float(score) if score is not None else None
    return None


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
        print(f"✗ No runs with knee_point.applied=true under {args.output_root}/")
        return

    n_runs = len(runs)
    n_with_band = sum(1 for _, _, _, b in runs if b is not None)
    n_fired = sum(1 for _, _, k, _ in runs if k["picked_idx"] != k["gepa_default_idx"])

    body_savings_when_fired: list[int] = []
    val_sacrifice_when_fired: list[float] = []
    holdout_picked_minus_default: list[float] = []
    n_holdout_comparisons = 0
    n_picked_beat_default = 0

    per_run_rows: list[dict[str, Any]] = []
    for skill, ts, knee, band in runs:
        fired = knee["picked_idx"] != knee["gepa_default_idx"]
        body_delta = None
        val_delta = None
        holdout_delta = None
        if fired:
            body_delta = knee["gepa_default_body_chars"] - knee["picked_body_chars"]
            body_savings_when_fired.append(body_delta)
            best_val = _best_val_in_band(knee)
            if best_val is not None:
                val_delta = best_val - float(knee["picked_val_score"])
                val_sacrifice_when_fired.append(val_delta)
            if band is not None:
                picked_h = _holdout_score_for_idx(band, knee["picked_idx"])
                default_h = _holdout_score_for_idx(band, knee["gepa_default_idx"])
                if picked_h is not None and default_h is not None:
                    holdout_delta = picked_h - default_h
                    holdout_picked_minus_default.append(holdout_delta)
                    n_holdout_comparisons += 1
                    if holdout_delta > 0:
                        n_picked_beat_default += 1
        per_run_rows.append({
            "skill": skill,
            "timestamp": ts,
            "fired": fired,
            "picked_idx": knee["picked_idx"],
            "gepa_default_idx": knee["gepa_default_idx"],
            "body_savings": body_delta,
            "val_sacrifice": val_delta,
            "holdout_picked_minus_default": holdout_delta,
        })

    holdout_coverage_ok = (
        n_with_band >= _HOLDOUT_COVERAGE_FLOOR
        and (n_with_band / n_runs) >= _HOLDOUT_COVERAGE_FRACTION
    )

    summary: dict[str, Any] = {
        "n_runs": n_runs,
        "n_with_band_holdout": n_with_band,
        "pick_rate": n_fired / n_runs if n_runs else 0.0,
        "n_fired": n_fired,
    }
    if body_savings_when_fired:
        summary["body_savings_when_fired"] = {
            "mean": statistics.mean(body_savings_when_fired),
            "median": statistics.median(body_savings_when_fired),
            "n": len(body_savings_when_fired),
        }
    if val_sacrifice_when_fired:
        summary["val_sacrifice_when_fired"] = {
            "mean": statistics.mean(val_sacrifice_when_fired),
            "median": statistics.median(val_sacrifice_when_fired),
            "n": len(val_sacrifice_when_fired),
        }
    if holdout_coverage_ok and holdout_picked_minus_default:
        summary["holdout_transfer"] = {
            "n_comparisons": n_holdout_comparisons,
            "n_picked_beat_default": n_picked_beat_default,
            "fraction_picked_beat_default": (
                n_picked_beat_default / n_holdout_comparisons
            ),
            "mean_picked_minus_default": statistics.mean(holdout_picked_minus_default),
            "median_picked_minus_default": statistics.median(holdout_picked_minus_default),
        }
    else:
        summary["holdout_transfer"] = (
            "insufficient holdout data — re-run with --evaluate-band-on-holdout "
            f"for richer audit (have {n_with_band}/{n_runs} runs with band_holdout.json; "
            f"need ≥{_HOLDOUT_COVERAGE_FLOOR} AND ≥{int(_HOLDOUT_COVERAGE_FRACTION*100)}%)"
        )

    payload: dict[str, Any] = {"summary": summary, "per_run": per_run_rows}
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "knee_point_audit.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"Loaded {n_runs} run(s) with knee_point.applied=true")
    print(f"  band_holdout.json available: {n_with_band}/{n_runs}")
    print()
    print(f"  {'Metric':<40} {'Value':>20}")
    print(f"  {'-'*40} {'-'*20}")
    print(f"  {'pick rate (picked != gepa_default)':<40} {summary['pick_rate']:>20.3f}")
    print(f"  {'fires':<40} {summary['n_fired']:>20}")
    if "body_savings_when_fired" in summary:
        b = summary["body_savings_when_fired"]
        print(f"  {'body chars saved (mean, when fired)':<40} {b['mean']:>20.1f}")
        print(f"  {'body chars saved (median, when fired)':<40} {b['median']:>20.1f}")
    if "val_sacrifice_when_fired" in summary:
        v = summary["val_sacrifice_when_fired"]
        print(f"  {'val score sacrificed (mean)':<40} {v['mean']:>20.4f}")
        print(f"  {'val score sacrificed (median)':<40} {v['median']:>20.4f}")
    if isinstance(summary["holdout_transfer"], dict):
        h = summary["holdout_transfer"]
        print(f"  {'holdout: picked beat default (count)':<40} "
              f"{h['n_picked_beat_default']}/{h['n_comparisons']:>15}")
        print(f"  {'holdout: mean (picked − default)':<40} {h['mean_picked_minus_default']:>20.4f}")
        print(f"  {'holdout: median (picked − default)':<40} "
              f"{h['median_picked_minus_default']:>20.4f}")
    else:
        print(f"\n  {summary['holdout_transfer']}")
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
