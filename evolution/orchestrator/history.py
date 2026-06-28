"""JSONL run history + summary rendering for the orchestrator.

Append-only ``run_history.jsonl`` (one row per executed phase) following the
``campaign.py`` ledger pattern: resume reads it back into a ``done`` set keyed on
``spec_index:phase:name`` and skips completed phases. ``summary.json`` /
``summary.md`` aggregate the run and surface the human-in-loop handoff — the
``deployable`` list of ``decision==deploy`` run dirs to review.
"""

from __future__ import annotations

import json
from pathlib import Path

HISTORY_SCHEMA_VERSION = "1"
LEDGER_NAME = "run_history.jsonl"


def row_key(row: dict) -> str:
    return f"{row['spec_index']}:{row['phase']}:{row['name']}"


def append_row(ledger_path: Path, row: dict) -> None:
    with Path(ledger_path).open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def load_done(ledger_path: Path) -> dict[str, dict]:
    """Map ``spec_index:phase:name`` → row for every prior run; last write wins."""
    path = Path(ledger_path)
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            done[row_key(row)] = row
    return done


def build_summary(rows: list[dict], *, run_id: str, stopped_early: bool) -> dict:
    by_status: dict[str, int] = {}
    by_decision: dict[str, int] = {}
    deployable = []
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        decision = r.get("decision")
        if decision:
            by_decision[decision] = by_decision.get(decision, 0) + 1
        if decision == "deploy":
            deployable.append(
                {"phase": r["phase"], "name": r["name"], "run_dir": r.get("run_dir")}
            )
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "run_id": run_id,
        "n_phases": len(rows),
        "by_status": by_status,
        "by_decision": by_decision,
        "deployable": deployable,
        "stopped_early": stopped_early,
        "phases": rows,
    }


def render_summary_md(summary: dict) -> str:
    status = ", ".join(f"{k} {v}" for k, v in sorted(summary["by_status"].items())) or "—"
    decisions = ", ".join(f"{k} {v}" for k, v in sorted(summary["by_decision"].items())) or "—"
    lines = [
        f"# Cross-phase evolution run {summary['run_id']}",
        "",
        f"{summary['n_phases']} phase(s): {status}.  Decisions: {decisions}."
        + ("  **Stopped early.**" if summary["stopped_early"] else ""),
        "",
        "| seq | phase | name | status | decision | run dir |",
        "|-----|-------|------|--------|----------|---------|",
    ]
    for r in summary["phases"]:
        lines.append(
            f"| {r['spec_index']} | {r['phase']} | `{r['name']}` | {r['status']} "
            f"| {r.get('decision') or '—'} | {r.get('run_dir') or '—'} |"
        )
    lines += ["", "_Propose-only: nothing was deployed._"]
    if summary["deployable"]:
        lines.append("Deploy-ready for human review:")
        for d in summary["deployable"]:
            lines.append(f"  - {d['phase']}/{d['name']} → {d['run_dir']}/gate_decision.json")
    return "\n".join(lines)
