"""JSONL ledger append/resume + summary aggregation."""

from __future__ import annotations

from evolution.orchestrator.history import (
    LEDGER_NAME,
    append_row,
    build_summary,
    load_done,
    render_summary_md,
)


def _row(spec_index, phase, name, status, decision, run_dir=None):
    return {"spec_index": spec_index, "phase": phase, "name": name,
            "status": status, "decision": decision, "run_dir": run_dir}


def test_append_and_load_done_keys_on_index_phase_name(tmp_path):
    ledger = tmp_path / LEDGER_NAME
    append_row(ledger, _row(0, "skills", "a", "passed", "deploy"))
    append_row(ledger, _row(1, "tools", "b", "passed", "reject"))
    done = load_done(ledger)
    assert set(done) == {"0:skills:a", "1:tools:b"}
    assert done["0:skills:a"]["decision"] == "deploy"


def test_load_done_missing_file_is_empty(tmp_path):
    assert load_done(tmp_path / "nope.jsonl") == {}


def test_build_summary_aggregates_and_lists_deployable():
    rows = [
        _row(0, "skills", "a", "passed", "deploy", run_dir="output/x"),
        _row(1, "tools", "b", "passed", "reject"),
        _row(2, "code", "c", "failed", "missing"),
    ]
    summary = build_summary(rows, run_id="RID", stopped_early=False)
    assert summary["by_status"] == {"passed": 2, "failed": 1}
    assert summary["by_decision"] == {"deploy": 1, "reject": 1, "missing": 1}
    assert summary["deployable"] == [{"phase": "skills", "name": "a", "run_dir": "output/x"}]
    assert summary["n_phases"] == 3


def test_deployable_requires_passed_status():
    # An aborted phase whose stale/partial gate says "deploy" must NOT be offered
    # for deploy review.
    rows = [_row(0, "code", "c", "aborted", "deploy", run_dir="output/x")]
    summary = build_summary(rows, run_id="R", stopped_early=False)
    assert summary["deployable"] == []


def test_render_summary_md_mentions_propose_only_and_deployable():
    rows = [_row(0, "tools", "fetch", "passed", "deploy", run_dir="output/x")]
    md = render_summary_md(build_summary(rows, run_id="RID", stopped_early=False))
    assert "Propose-only" in md
    assert "tools/fetch" in md
    assert "RID" in md
