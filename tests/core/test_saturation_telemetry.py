"""Saturation telemetry: per-run pre-flight band + scores, including aborts."""
from pathlib import Path

import pytest

from evolution.core.saturation_check import SaturationReport
from evolution.core.saturation_telemetry import (
    LEDGER_NAME,
    append_saturation_telemetry,
    build_saturation_telemetry_row,
    main,
    read_ledger,
    resolve_ledger_root,
    summarize_ledger,
)


def _report(
    band="healthy",
    holdout_score=0.6,
    holdout_n=20,
    closed_loop_score=None,
    closed_loop_n=None,
    floor_score=None,
    floor_n=None,
    noise=None,
):
    return SaturationReport(
        band=band,
        holdout_score=holdout_score,
        holdout_n=holdout_n,
        holdout_per_example=[holdout_score] * holdout_n,
        closed_loop_score=closed_loop_score,
        closed_loop_n=closed_loop_n,
        floor_score=floor_score,
        floor_n=floor_n,
        noise=noise,
    )


def test_proceed_row_carries_band_scores_and_decision():
    row = build_saturation_telemetry_row(
        _report(band="healthy", holdout_score=0.62, holdout_n=43),
        run_id="20260612_101500",
        artifact="my-skill",
        artifact_type="skill",
        proceeded=True,
        decision="deploy",
    )
    assert row.run_id == "20260612_101500"
    assert row.artifact == "my-skill"
    assert row.artifact_type == "skill"
    assert row.band == "healthy"
    assert row.holdout_score == pytest.approx(0.62)
    assert row.holdout_n == 43
    assert row.proceeded is True
    assert row.abort_reason is None
    assert row.decision == "deploy"


def test_abort_row_records_reason_and_no_decision():
    row = build_saturation_telemetry_row(
        _report(band="no_headroom", holdout_score=0.995),
        run_id="20260612_102000",
        artifact="write_file",
        artifact_type="tool",
        proceeded=False,
        abort_reason="non_interactive_deny",
    )
    assert row.band == "no_headroom"
    assert row.proceeded is False
    assert row.abort_reason == "non_interactive_deny"
    assert row.decision is None


def test_closed_loop_and_floor_scores_carried_when_present():
    row = build_saturation_telemetry_row(
        _report(
            band="weak_signal",
            holdout_score=0.96,
            closed_loop_score=0.5,
            closed_loop_n=8,
            floor_score=0.83,
            floor_n=8,
        ),
        run_id="ts",
        artifact="MEMORY_GUIDANCE",
        artifact_type="prompt_section",
        proceeded=True,
        decision="reject",
    )
    assert row.closed_loop_score == pytest.approx(0.5)
    assert row.closed_loop_n == 8
    assert row.floor_score == pytest.approx(0.83)
    assert row.floor_n == 8


def test_scores_absent_degrade_to_none():
    row = build_saturation_telemetry_row(
        _report(),  # synthetic-only run: no closed-loop, no floor
        run_id="ts",
        artifact="s",
        artifact_type="skill",
        proceeded=True,
    )
    assert row.closed_loop_score is None
    assert row.closed_loop_n is None
    assert row.floor_score is None
    assert row.noise_floor_passes is None


def test_noise_floor_passes_pulled_from_noise_sidecar():
    row = build_saturation_telemetry_row(
        _report(noise={"mean_per_task_flip": 1.5, "other": "ignored"}),
        run_id="ts",
        artifact="s",
        artifact_type="skill",
        proceeded=True,
    )
    assert row.noise_floor_passes == pytest.approx(1.5)


def test_malformed_noise_does_not_break_row():
    row = build_saturation_telemetry_row(
        _report(noise={"unexpected": "shape"}),
        run_id="ts",
        artifact="s",
        artifact_type="skill",
        proceeded=True,
    )
    assert row.noise_floor_passes is None


def test_append_writes_one_jsonl_row(tmp_path):
    row = build_saturation_telemetry_row(
        _report(), run_id="ts", artifact="s", artifact_type="skill",
        proceeded=True, decision="deploy",
    )
    path = append_saturation_telemetry(tmp_path, row=row)
    assert path == tmp_path / LEDGER_NAME
    rows = read_ledger(path)
    assert len(rows) == 1
    assert rows[0]["artifact"] == "s"
    assert rows[0]["decision"] == "deploy"
    assert rows[0]["proceeded"] is True


def test_append_is_additive_and_mixes_proceed_and_abort(tmp_path):
    append_saturation_telemetry(
        tmp_path,
        row=build_saturation_telemetry_row(
            _report(band="healthy"), run_id="a", artifact="a",
            artifact_type="skill", proceeded=True, decision="reject",
        ),
    )
    append_saturation_telemetry(
        tmp_path,
        row=build_saturation_telemetry_row(
            _report(band="no_headroom", holdout_score=1.0), run_id="b",
            artifact="b", artifact_type="tool", proceeded=False,
            abort_reason="user_decline",
        ),
    )
    rows = read_ledger(tmp_path / LEDGER_NAME)
    assert [r["run_id"] for r in rows] == ["a", "b"]
    assert [r["proceeded"] for r in rows] == [True, False]


def test_append_never_raises_on_unwritable_root(tmp_path):
    # A file where a directory is expected: mkdir fails. Must degrade to None.
    blocker = tmp_path / "blocked"
    blocker.write_text("not a dir")
    row = build_saturation_telemetry_row(
        _report(), run_id="ts", artifact="s", artifact_type="skill", proceeded=True,
    )
    assert append_saturation_telemetry(blocker / "sub", row=row) is None


def test_resolve_ledger_root_reused_from_search_telemetry():
    assert resolve_ledger_root(Path("output/tools/write_file/20260101")).name == "output"


def test_summarize_empty_ledger(tmp_path):
    assert "No saturation telemetry" in summarize_ledger(tmp_path / LEDGER_NAME)


def test_summarize_reports_bands_and_abort_count(tmp_path):
    for run_id, band, proceeded, reason in [
        ("a", "healthy", True, None),
        ("b", "no_headroom", False, "non_interactive_deny"),
        ("c", "no_headroom", False, "user_decline"),
    ]:
        append_saturation_telemetry(
            tmp_path,
            row=build_saturation_telemetry_row(
                _report(band=band, holdout_score=0.99 if not proceeded else 0.6),
                run_id=run_id, artifact=run_id, artifact_type="skill",
                proceeded=proceeded, abort_reason=reason,
            ),
        )
    out = summarize_ledger(tmp_path / LEDGER_NAME)
    assert "healthy" in out and "no_headroom" in out
    assert "2 aborted" in out


def test_main_prints_summary(tmp_path, capsys):
    append_saturation_telemetry(
        tmp_path,
        row=build_saturation_telemetry_row(
            _report(), run_id="ts", artifact="s", artifact_type="skill", proceeded=True,
        ),
    )
    rc = main(["--ledger", str(tmp_path / LEDGER_NAME)])
    assert rc == 0
    assert "Saturation telemetry" in capsys.readouterr().out
