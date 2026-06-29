"""The sequencer: reconcile, fault isolation, resume, dry-run, defaults merge."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from evolution.orchestrator.run import reconcile, run_pipeline
from evolution.orchestrator.spec import PhaseSpec, RunSpec

_CLOCK = lambda: datetime(2026, 1, 2, 3, 4, 5)  # noqa: E731 — deterministic run_id


def make_runner(plan=None, default=(0, {"decision": "deploy"})):
    """Fake phase_runner. ``plan`` maps an output-dir basename (e.g. 'skills-demo')
    to (exit_code, gate_dict_or_None); a None gate writes no file."""
    plan = plan or {}
    calls = []

    def runner(argv, *, env, cwd, timeout=None):
        calls.append(argv)
        out = Path(argv[argv.index("--output-dir") + 1])
        exit_code, gate = plan.get(out.name, default)
        if gate is not None:
            out.mkdir(parents=True, exist_ok=True)
            (out / "gate_decision.json").write_text(json.dumps(gate))
        return exit_code

    runner.calls = calls
    return runner


class TestReconcile:
    def test_no_gate_is_failed(self):
        assert reconcile(None, 1) == ("failed", "missing")
        assert reconcile(None, 0) == ("failed", "missing")

    def test_aborted_and_denied(self):
        assert reconcile({"decision": "aborted"}, 2) == ("aborted", "aborted")
        assert reconcile({"decision": "denied"}, 3) == ("denied", "denied")

    def test_reject_with_nonzero_exit_is_passed(self):
        assert reconcile({"decision": "reject"}, 1) == ("passed", "reject")

    def test_deploy_and_dry_run_pass(self):
        assert reconcile({"decision": "deploy"}, 0) == ("passed", "deploy")
        assert reconcile({"decision": "dry_run"}, 0) == ("passed", "dry_run")

    def test_gate_without_decision_key_recorded_as_unknown(self):
        assert reconcile({}, 0) == ("passed", "unknown")


def _spec(*phases):
    return RunSpec(phases=tuple(phases), defaults={})


class TestSequencer:
    def test_continue_on_error_runs_later_phases(self, tmp_path):
        spec = _spec(
            PhaseSpec("skills", "demo", {}),
            PhaseSpec("tools", "demo", {"manifest": "m.json"}),
        )
        runner = make_runner({"skills-demo": (1, None)})  # phase 0 produces no gate
        summary = run_pipeline(spec, run_root=tmp_path, phase_runner=runner, clock=_CLOCK)
        assert len(runner.calls) == 2
        assert summary["by_status"] == {"failed": 1, "passed": 1}
        assert summary["stopped_early"] is False

    def test_stop_on_error_halts_on_failed(self, tmp_path):
        spec = _spec(
            PhaseSpec("skills", "demo", {}),
            PhaseSpec("tools", "demo", {"manifest": "m.json"}),
        )
        runner = make_runner({"skills-demo": (1, None)})
        summary = run_pipeline(spec, run_root=tmp_path, phase_runner=runner,
                               stop_on_error=True, clock=_CLOCK)
        assert len(runner.calls) == 1
        assert summary["stopped_early"] is True

    def test_reject_does_not_halt_under_stop_on_error(self, tmp_path):
        spec = _spec(
            PhaseSpec("skills", "demo", {}),
            PhaseSpec("tools", "demo", {"manifest": "m.json"}),
        )
        runner = make_runner({"skills-demo": (1, {"decision": "reject"})})
        summary = run_pipeline(spec, run_root=tmp_path, phase_runner=runner,
                               stop_on_error=True, clock=_CLOCK)
        assert len(runner.calls) == 2  # reject is a clean pass, pipeline continues
        assert summary["stopped_early"] is False

    def test_dry_run_records_argv_without_calling_runner(self, tmp_path):
        spec = _spec(PhaseSpec("skills", "demo", {"iterations": 5}))
        runner = make_runner()
        summary = run_pipeline(spec, run_root=tmp_path, phase_runner=runner,
                               dry_run=True, clock=_CLOCK)
        assert runner.calls == []
        assert summary["phases"][0]["status"] == "skipped"
        assert summary["phases"][0]["decision"] == "dry_run"
        assert "--iterations" in summary["phases"][0]["argv"]

    def test_only_filters_phases(self, tmp_path):
        spec = _spec(
            PhaseSpec("skills", "demo", {}),
            PhaseSpec("tools", "demo", {"manifest": "m.json"}),
        )
        runner = make_runner()
        summary = run_pipeline(spec, run_root=tmp_path, phase_runner=runner,
                               only=("tools",), clock=_CLOCK)
        assert len(runner.calls) == 1
        assert summary["phases"][0]["phase"] == "tools"

    def test_resume_skips_done(self, tmp_path):
        spec = _spec(
            PhaseSpec("skills", "demo", {}),
            PhaseSpec("tools", "demo", {"manifest": "m.json"}),
        )
        run_pipeline(spec, run_root=tmp_path, phase_runner=make_runner(), clock=_CLOCK)
        second = make_runner()
        summary = run_pipeline(spec, run_root=tmp_path, phase_runner=second,
                               resume=True, clock=_CLOCK)
        assert second.calls == []  # both phases already in the ledger
        assert summary["n_phases"] == 2

    def test_defaults_merge_phase_args_win(self, tmp_path):
        spec = RunSpec(
            phases=(PhaseSpec("skills", "demo", {"iterations": 5}),),
            defaults={"iterations": 3, "seed": 42},
        )
        runner = make_runner()
        run_pipeline(spec, run_root=tmp_path, phase_runner=runner, clock=_CLOCK)
        argv = runner.calls[0]
        assert argv[argv.index("--iterations") + 1] == "5"  # phase wins over default 3
        assert "--seed" in argv

    def test_runner_exception_is_aborted_not_crash(self, tmp_path):
        spec = _spec(PhaseSpec("skills", "demo", {}))

        def boom(argv, *, env, cwd, timeout=None):
            raise RuntimeError("runner blew up")

        summary = run_pipeline(spec, run_root=tmp_path, phase_runner=boom, clock=_CLOCK)
        assert summary["phases"][0]["status"] == "aborted"
        assert "runner blew up" in summary["phases"][0]["error"]

    def test_writes_summary_files(self, tmp_path):
        spec = _spec(PhaseSpec("skills", "demo", {}))
        run_pipeline(spec, run_root=tmp_path, phase_runner=make_runner(), clock=_CLOCK)
        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "summary.md").exists()
        assert (tmp_path / "run_history.jsonl").exists()

    def test_aborted_gate_decision_halts_under_stop_on_error(self, tmp_path):
        spec = _spec(
            PhaseSpec("skills", "demo", {}),
            PhaseSpec("tools", "demo", {"manifest": "m.json"}),
        )
        runner = make_runner({"skills-demo": (2, {"decision": "aborted"})})  # cost-ceiling
        summary = run_pipeline(spec, run_root=tmp_path, phase_runner=runner,
                               stop_on_error=True, clock=_CLOCK)
        assert len(runner.calls) == 1 and summary["stopped_early"] is True
        assert summary["phases"][0]["status"] == "aborted"

    def test_denied_does_not_halt_under_stop_on_error(self, tmp_path):
        spec = _spec(
            PhaseSpec("skills", "demo", {}),
            PhaseSpec("tools", "demo", {"manifest": "m.json"}),
        )
        runner = make_runner({"skills-demo": (3, {"decision": "denied"})})  # saturation
        summary = run_pipeline(spec, run_root=tmp_path, phase_runner=runner,
                               stop_on_error=True, clock=_CLOCK)
        assert len(runner.calls) == 2 and summary["stopped_early"] is False
        assert summary["by_status"].get("denied") == 1

    def test_resume_retries_failed_phase_but_skips_passed(self, tmp_path):
        spec = _spec(
            PhaseSpec("skills", "demo", {}),
            PhaseSpec("tools", "demo", {"manifest": "m.json"}),
        )
        run_pipeline(spec, run_root=tmp_path,
                     phase_runner=make_runner({"skills-demo": (1, None)}), clock=_CLOCK)
        second = make_runner()  # this time skills would write a gate too
        summary = run_pipeline(spec, run_root=tmp_path, phase_runner=second,
                               resume=True, clock=_CLOCK)
        assert len(second.calls) == 1  # only the previously-failed phase is retried
        assert "--skill" in second.calls[0]
        statuses = {r["phase"]: r["status"] for r in summary["phases"]}
        assert statuses == {"skills": "passed", "tools": "passed"}

    def test_dry_run_then_resume_runs_for_real(self, tmp_path):
        spec = _spec(PhaseSpec("skills", "demo", {}))
        run_pipeline(spec, run_root=tmp_path, phase_runner=make_runner(),
                     dry_run=True, clock=_CLOCK)
        runner = make_runner()
        run_pipeline(spec, run_root=tmp_path, phase_runner=runner, resume=True, clock=_CLOCK)
        assert len(runner.calls) == 1  # a dry-run row is not treated as done

    def test_stale_gate_cleared_before_relaunch(self, tmp_path):
        spec = _spec(PhaseSpec("skills", "demo", {}))
        run_pipeline(spec, run_root=tmp_path,
                     phase_runner=make_runner({"skills-demo": (0, {"decision": "deploy"})}),
                     clock=_CLOCK)
        # Same run root, no --resume: the phase now "crashes" without writing a gate.
        summary = run_pipeline(spec, run_root=tmp_path,
                               phase_runner=make_runner({"skills-demo": (1, None)}),
                               clock=_CLOCK)
        assert summary["phases"][0]["status"] == "failed"  # not the stale deploy gate
        assert summary["deployable"] == []
