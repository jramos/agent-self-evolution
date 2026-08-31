"""A pytest run that did not produce an answer must not be scored as "no failures".

``failing_tests`` is the only seam ``run_code_oracle_gate`` consumes, and
``_parse_pytest_failures`` returns an empty set on any output it does not
recognise. An empty set reads as "nothing failed", so a run that timed out,
failed to collect, or was killed by a signal could certify a wrong repair as
correct — and because the campaign is the instrument behind the published
deploy-reachable rate, that inflates the number, biased toward the hard bugs
where hangs and OOMs cluster.
"""

import pytest

from evolution.code.worktree import TestRun, WorktreeEnv, WorktreeError

_WRONG_REPAIR = "def add(a, b):\n    return a + b + 999\n"


class _StubEnv(WorktreeEnv):
    """A worktree whose pytest result is dictated by the test."""

    def __init__(self, tmp_path, run: TestRun):
        root = tmp_path / "run"
        (root / "wt").mkdir(parents=True)
        super().__init__(tmp_path / "repo", root, root / "wt", root / "venv")
        self._run = run

    def run_test(self, *a, **k) -> TestRun:
        return self._run


@pytest.mark.parametrize(
    "label,run",
    [
        ("timeout", TestRun(passed=False, output="pytest timed out after 600s",
                            duration_seconds=600.0, exit_code=None)),
        ("usage_exit_4", TestRun(passed=False, output="ERROR: not found: tests/t.py::test_x",
                                duration_seconds=0.2, exit_code=4)),
        ("sigkill", TestRun(passed=False, output="", duration_seconds=1.0, exit_code=-9)),
        ("internal_error", TestRun(passed=False, output="INTERNALERROR> boom",
                                   duration_seconds=0.3, exit_code=3)),
    ],
)
def test_non_authoritative_runs_refuse_to_answer(tmp_path, label, run):
    """Each of these produced an empty failure set, i.e. "nothing failed"."""
    env = _StubEnv(tmp_path, run)

    with pytest.raises(WorktreeError, match="authoritative"):
        env.failing_tests("tests/test_x.py")


@pytest.mark.parametrize(
    "label,exit_code,output,expected",
    [
        ("all_passed", 0, "3 passed", set()),
        ("some_failed", 1, "FAILED tests/t.py::test_x - assert 1 == 2\n1 failed",
         {"tests/t.py::test_x"}),
        ("nothing_collected", 5, "no tests ran", set()),
    ],
)
def test_authoritative_runs_still_answer(tmp_path, label, exit_code, output, expected):
    """Negative control: the codes that do make a complete statement still work.

    Exit 5 is kept authoritative because callers already reason about "no tests
    collected" explicitly, and turning it into an error here would change how the
    harvester classifies a candidate rather than how the gate scores one.
    """
    env = _StubEnv(tmp_path, TestRun(passed=exit_code == 0, output=output,
                                     duration_seconds=0.1, exit_code=exit_code))

    assert env.failing_tests("tests/test_x.py") == expected


def test_oracle_gate_cannot_certify_a_timed_out_run(tmp_path, monkeypatch):
    """The end-to-end consequence, asserted at the gate rather than the seam.

    Before this change the same inputs returned decision='correct', deploy=True
    for a deliberately wrong repair.
    """
    from evolution.code import gate as gate_mod
    from evolution.code.repair import RepairResult

    env = _StubEnv(tmp_path, TestRun(passed=False, output="pytest timed out after 600s",
                                     duration_seconds=600.0, exit_code=None))
    monkeypatch.setattr(env.__class__, "read_tool", lambda self, p: _WRONG_REPAIR)

    with pytest.raises(WorktreeError, match="authoritative"):
        gate_mod.run_code_oracle_gate(
            env,
            tool_relpath="tools/calc.py",
            test_relpath="tests/test_calc.py",
            bug_tests=("tests/test_calc.py::test_add",),
            oracle_failures=frozenset(),
            base_src="def add(a, b):\n    return a\n",
            repair_result=RepairResult(
                fixed=True, fixed_round=1, rounds=[], final_source=_WRONG_REPAIR
            ),
        )


def test_campaign_records_an_inconclusive_run_honestly(monkeypatch, tmp_path):
    """The ledger must not blame worktree setup for a candidate that hung.

    campaign maps WorktreeError to Skip("worktree_failed"); an inconclusive run is
    a different thing, and mislabelling it would hide the real reason a candidate
    left the denominator.
    """
    from pathlib import Path

    from evolution.code import campaign as camp
    from evolution.code.worktree import NonAuthoritativeRunError

    class _Candidate:
        tool_path = "tools/x.py"
        fix_sha = "deadbeef"
        parent_sha = "cafebabe"
        test_path = "tests/test_x.py"

    class _Env:
        worktree = tmp_path
        def assert_authoritative(self, *a, **k): ...
        def destroy(self): ...
        def run_test(self, *a, **k):
            raise AssertionError("_failures must go through the failing_tests seam")
        def failing_tests(self, *a, **k):
            raise NonAuthoritativeRunError("pytest exited None")

    monkeypatch.setattr(camp, "_git_show", lambda *a, **k: "def add(a, b): return a")
    monkeypatch.setattr(camp.WorktreeEnv, "create", classmethod(lambda cls, *a, **k: _Env()))

    result = camp.run_organism(
        Path("/repo"), _Candidate(), engine=None, seeds=1, base_python=None
    )

    assert result.reason == "run_inconclusive"
