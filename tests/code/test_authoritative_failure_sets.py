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

    with pytest.raises(WorktreeError, match="established nothing"):
        env.failing_tests("tests/test_x.py")


@pytest.mark.parametrize(
    "label,exit_code,output,expected",
    [
        ("all_passed", 0, "3 passed", set()),
        ("some_failed", 1, "FAILED tests/t.py::test_x - assert 1 == 2\n1 failed",
         {"tests/t.py::test_x"}),

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
    with pytest.raises(WorktreeError, match="established nothing"):
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


class TestRegressionFloorRefusesToScoreAHang:
    """The same defect lived on the product deploy path, where it matters most.

    The floor runs the whole tests/tools subset under a 600s timeout on every
    deploy decision, making it the likeliest place in the gate to hang -- which
    is exactly where an LLM-introduced infinite loop lands.
    """

    def _staged(self, tmp_path, floor_run):
        """A gate env whose floor run is dictated, everything else healthy."""
        healthy = TestRun(passed=True, output="1 passed", duration_seconds=0.1, exit_code=0)
        env = _StubEnv(tmp_path, healthy)

        def run_test(*paths, **k):
            # Dispatch on the paths, not call order: which call is the floor
            # depends on gate internals this test should not encode.
            return floor_run if any("tests/tools" in p for p in paths) else healthy

        env.run_test = run_test
        return env

    @pytest.mark.parametrize("exit_code,label", [(None, "timeout"), (-9, "sigkill"), (3, "internal")])
    def test_floor_that_never_ran_is_not_green(self, tmp_path, monkeypatch, exit_code, label):
        from evolution.code import gate as gate_mod

        floor = TestRun(passed=False, output="", duration_seconds=600.0, exit_code=exit_code)
        env = self._staged(tmp_path, floor)
        monkeypatch.setattr(gate_mod, "_base_source", lambda *a, **k: "")
        monkeypatch.setattr(env.__class__, "read_tool", lambda self, p: _WRONG_REPAIR)
        monkeypatch.setattr(env.__class__, "changed_files", lambda self: ["tools/calc.py"])
        monkeypatch.setattr(env.__class__, "write_tool", lambda self, p, s: None)

        from evolution.code.repair import RepairResult

        result = gate_mod.run_code_gate(
            env, tool_relpath="tools/calc.py", visible_test_relpath="tests/test_v.py",
            holdout_test_relpath="tests/test_h.py",
            repair_result=RepairResult(fixed=True, fixed_round=1, rounds=[],
                                       final_source=_WRONG_REPAIR),
            floor_paths=("tests/tools",),
        )

        assert result.deploy is False
        assert "no statement" in result.reason


class TestParametrizedIdsDoNotCollapse:
    """Two failing ids must not merge into one key.

    The diff compares failure sets by identity, so a collapse lets a failure the
    repair introduced hide behind an unrelated pre-existing one -- silent
    certification, on a perfectly authoritative exit 1.
    """

    def test_ids_containing_the_separator_stay_distinct(self):
        from evolution.code.gate import _parse_pytest_failures

        out = (
            "FAILED t.py::test_dash[a - c] - AssertionError: boom\n"
            "FAILED t.py::test_dash[a - b] - AssertionError: boom\n"
            "FAILED t.py::test_space[x y] - assert False\n"
        )

        assert _parse_pytest_failures(out) == {
            "t.py::test_dash[a - c]",
            "t.py::test_dash[a - b]",
            "t.py::test_space[x y]",
        }

    def test_ordinary_lines_still_parse(self):
        from evolution.code.gate import _parse_pytest_failures

        out = "FAILED t.py::test_x - assert 1 == 2\nERROR t.py::test_y\n"
        assert _parse_pytest_failures(out) == {"t.py::test_x", "t.py::test_y"}


class TestEvidenceOutranksExitCode:
    """Named failures are a complete statement whatever the exit code says.

    The first version of this fix refused every code outside (0, 1, 5), which
    silently dropped a real bug class: a parent whose module fails to import
    exits 2 *and names the file it could not import*. That is evidence, and
    refusing it would have removed those candidates from the campaign's
    population -- an unremarked change to the denominator, in the direction that
    flatters the published rate.
    """

    def test_import_error_exit_2_is_kept(self, tmp_path):
        env = _StubEnv(tmp_path, TestRun(
            passed=False,
            output="ERROR tests/test_imp.py\n!!! Interrupted: 1 error during collection !!!",
            duration_seconds=0.3, exit_code=2,
        ))

        assert env.failing_tests("tests/test_imp.py") == {"tests/test_imp.py"}

    def test_clean_pass_still_returns_empty(self, tmp_path):
        env = _StubEnv(tmp_path, TestRun(passed=True, output="3 passed",
                                         duration_seconds=0.1, exit_code=0))

        assert env.failing_tests("tests/test_x.py") == set()

    @pytest.mark.parametrize("exit_code,output,label", [
        (5, "no tests ran", "nothing collected"),
        (4, "ERROR: not found: t.py::test_x", "node id not found"),
        (1, "", "ran but named nothing"),
    ])
    def test_no_evidence_is_refused_whatever_the_code(self, tmp_path, exit_code, output, label):
        """Including exit 5 and exit 1.

        Exit 5 could otherwise certify: the oracle gate records
        ``bug_tests_passed = not bug_fail``, so a zero-test run reads as passed.
        Exit 1 with nothing parsed means tests failed but none could be named,
        which is not something to score either.
        """
        env = _StubEnv(tmp_path, TestRun(passed=False, output=output,
                                         duration_seconds=0.2, exit_code=exit_code))

        with pytest.raises(WorktreeError, match="established nothing"):
            env.failing_tests("tests/test_x.py")


def test_inconclusive_while_scoring_is_a_failed_seed_not_a_skip(monkeypatch, tmp_path):
    """A repair that hangs the oracle scope is wrong, not unmeasurable.

    By that point the repair has already passed its bug tests, so the hang is its
    own doing on a sibling test. Dropping the organism would remove it from the
    denominator and inflate deploy-reachable -- the same direction as the bias
    this work exists to remove.
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
        def write_tool(self, *a, **k): ...

    monkeypatch.setattr(camp, "_git_show", lambda *a, **k: "def add(a, b): return a")
    monkeypatch.setattr(camp.WorktreeEnv, "create", classmethod(lambda cls, *a, **k: _Env()))
    # First call is the oracle (fix) run, second is the buggy parent: their
    # difference is what makes a valid organism with real bug tests.
    calls = {"n": 0}

    def failures(env, path):
        calls["n"] += 1
        return set() if calls["n"] == 1 else {"tests/test_x.py::test_bug"}

    monkeypatch.setattr(camp, "_failures", failures)

    def boom(*a, **k):
        raise NonAuthoritativeRunError("oracle scope hung")

    monkeypatch.setattr(camp, "run_code_oracle_gate", boom)

    class _Engine:
        def repair(self, *a, **k):
            return object()

    result = camp.run_organism(
        Path("/repo"), _Candidate(), engine=_Engine(), seeds=2, base_python=None
    )

    # organism retained, both seeds scored as not-deploy-reachable
    assert getattr(result, "seeds", None) == [False, False]
