"""Containment of the code-evolution test runner.

``run_test`` executes pytest against an LLM-modified worktree, so it is the one
place in the code path where autonomously-generated source runs. These tests
cover both halves of the posture: that confinement is applied where the OS
supports it, and that a *failure* of confinement can never be mistaken for a
test result.
"""

import pathlib
import subprocess

import pytest

from evolution.code.worktree import WorktreeEnv, WorktreeError
from pathlib import Path

from evolution.core.sandbox import (
    SandboxUnavailableError,
    sandbox_available,
    wrap_argv,
)


def _env(tmp_path, *, require_sandbox=False):
    root = tmp_path / "run"
    worktree = root / "wt"
    venv = root / "venv"
    worktree.mkdir(parents=True)
    env = WorktreeEnv(
        tmp_path / "repo", root, worktree, venv, require_sandbox=require_sandbox
    )
    env._dep_sites = []
    return env


def _capture(monkeypatch, *, returncode=0, stdout="1 passed", stderr=""):
    """Replace the subprocess call so we can inspect the argv run_test builds."""
    seen = {}

    def fake_run(args, *, cwd, timeout, env=None):
        seen["args"] = args
        return subprocess.CompletedProcess(args, returncode, stdout, stderr)

    monkeypatch.setattr("evolution.code.worktree._run", fake_run)
    return seen


def test_confines_writes_to_the_run_root_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr("evolution.code.worktree.sandbox_available", lambda: True)
    env = _env(tmp_path)
    seen = _capture(monkeypatch)

    env.run_test("tests/test_x.py")

    assert seen["args"][0] == "sandbox-exec"
    profile = seen["args"][2]
    assert f'(subpath "{env._root}")' in profile
    assert "(deny file-write*)" in profile
    assert env.sandboxed is True


def test_runs_unconfined_and_says_so_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("evolution.code.worktree.sandbox_available", lambda: False)
    env = _env(tmp_path)
    seen = _capture(monkeypatch)

    run = env.run_test("tests/test_x.py")

    assert seen["args"][0] != "sandbox-exec"
    assert env.sandboxed is False
    assert run.passed
    # The posture is recorded, never assumed -- an unconfined run must be
    # visible to whatever writes the run's evidence.
    assert env.containment()["sandboxed"] is False


def test_refuses_to_run_unconfined_when_required(tmp_path, monkeypatch):
    monkeypatch.setattr("evolution.code.worktree.sandbox_available", lambda: False)
    env = _env(tmp_path, require_sandbox=True)
    _capture(monkeypatch)

    with pytest.raises(SandboxUnavailableError):
        env.run_test("tests/test_x.py")


def test_sandbox_startup_failure_is_not_reported_as_a_test_result(tmp_path, monkeypatch):
    """The failure mode this whole change exists to prevent.

    ``sandbox-exec`` exits 65 with its own stderr when a profile fails to
    compile, having never run the child. The gate special-cases only exit 5, and
    its failure parser returns an empty set on unrecognised output -- so a run
    where *zero tests executed* would read as "no failures" and could be
    certified as correct. It must raise instead.
    """
    monkeypatch.setattr("evolution.code.worktree.sandbox_available", lambda: True)
    env = _env(tmp_path)
    _capture(
        monkeypatch,
        returncode=65,
        stdout="",
        stderr="sandbox-exec: syntax error: expecting ')'",
    )

    with pytest.raises(WorktreeError, match="sandbox"):
        env.run_test("tests/test_x.py")


@pytest.mark.parametrize("code", [0, 1, 5])
def test_real_pytest_exit_codes_still_return_a_result(tmp_path, monkeypatch, code):
    """Negative control: only *unexpected* codes may raise.

    0 passed, 1 tests failed, 5 nothing collected -- all are genuine pytest
    outcomes the gate interprets, and none may be turned into an exception by
    the containment check.
    """
    monkeypatch.setattr("evolution.code.worktree.sandbox_available", lambda: True)
    env = _env(tmp_path)
    _capture(monkeypatch, returncode=code, stdout="whatever")

    run = env.run_test("tests/test_x.py")

    assert run.exit_code == code
    assert run.passed is (code == 0)


def test_containment_block_shape(tmp_path, monkeypatch):
    monkeypatch.setattr("evolution.code.worktree.sandbox_available", lambda: False)
    env = _env(tmp_path)

    block = env.containment()

    assert set(block) == {"sandboxed", "mechanism", "platform"}
    assert block["mechanism"] is None


@pytest.mark.skipif(
    not sandbox_available(), reason="OS filesystem confinement is macOS-only"
)
def test_confinement_actually_denies_a_write_outside_the_root(tmp_path):
    """Proof by denied write, not by argv inspection.

    Every other test here checks the argv we *build*. This one runs a real
    confined process and asserts the kernel refuses it. The target must be a
    NON-temp path: the profile blanket-allows the temp roots, and the run root
    lives under one of them, so choosing a temp path here would pass while
    proving nothing.
    """
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.txt"          # under tmp_path, but NOT under root
    forbidden = Path.home() / ".containment_probe_should_not_exist"

    argv, sandboxed = wrap_argv(
        ["/bin/sh", "-c", f'echo denied > "{forbidden}"'],
        write_roots=[root], require=True,
    )
    assert sandboxed
    res = subprocess.run(argv, capture_output=True, text=True, timeout=30)

    assert res.returncode != 0, "a write to $HOME should have been denied"
    assert not forbidden.exists()

    # ...and the same write inside the root succeeds, so the profile is not
    # simply denying everything.
    inside = root / "allowed.txt"
    argv_ok, _ = wrap_argv(
        ["/bin/sh", "-c", f'echo ok > "{inside}"'], write_roots=[root], require=True
    )
    assert subprocess.run(argv_ok, capture_output=True, text=True, timeout=30).returncode == 0
    assert inside.read_text().strip() == "ok"
    assert not outside.exists()


class TestPostureIsRecorded:
    """An unconfined run must be visible in the run's evidence, not inferred."""

    def test_trace_carries_the_containment_block(self):
        from evolution.code.repair import RepairResult
        from evolution.code.trace import build_repair_trace

        trace = build_repair_trace(
            tool="tools/x.py", visible_test="t.py", holdout_test="h.py",
            result=RepairResult(fixed=True, fixed_round=1, rounds=[], final_source="x"),
            final_diff="", containment={"sandboxed": False, "mechanism": None,
                                        "platform": "linux"},
        )

        assert trace["containment"]["sandboxed"] is False

    def test_containment_of_tolerates_an_env_without_the_method(self):
        """SWEbenchEnv and the test fakes duck-type run_test only.

        Reading the posture off them must not explode, and must not silently
        claim confinement either -- unknown is None, not False.
        """
        from evolution.code.trace import containment_of

        class DuckTyped:
            def run_test(self, *a, **k): ...

        assert containment_of(DuckTyped()) is None

    def test_containment_of_reads_a_real_env(self, tmp_path, monkeypatch):
        from evolution.code.trace import containment_of

        monkeypatch.setattr("evolution.code.worktree.sandbox_available", lambda: False)
        assert containment_of(_env(tmp_path))["sandboxed"] is False


class TestRequireSandboxIsReachable:
    """Strict mode must reach every entry point that runs the LLM loop.

    Otherwise the flag documents a guarantee the user cannot actually get on the
    path that needs it most -- the gaming harness, whose proposer is built to
    game the gate.
    """

    LLM_LOOP_MODULES = [
        "evolution/code/evolve_code.py",
        "evolution/code/campaign.py",
        "evolution/code/audit_gaming.py",
    ]

    @pytest.mark.parametrize("relpath", LLM_LOOP_MODULES)
    def test_every_worktree_creation_passes_the_policy(self, relpath):
        """Checked structurally: audit_gaming builds its commands inside
        functions, so there is no importable command object to introspect."""
        import ast

        tree = ast.parse(pathlib.Path(relpath).read_text())
        creates = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "create"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "WorktreeEnv"
        ]
        assert creates, f"no WorktreeEnv.create call found in {relpath}"
        for call in creates:
            assert any(kw.arg == "require_sandbox" for kw in call.keywords), (
                f"{relpath}:{call.lineno} creates a worktree without passing "
                "require_sandbox, so strict mode cannot reach it"
            )

    @pytest.mark.parametrize("relpath", LLM_LOOP_MODULES)
    def test_cli_exposes_the_flag(self, relpath):
        assert "--require-sandbox" in pathlib.Path(relpath).read_text()
