"""Convention-adherence verdict: did the agent use the repo wrapper, not the default tool."""
from evolution.validation.agent_runner import AgentRunResult
from evolution.validation.report import score_task


def _run(cmds, error=None):
    return AgentRunResult(
        tool_calls_seq=["Bash"] * len(cmds),
        final_text_tail="",
        duration_seconds=1.0,
        error=error,
        tool_calls_with_args=[
            {"name": "Bash", "arguments": {"command": c}} for c in cmds
        ],
    )


def test_convention_adherent_when_wrapper_used_and_default_avoided():
    passed, abstained = score_task(
        expected_tools=(), forbidden_tools=(), run=_run(["./bin/check"]),
        expected_action="convention",
        required_cmd_substr=("bin/check",), forbidden_cmd_substr=("pytest",),
    )
    assert (passed, abstained) == (True, False)


def test_convention_fails_when_default_used():
    passed, _ = score_task(
        expected_tools=(), forbidden_tools=(), run=_run(["python -m pytest"]),
        expected_action="convention",
        required_cmd_substr=("bin/check",), forbidden_cmd_substr=("pytest",),
    )
    assert passed is False


def test_convention_fails_when_wrapper_used_but_default_also_used():
    passed, _ = score_task(
        expected_tools=(), forbidden_tools=(), run=_run(["./bin/check", "pytest -q"]),
        expected_action="convention",
        required_cmd_substr=("bin/check",), forbidden_cmd_substr=("pytest",),
    )
    assert passed is False


def test_convention_fails_when_no_bash_at_all():
    passed, abstained = score_task(
        expected_tools=(), forbidden_tools=(), run=_run([]),
        expected_action="convention",
        required_cmd_substr=("bin/check",), forbidden_cmd_substr=("pytest",),
    )
    assert (passed, abstained) == (False, False)


def test_convention_abstains_on_runner_error():
    passed, abstained = score_task(
        expected_tools=(), forbidden_tools=(), run=_run([], error="timeout"),
        expected_action="convention",
        required_cmd_substr=("bin/check",), forbidden_cmd_substr=("pytest",),
    )
    assert abstained is True
