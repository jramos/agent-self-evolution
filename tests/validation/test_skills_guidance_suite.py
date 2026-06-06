"""Sanity tests for the skills_guidance production suite.

Verifies that the suite loads correctly, every task's skills_src directory
contains the expected SKILL.md, and that patch tasks carry the action-level
fields while control tasks carry the forbidden_tools guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evolution.validation.task import TaskSuite


SUITE_PATH = (
    Path(__file__).resolve().parents[2]
    / "evolution"
    / "validation"
    / "suites"
    / "skills_guidance.jsonl"
)

SUITE_DIR = SUITE_PATH.parent

_PATCH_TASK_IDS = {"patch-stale-path", "patch-stale-api", "patch-stale-flag"}
_CONTROL_TASK_IDS = {"ctl-healthy", "ctl-cosmetic"}

_SKILL_NAME_BY_TASK: dict[str, str] = {
    "patch-stale-path": "csv-summarizer",
    "patch-stale-flag": "line-counter",
    "patch-stale-api": "parse-int",
    "ctl-healthy": "word-counter",
    "ctl-cosmetic": "greeter",
}


@pytest.fixture(scope="module")
def suite() -> TaskSuite:
    return TaskSuite.from_jsonl(SUITE_PATH)


class TestSuiteShape:
    def test_loads_five_tasks(self, suite):
        assert len(suite.tasks) == 5

    def test_task_ids_match_expected(self, suite):
        ids = {t.task_id for t in suite.tasks}
        assert ids == _PATCH_TASK_IDS | _CONTROL_TASK_IDS

    def test_patch_stale_flag_is_last(self, suite):
        assert suite.tasks[-1].task_id == "patch-stale-flag"


class TestSkillsSourceResolves:
    def test_every_skills_src_has_skill_md(self, suite):
        for task in suite.tasks:
            assert task.skills_src is not None, f"{task.task_id}: skills_src is None"
            skill_name = _SKILL_NAME_BY_TASK[task.task_id]
            skill_md = SUITE_DIR / task.skills_src / skill_name / "SKILL.md"
            assert skill_md.exists(), (
                f"{task.task_id}: SKILL.md not found at {skill_md}"
            )


class TestPatchTaskFields:
    def test_patch_tasks_have_expected_action_patch(self, suite):
        for task in suite.tasks:
            if task.task_id in _PATCH_TASK_IDS:
                assert task.expected_action == "patch", (
                    f"{task.task_id}: expected_action={task.expected_action!r}"
                )

    def test_patch_tasks_have_non_none_target_skill(self, suite):
        for task in suite.tasks:
            if task.task_id in _PATCH_TASK_IDS:
                assert task.target_skill is not None, (
                    f"{task.task_id}: target_skill is None"
                )

    def test_patch_tasks_have_non_none_stale_token(self, suite):
        for task in suite.tasks:
            if task.task_id in _PATCH_TASK_IDS:
                assert task.stale_token is not None, (
                    f"{task.task_id}: stale_token is None"
                )

    def test_patch_task_stale_tokens_match_skill_content(self, suite):
        for task in suite.tasks:
            if task.task_id in _PATCH_TASK_IDS:
                skill_name = _SKILL_NAME_BY_TASK[task.task_id]
                skill_md = SUITE_DIR / task.skills_src / skill_name / "SKILL.md"
                content = skill_md.read_text()
                assert task.stale_token in content, (
                    f"{task.task_id}: stale_token {task.stale_token!r} "
                    f"not found in {skill_md}"
                )


class TestControlTaskFields:
    def test_controls_have_skill_manage_forbidden(self, suite):
        for task in suite.tasks:
            if task.task_id in _CONTROL_TASK_IDS:
                assert "skill_manage" in task.forbidden_tools, (
                    f"{task.task_id}: skill_manage not in forbidden_tools"
                )

    def test_controls_have_no_expected_action(self, suite):
        for task in suite.tasks:
            if task.task_id in _CONTROL_TASK_IDS:
                assert task.expected_action is None, (
                    f"{task.task_id}: expected_action={task.expected_action!r}"
                )
