"""Per-phase adapter argv building, output-dir resolution, required fields."""

from __future__ import annotations

import sys
from pathlib import Path

from evolution.orchestrator.adapters import PHASE_ADAPTERS, _flagify, _slug
from evolution.orchestrator.spec import PhaseSpec


def _arg_value(argv, flag):
    return argv[argv.index(flag) + 1]


class TestFlagify:
    def test_bool_true_is_bare_flag(self):
        assert _flagify({"dry_run": True}) == ["--dry-run"]

    def test_false_and_none_omitted(self):
        assert _flagify({"a": False, "b": None, "c": 1}) == ["--c", "1"]

    def test_list_repeats_flag(self):
        assert _flagify({"floor_path": ["a", "b"]}) == ["--floor-path", "a", "--floor-path", "b"]

    def test_underscore_to_dash(self):
        assert _flagify({"max_total_cost_usd": 50}) == ["--max-total-cost-usd", "50"]


class TestSlug:
    def test_code_relpath_is_filesystem_safe(self):
        assert _slug("hermes/tools/fetch_url.py") == "hermes_tools_fetch_url.py"

    def test_empty_falls_back(self):
        assert _slug("///") == "phase"


class TestSkillsAdapter:
    adapter = PHASE_ADAPTERS["skills"]

    def test_argv_has_skill_and_output_dir_no_pr_by_default(self):
        ps = PhaseSpec("skills", "demo", {"iterations": 8}, create_pr=False)
        argv = self.adapter.build_argv(ps, Path("/run"))
        assert argv[:5] == [sys.executable, "-m", "evolution.skills.evolve_skill", "--skill", "demo"]
        assert _arg_value(argv, "--output-dir") == "/run/phases/skills-demo"
        # Explicit off-switch on the strip path (not mere omission).
        assert "--no-create-pr" in argv and "--create-pr" not in argv

    def test_create_pr_emits_create_pr(self):
        ps = PhaseSpec("skills", "demo", {}, create_pr=True)
        argv = self.adapter.build_argv(ps, Path("/run"))
        assert "--create-pr" in argv and "--no-create-pr" not in argv

    def test_required_fields_just_name(self):
        assert self.adapter.required_fields() == frozenset({"name"})


class TestCodeAdapter:
    adapter = PHASE_ADAPTERS["code"]

    def test_argv_tool_relpath_and_no_create_pr_default(self):
        ps = PhaseSpec(
            "code", "hermes/tools/x.py",
            {"repo": "/r", "visible_test": "v", "holdout_test": "h", "repair_rounds": 5},
            create_pr=False,
        )
        argv = self.adapter.build_argv(ps, Path("/run"))
        assert _arg_value(argv, "--tool") == "hermes/tools/x.py"
        assert _arg_value(argv, "--repo") == "/r"
        assert "--no-create-pr" in argv and "--create-pr" not in argv
        assert _arg_value(argv, "--output-dir") == "/run/phases/code-hermes_tools_x.py"

    def test_create_pr_emits_create_pr(self):
        ps = PhaseSpec("code", "x.py", {"repo": "/r", "visible_test": "v", "holdout_test": "h"},
                       create_pr=True)
        argv = self.adapter.build_argv(ps, Path("/run"))
        assert "--create-pr" in argv and "--no-create-pr" not in argv

    def test_required_fields(self):
        assert self.adapter.required_fields() == frozenset(
            {"name", "repo", "visible_test", "holdout_test"}
        )


class TestPromptsAdapter:
    def test_no_create_pr_flag_ever(self):
        adapter = PHASE_ADAPTERS["prompts"]
        ps = PhaseSpec("prompts", "sec", {"tasks": "t.jsonl"}, create_pr=False)
        argv = adapter.build_argv(ps, Path("/run"))
        assert "--create-pr" not in argv and "--no-create-pr" not in argv
        assert adapter.supports_create_pr is False
