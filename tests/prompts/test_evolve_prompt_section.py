"""Wiring tests for evolve_prompt_section — pure helpers + dry-run (no LM/agent)."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

from click.testing import CliRunner

from evolution.prompts.evolve_prompt_section import (
    _make_layer2_factory,
    _section_text_from_candidate,
    _split_train_holdout,
    evolve_prompt_section,
    main,
)
from evolution.prompts.prompt_module import PromptModule
from evolution.validation.task import Task


def _task(task_id: str, rubric: str | None = None) -> Task:
    return Task(
        task_id=task_id, user_message="m", expected_tools=("memory",),
        expected_save_content=rubric,
    )


def test_split_is_deterministic_and_non_empty():
    tasks = tuple(_task(f"t{i}") for i in range(10))
    train1, holdout1 = _split_train_holdout(tasks, holdout_ratio=0.5, seed=42)
    train2, holdout2 = _split_train_holdout(tasks, holdout_ratio=0.5, seed=42)
    assert [t.task_id for t in train1] == [t.task_id for t in train2]
    assert [t.task_id for t in holdout1] == [t.task_id for t in holdout2]
    assert train1 and holdout1
    assert len(train1) + len(holdout1) == 10


def test_split_keeps_both_sides_non_empty_at_extremes():
    tasks = tuple(_task(f"t{i}") for i in range(4))
    train, holdout = _split_train_holdout(tasks, holdout_ratio=1.0, seed=1)
    assert train and holdout  # never starve the train side


def test_layer2_factory_returns_none_without_rubric():
    factory = _make_layer2_factory(judge=None)
    assert factory(_task("t1", rubric=None)) is None
    assert callable(factory(_task("t2", rubric="a rubric")))


def test_section_text_from_candidate_module_and_dict():
    module = PromptModule("MEMORY_GUIDANCE", "candidate body")
    assert _section_text_from_candidate(module, "MEMORY_GUIDANCE") == "candidate body"
    instructions = module.passthrough.predict.signature.instructions
    assert (
        _section_text_from_candidate(
            {"passthrough.predict": instructions}, "MEMORY_GUIDANCE"
        )
        == "candidate body"
    )


def _fake_repo(tmp_path: Path) -> Path:
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "prompt_builder.py").write_text(textwrap.dedent('''\
        MEMORY_GUIDANCE = "Save durable facts about the user."
    '''))
    return tmp_path


def _suite(tmp_path: Path) -> Path:
    p = tmp_path / "suite.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"task_id": "s1", "user_message": "I use uv.",
         "expected_tools": ["memory"], "expected_save_content": "prefers uv"},
        {"task_id": "n1", "user_message": "summarize work",
         "expected_tools": [], "forbidden_tools": ["memory"]},
    ]) + "\n")
    return p


def test_dry_run_writes_gate_decision(tmp_path):
    repo = _fake_repo(tmp_path)
    suite = _suite(tmp_path)
    out = tmp_path / "out"
    result = evolve_prompt_section(
        section_name="MEMORY_GUIDANCE", hermes_repo=repo, tasks_path=suite,
        dry_run=True, output_dir=out,
    )
    assert result["decision"] == "dry_run"
    gate = json.loads((out / "gate_decision.json").read_text())
    assert gate["artifact_type"] == "prompt_section"
    assert gate["target_section"] == "MEMORY_GUIDANCE"
    # The baseline file must be byte-identical after a dry run (untouched).
    assert "Save durable facts about the user." in (
        repo / "agent" / "prompt_builder.py"
    ).read_text()


def test_cli_dry_run_exits_zero(tmp_path):
    repo = _fake_repo(tmp_path)
    suite = _suite(tmp_path)
    runner = CliRunner()
    res = runner.invoke(main, [
        "--section", "MEMORY_GUIDANCE",
        "--hermes-repo", str(repo),
        "--tasks", str(suite),
        "--dry-run",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert res.exit_code == 0, res.output
