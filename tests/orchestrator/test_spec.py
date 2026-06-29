"""Run-spec loading + validation (fail-fast SpecError at load)."""

from __future__ import annotations

import textwrap

import pytest

from evolution.orchestrator.spec import SpecError, load_spec


def _write(tmp_path, text):
    p = tmp_path / "run.yaml"
    p.write_text(textwrap.dedent(text))
    return p


def test_valid_spec_loads(tmp_path):
    spec = load_spec(_write(tmp_path, """
        defaults: { seed: 42 }
        phases:
          - { phase: skills, name: web_research, args: { iterations: 8 } }
          - { phase: tools,  name: fetch, args: { manifest: m.json } }
    """))
    assert spec.defaults == {"seed": 42}
    assert [p.phase for p in spec.phases] == ["skills", "tools"]
    assert spec.phases[0].name == "web_research"
    assert spec.phases[0].create_pr is False


def test_missing_required_arg_raises(tmp_path):
    with pytest.raises(SpecError, match="requires args.*manifest"):
        load_spec(_write(tmp_path, """
            phases:
              - { phase: tools, name: fetch }
        """))


def test_code_missing_holdout_raises(tmp_path):
    with pytest.raises(SpecError, match="holdout_test"):
        load_spec(_write(tmp_path, """
            phases:
              - { phase: code, name: x.py, args: { repo: /r, visible_test: v } }
        """))


def test_create_pr_inside_args_raises(tmp_path):
    with pytest.raises(SpecError, match="top-level phase field"):
        load_spec(_write(tmp_path, """
            phases:
              - { phase: skills, name: demo, args: { create_pr: true } }
        """))


def test_output_dir_inside_args_raises(tmp_path):
    with pytest.raises(SpecError, match="set by the orchestrator"):
        load_spec(_write(tmp_path, """
            phases:
              - { phase: skills, name: demo, args: { output_dir: /tmp/x } }
        """))


def test_create_pr_on_prompts_raises(tmp_path):
    with pytest.raises(SpecError, match="does not support create_pr"):
        load_spec(_write(tmp_path, """
            phases:
              - { phase: prompts, name: sec, args: { tasks: t.jsonl }, create_pr: true }
        """))


def test_unknown_top_level_key_raises(tmp_path):
    with pytest.raises(SpecError, match="unknown top-level"):
        load_spec(_write(tmp_path, """
            phasez:
              - { phase: skills, name: demo }
        """))


def test_bad_phase_name_raises(tmp_path):
    with pytest.raises(SpecError, match="'phase' must be one of"):
        load_spec(_write(tmp_path, """
            phases:
              - { phase: params, name: demo }
        """))


def test_empty_phases_raises(tmp_path):
    with pytest.raises(SpecError, match="non-empty list"):
        load_spec(_write(tmp_path, "phases: []"))
