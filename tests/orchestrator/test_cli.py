"""CLI: propose-only enforcement (--allow-pr strip + cost-cap guard) + dry-run."""

from __future__ import annotations

import json
import textwrap

from click.testing import CliRunner

from evolution.orchestrator.__main__ import main


def _spec_file(tmp_path, text):
    p = tmp_path / "run.yaml"
    p.write_text(textwrap.dedent(text))
    return p


def _argv_for(summary, phase):
    return next(r["argv"] for r in summary["phases"] if r["phase"] == phase)


def _run(tmp_path, spec_text, *extra):
    spec = _spec_file(tmp_path, spec_text)
    out = tmp_path / "run_root"
    result = CliRunner().invoke(
        main, ["--spec", str(spec), "--base-output", str(out), "--dry-run", *extra]
    )
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return result, summary


def test_allow_pr_off_strips_create_pr(tmp_path):
    result, summary = _run(tmp_path, """
        phases:
          - { phase: code, name: x.py,
              args: { repo: /r, visible_test: v, holdout_test: h }, create_pr: true }
    """)
    assert result.exit_code == 0, result.output
    argv = _argv_for(summary, "code")
    assert "--no-create-pr" in argv
    assert "--create-pr" not in argv  # exact membership; --no-create-pr is distinct


def test_allow_pr_on_honors_create_pr(tmp_path):
    result, summary = _run(tmp_path, """
        phases:
          - { phase: code, name: x.py,
              args: { repo: /r, visible_test: v, holdout_test: h }, create_pr: true }
    """, "--allow-pr")
    assert result.exit_code == 0, result.output
    argv = _argv_for(summary, "code")
    assert "--create-pr" in argv and "--no-create-pr" not in argv


def test_allow_pr_without_cost_cap_raises_usage_error(tmp_path):
    result, _ = _run(tmp_path, """
        phases:
          - { phase: skills, name: demo, create_pr: true }
    """, "--allow-pr")
    assert result.exit_code != 0
    assert "spend ceiling" in result.output


def test_allow_pr_with_cost_cap_ok(tmp_path):
    result, summary = _run(tmp_path, """
        phases:
          - { phase: skills, name: demo, args: { max_total_cost_usd: 50 }, create_pr: true }
    """, "--allow-pr")
    assert result.exit_code == 0, result.output
    assert "--create-pr" in _argv_for(summary, "skills")


def test_dry_run_smoke(tmp_path):
    result, summary = _run(tmp_path, """
        phases:
          - { phase: skills, name: demo, args: { iterations: 8 } }
          - { phase: tools, name: fetch, args: { manifest: m.json } }
    """)
    assert result.exit_code == 0, result.output
    assert summary["n_phases"] == 2
    assert all(r["status"] == "skipped" for r in summary["phases"])
