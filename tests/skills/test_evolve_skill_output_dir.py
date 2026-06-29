"""The --output-dir CLI option forwards an exact directory to evolve().

The cross-phase orchestrator relies on every evolver honoring --output-dir as
an exact run dir (no <timestamp> appended) so it can capture each phase's
gate_decision.json at a known path. The full evolve() loop needs real LM spend,
so we pin the contract at the CLI boundary: the flag reaches evolve() as the
Path given, and defaults to None (the legacy output/<skill>/<timestamp>/ path).
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from evolution.skills.evolve_skill import main as evolve_skill_main


def test_output_dir_forwarded_to_evolve(tmp_path):
    target = tmp_path / "run_here"
    with patch("evolution.skills.evolve_skill.evolve") as mock_evolve:
        result = CliRunner().invoke(
            evolve_skill_main, ["--skill", "demo", "--output-dir", str(target)]
        )
    assert result.exit_code == 0, result.output
    assert mock_evolve.call_args.kwargs["output_dir"] == target


def test_output_dir_defaults_to_none(tmp_path):
    with patch("evolution.skills.evolve_skill.evolve") as mock_evolve:
        result = CliRunner().invoke(evolve_skill_main, ["--skill", "demo"])
    assert result.exit_code == 0, result.output
    assert mock_evolve.call_args.kwargs["output_dir"] is None
