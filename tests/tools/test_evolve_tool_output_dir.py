"""The --output-dir CLI option forwards an exact directory to evolve().

Mirror of the skills output-dir contract: the cross-phase orchestrator drives
each evolver's CLI and needs --output-dir honored as an exact run dir so it can
capture gate_decision.json at a known path. The full evolve() loop needs real
LM spend, so we pin the contract at the CLI boundary.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from evolution.tools.evolve_tool import main as evolve_tool_main


def test_output_dir_forwarded_to_evolve(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    target = tmp_path / "run_here"
    with patch("evolution.tools.evolve_tool.evolve") as mock_evolve:
        result = CliRunner().invoke(
            evolve_tool_main,
            ["--tool", "demo", "--manifest", str(manifest), "--output-dir", str(target)],
        )
    assert result.exit_code == 0, result.output
    assert mock_evolve.call_args.kwargs["output_dir"] == target


def test_output_dir_defaults_to_none(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with patch("evolution.tools.evolve_tool.evolve") as mock_evolve:
        result = CliRunner().invoke(
            evolve_tool_main, ["--tool", "demo", "--manifest", str(manifest)]
        )
    assert result.exit_code == 0, result.output
    assert mock_evolve.call_args.kwargs["output_dir"] is None
