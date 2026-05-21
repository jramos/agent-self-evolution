"""Integration tests for the saturation pre-flight wiring in evolve_tool.

Mocks the LM and the dataset builder so each test runs in ≤2s —
zero real LM spend. Mirrors tests/tools/test_evolve_tool_closed_loop.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from evolution.tools.evolve_tool import main as evolve_tool_main


def _minimal_manifest_dir(tmp_path: Path) -> Path:
    """Write a one-tool _SCHEMA file so the manifest loads."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "__init__.py").write_text("")
    (tools_dir / "my_tools.py").write_text(
        'WRITE_FILE_SCHEMA = {\n'
        '    "name": "write_file",\n'
        '    "description": "Write to a file.",\n'
        '    "input_schema": {"type": "object", "properties": {}},\n'
        '}\n'
    )
    return tools_dir


@pytest.fixture
def manifest_dir(tmp_path):
    return _minimal_manifest_dir(tmp_path)


class TestSaturationPreflightCLI:
    def test_no_saturation_check_flag_skips_helper(self, manifest_dir):
        """--no-saturation-check skips the preflight helper entirely."""
        with patch(
            "evolution.tools.evolve_tool.saturation_preflight"
        ) as mock_preflight, patch(
            "evolution.tools.evolve_tool._preflight_lm_credentials"
        ), patch("evolution.tools.evolve_tool.dspy.GEPA"):
            runner = CliRunner()
            runner.invoke(
                evolve_tool_main,
                ["--tool", "write_file", "--manifest", str(manifest_dir),
                 "--iterations", "1", "--no-saturation-check", "--no-preflight"],
            )
            mock_preflight.assert_not_called()

    def test_healthy_band_does_not_prompt(self, manifest_dir):
        """When preflight returns healthy, no panel, no prompt; GEPA proceeds."""
        from evolution.core.saturation_check import SaturationReport
        healthy = SaturationReport(
            band="healthy", holdout_score=0.5, holdout_n=10,
            holdout_per_example=[0.5] * 10, suggestions=[], thresholds={},
        )
        with patch(
            "evolution.tools.evolve_tool.saturation_preflight", return_value=healthy
        ), patch(
            "evolution.tools.evolve_tool._preflight_lm_credentials"
        ), patch(
            "evolution.tools.evolve_tool.interactive_confirm"
        ) as mock_confirm, patch("evolution.tools.evolve_tool.dspy.GEPA"):
            runner = CliRunner()
            runner.invoke(
                evolve_tool_main,
                ["--tool", "write_file", "--manifest", str(manifest_dir),
                 "--iterations", "1", "--no-preflight"],
            )
            mock_confirm.assert_not_called()

    def test_saturated_band_non_interactive_aborts(self, manifest_dir):
        """no_headroom band in non-interactive context exits cleanly without GEPA."""
        from evolution.core.saturation_check import SaturationReport
        saturated = SaturationReport(
            band="no_headroom", holdout_score=0.99, holdout_n=50,
            holdout_per_example=[1.0] * 50,
            suggestions=["Try a harder suite"], thresholds={},
        )
        gepa_mock = MagicMock()
        with patch(
            "evolution.tools.evolve_tool.saturation_preflight", return_value=saturated
        ), patch(
            "evolution.tools.evolve_tool._preflight_lm_credentials"
        ), patch(
            "evolution.tools.evolve_tool.is_non_interactive", return_value=True
        ), patch("evolution.tools.evolve_tool.dspy.GEPA", gepa_mock):
            runner = CliRunner()
            result = runner.invoke(
                evolve_tool_main,
                ["--tool", "write_file", "--manifest", str(manifest_dir),
                 "--iterations", "1", "--no-preflight"],
            )
            gepa_mock.assert_not_called()
            assert "force-saturation-check" in result.output

    def test_force_saturation_check_overrides_abort(self, manifest_dir):
        """--force-saturation-check renders panel but lets GEPA run."""
        from evolution.core.saturation_check import SaturationReport
        saturated = SaturationReport(
            band="no_headroom", holdout_score=0.99, holdout_n=50,
            holdout_per_example=[1.0] * 50,
            suggestions=["x"], thresholds={},
        )
        with patch(
            "evolution.tools.evolve_tool.saturation_preflight", return_value=saturated
        ), patch(
            "evolution.tools.evolve_tool._preflight_lm_credentials"
        ), patch(
            "evolution.tools.evolve_tool.interactive_confirm"
        ) as mock_confirm, patch("evolution.tools.evolve_tool.dspy.GEPA"):
            runner = CliRunner()
            runner.invoke(
                evolve_tool_main,
                ["--tool", "write_file", "--manifest", str(manifest_dir),
                 "--iterations", "1", "--force-saturation-check", "--no-preflight"],
            )
            # confirm is bypassed when --force-saturation-check is set
            mock_confirm.assert_not_called()
