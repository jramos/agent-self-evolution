"""Integration tests for saturation pre-flight wiring in evolve_skill.

Symmetric to tests/tools/test_evolve_tool_saturation_preflight.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from evolution.skills.evolve_skill import main as evolve_skill_main


@pytest.fixture
def skill_dir(tmp_path):
    """Write a minimal SKILL.md so skill discovery succeeds."""
    skills_root = tmp_path / "skills"
    skill_path = skills_root / "demo-skill"
    skill_path.mkdir(parents=True)
    (skill_path / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: a test skill\n---\n\nDo X.\n"
    )
    return skills_root


class TestSaturationPreflightCLI:
    def test_no_saturation_check_flag_skips_helper(self, skill_dir):
        with patch(
            "evolution.skills.evolve_skill.saturation_preflight"
        ) as mock_preflight, patch(
            "evolution.skills.evolve_skill._preflight_lm_credentials"
        ), patch("evolution.skills.evolve_skill.dspy.GEPA"):
            runner = CliRunner()
            runner.invoke(
                evolve_skill_main,
                ["--skill", "demo-skill", "--skill-source-dir", str(skill_dir),
                 "--iterations", "1", "--no-saturation-check", "--no-preflight"],
            )
            mock_preflight.assert_not_called()

    def test_healthy_band_does_not_prompt(self, skill_dir):
        from evolution.core.saturation_check import SaturationReport
        healthy = SaturationReport(
            band="healthy", holdout_score=0.5, holdout_n=10,
            holdout_per_example=[0.5] * 10, suggestions=[], thresholds={},
        )
        with patch(
            "evolution.skills.evolve_skill.saturation_preflight", return_value=healthy
        ), patch(
            "evolution.skills.evolve_skill._preflight_lm_credentials"
        ), patch(
            "evolution.skills.evolve_skill.interactive_confirm"
        ) as mock_confirm, patch("evolution.skills.evolve_skill.dspy.GEPA"):
            runner = CliRunner()
            runner.invoke(
                evolve_skill_main,
                ["--skill", "demo-skill", "--skill-source-dir", str(skill_dir),
                 "--iterations", "1", "--no-preflight"],
            )
            mock_confirm.assert_not_called()

    def test_saturated_band_non_interactive_aborts(self, skill_dir):
        from evolution.core.saturation_check import SaturationReport
        saturated = SaturationReport(
            band="no_headroom", holdout_score=0.99, holdout_n=50,
            holdout_per_example=[1.0] * 50, suggestions=["x"], thresholds={},
        )
        gepa_mock = MagicMock()
        with patch(
            "evolution.skills.evolve_skill.saturation_preflight", return_value=saturated
        ), patch(
            "evolution.skills.evolve_skill._preflight_lm_credentials"
        ), patch(
            "evolution.skills.evolve_skill.is_non_interactive", return_value=True
        ), patch("evolution.skills.evolve_skill.dspy.GEPA", gepa_mock):
            runner = CliRunner()
            result = runner.invoke(
                evolve_skill_main,
                ["--skill", "demo-skill", "--skill-source-dir", str(skill_dir),
                 "--iterations", "1", "--no-preflight"],
            )
            gepa_mock.assert_not_called()
            assert "force-saturation-check" in result.output

    def test_force_saturation_check_overrides_abort(self, skill_dir):
        from evolution.core.saturation_check import SaturationReport
        saturated = SaturationReport(
            band="no_headroom", holdout_score=0.99, holdout_n=50,
            holdout_per_example=[1.0] * 50, suggestions=["x"], thresholds={},
        )
        with patch(
            "evolution.skills.evolve_skill.saturation_preflight", return_value=saturated
        ), patch(
            "evolution.skills.evolve_skill._preflight_lm_credentials"
        ), patch(
            "evolution.skills.evolve_skill.interactive_confirm"
        ) as mock_confirm, patch("evolution.skills.evolve_skill.dspy.GEPA"):
            runner = CliRunner()
            runner.invoke(
                evolve_skill_main,
                ["--skill", "demo-skill", "--skill-source-dir", str(skill_dir),
                 "--iterations", "1", "--force-saturation-check", "--no-preflight"],
            )
            mock_confirm.assert_not_called()
