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

    def test_cache_reuse_skips_baseline_re_eval_after_gepa(self, skill_dir):
        """When the saturation preflight runs, the cached baseline holdout
        scores must be reused at the post-GEPA evaluation site — the baseline
        module should NOT be re-scored on the holdout after GEPA finishes.
        This is the 'net cost ~zero' contract."""
        from evolution.core.saturation_check import SaturationReport
        from evolution.skills.knee_point import CandidatePick
        from unittest.mock import MagicMock

        healthy = SaturationReport(
            band="healthy", holdout_score=0.6, holdout_n=10,
            holdout_per_example=[0.6] * 10, suggestions=[], thresholds={},
        )
        # Fake knee-point result so execution reaches the holdout site.
        # skill_text must be a non-empty string so SkillModule can be built.
        fake_module = MagicMock()
        fake_module.skill_text = "evolved skill text"
        knee_pick = CandidatePick(
            module=fake_module, skill_text="evolved skill text", body_chars=18,
            val_score=0.8, val_rank_in_band=1, band_size=1, epsilon=0.1,
            fallback="knee", picked_idx=0, gepa_default_idx=0,
            gepa_default_body_chars=18, band_roster=[],
        )
        with patch(
            "evolution.skills.evolve_skill.saturation_preflight", return_value=healthy
        ), patch(
            "evolution.skills.evolve_skill._preflight_lm_credentials"
        ), patch("evolution.skills.evolve_skill.dspy.GEPA"), patch(
            "evolution.skills.evolve_skill.select_knee_point", return_value=knee_pick
        ), patch(
            "evolution.skills.evolve_skill._holdout_evaluate_with_metric"
        ) as mock_holdout_eval:
            mock_holdout_eval.return_value = (0.6, [0.6] * 10)
            runner = CliRunner()
            runner.invoke(
                evolve_skill_main,
                ["--skill", "demo-skill", "--skill-source-dir", str(skill_dir),
                 "--iterations", "1", "--no-preflight"],
            )
            # With preflight populating the cache, baseline should NOT be
            # re-evaluated post-GEPA. Only evolved should be evaluated, so
            # _holdout_evaluate_with_metric is called exactly once.
            assert mock_holdout_eval.call_count == 1, (
                f"Expected baseline holdout to be reused from preflight cache "
                f"(1 call for evolved only), got {mock_holdout_eval.call_count}"
            )
