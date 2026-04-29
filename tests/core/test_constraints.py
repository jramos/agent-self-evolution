"""Tests for constraint validators."""

import pytest

from evolution.core.constraints import ConstraintValidator
from evolution.core.config import EvolutionConfig


@pytest.fixture(autouse=True, scope="session")
def _hermes_repo_env(tmp_path_factory):
    """Satisfy EvolutionConfig.hermes_agent_path's requirement without
    needing a real hermes-agent checkout. Without this the 16 tests in
    this file error at collection time when the env var is unset and no
    fallback paths exist on the running machine.
    """
    import os
    fake_repo = tmp_path_factory.mktemp("fake_hermes_repo")
    os.environ["HERMES_AGENT_REPO"] = str(fake_repo)
    yield


@pytest.fixture
def validator():
    config = EvolutionConfig()
    return ConstraintValidator(config)


class TestSizeConstraints:
    def test_skill_under_limit(self, validator):
        result = validator._check_size("x" * 1000, "skill")
        assert result.passed

    def test_skill_over_limit(self, validator):
        result = validator._check_size("x" * 20_000, "skill")
        assert not result.passed
        assert "exceeded" in result.message

    def test_tool_description_under_limit(self, validator):
        result = validator._check_size("Search files by content", "tool_description")
        assert result.passed

    def test_tool_description_over_limit(self, validator):
        result = validator._check_size("x" * 600, "tool_description")
        assert not result.passed




class TestNonEmpty:
    def test_non_empty_passes(self, validator):
        result = validator._check_non_empty("some content")
        assert result.passed

    def test_empty_fails(self, validator):
        result = validator._check_non_empty("")
        assert not result.passed

    def test_whitespace_only_fails(self, validator):
        result = validator._check_non_empty("   \n  ")
        assert not result.passed


class TestSkillStructure:
    def test_valid_skill(self, validator):
        skill = "---\nname: test-skill\ndescription: A test skill\n---\n\n# Test\nContent here"
        result = validator._check_skill_structure(skill)
        assert result.passed

    def test_missing_frontmatter(self, validator):
        skill = "# Test\nContent without frontmatter"
        result = validator._check_skill_structure(skill)
        assert not result.passed

    def test_missing_name(self, validator):
        skill = "---\ndescription: A test skill\n---\n\n# Test"
        result = validator._check_skill_structure(skill)
        assert not result.passed

    def test_missing_description(self, validator):
        skill = "---\nname: test-skill\n---\n\n# Test"
        result = validator._check_skill_structure(skill)
        assert not result.passed


class TestValidateStatic:
    """Static checks only — no growth check, no quality gate."""

    def test_skips_growth_check(self, validator):
        # Even with a clearly-bloated artifact, validate_static doesn't
        # fire the growth check (it has no baseline to compare to).
        evolved = "---\nname: x\ndescription: y\n---\n" + "z" * 5000
        results = validator.validate_static(evolved, "skill")
        names = {r.constraint_name for r in results}
        assert "growth_quality_gate" not in names
        assert "growth_limit" not in names

    def test_runs_size_non_empty_structure(self, validator):
        skill = "---\nname: t\ndescription: d\n---\n\n# Body\nstuff"
        results = validator.validate_static(skill, "skill")
        names = {r.constraint_name for r in results}
        assert "size_limit" in names
        assert "non_empty" in names
        assert "skill_structure" in names


class TestGrowthQualityGate:
    """Continuous quality-gated growth check.

    Curve: required_improvement = max(0, slope * (growth - growth_free)).
    Defaults: growth_free=0.20, slope=0.30.
    """

    @staticmethod
    def _bootstrap(mean: float, lower: float, upper: float = None, n: int = 12) -> dict:
        """Build a stub bootstrap_result dict matching paired_bootstrap's shape."""
        return {
            "mean": mean,
            "lower_bound": lower,
            "upper_bound": upper if upper is not None else max(mean, lower),
            "n_examples": n,
            "n_resamples": 2000,
            "confidence": 0.90,
        }

    def test_no_regression_branch_passes_at_zero_improvement(self, validator):
        # Growth +15% (under 20% free threshold) → required = 0;
        # no_regression_only branch passes on mean ≥ 0.
        baseline = "x" * 1000
        evolved = "x" * 1150
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.0, lower=-0.05),
        )
        assert result.passed
        assert "no improvement required" in result.message

    def test_no_regression_branch_fails_on_negative_mean(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1150  # +15%, no required improvement
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=-0.05, lower=-0.10),
        )
        assert not result.passed
        assert "regression" in result.message

    def test_negative_growth_falls_into_no_regression_branch(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 800  # -20%
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.0, lower=-0.10),
        )
        assert result.passed

    def test_zero_baseline_treated_as_zero_growth(self, validator):
        result = validator._check_growth_with_quality_gate(
            "anything", "", self._bootstrap(mean=0.0, lower=0.0),
        )
        assert result.passed

    def test_dual_check_passes(self, validator):
        # Growth +40% → required = 0.30 * (0.40 - 0.20) = 0.06.
        # mean +0.07 ≥ required 0.06 AND lower +0.005 > 0 → pass.
        baseline = "x" * 1000
        evolved = "x" * 1400
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.07, lower=0.005),
        )
        assert result.passed
        assert "lower-bound +0.005 > 0" in result.message

    def test_dual_check_fails_on_mean_below_required(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1400  # +40%, required +0.06
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.04, lower=0.01),
        )
        assert not result.passed
        assert "mean +0.040 < required +0.060" in result.message

    def test_dual_check_fails_on_lower_bound_at_zero(self, validator):
        # Even a fat mean fails when the bootstrap lower bound is at the
        # noise floor — this is the regression-risk failure mode.
        baseline = "x" * 1000
        evolved = "x" * 1400  # +40%, required +0.06
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.10, lower=-0.02),
        )
        assert not result.passed
        assert "regression risk" in result.message

    def test_pr6_obsidian_data_under_principled_gate(self, validator):
        # PR #6 deployed obsidian at +24.2% growth, mean +0.015,
        # lower_bound ≈ -0.060 (per the actual paired_bootstrap on its
        # per-example scores). Under the principled gate this should
        # REJECT — that's the entire point of this PR.
        baseline = "x" * 1264
        evolved = "x" * 1570
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.015, lower=-0.060),
        )
        assert not result.passed
        assert "regression risk" in result.message


class TestAbsoluteCharCeiling:
    def test_under_ceiling_passes(self, validator):
        result = validator._check_absolute_chars("x" * 4000)
        assert result.passed

    def test_over_ceiling_fails(self, validator):
        # Default max_absolute_chars=5000.
        result = validator._check_absolute_chars("x" * 6000)
        assert not result.passed
        assert "1000 over" in result.message

    def test_ceiling_independent_of_growth(self, validator):
        # An artifact with low growth (+3.4%) but huge absolute size
        # still fails the absolute ceiling — escape hatch wouldn't apply.
        baseline = "x" * 5800
        evolved = "x" * 6000  # +3.4%, would pass quality gate even at 0 improvement
        # Quality gate: passes (no_regression branch, mean ≥ 0).
        bootstrap = TestGrowthQualityGate._bootstrap(mean=0.0, lower=-0.05)
        gate = validator._check_growth_with_quality_gate(evolved, baseline, bootstrap)
        assert gate.passed
        # Absolute ceiling: fails.
        ceiling = validator._check_absolute_chars(evolved)
        assert not ceiling.passed


class TestValidateGrowthWithQuality:
    """The combined entry point: growth-with-quality + absolute ceiling."""

    def test_runs_both_checks(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1100  # safe growth, safe absolute
        bootstrap = TestGrowthQualityGate._bootstrap(mean=0.0, lower=-0.02)
        results = validator.validate_growth_with_quality(evolved, baseline, bootstrap)
        names = {r.constraint_name for r in results}
        assert "growth_quality_gate" in names
        assert "absolute_char_ceiling" in names

    def test_absolute_ceiling_blocks_even_when_growth_ok(self, validator):
        # +200% growth from a tiny baseline that would normally need
        # significant improvement, but absolute size is what kills it.
        baseline = "x" * 100
        evolved = "x" * 6000  # +5900%; absolute ceiling 5000 fails.
        bootstrap = TestGrowthQualityGate._bootstrap(mean=0.0, lower=-0.05)
        results = validator.validate_growth_with_quality(evolved, baseline, bootstrap)
        # Both fail in this case (huge growth + huge size).
        assert any(not r.passed for r in results)
