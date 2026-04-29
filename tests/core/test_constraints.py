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


class TestLegacyGrowthConstraint:
    """The single-ratio cliff is kept for back-compat in validate_all when
    no holdout improvement signal is available. Primary deploy gate is
    the continuous quality-gated curve below.
    """

    def test_acceptable_growth(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1100  # +10%, under default 30%
        result = validator._check_growth_legacy(evolved, baseline)
        assert result.passed

    def test_excessive_growth(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1500  # +50%, over default 30%
        result = validator._check_growth_legacy(evolved, baseline)
        assert not result.passed

    def test_shrinkage_is_ok(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 800  # -20%
        result = validator._check_growth_legacy(evolved, baseline)
        assert result.passed


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


class TestValidateAll:
    """Back-compat path: static + legacy growth (no quality signal)."""

    def test_valid_skill_passes_all(self, validator):
        skill = "---\nname: test\ndescription: Test skill\n---\n\n# Procedure\n1. Do thing"
        results = validator.validate_all(skill, "skill")
        assert all(r.passed for r in results), [r.message for r in results if not r.passed]

    def test_empty_skill_fails(self, validator):
        results = validator.validate_all("", "skill")
        failed = [r for r in results if not r.passed]
        assert len(failed) > 0


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

    def test_passes_below_free_threshold_with_zero_improvement(self, validator):
        # Growth +15% (under 20% free threshold) — required = 0; pass.
        baseline = "x" * 1000
        evolved = "x" * 1150
        result = validator._check_growth_with_quality_gate(evolved, baseline, 0.0)
        assert result.passed
        assert "+15.0%" in result.message

    def test_negative_growth_always_passes(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 800  # -20% growth
        result = validator._check_growth_with_quality_gate(evolved, baseline, 0.0)
        assert result.passed

    def test_zero_baseline_treated_as_zero_growth(self, validator):
        # Defensive: division-by-zero guard. growth=0, no improvement
        # required, passes regardless of artifact length.
        result = validator._check_growth_with_quality_gate("anything", "", 0.0)
        assert result.passed

    def test_above_free_threshold_requires_improvement(self, validator):
        # Growth +40% → required = 0.30 * (0.40 - 0.20) = 0.06.
        baseline = "x" * 1000
        evolved = "x" * 1400
        # Improvement +0.05 < required 0.06 → fail.
        result = validator._check_growth_with_quality_gate(evolved, baseline, 0.05)
        assert not result.passed
        assert "+0.060" in result.message
        # Improvement +0.07 >= required 0.06 → pass.
        result = validator._check_growth_with_quality_gate(evolved, baseline, 0.07)
        assert result.passed

    def test_pr5_obsidian_data_point_with_default_curve(self, validator):
        # PR #5 obsidian: 1264 → 1570 chars (+24.2% growth).
        # Required improvement = 0.30 * (0.242 - 0.20) = 0.0126.
        # We don't know the actual holdout improvement (PR #5's holdout
        # never ran), but if it's anywhere near the typical +0.05-0.10
        # range we'd expect from a 1.0-valset run, it deploys.
        baseline = "x" * 1264
        evolved = "x" * 1570
        result = validator._check_growth_with_quality_gate(evolved, baseline, 0.05)
        assert result.passed
        # Conversely, an essentially-flat improvement fails.
        result = validator._check_growth_with_quality_gate(evolved, baseline, 0.005)
        assert not result.passed


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
        # Quality gate: passes.
        gate = validator._check_growth_with_quality_gate(evolved, baseline, 0.0)
        assert gate.passed
        # Absolute ceiling: fails.
        ceiling = validator._check_absolute_chars(evolved)
        assert not ceiling.passed


class TestValidateGrowthWithQuality:
    """The combined entry point: growth-with-quality + absolute ceiling."""

    def test_runs_both_checks(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1100  # safe growth, safe absolute
        results = validator.validate_growth_with_quality(evolved, baseline, 0.0)
        names = {r.constraint_name for r in results}
        assert "growth_quality_gate" in names
        assert "absolute_char_ceiling" in names

    def test_absolute_ceiling_blocks_even_when_growth_ok(self, validator):
        # +200% growth from a tiny baseline that would normally need
        # significant improvement, but absolute size is what kills it.
        baseline = "x" * 100
        evolved = "x" * 6000  # +5900%; absolute ceiling 5000 fails.
        results = validator.validate_growth_with_quality(evolved, baseline, 0.0)
        # Both fail in this case (huge growth + huge size).
        assert any(not r.passed for r in results)
