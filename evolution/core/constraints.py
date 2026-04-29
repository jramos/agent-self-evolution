"""Constraint validators for evolved artifacts.

Every candidate variant must pass ALL constraints before it can be
considered valid. Failed constraints = immediate rejection.
"""

import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig


@dataclass
class ConstraintResult:
    """Result of constraint validation."""
    passed: bool
    constraint_name: str
    message: str
    details: Optional[str] = None


class ConstraintValidator:
    """Validates evolved artifacts against hard constraints."""

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def validate_all(
        self,
        artifact_text: str,
        artifact_type: str,
        baseline_text: Optional[str] = None,
    ) -> list[ConstraintResult]:
        """Back-compat shim. New code should call validate_static() and
        validate_growth_with_quality() explicitly so the deploy decision
        sees the holdout improvement signal. Without that signal the
        legacy growth check (a single-ratio cliff) is the only thing
        available — kept here so external callers don't break.
        """
        results = self.validate_static(artifact_text, artifact_type)

        if baseline_text is not None:
            # Fall back to the legacy ratio-only check when no quality
            # signal is available. Continuous quality gate requires
            # holdout_improvement; use validate_growth_with_quality() for
            # that path.
            results.append(self._check_growth_legacy(artifact_text, baseline_text))

        return results

    def validate_static(
        self,
        artifact_text: str,
        artifact_type: str,
    ) -> list[ConstraintResult]:
        """Static checks that don't need a baseline or holdout signal:
        size, non-empty, structural integrity. Run these first in the
        evolve flow so a broken artifact short-circuits before we spend
        judge-call budget on the holdout pass.
        """
        results = [self._check_size(artifact_text, artifact_type)]
        results.append(self._check_non_empty(artifact_text))
        if artifact_type == "skill":
            results.append(self._check_skill_structure(artifact_text))
        return results

    def validate_growth_with_quality(
        self,
        artifact_text: str,
        baseline_text: str,
        holdout_improvement: float,
    ) -> list[ConstraintResult]:
        """Quality-gated growth check + absolute-char ceiling.

        The growth check uses a continuous curve where required holdout
        improvement scales linearly with growth above the free threshold.
        The absolute-char ceiling is independent of growth — applies even
        when the curve would pass — to backstop runaway absolute size.
        """
        return [
            self._check_growth_with_quality_gate(
                artifact_text, baseline_text, holdout_improvement
            ),
            self._check_absolute_chars(artifact_text),
        ]

    def run_test_suite(self, hermes_repo: Path) -> ConstraintResult:
        """Run the full hermes-agent test suite. Must pass 100%."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-q", "--tb=no"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(hermes_repo),
            )

            if result.returncode == 0:
                return ConstraintResult(
                    passed=True,
                    constraint_name="test_suite",
                    message="All tests passed",
                    details=result.stdout.strip().split("\n")[-1] if result.stdout else "",
                )
            else:
                # Extract failure summary
                last_lines = result.stdout.strip().split("\n")[-5:] if result.stdout else []
                return ConstraintResult(
                    passed=False,
                    constraint_name="test_suite",
                    message="Test suite failed",
                    details="\n".join(last_lines),
                )
        except subprocess.TimeoutExpired:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message="Test suite timed out (300s)",
            )
        except Exception as e:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message=f"Failed to run tests: {e}",
            )

    def _check_size(self, text: str, artifact_type: str) -> ConstraintResult:
        size = len(text)
        if artifact_type == "skill":
            limit = self.config.max_skill_size
        elif artifact_type == "tool_description":
            limit = self.config.max_tool_desc_size
        elif artifact_type == "param_description":
            limit = self.config.max_param_desc_size
        else:
            limit = self.config.max_skill_size  # Default

        if size <= limit:
            return ConstraintResult(
                passed=True,
                constraint_name="size_limit",
                message=f"Size OK: {size}/{limit} chars",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="size_limit",
                message=f"Size exceeded: {size}/{limit} chars ({size - limit} over)",
            )

    def _check_growth_legacy(self, text: str, baseline: str) -> ConstraintResult:
        """Single-ratio cliff for the back-compat validate_all path.

        Used only when no holdout improvement signal is available. The
        primary deploy gate is _check_growth_with_quality_gate.
        """
        growth = (len(text) - len(baseline)) / max(1, len(baseline))
        max_growth = self.config.max_prompt_growth

        if growth <= max_growth:
            return ConstraintResult(
                passed=True,
                constraint_name="growth_limit",
                message=f"Growth OK: {growth:+.1%} (max {max_growth:+.1%}, no quality signal available)",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="growth_limit",
            message=f"Growth exceeded: {growth:+.1%} (max {max_growth:+.1%}, no quality signal available)",
        )

    def _check_growth_with_quality_gate(
        self,
        text: str,
        baseline: str,
        holdout_improvement: float,
    ) -> ConstraintResult:
        """Continuous quality-gated growth check.

        required_improvement(growth) = max(0, slope * (growth - growth_free))
        Pass when actual holdout improvement >= required.

        Negative growth (the LM produced a shorter artifact) always
        passes because shrinkage doesn't need quality justification.
        """
        baseline_len = len(baseline)
        if baseline_len == 0:
            growth = 0.0
        else:
            growth = (len(text) - baseline_len) / baseline_len

        free = self.config.growth_free_threshold
        slope = self.config.growth_quality_slope
        required = max(0.0, slope * (growth - free))

        if holdout_improvement >= required:
            return ConstraintResult(
                passed=True,
                constraint_name="growth_quality_gate",
                message=(
                    f"Growth {growth:+.1%} OK: holdout improvement "
                    f"{holdout_improvement:+.3f} ≥ required {required:+.3f}"
                ),
            )
        return ConstraintResult(
            passed=False,
            constraint_name="growth_quality_gate",
            message=(
                f"Growth {growth:+.1%} requires improvement ≥{required:+.3f}; "
                f"got {holdout_improvement:+.3f}"
            ),
        )

    def _check_absolute_chars(self, text: str) -> ConstraintResult:
        """Hard absolute-char ceiling on the evolved artifact.

        Independent of growth. Backstops runaway absolute size that the
        relative growth curve can't catch (e.g., a 200-char baseline
        growing to 1500 chars is +650% but only 1500 chars absolute).
        """
        size = len(text)
        ceiling = self.config.max_absolute_chars
        if size <= ceiling:
            return ConstraintResult(
                passed=True,
                constraint_name="absolute_char_ceiling",
                message=f"Size {size} ≤ {ceiling} chars (absolute ceiling)",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="absolute_char_ceiling",
            message=f"Size {size} exceeds absolute ceiling {ceiling} chars ({size - ceiling} over)",
        )

    def _check_non_empty(self, text: str) -> ConstraintResult:
        if text.strip():
            return ConstraintResult(
                passed=True,
                constraint_name="non_empty",
                message="Artifact is non-empty",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="non_empty",
                message="Artifact is empty",
            )

    def _check_skill_structure(self, text: str) -> ConstraintResult:
        """Check that a skill file has valid YAML frontmatter and markdown body."""
        has_frontmatter = text.strip().startswith("---")
        has_name = "name:" in text[:500] if has_frontmatter else False
        has_description = "description:" in text[:500] if has_frontmatter else False

        if has_frontmatter and has_name and has_description:
            return ConstraintResult(
                passed=True,
                constraint_name="skill_structure",
                message="Skill has valid frontmatter (name + description)",
            )
        else:
            missing = []
            if not has_frontmatter:
                missing.append("YAML frontmatter (---)")
            if not has_name:
                missing.append("name field")
            if not has_description:
                missing.append("description field")
            return ConstraintResult(
                passed=False,
                constraint_name="skill_structure",
                message=f"Skill missing: {', '.join(missing)}",
            )
