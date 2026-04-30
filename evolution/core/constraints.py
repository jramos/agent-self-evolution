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

    def validate_static(
        self,
        artifact_text: str,
        artifact_type: str,
    ) -> list[ConstraintResult]:
        """Static checks that don't need a baseline or holdout signal:
        size, non-empty, structural integrity. Runs first in the evolve
        flow so broken artifacts short-circuit before judge-call spend.
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
        bootstrap_result: dict,
    ) -> list[ConstraintResult]:
        """Bootstrap-gated growth check + absolute-char ceiling.

        bootstrap_result is the dict returned by
        evolution.core.stats.paired_bootstrap on the per-example
        holdout improvement scores. The growth check uses a continuous
        curve where required holdout improvement scales linearly with
        growth above the free threshold; the gate passes when both the
        sample mean meets the requirement AND the bootstrap lower bound
        is positive (no-regression). The absolute-char ceiling is
        independent of growth — applies even when the curve passes — to
        backstop runaway absolute size.
        """
        return [
            self._check_growth_with_quality_gate(
                artifact_text, baseline_text, bootstrap_result
            ),
            self._check_absolute_chars(artifact_text),
        ]

    def run_test_suite(self, repo_path: Path) -> ConstraintResult:
        """Run the target repo's full pytest suite. Must pass 100%."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-q", "--tb=no"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(repo_path),
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

    def _check_growth_with_quality_gate(
        self,
        text: str,
        baseline: str,
        bootstrap_result: dict,
    ) -> ConstraintResult:
        """Bootstrap-gated quality check on the holdout improvement.

        Required improvement scales linearly with growth above the free
        threshold:
            required_improvement(growth) = max(0, slope * (growth - growth_free))

        Decision rule:
          - When required > 0: pass when both the sample mean meets the
            requirement AND the bootstrap lower bound is positive
            (Statsig "guardrail" pattern: effect size + no-regression).
          - When required == 0 (growth ≤ free threshold): pass on
            mean ≥ 0 alone. The optimizer doesn't need to justify a
            decrease in length, only that the candidate doesn't regress.

        Negative growth (the LM produced a shorter artifact) always
        falls into the required==0 branch.
        """
        baseline_len = len(baseline)
        growth = (len(text) - baseline_len) / baseline_len if baseline_len else 0.0

        free = self.config.growth_free_threshold
        slope = self.config.growth_quality_slope
        required = max(0.0, slope * (growth - free))

        mean = bootstrap_result["mean"]
        lower = bootstrap_result["lower_bound"]
        n = bootstrap_result["n_examples"]
        conf = bootstrap_result["confidence"]

        if required == 0.0:
            # No-regression branch: just confirm we didn't break things.
            if mean >= 0.0:
                return ConstraintResult(
                    passed=True,
                    constraint_name="growth_quality_gate",
                    message=(
                        f"Growth {growth:+.1%}: no improvement required; "
                        f"mean {mean:+.3f} ≥ 0 (n={n})"
                    ),
                )
            return ConstraintResult(
                passed=False,
                constraint_name="growth_quality_gate",
                message=(
                    f"Growth {growth:+.1%}: regression — mean {mean:+.3f} < 0 (n={n})"
                ),
            )

        # Dual check: effect size AND no-regression CI.
        mean_ok = mean >= required
        lower_ok = lower > 0.0

        if mean_ok and lower_ok:
            return ConstraintResult(
                passed=True,
                constraint_name="growth_quality_gate",
                message=(
                    f"Growth {growth:+.1%}: mean {mean:+.3f} ≥ required {required:+.3f}; "
                    f"lower-bound {lower:+.3f} > 0 (n={n}, conf={conf:.2f})"
                ),
            )
        if not mean_ok:
            return ConstraintResult(
                passed=False,
                constraint_name="growth_quality_gate",
                message=(
                    f"Growth {growth:+.1%}: mean {mean:+.3f} < required {required:+.3f} (n={n})"
                ),
            )
        return ConstraintResult(
            passed=False,
            constraint_name="growth_quality_gate",
            message=(
                f"Growth {growth:+.1%}: lower-bound {lower:+.3f} ≤ 0 "
                f"(regression risk; n={n}, conf={conf:.2f})"
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
