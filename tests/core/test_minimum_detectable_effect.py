"""Minimum detectable effect: what a sample size could not have seen.

These are diagnostics, never gate inputs. The last test in this file is the one
that matters most — it pins that adding them changed no deploy decision.
"""

import json
import math
from pathlib import Path

import pytest

from evolution.core.stats import (
    min_detectable_effect_paired,
    min_detectable_shift_paired_binary,
)


class TestKnownAnswers:
    """Checked against values derived independently, not against the code."""

    def test_continuous_matches_a_hand_computed_value(self):
        # one-sided alpha 0.05 -> z 1.644854; power 0.80 -> z 0.841621
        # (1.644854 + 0.841621) * 1.0 / sqrt(16) = 0.621619
        result = min_detectable_effect_paired(16, 1.0)
        assert result["mde"] == pytest.approx(0.621619, abs=1e-5)

    def test_binary_matches_a_hand_computed_value(self):
        # (1.644854 + 0.841621) * sqrt(0.25 / 16) = 0.310809
        result = min_detectable_shift_paired_binary(16, discordance_rate=0.25)
        assert result["mde"] == pytest.approx(0.310809, abs=1e-5)

    def test_effect_shrinks_with_the_square_root_of_n(self):
        """Quadrupling n should halve the detectable effect."""
        small = min_detectable_effect_paired(8, 1.0)["mde"]
        large = min_detectable_effect_paired(32, 1.0)["mde"]
        assert large == pytest.approx(small / 2.0, rel=1e-9)


class TestAlphaIsDerivedFromTheGate:
    """The diagnostic must describe the decision it sits beside, not a default."""

    def test_alpha_follows_the_confidence_argument(self):
        """paired_bootstrap's 0.90 two-sided interval is a one-sided 0.05 decision."""
        assert min_detectable_effect_paired(16, 1.0)["alpha_one_sided"] == pytest.approx(0.05)
        assert min_detectable_effect_paired(
            16, 1.0, confidence=0.95
        )["alpha_one_sided"] == pytest.approx(0.025)

    def test_a_stricter_confidence_demands_a_larger_effect(self):
        loose = min_detectable_effect_paired(16, 1.0, confidence=0.90)["mde"]
        strict = min_detectable_effect_paired(16, 1.0, confidence=0.99)["mde"]
        assert strict > loose

    def test_hardcoding_two_sided_005_would_have_inflated_it(self):
        """Guards the specific mistake: a two-sided 0.05 against a 0.90 interval.

        That reads z=1.96 where the gate uses 1.645. Note the inflation is ~13%,
        not the ~19% the z-terms alone suggest, because the power term is common
        to both -- the sort of number that gets quoted, so it is pinned here.
        """
        derived = min_detectable_effect_paired(16, 1.0)["mde"]
        two_sided_005 = (1.959964 + 0.841621) * 1.0 / math.sqrt(16)
        assert derived < two_sided_005
        assert two_sided_005 / derived == pytest.approx(1.127, abs=0.005)


class TestHonestyOfTheNumber:
    def test_it_declares_itself_a_lower_bound(self):
        """z understates the true MDE at our operating n; the caveat must travel."""
        for result in (min_detectable_effect_paired(8, 1.0),
                       min_detectable_shift_paired_binary(8, discordance_rate=0.3)):
            assert result["is_lower_bound"] is True
            assert result["method"] == "normal-approximation"

    def test_ddof_is_recorded(self):
        """At n=8 the ddof choice moves sd_diff ~7%, the size of the effects judged."""
        assert min_detectable_effect_paired(8, 1.0, ddof=0)["ddof"] == 0

    def test_zero_variance_is_a_number_not_an_error(self):
        """Identical arms are a valid observation: nothing varies, so any effect shows."""
        assert min_detectable_effect_paired(16, 0.0)["mde"] == 0.0


class TestDegenerateInput:
    @pytest.mark.parametrize("n", [0, 1, -3])
    def test_n_too_small(self, n):
        with pytest.raises(ValueError, match="n > 1"):
            min_detectable_effect_paired(n, 1.0)

    def test_negative_sd(self):
        with pytest.raises(ValueError, match="non-negative"):
            min_detectable_effect_paired(16, -1.0)

    @pytest.mark.parametrize("rate", [0.0, -0.1, 1.5])
    def test_discordance_rate_out_of_range(self, rate):
        """The message names the confusion it exists to prevent."""
        with pytest.raises(ValueError, match="not the pass rate"):
            min_detectable_shift_paired_binary(16, discordance_rate=rate)

    def test_confidence_out_of_range(self):
        with pytest.raises(ValueError, match="confidence"):
            min_detectable_effect_paired(16, 1.0, confidence=1.0)


def test_adding_these_diagnostics_changed_no_deploy_decision():
    """The load-bearing invariant, against a golden captured BEFORE this work.

    A golden generated from the changed tree would pass unconditionally. This one
    was committed from the prior commit, so a new key in the decision payload --
    the way a diagnostic leaks into a verdict -- fails here.
    """
    from evolution.code.gate import run_code_gate
    from evolution.code.repair import RepairResult, RoundRecord
    from tests.code.conftest import (
        BUGGY_CALC,
        FIXED_CALC,
        HOLDOUT_TEST,
        VISIBLE_TEST,
        StagedRepo,
    )

    def key_paths(obj, prefix=""):
        if isinstance(obj, dict):
            out = []
            for k in sorted(obj):
                out += key_paths(obj[k], f"{prefix}.{k}" if prefix else k)
            return out
        if isinstance(obj, list):
            return [f"{prefix}[]"]
        return [prefix]

    import tempfile

    tool, vis, hold = ("tools/calc.py", "tests/tools/test_calc_visible.py",
                       "tests/tools/test_calc_holdout.py")
    with tempfile.TemporaryDirectory() as td:
        repo = StagedRepo(Path(td))
        repo.write(tool, BUGGY_CALC)
        repo.write(vis, VISIBLE_TEST)
        repo.write(hold, HOLDOUT_TEST)
        repo.git_init_commit()
        repo.write_tool(tool, FIXED_CALC)
        result = run_code_gate(
            repo, tool_relpath=tool, visible_test_relpath=vis, holdout_test_relpath=hold,
            repair_result=RepairResult(
                fixed=True, fixed_round=1, final_source=FIXED_CALC,
                rounds=[RoundRecord(round=1, proposed=True, test_passed=True)],
            ),
            floor_paths=("tests/tools",),
        )

    golden = json.loads(
        Path("tests/fixtures/gate_goldens/decision_keys.json").read_text()
    )
    assert key_paths(result.decision) == golden


class TestTheArtifact:
    """The diagnostics land beside the decision, and say when they didn't."""

    def test_writes_both_regimes_from_paired_arrays(self, tmp_path):
        from evolution.core.power_report import write_power_diagnostics

        path = write_power_diagnostics(
            tmp_path, [0.5] * 10, [0.6, 0.5, 0.7, 0.5, 0.6, 0.6, 0.5, 0.5, 0.7, 0.6]
        )

        payload = json.loads(path.read_text())
        assert payload["n_examples"] == 10
        assert payload["discordant_pairs"] == 6
        assert payload["continuous"]["is_lower_bound"] is True
        # the binary regime is powered by disagreement, not by the pass rate
        assert payload["paired_binary"]["discordance_rate"] == pytest.approx(0.6)

    def test_no_discordance_reports_none_rather_than_inventing_a_rate(self, tmp_path):
        """Identical arms cannot power a paired-binary test.

        Reporting a number here would be worse than reporting nothing: there is
        no disagreement for the test to be about.
        """
        from evolution.core.power_report import write_power_diagnostics

        payload = json.loads(
            write_power_diagnostics(tmp_path, [0.5] * 6, [0.5] * 6).read_text()
        )

        assert payload["discordant_pairs"] == 0
        assert payload["paired_binary"] is None

    def test_absent_when_there_is_nothing_to_score(self, tmp_path):
        """A missing file means "not computed", not "nothing to detect"."""
        from evolution.core.power_report import write_power_diagnostics

        assert write_power_diagnostics(tmp_path, [], []) is None
        assert write_power_diagnostics(None, [0.1], [0.2]) is None
        assert not (tmp_path / "power_diagnostics.json").exists()

    def test_console_line_states_the_comparison(self, tmp_path):
        from evolution.core.power_report import build_power_diagnostics, format_power_line

        # A tiny mean difference buried in large per-example spread: exactly the
        # case where a gate could certify a win the sample could not have seen.
        noisy = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -0.9]
        line = format_power_line(build_power_diagnostics([0.0] * 8, noisy))

        assert "smallest detectable effect" in line
        # the point of the line: the observed effect sits below what n could see
        assert "below it" in line
