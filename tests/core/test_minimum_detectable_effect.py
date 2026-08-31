"""Minimum detectable effect: what a sample size could not have seen.

These are diagnostics, never gate inputs. The last test in this file is the one
that matters most — it pins that adding them changed no deploy decision.
"""

import json
import math
from pathlib import Path

import pytest

from evolution.core.stats import min_detectable_effect_paired


def _unit_diffs(n: int) -> list[float]:
    """Per-example differences whose sample sd (ddof=1) is exactly 1.0.

    The function takes raw differences now, so the known-answer cases construct a
    sample with a known spread instead of passing one in. Alternating ±√((n-1)/n)
    about zero gives sd exactly 1 and mean 0.
    """
    a = math.sqrt((n - 1) / n)
    return [a if i % 2 == 0 else -a for i in range(n)]


class TestKnownAnswers:
    """Checked against values derived independently, not against the code."""

    def test_continuous_matches_a_hand_computed_value(self):
        # one-sided alpha 0.05 -> z 1.644854; power 0.80 -> z 0.841621
        # (1.644854 + 0.841621) * 1.0 / sqrt(16) = 0.621619
        result = min_detectable_effect_paired(_unit_diffs(16))
        assert result["mde"] == pytest.approx(0.621619, abs=1e-5)

    def test_effect_shrinks_with_the_square_root_of_n(self):
        """Quadrupling n should halve the detectable effect."""
        small = min_detectable_effect_paired(_unit_diffs(8))["mde"]
        large = min_detectable_effect_paired(_unit_diffs(32))["mde"]
        assert large == pytest.approx(small / 2.0, rel=1e-9)


class TestAlphaIsDerivedFromTheGate:
    """The diagnostic must describe the decision it sits beside, not a default."""

    def test_alpha_follows_the_confidence_argument(self):
        """paired_bootstrap's 0.90 two-sided interval is a one-sided 0.05 decision."""
        assert min_detectable_effect_paired(_unit_diffs(16))["alpha_one_sided"] == pytest.approx(0.05)
        assert min_detectable_effect_paired(_unit_diffs(16), confidence=0.95)["alpha_one_sided"] == pytest.approx(0.025)

    def test_a_stricter_confidence_demands_a_larger_effect(self):
        loose = min_detectable_effect_paired(_unit_diffs(16), confidence=0.90)["mde"]
        strict = min_detectable_effect_paired(_unit_diffs(16), confidence=0.99)["mde"]
        assert strict > loose

    def test_hardcoding_two_sided_005_would_have_inflated_it(self):
        """Guards the specific mistake: a two-sided 0.05 against a 0.90 interval.

        That reads z=1.96 where the gate uses 1.645. Note the inflation is ~13%,
        not the ~19% the z-terms alone suggest, because the power term is common
        to both -- the sort of number that gets quoted, so it is pinned here.
        """
        derived = min_detectable_effect_paired(_unit_diffs(16))["mde"]
        two_sided_005 = (1.959964 + 0.841621) * 1.0 / math.sqrt(16)
        assert derived < two_sided_005
        assert two_sided_005 / derived == pytest.approx(1.127, abs=0.005)


class TestHonestyOfTheNumber:
    def test_it_declares_itself_a_lower_bound(self):
        """z understates the true MDE at our operating n; the caveat must travel.

        Measured against the exact noncentral t: the normal approximation is low
        by ~11% at n=8 and ~5% at n=16, so the flag points the right way here --
        unlike the withdrawn paired-binary variant, where the same flag would
        have been wrong-signed by up to 60%.
        """
        result = min_detectable_effect_paired(_unit_diffs(8))
        assert result["is_lower_bound"] is True
        assert result["method"] == "normal-approximation"

    def test_ddof_is_recorded(self):
        """At n=8 the ddof choice moves sd_diff ~7%, the size of the effects judged."""
        assert min_detectable_effect_paired(_unit_diffs(8), ddof=0)["ddof"] == 0

    def test_zero_variance_is_a_number_not_an_error(self):
        """Identical arms are a valid observation: nothing varies, so any effect shows."""
        assert min_detectable_effect_paired([0.0] * 16)["mde"] == 0.0


class TestDegenerateInput:
    @pytest.mark.parametrize("n", [0, 1, -3])
    def test_n_too_small(self, n):
        with pytest.raises(ValueError, match="n > 1"):
            min_detectable_effect_paired([1.0] * max(n, 0))

    def test_ddof_must_be_less_than_n(self):
        with pytest.raises(ValueError, match="ddof"):
            min_detectable_effect_paired([1.0, 2.0], ddof=2)

    def test_confidence_out_of_range(self):
        with pytest.raises(ValueError, match="confidence"):
            min_detectable_effect_paired(_unit_diffs(16), confidence=1.0)


def test_the_diagnostic_never_enters_a_verdict():
    """The load-bearing claim, checked where the diagnostics actually are.

    An earlier version of this test compared a decision payload from
    ``evolve_code``'s gate. That payload is real and pre-change, but it belongs to
    a path this work never touches -- the diagnostics are wired into the skill and
    tool evolvers, which build their verdicts independently. It proved something
    true and irrelevant.

    Those payloads cannot be constructed without a full evolution run, so the
    check is structural instead: the values returned by the writer must never be
    read by anything except the console line. If someone feeds them into a gate,
    this fails.
    """
    import ast

    for relpath in ("evolution/skills/evolve_skill.py", "evolution/tools/evolve_tool.py"):
        tree = ast.parse(Path(relpath).read_text())
        reads = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id in {"power_payload", "power_path"}
            and isinstance(node.ctx, ast.Load)
        ]
        assert reads, f"{relpath}: diagnostics not wired at all"

        enclosing = []
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                if child in reads:
                    enclosing.append(node)
        allowed = (ast.Compare, ast.Call, ast.If)
        for node in enclosing:
            assert isinstance(node, allowed), (
                f"{relpath}: a power diagnostic is consumed by {type(node).__name__}, "
                "which is not the console line — a diagnostic must not reach a verdict"
            )
        # and it must never be named in the decision payload
        source = Path(relpath).read_text()
        payload_region = source[source.index("decision_payload"):]
        head = payload_region[:4000]
        assert "power_payload" not in head and "mde" not in head, (
            f"{relpath}: the decision payload references a power diagnostic"
        )


def test_evolve_code_decision_payload_is_unchanged():
    """A general regression guard on the code gate's payload shape.

    Kept because it is a genuine golden captured before this work, and the
    payload's stability is worth pinning -- but note it guards ``evolve_code``,
    which these diagnostics do not touch. The claim about *this* change is the
    structural test above.
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
