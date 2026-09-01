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
        """Against the rule the gate runs, not a textbook t-test.

        The bootstrap's percentile lower bound uses the divisor-n resample sd, so
        the effective threshold is z_0.95 * sqrt((n-1)/n) = 1.5926228 at n=16, not
        1.6448536. (1.5926228 + 0.8416212) * 1.0 / sqrt(16) = 0.6085611.
        """
        result = min_detectable_effect_paired(_unit_diffs(16))
        assert result["mde"] == pytest.approx(0.6085611, abs=1e-6)
        assert result["critical_multiplier"] == pytest.approx(1.5926228, abs=1e-6)

    def test_models_the_gate_rule_not_the_nominal_quantile(self):
        """Guards the correction that prompted this shape.

        Using the nominal quantile would describe a test the gate does not run,
        and would report a larger detectable effect than the rule's own -- the
        error the first version shipped.
        """
        import math
        from statistics import NormalDist

        r = min_detectable_effect_paired(_unit_diffs(16))
        nominal = (NormalDist().inv_cdf(0.95) + NormalDist().inv_cdf(0.80)) / math.sqrt(16)
        assert r["mde"] < nominal, "the gate's rule is anti-conservative; the MDE must reflect that"

    def test_effect_shrinks_with_n_but_not_purely_as_root_n(self):
        """Quadrupling n nearly halves it -- and the gap is the point.

        Pure 1/sqrt(n) scaling would hold for a fixed threshold, but the gate's
        effective threshold z*sqrt((n-1)/n) itself rises with n, approaching the
        nominal quantile. So the effect shrinks slightly more slowly than root-n,
        which is a property of the rule being modelled rather than an error.
        """
        small = min_detectable_effect_paired(_unit_diffs(8))["mde"]
        large = min_detectable_effect_paired(_unit_diffs(32))["mde"]

        assert large < small
        ratio = small / large
        assert 1.9 < ratio < 2.0, f"expected just under root-n scaling, got {ratio}"


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
        # same critical-multiplier correction on both sides, so the comparison is
        # between alphas rather than between rules
        two_sided_005 = (1.959964 * math.sqrt(15 / 16) + 0.841621) / math.sqrt(16)
        assert derived < two_sided_005
        assert two_sided_005 / derived == pytest.approx(1.125, abs=0.005)


class TestHonestyOfTheNumber:
    def test_it_claims_no_bound_it_cannot_support(self):
        """No direction claim, because the number cannot support one.

        Against an exact paired t-test the normal approximation understates; but
        the gate's own percentile rule is anti-conservative, so relative to *that*
        the figure sits on the other side. An earlier version asserted a lower
        bound, which was wrong-signed for the decision it sits beside -- the same
        defect that got the paired-binary regime withdrawn.
        """
        result = min_detectable_effect_paired(_unit_diffs(8))
        assert "is_lower_bound" not in result
        assert result["method"] == "normal-approximation"
        assert "gate" in result["models_rule"]

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
    """The load-bearing claim: the payload reaches the console line and nothing else.

    Allowlisted by *call target*, not by AST node type. An earlier version allowed
    any ``ast.Call``, which is precisely how a diagnostic would reach a verdict --
    ``growth_pass = _veto(power_payload)`` passed it. Only the two reporting
    functions may receive these values.
    """
    import ast

    ALLOWED_CALLEES = {"format_power_line", "write_power_diagnostics"}

    for relpath in ("evolution/skills/evolve_skill.py", "evolution/tools/evolve_tool.py"):
        tree = ast.parse(Path(relpath).read_text())
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        reads = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Name)
            and n.id in {"power_payload", "power_path"}
            and isinstance(n.ctx, ast.Load)
        ]
        assert reads, f"{relpath}: diagnostics not wired at all"

        for node in reads:
            parent = parents[node]
            if isinstance(parent, ast.Compare):
                continue  # `is not None` guard
            if isinstance(parent, ast.Call):
                callee = parent.func
                name = getattr(callee, "id", None) or getattr(callee, "attr", None)
                assert name in ALLOWED_CALLEES, (
                    f"{relpath}:{node.lineno}: a power diagnostic is passed to "
                    f"{name!r}, which is not a reporting function — a diagnostic "
                    "must not reach a verdict"
                )
                continue
            raise AssertionError(
                f"{relpath}:{node.lineno}: a power diagnostic is consumed by "
                f"{type(parent).__name__}, which is not the console line"
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


class TestTheArtifact:
    """Coverage for the writer itself, which the module rewrite left untested."""

    def test_writes_the_diagnostic_beside_the_decision(self, tmp_path):
        from evolution.core.power_report import write_power_diagnostics

        path, payload = write_power_diagnostics(
            tmp_path, [0.5] * 10, [0.6, 0.5, 0.7, 0.5, 0.6, 0.6, 0.5, 0.5, 0.7, 0.6]
        )

        assert json.loads(path.read_text()) == payload
        assert payload["n_examples"] == 10
        assert payload["continuous"]["models_rule"]

    def test_returns_a_pair_so_a_single_assignment_cannot_look_like_a_path(self, tmp_path):
        """Pins the shape, because getting it wrong fails far from the cause.

        The writer returns (path, payload). A caller writing ``path = write(...)``
        gets a tuple that is *always* truthy, so an ``is not None`` guard always
        fires and the mistake surfaces later as an AttributeError.
        """
        from evolution.core.power_report import write_power_diagnostics

        result = write_power_diagnostics(tmp_path, [0.1, 0.2], [0.2, 0.3])

        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] is not None and isinstance(result[1], dict)

    def test_absent_when_there_is_nothing_to_score(self, tmp_path):
        """A missing file means "not computed", never "nothing to detect"."""
        from evolution.core.power_report import write_power_diagnostics

        assert write_power_diagnostics(tmp_path, [], []) == (None, None)
        assert write_power_diagnostics(None, [0.1], [0.2]) == (None, None)
        assert not (tmp_path / "power_diagnostics.json").exists()

    @pytest.mark.parametrize("base,evolved", [([0.5] * 3, [0.5] * 2), ([], [0.1])])
    def test_mismatched_arrays_raise_rather_than_truncate(self, base, evolved):
        """Including the empty-baseline mirror, which the guard order once let slip."""
        from evolution.core.power_report import build_power_diagnostics

        with pytest.raises(ValueError, match="equal length"):
            build_power_diagnostics(base, evolved)

    def test_console_line_keeps_the_sign_of_a_regression(self):
        """A gate that only certifies improvements must not read a loss as a win."""
        from evolution.core.power_report import build_power_diagnostics, format_power_line

        worse = [-0.4, -0.5, -0.3, -0.4, -0.6, -0.4, -0.3, -0.5]
        line = format_power_line(build_power_diagnostics([0.0] * 8, worse))

        assert "negative" in line and "regression" in line

    def test_identical_arms_report_nothing_to_detect(self):
        """Zero spread and zero effect must not render as an effect above the floor.

        With a strict comparison this printed "observed Δ=+0.000 is above it" --
        the exact misreading the sign handling exists to prevent, in the project's
        documented saturation regime.
        """
        from evolution.core.power_report import build_power_diagnostics, format_power_line

        line = format_power_line(build_power_diagnostics([0.5] * 8, [0.5] * 8))

        assert "nothing to detect" in line
        assert "above it" not in line
