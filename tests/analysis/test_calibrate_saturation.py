"""Saturation calibration: guards the honesty logic (binning, Wilson, sweep).

The script lives in scripts/analysis/ (not an importable package), so load it
by path.
"""
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "analysis" / "calibrate_saturation.py"
)
_spec = importlib.util.spec_from_file_location("calibrate_saturation", _SCRIPT)
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def _gate(avg_baseline, lower_bound=None, avg_evolved=None, decision="reject",
          fitness_profile="balanced", gate_mode="no_regression",
          decision_signal="synthetic", n_examples=50):
    return {
        "avg_baseline": avg_baseline,
        "avg_evolved": avg_evolved if avg_evolved is not None else avg_baseline,
        "decision": decision,
        "fitness_profile": fitness_profile,
        "gate_mode": gate_mode,
        "decision_signal": decision_signal,
        "bootstrap": {"lower_bound": lower_bound, "n_examples": n_examples},
    }


def test_assign_bin_edges():
    assert cs.assign_bin(0.80) == "<0.90"
    assert cs.assign_bin(0.90) == "0.90-0.95"
    assert cs.assign_bin(0.949) == "0.90-0.95"
    assert cs.assign_bin(0.95) == "0.95-0.99"
    assert cs.assign_bin(0.989) == "0.95-0.99"
    assert cs.assign_bin(0.99) == ">=0.99"
    assert cs.assign_bin(1.0) == ">=0.99"


def test_wilson_zero_successes_has_wide_upper_bound():
    # 0/17 — the data-starved gated region. Point estimate 0, but upper bound
    # must be substantial (≈18%), which is the honest read the report leads with.
    lo, hi = cs.wilson_interval(0, 17)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert 0.15 < hi < 0.25


def test_wilson_empty_is_fully_uncertain():
    assert cs.wilson_interval(0, 0) == (0.0, 1.0)


def test_real_improvement_primary_is_lower_bound_positive():
    assert cs.real_improvement(_gate(0.6, lower_bound=0.04)) is True
    assert cs.real_improvement(_gate(0.6, lower_bound=0.0)) is False
    assert cs.real_improvement(_gate(0.6, lower_bound=-0.02)) is False
    assert cs.real_improvement(_gate(0.6, lower_bound=None)) is False


def test_real_improvement_decision_definition_is_circular():
    g = _gate(0.6, lower_bound=-0.1, decision="deploy")  # deployed but regressed
    assert cs.real_improvement(g, definition="lower_bound") is False
    assert cs.real_improvement(g, definition="decision") is True


def test_real_improvement_gain_definition_respects_delta():
    g = _gate(0.60, avg_evolved=0.63)  # +0.03
    assert cs.real_improvement(g, definition="gain", delta=0.02) is True
    assert cs.real_improvement(g, definition="gain", delta=0.05) is False


def test_false_abort_sweep_degenerate_gated_region():
    # Mirrors the real archive: every positive-signal run sits below 0.95,
    # the gated region has runs but none improve.
    runs = (
        [_gate(0.6, lower_bound=0.05) for _ in range(40)]   # positive, below gate
        + [_gate(0.97, lower_bound=0.0) for _ in range(15)]  # in gate, no signal
        + [_gate(0.995, lower_bound=None) for _ in range(5)]
    )
    sweep = {r["threshold"]: r for r in cs.false_abort_sweep(runs)}
    at95 = sweep[0.95]
    assert at95["would_abort_n"] == 20            # 15 + 5 at/above 0.95
    assert at95["n_real_improvement"] == 0
    assert at95["false_abort_rate"] == 0.0
    assert at95["wilson_upper"] > 0.0             # absence of evidence, not safety
    # The 40 positive runs (0.6) are below every candidate threshold, and even
    # the saturated runs (0.97/0.995) sit below τ=1.0 → nothing aborts at 1.0.
    assert sweep[1.0]["would_abort_n"] == 0


def test_false_abort_sweep_detects_a_real_false_abort():
    # A high-baseline run that genuinely improved — the case the threshold would
    # wrongly kill. The sweep must surface a nonzero false-abort rate.
    runs = [
        _gate(0.97, lower_bound=0.06),   # gated AND improved → false abort
        _gate(0.97, lower_bound=0.0),    # gated, no signal
        _gate(0.6, lower_bound=0.05),    # below gate
    ]
    at95 = {r["threshold"]: r for r in cs.false_abort_sweep(runs)}[0.95]
    assert at95["would_abort_n"] == 2
    assert at95["n_real_improvement"] == 1
    assert at95["false_abort_rate"] == pytest.approx(0.5)


def test_analyze_excludes_closed_loop_and_splits_strata():
    runs = (
        [_gate(0.6, lower_bound=0.05) for _ in range(3)]
        + [_gate(0.6, lower_bound=0.05, decision_signal="closed_loop")]
        + [_gate(0.6, lower_bound=0.05, fitness_profile="compression")]  # off-profile
    )
    a = cs.analyze(runs)
    assert a["n_total"] == 5
    assert a["n_closed_loop_excluded"] == 1
    assert a["n_paired_pool"] == 4          # closed-loop dropped from pool
    assert a["n_homogeneous"] == 3          # compression run is off-profile
    assert a["n_off_profile"] == 1


def test_analyze_handles_empty_pool():
    # Only closed-loop runs: synthetic pool is empty, must not raise.
    a = cs.analyze([_gate(0.6, lower_bound=0.05, decision_signal="closed_loop")])
    assert a["n_paired_pool"] == 0
    assert a["n_homogeneous"] == 0
    assert all(r["would_abort_n"] == 0 for r in a["false_abort_primary"])
