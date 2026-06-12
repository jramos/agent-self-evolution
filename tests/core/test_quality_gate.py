"""Tests for the extracted quality_gate helpers.

These tests live alongside the new module rather than under tests/skills/
because the helpers are now artifact-agnostic. Skill-pipeline behavior is
still covered by tests/skills/test_evolve_skill_validation_flow.py.
"""

import json
from pathlib import Path

import pytest

from evolution.core.constraints import ConstraintResult
from evolution.core.quality_gate import (
    QUALITY_GATE_PRESETS,
    _check_cl_primary_gate,
    resolve_floor_fallback,
    resolve_proposer_mode,
    write_gate_decision,
)


def _gate(passed: bool) -> ConstraintResult:
    return ConstraintResult(passed=passed, constraint_name="cl_primary_gate", message="")


class TestResolveFloorFallback:
    def test_evolved_passes_wins_regardless_of_floor(self):
        # A clearing evolved candidate is always preferred; the floor is a
        # fallback, never a competitor.
        assert resolve_floor_fallback(
            evolved_gate=_gate(True), floor_gate=_gate(True)
        ) == "evolved"
        assert resolve_floor_fallback(
            evolved_gate=_gate(True), floor_gate=_gate(False)
        ) == "evolved"
        assert resolve_floor_fallback(
            evolved_gate=_gate(True), floor_gate=None
        ) == "evolved"

    def test_evolved_fails_floor_clears_deploys_floor(self):
        assert resolve_floor_fallback(
            evolved_gate=_gate(False), floor_gate=_gate(True)
        ) == "floor"

    def test_both_fail_rejects(self):
        assert resolve_floor_fallback(
            evolved_gate=_gate(False), floor_gate=_gate(False)
        ) == "reject"

    def test_no_floor_degrades_to_reject(self):
        # floor_gate=None (uncompilable/empty/not requested) → byte-identical to
        # the no-floor path: evolved-or-reject only.
        assert resolve_floor_fallback(
            evolved_gate=_gate(False), floor_gate=None
        ) == "reject"


class TestFloorJudgedBySameRule:
    def test_floor_obeys_noise_aware_required_gain(self):
        # The floor is judged by the SAME _check_cl_primary_gate as evolved.
        # With an A/A noise floor of 1.5, required gain is floor(1.5)+1 = 2, so a
        # +1 floor win is rejected — proving the noise-aware rule applies
        # symmetrically to the floor challenger. The floor is zero-LM, so its
        # synth mean equals baseline's (synth Δ = 0, trivially within tolerance).
        gate = _check_cl_primary_gate(
            baseline_cl_passes=5,
            evolved_cl_passes=6,  # +1 (the "floor" arm)
            baseline_synth_mean=0.6,
            evolved_synth_mean=0.6,
            growth_pct=0.05,
            noise_floor_passes=1.5,
        )
        assert gate.passed is False
        # +2 clears the same gate.
        assert _check_cl_primary_gate(
            baseline_cl_passes=5, evolved_cl_passes=7,
            baseline_synth_mean=0.6, evolved_synth_mean=0.6,
            growth_pct=0.05, noise_floor_passes=1.5,
        ).passed is True


class TestResolveProposerMode:
    def test_compression_profile(self):
        assert resolve_proposer_mode("compression") == "compression"

    def test_balanced_profile(self):
        assert resolve_proposer_mode("balanced") == "balanced"

    def test_growth_profile(self):
        assert resolve_proposer_mode("growth") == "growth"

    def test_unknown_profile_falls_back_to_compression(self):
        assert resolve_proposer_mode("bogus") == "compression"


class TestQualityGatePresets:
    def test_default_preset_exists(self):
        assert "default" in QUALITY_GATE_PRESETS

    def test_each_preset_has_required_keys(self):
        for name, preset in QUALITY_GATE_PRESETS.items():
            assert "growth_free_threshold" in preset, f"{name} missing growth_free_threshold"
            assert "growth_quality_slope" in preset, f"{name} missing growth_quality_slope"
            assert "max_absolute_chars" in preset, f"{name} missing max_absolute_chars"


class TestWriteGateDecision:
    def test_writes_json_file(self, tmp_path: Path):
        payload = {"decision": "deploy", "reason": "test"}
        output_path = write_gate_decision(tmp_path, payload)

        assert output_path.exists()
        assert output_path.name == "gate_decision.json"
        loaded = json.loads(output_path.read_text())
        assert loaded["decision"] == "deploy"
        assert loaded["reason"] == "test"

    def test_preserves_all_payload_fields(self, tmp_path: Path):
        payload = {
            "decision": "reject",
            "reason": "growth_quality_gate",
            "growth_pct": 0.5,
            "baseline_chars": 100,
            "evolved_chars": 150,
        }
        write_gate_decision(tmp_path, payload)
        loaded = json.loads((tmp_path / "gate_decision.json").read_text())
        for key, value in payload.items():
            assert loaded[key] == value


class TestBackCompatAliases:
    def test_evolve_skill_reexports_underscored_names(self):
        from evolution.skills import evolve_skill
        assert evolve_skill._QUALITY_GATE_PRESETS is QUALITY_GATE_PRESETS
        assert evolve_skill._resolve_proposer_mode is resolve_proposer_mode
        assert evolve_skill._write_gate_decision is write_gate_decision
