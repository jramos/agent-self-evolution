"""Tests for knee-point Pareto selection.

Pure-Python, no LM. Uses lightweight stand-ins for SkillModule (only the
`skill_text` property is read) and ConstraintResult (only `.passed` is read).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from evolution.skills.knee_point import select_knee_point


@dataclass
class _FakeModule:
    skill_text: str


@dataclass
class _FakeResult:
    passed: bool
    constraint_name: str = "test"
    message: str = ""


def _all_pass(_text: str) -> list[_FakeResult]:
    return [_FakeResult(passed=True)]


def _all_fail(_text: str) -> list[_FakeResult]:
    return [_FakeResult(passed=False, message="failed")]


def _fail_when_short(threshold: int):
    def validator(text: str) -> list[_FakeResult]:
        return [_FakeResult(passed=len(text) >= threshold)]
    return validator


class TestEpsilon:
    def test_default_epsilon_is_one_over_n_val_for_n6(self):
        candidates = [_FakeModule("x" * 100), _FakeModule("x" * 50)]
        scores = [1.0, 0.85]  # 0.85 inside ε=0.167 band
        pick = select_knee_point(
            candidates, scores, n_val=6,
            static_validator=_all_pass, gepa_default_idx=0,
        )
        assert pick.epsilon == pytest.approx(1.0 / 6.0)
        assert pick.band_size == 2

    def test_default_epsilon_is_one_over_n_val_for_n12(self):
        candidates = [_FakeModule("x" * 100), _FakeModule("x" * 50)]
        scores = [1.0, 0.85]  # 0.85 outside ε≈0.083 band
        pick = select_knee_point(
            candidates, scores, n_val=12,
            static_validator=_all_pass, gepa_default_idx=0,
        )
        assert pick.epsilon == pytest.approx(1.0 / 12.0)
        assert pick.band_size == 1
        assert pick.picked_idx == 0

    def test_explicit_epsilon_override(self):
        candidates = [_FakeModule("x" * 100), _FakeModule("x" * 50)]
        scores = [1.0, 0.95]
        pick = select_knee_point(
            candidates, scores, n_val=6,
            static_validator=_all_pass, gepa_default_idx=0,
            epsilon=0.02,
        )
        # ε=0.02; 0.95 not in [0.98, 1.0]
        assert pick.epsilon == 0.02
        assert pick.band_size == 1


class TestParsimonyPick:
    def test_picks_smallest_in_band(self):
        # Per the plan's verification example.
        candidates = [
            _FakeModule("x" * 1572),  # idx 0, val 0.997 — best, in band
            _FakeModule("x" * 412),   # idx 1, val 0.95  — in band
            _FakeModule("x" * 800),   # idx 2, val 0.95  — in band
            _FakeModule("x" * 1100),  # idx 3, val 0.93  — in band
            _FakeModule("x" * 200),   # idx 4, val 0.80  — outside ε=0.167
        ]
        scores = [0.997, 0.95, 0.95, 0.93, 0.80]
        pick = select_knee_point(
            candidates, scores, n_val=6,
            static_validator=_all_pass, gepa_default_idx=0,
        )
        assert pick.picked_idx == 1
        assert pick.body_chars == 412
        assert pick.fallback == "knee"
        assert pick.band_size == 4

    def test_singleton_band_returns_only_candidate(self):
        # Only one candidate within ε of the unanimous best.
        candidates = [_FakeModule("a" * 500), _FakeModule("b" * 100)]
        scores = [1.0, 0.5]  # 0.5 outside ε=0.167 band
        pick = select_knee_point(
            candidates, scores, n_val=6,
            static_validator=_all_pass, gepa_default_idx=0,
        )
        assert pick.band_size == 1
        assert pick.picked_idx == 0
        assert pick.fallback == "knee"

    def test_deterministic_tiebreak_by_val_then_idx(self):
        # Two candidates with identical body_chars: higher val_score wins;
        # if val also tied, lower idx wins.
        candidates = [
            _FakeModule("x" * 100),  # idx 0, val 0.90
            _FakeModule("x" * 100),  # idx 1, val 0.95
            _FakeModule("x" * 100),  # idx 2, val 0.95 (ties idx 1 on val + chars)
        ]
        scores = [0.90, 0.95, 0.95]
        pick = select_knee_point(
            candidates, scores, n_val=6,
            static_validator=_all_pass, gepa_default_idx=0,
        )
        # idx 1 wins: same chars + higher val than 0; same as idx 2 but lower idx.
        assert pick.picked_idx == 1


class TestStaticFallback:
    def test_skips_static_failures_in_band(self):
        # Smallest candidate fails static; next-smallest wins.
        candidates = [
            _FakeModule("x" * 50),    # idx 0, val 1.0 — too short, fails static
            _FakeModule("x" * 200),   # idx 1, val 0.95 — passes
            _FakeModule("x" * 1000),  # idx 2, val 0.99 — best val, passes
        ]
        scores = [1.0, 0.95, 0.99]
        pick = select_knee_point(
            candidates, scores, n_val=6,
            static_validator=_fail_when_short(threshold=100),
            gepa_default_idx=0,
        )
        assert pick.picked_idx == 1
        assert pick.fallback == "knee"

    def test_falls_back_to_gepa_default_when_band_all_fails(self):
        # Every band candidate fails static; fall back to GEPA's default,
        # even though that fails too — caller handles downstream.
        candidates = [
            _FakeModule("x" * 50),
            _FakeModule("x" * 60),
            _FakeModule("x" * 70),
        ]
        scores = [1.0, 0.95, 0.92]
        pick = select_knee_point(
            candidates, scores, n_val=6,
            static_validator=_all_fail,
            gepa_default_idx=0,
        )
        assert pick.fallback == "static_failed_all"
        assert pick.picked_idx == 0  # gepa_default_idx


class TestRosterAndRank:
    def test_band_roster_sorted_by_val_score_desc(self):
        candidates = [
            _FakeModule("x" * 1572),
            _FakeModule("x" * 412),
            _FakeModule("x" * 800),
        ]
        scores = [0.997, 0.95, 0.96]
        pick = select_knee_point(
            candidates, scores, n_val=6,
            static_validator=_all_pass, gepa_default_idx=0,
        )
        # All three in band (ε=0.167); roster sorted by val desc.
        assert [r["idx"] for r in pick.band_roster] == [0, 2, 1]
        assert [r["val_score"] for r in pick.band_roster] == [0.997, 0.96, 0.95]

    def test_val_rank_in_band_is_one_indexed(self):
        candidates = [
            _FakeModule("x" * 1572),
            _FakeModule("x" * 412),
            _FakeModule("x" * 800),
        ]
        scores = [0.997, 0.95, 0.96]
        pick = select_knee_point(
            candidates, scores, n_val=6,
            static_validator=_all_pass, gepa_default_idx=0,
        )
        # picked = idx 1 (smallest body); val 0.95 is rank 3 of 3 in band.
        assert pick.picked_idx == 1
        assert pick.val_rank_in_band == 3

    def test_telemetry_records_gepa_default(self):
        candidates = [_FakeModule("x" * 1572), _FakeModule("x" * 412)]
        scores = [0.997, 0.95]
        pick = select_knee_point(
            candidates, scores, n_val=6,
            static_validator=_all_pass, gepa_default_idx=0,
        )
        assert pick.gepa_default_idx == 0
        assert pick.gepa_default_body_chars == 1572


class TestDefensive:
    def test_raises_on_empty_candidates(self):
        with pytest.raises(ValueError, match="empty"):
            select_knee_point(
                [], [], n_val=6,
                static_validator=_all_pass, gepa_default_idx=0,
            )

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError, match="length mismatch"):
            select_knee_point(
                [_FakeModule("x")], [1.0, 0.9],
                n_val=6, static_validator=_all_pass, gepa_default_idx=0,
            )

    def test_raises_on_zero_n_val(self):
        with pytest.raises(ValueError, match="n_val"):
            select_knee_point(
                [_FakeModule("x")], [1.0],
                n_val=0, static_validator=_all_pass, gepa_default_idx=0,
            )
