"""Tests for the SaveCallJudge — scores memory-save args against MEMORY_GUIDANCE rules."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evolution.prompts.prompt_judge import SaveCallJudge, judge_save_calls


def test_no_save_calls_yields_default():
    """No save calls at all → score 1.0 (vacuously correct). Layer 1 catches
    'should have saved but didn't'; Layer 2 only scores content of calls made."""
    assert judge_save_calls(judge=None, calls=[], expected_content=None) == 1.0


def test_invokes_judge_per_call_and_means():
    fake_judge = MagicMock(spec=SaveCallJudge)
    fake_judge.score.side_effect = [0.8, 0.6]
    calls = [
        {"action": "save", "content": "user prefers concise responses"},
        {"action": "save", "content": "completed phase 3"},
    ]
    score = judge_save_calls(
        judge=fake_judge, calls=calls,
        expected_content="user preference about response style",
    )
    assert score == pytest.approx(0.7)
    assert fake_judge.score.call_count == 2


def test_caps_at_five_calls():
    """Pathological: agent saves on every turn. Judge at most 5; excess score 0."""
    fake_judge = MagicMock(spec=SaveCallJudge)
    fake_judge.score.return_value = 1.0
    calls = [{"action": "save", "content": f"item {i}"} for i in range(10)]
    score = judge_save_calls(judge=fake_judge, calls=calls, expected_content="any")
    # 5 scored 1.0, 5 unjudged scored 0 → mean 0.5
    assert score == pytest.approx(0.5)
    assert fake_judge.score.call_count == 5


def test_filters_non_save_actions():
    fake_judge = MagicMock(spec=SaveCallJudge)
    fake_judge.score.return_value = 1.0
    calls = [
        {"action": "delete", "key": "x"},
        {"action": "save", "content": "real save"},
    ]
    score = judge_save_calls(judge=fake_judge, calls=calls, expected_content="any")
    assert score == pytest.approx(1.0)
    assert fake_judge.score.call_count == 1


def test_none_judge_or_expected_is_vacuous_pass():
    """A save call exists but no judge/rubric configured → don't penalize."""
    calls = [{"action": "save", "content": "x"}]
    assert judge_save_calls(judge=None, calls=calls, expected_content="r") == 1.0
    fake = MagicMock(spec=SaveCallJudge)
    assert judge_save_calls(judge=fake, calls=calls, expected_content=None) == 1.0
    fake.score.assert_not_called()
