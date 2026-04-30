"""Tests for LMTimingCallback + litellm failure callback registration.

Pure-Python; no real LM, no real Timer. The `FakeTimer` double makes
heartbeat tests deterministic — `Timer.advance(seconds)` fires
elapsed-eligible timers immediately instead of relying on time.sleep
on a slow CI machine.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from unittest.mock import patch

import litellm
import pytest

from evolution.core.lm_timing_callback import (
    LMTimingCallback,
    _log_litellm_failure,
    register_litellm_failure_callback,
)


class FakeTimer:
    """Drop-in for threading.Timer that defers fire until .advance() is called.

    Each constructed instance registers itself in the class-level
    `_instances` list; tests use `FakeTimer.advance_all(seconds)` to
    fire any timer whose interval has elapsed.
    """

    _instances: list["FakeTimer"] = []

    def __init__(self, interval, function, args=None, kwargs=None) -> None:
        self.interval = interval
        self.function = function
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.daemon = False
        self.canceled = False
        self.fired = False
        self._elapsed = 0.0
        FakeTimer._instances.append(self)

    def start(self) -> None:
        pass  # No real thread; advance() drives execution.

    def cancel(self) -> None:
        self.canceled = True

    @classmethod
    def reset(cls) -> None:
        cls._instances.clear()

    @classmethod
    def advance_all(cls, seconds: float) -> None:
        """Add `seconds` to every live timer; fire those that crossed
        their interval, in interval-ascending order (matches real-thread
        firing order)."""
        for t in cls._instances:
            if not t.canceled and not t.fired:
                t._elapsed += seconds
        eligible = sorted(
            (t for t in cls._instances if t._elapsed >= t.interval and not t.canceled and not t.fired),
            key=lambda t: t.interval,
        )
        for t in eligible:
            t.fired = True
            t.function(*t.args, **t.kwargs)


@pytest.fixture(autouse=True)
def _reset_fake_timer():
    FakeTimer.reset()
    yield
    FakeTimer.reset()


def _make_lm_instance(model: str = "openai/gpt-4.1-mini") -> SimpleNamespace:
    return SimpleNamespace(model=model)


class TestLMStartEndLogging:
    def test_on_lm_start_logs_with_model_and_prompt_chars(self, caplog):
        cb = LMTimingCallback(timer_factory=FakeTimer)
        with caplog.at_level(logging.INFO, logger="evolution.core.lm_timing_callback"):
            cb.on_lm_start(
                "abc12345",
                _make_lm_instance("openai/gpt-5-mini"),
                {"messages": [{"content": "hello"}, {"content": "world"}]},
            )
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[LM #1 start]" in m and "model=openai/gpt-5-mini" in m and "prompt_chars=10" in m for m in msgs)

    def test_on_lm_end_under_30s_logs_at_info(self, caplog):
        cb = LMTimingCallback(timer_factory=FakeTimer)
        cb.on_lm_start("c1", _make_lm_instance(), {"messages": []})
        with caplog.at_level(logging.DEBUG, logger="evolution.core.lm_timing_callback"):
            cb.on_lm_end("c1", outputs=[], exception=None)
        end_records = [r for r in caplog.records if "[LM end]" in r.getMessage()]
        assert len(end_records) == 1
        assert end_records[0].levelno == logging.INFO

    def test_on_lm_end_over_30s_logs_at_warning(self, caplog):
        cb = LMTimingCallback(timer_factory=FakeTimer)
        cb.on_lm_start("c1", _make_lm_instance(), {"messages": []})
        # Backdate the in-flight start_time so duration computes > 30s.
        with cb._lock:
            start, model, timers = cb._inflight["c1"]
            cb._inflight["c1"] = (start - 35.0, model, timers)
        with caplog.at_level(logging.INFO, logger="evolution.core.lm_timing_callback"):
            cb.on_lm_end("c1", outputs=[], exception=None)
        end_records = [r for r in caplog.records if "[LM end]" in r.getMessage()]
        assert len(end_records) == 1
        assert end_records[0].levelno == logging.WARNING

    def test_on_lm_end_with_exception_logs_at_warning(self, caplog):
        cb = LMTimingCallback(timer_factory=FakeTimer)
        cb.on_lm_start("c1", _make_lm_instance(), {"messages": []})
        with caplog.at_level(logging.INFO, logger="evolution.core.lm_timing_callback"):
            cb.on_lm_end("c1", outputs=None, exception=TimeoutError("hung"))
        exc_records = [r for r in caplog.records if "[LM end EXC]" in r.getMessage()]
        assert len(exc_records) == 1
        assert exc_records[0].levelno == logging.WARNING
        assert "TimeoutError" in exc_records[0].getMessage()


class TestHeartbeats:
    def test_heartbeat_at_60s_logs_at_debug(self, caplog):
        cb = LMTimingCallback(timer_factory=FakeTimer)
        cb.on_lm_start("c1", _make_lm_instance(), {"messages": []})
        with caplog.at_level(logging.DEBUG, logger="evolution.core.lm_timing_callback"):
            FakeTimer.advance_all(60)
        hb = [r for r in caplog.records if "HEARTBEAT" in r.getMessage()]
        assert len(hb) == 1
        assert hb[0].levelno == logging.DEBUG
        assert "after 60s" in hb[0].getMessage()

    def test_heartbeat_at_180s_logs_at_warning(self, caplog):
        cb = LMTimingCallback(timer_factory=FakeTimer)
        cb.on_lm_start("c1", _make_lm_instance(), {"messages": []})
        with caplog.at_level(logging.DEBUG, logger="evolution.core.lm_timing_callback"):
            FakeTimer.advance_all(180)
        hb_180 = [r for r in caplog.records if "after 180s" in r.getMessage()]
        assert len(hb_180) == 1
        assert hb_180[0].levelno == logging.WARNING

    def test_heartbeat_canceled_when_call_completes_before_interval(self, caplog):
        cb = LMTimingCallback(timer_factory=FakeTimer)
        cb.on_lm_start("c1", _make_lm_instance(), {"messages": []})
        cb.on_lm_end("c1", outputs=[], exception=None)
        with caplog.at_level(logging.DEBUG, logger="evolution.core.lm_timing_callback"):
            FakeTimer.advance_all(600)
        hb = [r for r in caplog.records if "HEARTBEAT" in r.getMessage()]
        # All four tier timers were canceled at on_lm_end → no heartbeat fires.
        assert hb == []


class TestLitellmFailureCallback:
    def test_register_idempotent(self):
        # Snapshot whatever litellm has + clear any prior registrations
        # of our callback to make the assertion stable across test order.
        original = list(litellm.failure_callback or [])
        litellm.failure_callback = [
            cb for cb in original if cb is not _log_litellm_failure
        ]
        try:
            register_litellm_failure_callback()
            register_litellm_failure_callback()
            register_litellm_failure_callback()
            count = sum(
                1 for cb in litellm.failure_callback or []
                if cb is _log_litellm_failure
            )
            assert count == 1
        finally:
            litellm.failure_callback = original
