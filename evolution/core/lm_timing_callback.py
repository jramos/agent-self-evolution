"""LM-call observability: timing + heartbeat warnings + per-attempt failures.

Surfaces LM-call latency, mid-call heartbeats on stalls, and per-attempt
failures (which DSPy's BaseCallback hides behind tenacity retries).
Required for diagnosing OpenAI capacity degradation, GEPA reflection-LM
stalls, and silent retry behavior — without this, hung calls are
indistinguishable from "still optimizing" until hours pass.

Two surfaces:

1. `LMTimingCallback` — DSPy `BaseCallback` registered globally via
   `dspy.configure(callbacks=[...])`. Logs every LM call's start/end
   with model + duration. Heartbeat warnings fire at 60s/180s/300s/600s
   for any call that hasn't returned (60s = DEBUG since cold-cache calls
   commonly cross it; 180s+ = WARNING).
2. `register_litellm_failure_callback()` — installs `_log_litellm_failure`
   into `litellm.failure_callback` so each retry attempt is logged
   (BaseCallback only fires once per logical call, hiding retries).
   Idempotent + lock-guarded against TOCTOU on concurrent imports.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import litellm
from dspy.utils.callback import BaseCallback

logger = logging.getLogger(__name__)

# 60s tier intentionally DEBUG: cold-cache calls (e.g., gpt-5-mini reasoning
# on first call of a run) commonly cross 60s legitimately. WARNING here
# would train the user to ignore heartbeats. 180s+ are real-stall territory.
_HEARTBEAT_TIERS: tuple[tuple[int, int], ...] = (
    (60, logging.DEBUG),
    (180, logging.WARNING),
    (300, logging.WARNING),
    (600, logging.WARNING),
)


class LMTimingCallback(BaseCallback):
    """Log every dspy.LM call with timing + tiered heartbeat warnings."""

    def __init__(
        self,
        timer_factory: Callable[..., threading.Timer] = threading.Timer,
    ) -> None:
        # Injectable so tests can use a FakeTimer.advance(seconds) double
        # instead of monkeypatching intervals + time.sleep (flaky on slow CI).
        self._timer_factory = timer_factory
        self._inflight: dict[str, tuple[float, str, list[Any]]] = {}
        self._lock = threading.Lock()
        self._call_seq = 0

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        model = getattr(instance, "model", "<unknown>")
        with self._lock:
            self._call_seq += 1
            seq = self._call_seq
        prompt_chars = sum(
            len(str(m.get("content", ""))) for m in (inputs.get("messages") or [])
        )
        logger.info(
            "[LM #%d start] model=%s call_id=%s prompt_chars=%d",
            seq, model, call_id[:8], prompt_chars,
        )
        timers: list[Any] = []
        for interval, level in _HEARTBEAT_TIERS:
            t = self._timer_factory(
                interval,
                self._emit_heartbeat,
                args=(call_id, seq, model, interval, level),
            )
            t.daemon = True
            t.start()
            timers.append(t)
        with self._lock:
            self._inflight[call_id] = (time.time(), model, timers)

    def on_lm_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        with self._lock:
            entry = self._inflight.pop(call_id, None)
        if entry is None:
            return
        start, model, timers = entry
        for t in timers:
            t.cancel()
        duration = time.time() - start
        if exception is not None:
            logger.warning(
                "[LM end EXC] model=%s call_id=%s duration=%.1fs exception=%s",
                model, call_id[:8], duration, type(exception).__name__,
            )
        else:
            level = logging.WARNING if duration > 30 else logging.INFO
            logger.log(
                level,
                "[LM end] model=%s call_id=%s duration=%.1fs",
                model, call_id[:8], duration,
            )

    def _emit_heartbeat(
        self,
        call_id: str,
        seq: int,
        model: str,
        elapsed: int,
        level: int,
    ) -> None:
        with self._lock:
            still_inflight = call_id in self._inflight
        if still_inflight:
            logger.log(
                level,
                "[LM #%d HEARTBEAT] model=%s call_id=%s still running after %ds",
                seq, model, call_id[:8], elapsed,
            )


def _log_litellm_failure(kwargs, exception, start_time, end_time) -> None:
    """litellm failure_callback fires once per failed *attempt* (not per
    logical call), exposing intermediate retries that BaseCallback hides
    behind a single `on_lm_end`. Without this, a 5×60s retry loop on a
    flaky API looks like a single 5-minute LM call.
    """
    model = kwargs.get("model", "<unknown>")
    duration = (end_time - start_time).total_seconds() if end_time else -1.0
    logger.warning(
        "[litellm RETRY/FAIL] model=%s duration=%.1fs exception=%s: %s",
        model, duration, type(exception).__name__, str(exception)[:200],
    )


_register_lock = threading.Lock()


def register_litellm_failure_callback() -> None:
    """Install `_log_litellm_failure` into `litellm.failure_callback`,
    idempotently. Lock-guarded against TOCTOU on concurrent first-call
    (the dedup check + append must be atomic).

    Call this once per `evolve()` invocation rather than at import time
    so test isolation isn't compromised by module-level mutation of a
    third-party global.
    """
    with _register_lock:
        callbacks = litellm.failure_callback or []
        if _log_litellm_failure not in callbacks:
            litellm.failure_callback = callbacks + [_log_litellm_failure]
