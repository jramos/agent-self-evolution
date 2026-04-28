"""Tests for the budget resolver and optimizer fallback control flow.

These two helpers were extracted from `evolve` so the branching logic
(budget precedence, GEPA→MIPROv2 fallback, --no-fallback re-raise,
ImportError install-hint chaining) can be unit-tested without an LM.
"""

import dspy
import pytest

from evolution.skills.evolve_skill import (
    _build_optimizer_and_compile,
    _default_mipro_runner,
    _resolve_budget,
)


class TestResolveBudget:
    def test_explicit_budget_wins_over_iterations(self):
        assert _resolve_budget(iterations=1, budget="heavy") == "heavy"
        assert _resolve_budget(iterations=10, budget="light") == "light"

    def test_iterations_1_2_3_map_to_budgets(self):
        assert _resolve_budget(iterations=1, budget=None) == "light"
        assert _resolve_budget(iterations=2, budget=None) == "medium"
        assert _resolve_budget(iterations=3, budget=None) == "heavy"

    def test_other_iteration_values_collapse_to_light(self):
        # The default --iterations is 10; this confirms the silent collapse
        # is the documented behavior (rather than e.g. raising).
        assert _resolve_budget(iterations=10, budget=None) == "light"
        assert _resolve_budget(iterations=0, budget=None) == "light"
        assert _resolve_budget(iterations=4, budget=None) == "light"


class _FakeOptimized:
    """Stand-in for the optimized module returned by .compile()."""


def _gepa_runner_succeeds(**kwargs):
    return _FakeOptimized()


def _gepa_runner_raises(**kwargs):
    raise RuntimeError("simulated GEPA failure")


def _mipro_runner_succeeds(**kwargs):
    return _FakeOptimized()


def _mipro_runner_missing_optuna(**kwargs):
    # MIPROv2 lazy-imports optuna inside .compile(); this mirrors that.
    raise ImportError("optuna is required for MIPROv2")


class TestBuildOptimizerAndCompile:
    def _common_kwargs(self, **overrides):
        kwargs = dict(
            baseline_module=object(),
            trainset=[],
            valset=[],
            metric=lambda *a, **kw: 0.0,
            gepa_budget="light",
            optimizer_model="openai/gpt-4o-mini",
            seed=42,
            no_fallback=False,
            failure_log_path=None,
        )
        kwargs.update(overrides)
        return kwargs

    def test_gepa_succeeds_no_fallback_invoked(self):
        result, name = _build_optimizer_and_compile(
            **self._common_kwargs(),
            _gepa_runner=_gepa_runner_succeeds,
            _mipro_runner=_mipro_runner_missing_optuna,
        )
        assert isinstance(result, _FakeOptimized)
        assert name == "GEPA"

    def test_gepa_fails_falls_back_to_miprov2(self):
        result, name = _build_optimizer_and_compile(
            **self._common_kwargs(),
            _gepa_runner=_gepa_runner_raises,
            _mipro_runner=_mipro_runner_succeeds,
        )
        assert isinstance(result, _FakeOptimized)
        assert name == "MIPROv2"

    def test_no_fallback_reraises_gepa_failure(self):
        with pytest.raises(RuntimeError, match="simulated GEPA failure"):
            _build_optimizer_and_compile(
                **self._common_kwargs(no_fallback=True),
                _gepa_runner=_gepa_runner_raises,
                _mipro_runner=_mipro_runner_succeeds,
            )

    def test_miprov2_importerror_chains_through_gepa_cause(self):
        # When MIPROv2 raises ImportError (missing optuna), the user must
        # still see the underlying GEPA failure that triggered the fallback;
        # losing that cause forces a re-debug from scratch.
        with pytest.raises(ImportError) as excinfo:
            _build_optimizer_and_compile(
                **self._common_kwargs(),
                _gepa_runner=_gepa_runner_raises,
                _mipro_runner=_mipro_runner_missing_optuna,
            )
        assert excinfo.value.__cause__ is not None
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert "simulated GEPA failure" in str(excinfo.value.__cause__)

    def test_failure_log_written_when_path_provided(self, tmp_path):
        log_path = tmp_path / "gepa_failure.log"
        _build_optimizer_and_compile(
            **self._common_kwargs(failure_log_path=log_path),
            _gepa_runner=_gepa_runner_raises,
            _mipro_runner=_mipro_runner_succeeds,
        )
        assert log_path.exists()
        contents = log_path.read_text()
        assert "RuntimeError" in contents
        assert "simulated GEPA failure" in contents

    def test_fallback_unwraps_prediction_returning_metric(self):
        """When GEPA fails and we fall back to MIPROv2, the metric the
        caller passed is GEPA-shaped (returns dspy.Prediction(score, ...)),
        but MIPROv2 expects a float. _default_mipro_runner must adapt the
        metric so the fallback path doesn't crash on score aggregation."""

        prediction_metric_calls = []

        def prediction_returning_metric(*args, **kwargs):
            prediction_metric_calls.append(args)
            return dspy.Prediction(score=0.7, feedback="test feedback")

        captured_metric = {}

        class _FakeMIPRO:
            def __init__(self, *, metric, **kwargs):
                captured_metric["fn"] = metric

            def compile(self, baseline_module, *, trainset):
                # MIPROv2 calls the metric with a (gold, pred, ...) tuple
                # and expects a float; verify the wrapper produces one.
                fake_pred = type("P", (), {"output": "x"})()
                fake_ex = type("E", (), {})()
                value = captured_metric["fn"](fake_ex, fake_pred)
                assert isinstance(value, float)
                assert value == 0.7
                return _FakeOptimized()

        # Exercise the runner directly so we exercise the wrapping logic,
        # not the broader fallback control flow.
        original_mipro = dspy.MIPROv2
        dspy.MIPROv2 = _FakeMIPRO
        try:
            result = _default_mipro_runner(
                baseline_module=object(),
                trainset=[],
                metric=prediction_returning_metric,
                seed=42,
            )
        finally:
            dspy.MIPROv2 = original_mipro

        assert isinstance(result, _FakeOptimized)
        assert prediction_metric_calls, "metric should have been invoked"
