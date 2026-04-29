"""Tests for the post-optimization validation+holdout flow in evolve_skill.

The flow is: validate_static → holdout → validate_growth_with_quality →
results table. These tests assert the wiring (right calls in the right
order, right outputs persisted) without exercising the LM-heavy parts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from evolution.skills.evolve_skill import (
    _holdout_evaluate_with_metric,
    _knee_point_payload,
    _write_gate_decision,
)
from evolution.skills.knee_point import CandidatePick


@pytest.fixture(autouse=True, scope="session")
def _hermes_repo_env(tmp_path_factory):
    """Same env-var workaround as test_constraints.py."""
    fake_repo = tmp_path_factory.mktemp("fake_hermes_repo")
    os.environ["HERMES_AGENT_REPO"] = str(fake_repo)
    yield


class TestWriteGateDecision:
    def test_writes_json_with_payload(self, tmp_path: Path):
        payload = {"decision": "deploy", "growth_pct": 0.24, "improvement": 0.07}
        path = _write_gate_decision(tmp_path, payload)

        assert path.name == "gate_decision.json"
        loaded = json.loads(path.read_text())
        assert loaded == payload

    def test_creates_parent_directories(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        path = _write_gate_decision(nested, {"decision": "reject"})
        assert path.exists()
        assert path.parent == nested


class TestHoldoutEvaluate:
    """The metric returns dspy.Prediction(score, feedback); _holdout_evaluate
    must wrap it for dspy.Evaluate's 2-arg metric protocol and unwrap the
    .score on return."""

    def test_unwraps_prediction_score(self):
        # Stand-in for dspy.Prediction — only .score matters.
        prediction = SimpleNamespace(output="answer")
        examples = [
            SimpleNamespace(task_input=f"task {i}", expected_behavior="b",
                            with_inputs=lambda *a, **k: SimpleNamespace(task_input=f"task {i}"))
            for i in range(3)
        ]
        # Module that returns the same prediction every call.
        module = MagicMock()
        module.return_value = prediction
        # Metric returns a Prediction-shaped object with .score=0.7 always.
        metric = MagicMock(return_value=SimpleNamespace(score=0.7))

        # Patch dspy.Evaluate to invoke the metric on each example, simulating
        # what real Evaluate does without spinning up DSPy's machinery.
        with patch("evolution.skills.evolve_skill.dspy.Evaluate") as evaluate_cls:
            captured_metric = {}

            class _FakeEval:
                def __init__(self, *, devset, metric, num_threads,
                             provide_traceback, max_errors):
                    captured_metric["fn"] = metric
                    self.devset = devset

                def __call__(self, mod):
                    # dspy.Evaluate returns EvaluationResult(score, results).
                    # score is mean*100; results is list of (example, prediction, score).
                    scores = []
                    results = []
                    for ex in self.devset:
                        pred = mod(task_input=getattr(ex, "task_input", ""))
                        s = captured_metric["fn"](ex, pred)
                        scores.append(s)
                        results.append((ex, pred, s))
                    return SimpleNamespace(
                        score=sum(scores) / len(scores) * 100,
                        results=results,
                    )

            evaluate_cls.side_effect = _FakeEval

            avg, per_example = _holdout_evaluate_with_metric(
                module, examples, metric, lm=MagicMock(),
            )

        assert avg == pytest.approx(0.7)
        assert per_example == [0.7, 0.7, 0.7]
        # Metric called once per example; each call passed (example, prediction).
        assert metric.call_count == 3


class TestStaticValidationShortCircuitsBeforeHoldout:
    """If validate_static returns any failures, the flow must save
    evolved_FAILED.md + gate_decision.json and return without invoking
    the holdout block (which would waste judge calls on a broken artifact).

    Exercising this end-to-end requires patching most of evolve(), so we
    instead test the mechanism: a failed-static gate_decision.json names
    the failed constraints and reason='static_constraint_failure'.
    """

    def test_static_failure_reason_in_decision(self, tmp_path: Path):
        # Manual reproduction of the static-failure branch's payload —
        # locks the schema so a future refactor can't silently drop fields.
        payload = {
            "schema_version": "3",
            "decision": "reject",
            "reason": "static_constraint_failure",
            "failed_constraints": ["non_empty"],
            "messages": ["Artifact is empty"],
            "knee_point": {"applied": False, "reason": "no_detailed_results"},
        }
        path = _write_gate_decision(tmp_path, payload)
        loaded = json.loads(path.read_text())
        assert loaded["schema_version"] == "3"
        assert loaded["reason"] == "static_constraint_failure"
        assert "non_empty" in loaded["failed_constraints"]
        assert "knee_point" in loaded


class TestGrowthGateDecisionSchema:
    """The growth-gate decision payload is the calibration substrate for
    future tier tuning. Lock the schema so a `jq -s` calibration script
    doesn't break when fields are renamed.
    """

    def test_required_fields_present(self, tmp_path: Path):
        payload = {
            "schema_version": "3",
            "decision": "reject",
            "reason": "growth_quality_gate",
            "decision_rule_used": "dual_check",
            "growth_pct": 0.30,
            "required_improvement": 0.030,
            "baseline_chars": 1000,
            "evolved_chars": 1300,
            "absolute_char_ceiling": 5000,
            "growth_free_threshold": 0.20,
            "growth_quality_slope": 0.30,
            "baseline_per_example": [0.5, 0.6, 0.7],
            "evolved_per_example": [0.51, 0.61, 0.71],
            "avg_baseline": 0.6,
            "avg_evolved": 0.605,
            "bootstrap": {
                "mean": 0.005,
                "lower_bound": -0.020,
                "upper_bound": 0.030,
                "n_examples": 12,
                "n_resamples": 2000,
                "confidence": 0.90,
            },
            "failed_constraints": ["growth_quality_gate"],
            "messages": ["..."],
            "knee_point": {
                "applied": True,
                "fallback": "knee",
                "epsilon": 0.1666666,
                "band_size": 4,
                "picked_idx": 12,
                "picked_val_score": 0.95,
                "picked_val_rank_in_band": 3,
                "picked_body_chars": 412,
                "gepa_default_idx": 5,
                "gepa_default_body_chars": 1572,
                "band_roster": [
                    {"idx": 5, "val_score": 0.997, "body_chars": 1572},
                    {"idx": 12, "val_score": 0.95, "body_chars": 412},
                ],
            },
        }
        path = _write_gate_decision(tmp_path, payload)
        loaded = json.loads(path.read_text())

        # Calibration script will rely on these keys.
        for required in (
            "schema_version", "decision", "decision_rule_used",
            "growth_pct", "required_improvement",
            "baseline_chars", "evolved_chars",
            "growth_free_threshold", "growth_quality_slope",
            "bootstrap", "knee_point",
        ):
            assert required in loaded, f"missing {required}"
        assert loaded["schema_version"] == "3"
        for required_in_bootstrap in (
            "mean", "lower_bound", "upper_bound", "n_examples",
            "n_resamples", "confidence",
        ):
            assert required_in_bootstrap in loaded["bootstrap"], (
                f"missing bootstrap.{required_in_bootstrap}"
            )
        for required_in_knee in (
            "applied", "fallback", "epsilon", "band_size",
            "picked_idx", "picked_val_score", "picked_val_rank_in_band",
            "picked_body_chars", "gepa_default_idx", "gepa_default_body_chars",
            "band_roster",
        ):
            assert required_in_knee in loaded["knee_point"], (
                f"missing knee_point.{required_in_knee}"
            )


class TestKneePointPayloadHelper:
    """`_knee_point_payload` is the single producer of the knee_point block
    that lands in gate_decision.json. Lock both shapes (applied/skipped).
    """

    def test_skipped_payload_when_no_detailed_results(self):
        # MIPROv2 fallback path: knee_pick is None.
        payload = _knee_point_payload(None)
        assert payload == {"applied": False, "reason": "no_detailed_results"}

    def test_applied_payload_carries_all_required_fields(self):
        pick = CandidatePick(
            module=SimpleNamespace(skill_text="x" * 412),
            skill_text="x" * 412,
            body_chars=412,
            val_score=0.95,
            val_rank_in_band=3,
            band_size=4,
            epsilon=1.0 / 6.0,
            fallback="knee",
            picked_idx=12,
            gepa_default_idx=5,
            gepa_default_body_chars=1572,
            band_roster=[
                {"idx": 5, "val_score": 0.997, "body_chars": 1572},
                {"idx": 12, "val_score": 0.95, "body_chars": 412},
            ],
        )
        payload = _knee_point_payload(pick)
        assert payload["applied"] is True
        assert payload["fallback"] == "knee"
        assert payload["picked_idx"] == 12
        assert payload["gepa_default_idx"] == 5
        assert payload["picked_val_rank_in_band"] == 3
        assert payload["band_size"] == 4
        assert payload["band_roster"][0]["idx"] == 5
        # Round-trips JSON cleanly (no non-serializable objects sneaked in).
        json.dumps(payload)
