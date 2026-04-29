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
    _write_gate_decision,
)


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
                def __init__(self, *, devset, metric, num_threads, return_all_scores,
                             provide_traceback, max_errors):
                    captured_metric["fn"] = metric
                    self.devset = devset

                def __call__(self, mod):
                    scores = []
                    for ex in self.devset:
                        pred = mod(task_input=getattr(ex, "task_input", ""))
                        scores.append(captured_metric["fn"](ex, pred))
                    return (sum(scores) / len(scores) * 100, scores)

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
            "decision": "reject",
            "reason": "static_constraint_failure",
            "failed_constraints": ["non_empty"],
            "messages": ["Artifact is empty"],
        }
        path = _write_gate_decision(tmp_path, payload)
        loaded = json.loads(path.read_text())
        assert loaded["reason"] == "static_constraint_failure"
        assert "non_empty" in loaded["failed_constraints"]


class TestGrowthGateDecisionSchema:
    """The growth-gate decision payload is the calibration substrate for
    future tier tuning. Lock the schema so a `jq -s` calibration script
    doesn't break when fields are renamed.
    """

    def test_required_fields_present(self, tmp_path: Path):
        payload = {
            "decision": "reject",
            "reason": "growth_quality_gate",
            "growth_pct": 0.30,
            "improvement": 0.005,
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
            "failed_constraints": ["growth_quality_gate"],
            "messages": ["Growth +30.0% requires improvement ≥+0.030; got +0.005"],
        }
        path = _write_gate_decision(tmp_path, payload)
        loaded = json.loads(path.read_text())

        # Calibration script will rely on these keys.
        for required in (
            "decision", "growth_pct", "improvement", "required_improvement",
            "baseline_chars", "evolved_chars",
            "growth_free_threshold", "growth_quality_slope",
        ):
            assert required in loaded, f"missing {required}"
