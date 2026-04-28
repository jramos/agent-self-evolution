"""Tests for the skill fitness metric."""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import dspy

from evolution.core.fitness import (
    FitnessScore,
    LLMJudge,
    _clamp_to_unit,
    make_skill_fitness_metric,
)


def _stub_judge(score: float = 0.7, feedback: str = "explanatory feedback") -> Mock:
    """Build a Mock LLMJudge whose .score() returns a deterministic FitnessScore."""
    judge = Mock(spec=LLMJudge)
    judge.score.return_value = FitnessScore(
        correctness=score,
        procedure_following=score,
        conciseness=score,
        length_penalty=0.0,
        feedback=feedback,
    )
    return judge


class TestMetricArity:
    """Lock the metric signature against DSPy GEPA's runtime arity check.

    DSPy 3.2.0's dspy.GEPA.__init__ runs:
        inspect.signature(metric).bind(None, None, None, None, None)
    and raises TypeError if the metric can't accept five positional args.
    Required signature: (gold, pred, trace=None, pred_name=None, pred_trace=None).
    """

    def test_factory_metric_accepts_five_positional_args(self):
        metric = make_skill_fitness_metric(_stub_judge())
        inspect.signature(metric).bind(None, None, None, None, None)

    def test_factory_metric_accepts_two_positional_args(self):
        metric = make_skill_fitness_metric(_stub_judge())
        inspect.signature(metric).bind(None, None)


class TestFactoryMetricBehavior:
    def test_returns_prediction_with_score_and_feedback(self):
        judge = _stub_judge(score=0.6, feedback="agent missed step 3")
        metric = make_skill_fitness_metric(judge)

        example = SimpleNamespace(task_input="t", expected_behavior="b")
        prediction = SimpleNamespace(output="some non-empty agent response")

        result = metric(example, prediction)

        assert isinstance(result, dspy.Prediction)
        # FitnessScore.composite for all-0.6 inputs: 0.5*0.6 + 0.3*0.6 + 0.2*0.6 = 0.6
        assert result.score == 0.6
        assert result.feedback == "agent missed step 3"
        # Judge was actually invoked with the right inputs (no skill_text).
        judge.score.assert_called_once_with(
            task_input="t",
            expected_behavior="b",
            agent_output="some non-empty agent response",
        )

    def test_empty_output_returns_zero_without_calling_judge(self, caplog):
        judge = _stub_judge()
        metric = make_skill_fitness_metric(judge)

        example = SimpleNamespace(task_input="t", expected_behavior="b")
        prediction = SimpleNamespace(output="   ")  # whitespace-only

        with caplog.at_level("WARNING", logger="evolution.core.fitness"):
            result = metric(example, prediction)

        assert isinstance(result, dspy.Prediction)
        assert result.score == 0.0
        assert "empty" in result.feedback.lower()
        judge.score.assert_not_called()
        assert any("empty agent output" in rec.message for rec in caplog.records)


class TestJudgeSignatureFieldTypes:
    """LLM scores arrive as strings; declaring float OutputFields used to
    trigger typeguard warnings on every call. Lock the declared types as
    str so the warning suppression isn't load-bearing for this surface."""

    def test_score_outputs_are_str(self):
        sig = LLMJudge.JudgeSignature
        assert sig.output_fields["correctness"].annotation is str
        assert sig.output_fields["procedure_following"].annotation is str
        assert sig.output_fields["conciseness"].annotation is str

    def test_feedback_output_is_str(self):
        sig = LLMJudge.JudgeSignature
        assert sig.output_fields["feedback"].annotation is str

    def test_skill_text_input_was_dropped(self):
        # The judge no longer reads the skill body — that risked rewarding
        # outputs that quote the skill back instead of solving the task.
        sig = LLMJudge.JudgeSignature
        assert "skill_text" not in sig.input_fields


class TestClampToUnit:
    def test_parses_valid_floats(self):
        assert _clamp_to_unit("0.7") == 0.7
        assert _clamp_to_unit("0.0") == 0.0
        assert _clamp_to_unit("1.0") == 1.0

    def test_clamps_above_one(self):
        assert _clamp_to_unit("1.5") == 1.0
        assert _clamp_to_unit("99") == 1.0

    def test_clamps_below_zero(self):
        assert _clamp_to_unit("-0.3") == 0.0

    def test_handles_whitespace(self):
        assert _clamp_to_unit("  0.42  ") == 0.42

    def test_neutral_fallback_on_garbage(self):
        # LLMs occasionally emit explanations instead of bare floats.
        assert _clamp_to_unit("not a number") == 0.5
        assert _clamp_to_unit("") == 0.5
        assert _clamp_to_unit(None) == 0.5
