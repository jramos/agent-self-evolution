"""Tests for the skill fitness metric."""

import inspect

from evolution.core.fitness import (
    LLMJudge,
    _clamp_to_unit,
    skill_fitness_metric,
)


class TestMetricArity:
    """Lock the metric signature against DSPy GEPA's runtime arity check.

    DSPy 3.2.0's dspy.GEPA.__init__ runs:
        inspect.signature(metric).bind(None, None, None, None, None)
    and raises TypeError if the metric can't accept five positional args.
    Required signature: (gold, pred, trace=None, pred_name=None, pred_trace=None).
    """

    def test_skill_fitness_metric_accepts_five_positional_args(self):
        inspect.signature(skill_fitness_metric).bind(None, None, None, None, None)

    def test_skill_fitness_metric_accepts_two_positional_args(self):
        inspect.signature(skill_fitness_metric).bind(None, None)


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
