"""Tests for the skill fitness metric."""

import inspect

from evolution.core.fitness import skill_fitness_metric


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
