"""Fitness functions for evaluating evolved artifacts.

Uses LLM-as-judge with rubrics to score agent outputs.
Supports length penalties and multi-dimensional scoring.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig

logger = logging.getLogger(__name__)


@dataclass
class FitnessScore:
    """Multi-dimensional fitness score."""
    correctness: float = 0.0  # Did the agent produce correct output? (0-1)
    procedure_following: float = 0.0  # Did it follow the skill's procedure? (0-1)
    conciseness: float = 0.0  # Was it appropriately concise? (0-1)
    length_penalty: float = 0.0  # Penalty for being too verbose (0-1, 0 = no penalty)
    feedback: str = ""  # Textual feedback for GEPA's reflective analysis

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        raw = (
            0.5 * self.correctness
            + 0.3 * self.procedure_following
            + 0.2 * self.conciseness
        )
        return max(0.0, raw - self.length_penalty)


class LLMJudge:
    """LLM-as-judge scorer with rubric-based evaluation.

    Scores agent outputs on multiple dimensions and provides
    textual feedback that GEPA can use for reflective mutation.
    """

    class JudgeSignature(dspy.Signature):
        """Evaluate an agent's response against an expected behavior rubric.

        Score the response on three dimensions (0.0 to 1.0 each):
        1. correctness: Did the response correctly address the task?
        2. procedure_following: Did it follow the expected approach/procedure?
        3. conciseness: Was it appropriately concise without omitting important info?

        Also provide specific, actionable feedback on what could be improved.
        """
        task_input: str = dspy.InputField(desc="The task the agent was given")
        expected_behavior: str = dspy.InputField(desc="Rubric describing what a good response looks like")
        agent_output: str = dspy.InputField(desc="The agent's actual response")
        skill_text: str = dspy.InputField(desc="The skill/instructions the agent was following")
        # Scores arrive as strings from the LLM and are clamped to [0,1] in
        # score(); declaring them as `str` keeps the typeguard quiet without
        # the ceremony of float-coercion fields.
        correctness: str = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
        procedure_following: str = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
        conciseness: str = dspy.OutputField(desc="Score 0.0-1.0: Appropriately concise?")
        feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.judge = dspy.ChainOfThought(self.JudgeSignature)

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> FitnessScore:
        """Score an agent output using LLM-as-judge."""

        lm = dspy.LM(self.config.eval_model, temperature=0.0, max_tokens=4000)

        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )

        correctness = _clamp_to_unit(result.correctness)
        procedure_following = _clamp_to_unit(result.procedure_following)
        conciseness = _clamp_to_unit(result.conciseness)

        # Length penalty
        length_penalty = 0.0
        if artifact_size is not None and max_size is not None:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                # Penalty ramps from 0 at 90% to 0.3 at 100%+
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        return FitnessScore(
            correctness=correctness,
            procedure_following=procedure_following,
            conciseness=conciseness,
            length_penalty=length_penalty,
            feedback=str(result.feedback),
        )


def skill_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> float:
    """DSPy-compatible metric function for skill optimization.

    This is what gets passed to dspy.GEPA(metric=...).
    Returns a float 0-1 score.

    GEPA binds the metric with five positional args:
    (gold, pred, trace, pred_name, pred_trace). pred_name and pred_trace
    are accepted to satisfy GEPA's runtime arity check; the metric is
    keyword-stable across DSPy 3.2.x.
    """
    # The prediction should have an 'output' field with the agent's response
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""

    if not agent_output.strip():
        # Empty output is a real failure signal (timeout, content filter,
        # malformed prompt) that GEPA's reflective loop can't distinguish
        # from a wrong answer. Logging it lets us diagnose plateaus at 0.
        logger.warning(
            "skill_fitness_metric: empty agent output for task=%r",
            task[:80],
        )
        return 0.0

    # Quick heuristic scoring (for speed during optimization)
    # Full LLM-as-judge scoring is expensive — use it selectively
    score = 0.5  # Base score for non-empty output

    # Check if key phrases from expected behavior appear
    expected_lower = expected.lower()
    output_lower = agent_output.lower()

    # Simple keyword overlap as a fast proxy
    expected_words = set(expected_lower.split())
    output_words = set(output_lower.split())
    if expected_words:
        overlap = len(expected_words & output_words) / len(expected_words)
        score = 0.3 + (0.7 * overlap)

    return min(1.0, max(0.0, score))


def _clamp_to_unit(value: str) -> float:
    """Parse an LLM-emitted score string and clamp it to [0, 1].

    Falls back to 0.5 (neutral) on malformed output rather than raising —
    LLMs occasionally emit explanatory text or confidence ranges instead
    of a clean float. Loud failure here would crash an optimization run
    over a single noisy judge call.
    """
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5
