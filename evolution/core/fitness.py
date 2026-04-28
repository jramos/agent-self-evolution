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


def make_skill_fitness_metric(judge: "LLMJudge"):
    """Build a GEPA-compatible metric closed over an LLMJudge instance.

    The returned callable matches GEPA's metric protocol:
    (gold, pred, trace=None, pred_name=None, pred_trace=None) ->
    dspy.Prediction(score: float, feedback: str).

    Returning Prediction(score, feedback) lets GEPA's reflective loop
    consume the judge's natural-language critique directly instead of
    falling back to its canned "trajectory got a score of {x}" template
    (see dspy/teleprompt/gepa/gepa.py:537).
    """

    def skill_fitness_metric(
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace=None,
        pred_name=None,
        pred_trace=None,
    ):
        agent_output = getattr(prediction, "output", "") or ""
        task = getattr(example, "task_input", "") or ""

        if not agent_output.strip():
            # Empty output is a real upstream failure signal (timeout,
            # content filter, malformed prompt) that GEPA can't otherwise
            # distinguish from a wrong answer. Surface it.
            logger.warning(
                "skill_fitness_metric: empty agent output for task=%r",
                task[:80],
            )
            return dspy.Prediction(
                score=0.0,
                feedback="Agent produced empty output (no characters returned).",
            )

        score = judge.score(
            task_input=task,
            expected_behavior=getattr(example, "expected_behavior", "") or "",
            agent_output=agent_output,
        )
        return dspy.Prediction(score=score.composite, feedback=score.feedback)

    return skill_fitness_metric


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
