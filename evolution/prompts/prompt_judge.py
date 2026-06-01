"""LLM-as-judge for memory-save calls — scores args against MEMORY_GUIDANCE rules.

Layer 2 of the compound verdict. Layer 1 (trigger membership) is handled
by ``score_task``'s existing expected_tools / forbidden_tools logic.
"""

from __future__ import annotations

import logging
from typing import Any

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.fitness import _clamp_to_unit

logger = logging.getLogger(__name__)

MAX_JUDGED_CALLS_PER_TASK = 5
"""Cap on how many save calls per task the judge will score. Excess calls
beyond the cap score 0 each — bounds cost on pathological cases where the
agent saves on every turn."""


class SaveCallSignature(dspy.Signature):
    """Score a memory-save call against MEMORY_GUIDANCE's rules.

    Output ``quality`` (0.0-1.0): how well ``saved_content`` follows the
    rules — durable (not stale in a week), declarative phrasing (not
    imperative), focused on facts that prevent future correction, and NOT
    task progress, PR numbers, or completed-work logs.
    """

    task: str = dspy.InputField(desc="The user task that prompted the save")
    expected_content: str = dspy.InputField(
        desc="A rubric for what the saved content should resemble (not exact text)"
    )
    saved_content: str = dspy.InputField(desc="The content the agent actually saved")
    quality: str = dspy.OutputField(
        desc="0.0-1.0 quality score per MEMORY_GUIDANCE rules"
    )
    feedback: str = dspy.OutputField(
        desc="One-sentence diagnosis of any rule violation; empty if quality is 1.0"
    )


class SaveCallJudge:
    """LLM scorer for individual memory-save calls."""

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.judge = dspy.ChainOfThought(SaveCallSignature)

    def score(self, *, task: str, expected_content: str, saved_content: str) -> float:
        _lm = self.config.get_lm("eval")
        lm = dspy.LM(
            _lm.model,
            **_lm.lm_kwargs,
            temperature=0.0,
            max_tokens=1000,
            request_timeout=60,
            num_retries=5,
        )
        with dspy.context(lm=lm):
            result = self.judge(
                task=task,
                expected_content=expected_content,
                saved_content=saved_content,
            )
        return _clamp_to_unit(result.quality)


def judge_save_calls(
    *,
    judge: SaveCallJudge | None,
    calls: list[dict[str, Any]],
    expected_content: str | None,
    task_text: str = "",
) -> float:
    """Aggregate the Layer 2 score across a task's memory-save calls.

    ``calls`` is the subset of ``tool_calls_with_args`` whose name is
    ``memory`` — each item the call's ``arguments`` dict. Only
    ``action == 'save'`` calls are judged.

    Returns 1.0 when no save calls were made (Layer 1 catches the
    "should-have-saved-but-didn't" failure; Layer 2 only scores what
    actually happened) and also when no judge/rubric is configured.
    """
    save_calls = [c for c in calls if c.get("action") == "save"]
    if not save_calls:
        return 1.0
    if judge is None or expected_content is None:
        return 1.0

    judged = save_calls[:MAX_JUDGED_CALLS_PER_TASK]
    unjudged_count = max(0, len(save_calls) - MAX_JUDGED_CALLS_PER_TASK)

    scores: list[float] = []
    for call in judged:
        scores.append(judge.score(
            task=task_text,
            expected_content=expected_content,
            saved_content=str(call.get("content", "")),
        ))
    scores.extend([0.0] * unjudged_count)
    return sum(scores) / len(scores)
