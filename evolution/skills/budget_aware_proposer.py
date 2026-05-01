"""Custom GEPA instruction proposer that bakes a length budget into the
reflection prompt.

Why this exists: `gepa.optimize`'s `reflection_prompt_template` kwarg is
unconditionally rejected when the DSPy adapter is in use
(`gepa/api.py:317-321` asserts that the adapter does not provide its own
`propose_new_texts`, which `DspyAdapter` always does). DSPy's documented
extension point is `instruction_proposer: ProposalFn` on `dspy.GEPA`, so
that's what we use to inject growth-budget awareness into the prompt the
reflection LM sees on every iteration.

The ProposalFn protocol lives at `gepa.core.adapter.ProposalFn` and is
implemented here via duck typing — it's a `typing.Protocol`, so we don't
need to inherit.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import dspy

logger = logging.getLogger(__name__)


class _BudgetAwareInstructionProposal(dspy.Signature):
    """Placeholder docstring — overwritten per-instance via with_instructions
    in BudgetAwareProposer.__init__ so the budget numbers can be baked in."""

    current_instruction: str = dspy.InputField(
        desc="The current instruction text the assistant was given."
    )
    examples_with_feedback: str = dspy.InputField(
        desc="Task examples showing the assistant's outputs and the feedback "
        "explaining what should improve."
    )
    improved_instruction: str = dspy.OutputField(
        desc="A better instruction that fixes the issues raised in the "
        "feedback while staying within the length budget."
    )


# The phrasings below are picked from length-constrained-generation
# literature: countdown framing + "at most N" + loss-frame ("each char
# above N is wasted") + a one-shot example at the target length. Together
# these raised exact-length compliance from <30% to >95% on GPT-4.1
# (arXiv:2508.13805). The example is from the terminal/CLI domain to
# match our typical input style.
_TIGHT_INSTRUCTION_EXAMPLE = """\
Read, list, and search markdown notes in $VAULT_PATH (default ~/notes).

Read: cat "$VAULT_PATH/<name>.md"
List: find "$VAULT_PATH" -name '*.md'
Search by name: find "$VAULT_PATH" -iname '*<term>*.md'
Search by content: grep -rli "<term>" "$VAULT_PATH" --include='*.md'
Always quote paths to handle spaces."""


_BUDGET_AWARE_INSTRUCTIONS = """\
Length budget: at most {target_chars} characters. Each character above {target_chars} is wasted.
The current baseline instruction is {baseline_chars} characters; you are revising it.

Example of an instruction at the target length:
\"\"\"
{tight_example}
\"\"\"

Your task: rewrite the current instruction to fix the failures shown below, staying at most {target_chars} characters.

Steps:
1. Read the failure modes in the feedback below.
2. Keep the niche / domain-specific facts the assistant needs.
3. Cut redundant phrasing, headings, ceremony, and explanatory prose. Imperative mood. No preamble.
4. If a fix needs new content, delete equal-or-greater redundant content elsewhere first.

Output the new instruction text only — no preamble, no markdown fences, no explanation."""


class BudgetAwareProposer:
    """GEPA-compatible ProposalFn that enforces a per-skill char budget.

    Construct one per evolve() invocation: the budget is closed over the
    baseline skill body length, so the same instance can't be reused
    across skills. GEPA calls __call__ once per iteration with the current
    candidate text and a reflective dataset summarizing the latest
    failures.

    Why `safety_margin`: in observed runs the reflection LM overshoots
    the requested target by ~8-9% — asking for +20% growth lands at
    +30%, busting the validator. We tell the LM a *tighter* target than
    the validator's bar so the observed overshoot still lands inside it.
    Default 0.10 means: if max_growth=0.20 (validator bar at +20%), the
    prompt requests +10%, expected actual ~+18-20%, passes.
    """

    def __init__(
        self,
        baseline_chars: int,
        max_growth: float = 0.2,
        safety_margin: float = 0.10,
    ):
        self.baseline_chars = baseline_chars
        prompt_growth = max(0.0, max_growth - safety_margin)
        # Floor at 1 so the LM never gets handed a literal 0; baseline_chars=0
        # is the documented "disable the budget" case anyway.
        self.target_chars = max(1, int(baseline_chars * (1 + prompt_growth)))
        signature = _BudgetAwareInstructionProposal.with_instructions(
            _BUDGET_AWARE_INSTRUCTIONS.format(
                baseline_chars=baseline_chars,
                target_chars=self.target_chars,
                tight_example=_TIGHT_INSTRUCTION_EXAMPLE,
            )
        )
        self.propose = dspy.Predict(signature)

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        updated: dict[str, str] = {}
        for name in components_to_update:
            if name not in candidate or name not in reflective_dataset:
                continue
            examples_text = self._format_examples(reflective_dataset[name])
            result = self.propose(
                current_instruction=candidate[name],
                examples_with_feedback=examples_text,
            )
            new_text = result.improved_instruction
            new_len = len(new_text)
            pct_of_target = (new_len / self.target_chars * 100) if self.target_chars else 0
            # GEPA's own logs print proposer outputs but not inputs; this is
            # the only way to confirm the budget signal reached the LM.
            logger.info(
                "BudgetAwareProposer iter: target=%d, proposed[%s]=%d chars (%.1f%% of target)",
                self.target_chars, name, new_len, pct_of_target,
            )
            if new_len > self.target_chars:
                # Soft enforcement only: hard truncation would corrupt
                # mid-sentence and could lose the very change that helped.
                logger.warning(
                    "BudgetAwareProposer: %s came back at %d chars (target %d)",
                    name, new_len, self.target_chars,
                )
            updated[name] = new_text
        return updated

    @staticmethod
    def _format_examples(reflective_examples: Sequence[Mapping[str, Any]]) -> str:
        """Render the reflective dataset as readable markdown for the LM.

        Each entry is a dict like {"Inputs": ..., "Generated Outputs": ...,
        "Feedback": ...} (see `dspy/teleprompt/gepa/gepa_utils.py`
        make_reflective_dataset).
        """
        chunks: list[str] = []
        for i, example in enumerate(reflective_examples, start=1):
            chunks.append(f"### Example {i}")
            for key, value in example.items():
                chunks.append(f"**{key}:** {value}")
            chunks.append("")
        return "\n".join(chunks).strip()
