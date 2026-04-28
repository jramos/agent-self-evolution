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


_BUDGET_AWARE_INSTRUCTIONS = """\
You are improving an assistant's task instructions based on feedback from real
runs. Read the current instruction and the examples-with-feedback below, then
write a new instruction that fixes the issues identified in the feedback.

Steps:
1. Identify the failure modes shown in the feedback.
2. Identify the niche / domain-specific facts the assistant needs to keep
   getting right; preserve them.
3. Compose a new instruction that targets the failures concisely.

Length budget (HARD REQUIREMENT, not a preference):
- The original baseline instruction was roughly {baseline_chars} characters.
- The new instruction MUST be at most {target_chars} characters.
- If you need to add a sentence, find redundant phrasing elsewhere to remove.
- Verbosity that does not directly fix the failures shown above will be rejected.
- Prefer terse, direct prose over headings, bullet lists, and ceremony.

Output the new instruction text only — no preamble, no markdown fences,
no explanation."""


class BudgetAwareProposer:
    """GEPA-compatible ProposalFn that enforces a per-skill char budget.

    Construct one per evolve() invocation: the budget is closed over the
    baseline skill body length, so the same instance can't be reused
    across skills. GEPA calls __call__ once per iteration with the current
    candidate text and a reflective dataset summarizing the latest
    failures.
    """

    def __init__(self, baseline_chars: int, max_growth: float = 0.2):
        self.baseline_chars = baseline_chars
        # Floor the target at 1 to avoid handing the LM a literal 0; if the
        # caller passes baseline_chars=0 they're disabling the budget anyway.
        self.target_chars = max(1, int(baseline_chars * (1 + max_growth)))
        signature = _BudgetAwareInstructionProposal.with_instructions(
            _BUDGET_AWARE_INSTRUCTIONS.format(
                baseline_chars=baseline_chars,
                target_chars=self.target_chars,
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
            if len(new_text) > self.target_chars:
                # Soft enforcement: the prompt asks for ≤target_chars, but
                # LMs occasionally overshoot. Log so we can see whether the
                # signal is actually being respected before deciding to add
                # hard truncation.
                logger.warning(
                    "BudgetAwareProposer: %s came back at %d chars (target %d)",
                    name, len(new_text), self.target_chars,
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
