"""Tests for the GEPA-compatible BudgetAwareProposer.

The proposer must:
- Conform to gepa.core.adapter.ProposalFn (callable returning dict[str, str]).
- Bake real char numbers into the predictor's signature instructions
  (not leave format placeholders) so the LM sees a concrete budget.
- Skip components missing from candidate or reflective_dataset (defensive).
- Warn (but not raise) when the LM's proposal exceeds the target budget.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evolution.skills.budget_aware_proposer import (
    _BUDGET_AWARE_INSTRUCTIONS,
    BudgetAwareProposer,
)


class TestProposerConstruction:
    def test_target_chars_is_baseline_times_one_plus_growth(self):
        proposer = BudgetAwareProposer(baseline_chars=1000, max_growth=0.2)
        assert proposer.target_chars == 1200

    def test_target_chars_floored_at_one_for_zero_baseline(self):
        # Defensive: a 0 baseline shouldn't yield target_chars=0.
        proposer = BudgetAwareProposer(baseline_chars=0, max_growth=0.2)
        assert proposer.target_chars == 1

    def test_signature_instructions_have_baked_in_numbers(self):
        proposer = BudgetAwareProposer(baseline_chars=1264, max_growth=0.2)
        # 1264 * 1.2 = 1516.8 → 1516 after int().
        instructions = proposer.propose.signature.instructions
        assert "1264" in instructions
        assert "1516" in instructions
        # No leaked format placeholders.
        assert "{baseline_chars}" not in instructions
        assert "{target_chars}" not in instructions


class TestProposerInstructions:
    """The static instruction template should hard-code the budget framing."""

    def test_template_has_required_format_keys(self):
        # If we rename a key in the f-string the .format() call below would
        # raise — surface it as a test rather than at GEPA call time.
        formatted = _BUDGET_AWARE_INSTRUCTIONS.format(baseline_chars=100, target_chars=120)
        assert "100" in formatted
        assert "120" in formatted

    def test_template_calls_budget_a_hard_requirement(self):
        # The wording matters — "preference" gets ignored, "HARD REQUIREMENT"
        # is what the literature suggests survives instruction-following.
        assert "HARD REQUIREMENT" in _BUDGET_AWARE_INSTRUCTIONS


class TestProposerCallable:
    def _patched_proposer(self, returned_text: str) -> BudgetAwareProposer:
        proposer = BudgetAwareProposer(baseline_chars=100, max_growth=0.2)
        proposer.propose = MagicMock(
            return_value=SimpleNamespace(improved_instruction=returned_text)
        )
        return proposer

    def test_returns_dict_mapping_component_to_new_text(self):
        proposer = self._patched_proposer("new short instruction")
        result = proposer(
            candidate={"predict": "old instruction"},
            reflective_dataset={"predict": [{"Inputs": "x", "Generated Outputs": "y", "Feedback": "z"}]},
            components_to_update=["predict"],
        )
        assert result == {"predict": "new short instruction"}
        # And the predictor was called once with the rendered examples.
        proposer.propose.assert_called_once()
        kwargs = proposer.propose.call_args.kwargs
        assert kwargs["current_instruction"] == "old instruction"
        assert "Example 1" in kwargs["examples_with_feedback"]

    def test_skips_component_missing_from_candidate(self):
        proposer = self._patched_proposer("ignored")
        result = proposer(
            candidate={"predict": "x"},
            reflective_dataset={"predict": [{"k": "v"}], "missing": [{"k": "v"}]},
            components_to_update=["missing"],
        )
        assert result == {}
        proposer.propose.assert_not_called()

    def test_skips_component_missing_from_reflective_dataset(self):
        proposer = self._patched_proposer("ignored")
        result = proposer(
            candidate={"predict": "x", "missing": "y"},
            reflective_dataset={"predict": [{"k": "v"}]},
            components_to_update=["missing"],
        )
        assert result == {}
        proposer.propose.assert_not_called()

    def test_warns_on_oversized_proposal(self, caplog):
        # target = 100 * 1.2 = 120 chars; proposal is 200 chars.
        proposer = self._patched_proposer("x" * 200)
        with caplog.at_level(logging.WARNING, logger="evolution.skills.budget_aware_proposer"):
            result = proposer(
                candidate={"predict": "old"},
                reflective_dataset={"predict": [{"Inputs": "i", "Generated Outputs": "o", "Feedback": "f"}]},
                components_to_update=["predict"],
            )

        # Soft enforcement: the proposal is still returned (truncating
        # mid-sentence would corrupt instructions); we just log so we
        # can see whether the LM is honoring the budget across runs.
        assert result == {"predict": "x" * 200}
        assert any("came back at 200 chars" in rec.message for rec in caplog.records)
        assert any("target 120" in rec.message for rec in caplog.records)


class TestExampleFormatting:
    def test_renders_example_dict_as_markdown(self):
        proposer = BudgetAwareProposer(baseline_chars=100, max_growth=0.2)
        out = proposer._format_examples([
            {"Inputs": "the input", "Generated Outputs": "the output", "Feedback": "the critique"},
            {"Inputs": "input2", "Generated Outputs": "output2", "Feedback": "critique2"},
        ])
        assert "### Example 1" in out
        assert "**Inputs:** the input" in out
        assert "### Example 2" in out
        assert "**Feedback:** critique2" in out

    def test_handles_empty_dataset(self):
        proposer = BudgetAwareProposer(baseline_chars=100, max_growth=0.2)
        assert proposer._format_examples([]) == ""
