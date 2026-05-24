"""Unit tests for `build_run_inputs`.

The helper centralizes the construction of the `run_inputs` block written
into every gate_decision.json. Both `evolve_skill` and `evolve_tool` build
the same nine-key core; the tool side adds two extra keys
(`fitness_profile`, `enable_confusable_bucket`). Lock the shape so future
refactors can't silently drop a key — the cost-ceiling fallback in
`evolve_tool` historically dropped `enable_confusable_bucket`, breaking
the deploy-gate contract asserted by `TestGateDecisionSchemaOnDeploy`.
"""

from __future__ import annotations

from evolution.core.config import EvolutionConfig
from evolution.core.hermes_provider import resolved_lms_dump
from evolution.core.run_inputs import build_run_inputs


def _fake_config() -> EvolutionConfig:
    return EvolutionConfig(
        seed=42,
        reflection_model="openai/gpt-4.1",
        eval_model="openai/gpt-4.1-mini",
        eval_dataset_size=150,
        holdout_ratio=0.5,
        enable_confusable_bucket=False,
    )


class TestBuildRunInputs:
    def test_skill_side_has_nine_keys(self):
        config = _fake_config()
        result = build_run_inputs(
            config=config,
            iterations=10,
            optimizer_model="openai/gpt-4.1",
            quality_gate_preset="default",
            eval_source="synthetic",
        )
        assert set(result.keys()) == {
            "seed",
            "iterations",
            "optimizer_model",
            "reflection_model",
            "eval_model",
            "resolved_lms",
            "eval_dataset_size",
            "holdout_ratio",
            "quality_gate_preset",
            "eval_source",
        }
        assert len(result) == 10

    def test_tool_side_adds_fitness_profile_and_confusable_bucket(self):
        config = _fake_config()
        config.enable_confusable_bucket = True
        result = build_run_inputs(
            config=config,
            iterations=10,
            optimizer_model="openai/gpt-4.1",
            quality_gate_preset="default",
            eval_source="synthetic",
            fitness_profile="balanced",
            enable_confusable_bucket=True,
        )
        assert set(result.keys()) == {
            "seed",
            "iterations",
            "optimizer_model",
            "reflection_model",
            "eval_model",
            "resolved_lms",
            "eval_dataset_size",
            "holdout_ratio",
            "quality_gate_preset",
            "eval_source",
            "fitness_profile",
            "enable_confusable_bucket",
        }
        assert len(result) == 12
        assert result["fitness_profile"] == "balanced"
        assert result["enable_confusable_bucket"] is True

    def test_resolved_lms_matches_helper_output(self):
        config = _fake_config()
        result = build_run_inputs(
            config=config,
            iterations=10,
            optimizer_model="openai/gpt-4.1",
            quality_gate_preset="default",
            eval_source="synthetic",
        )
        expected = resolved_lms_dump(
            optimizer="openai/gpt-4.1",
            reflection=config.reflection_model,
            eval_=config.eval_model,
        )
        assert result["resolved_lms"] == expected

    def test_enable_confusable_bucket_round_trips_when_passed(self):
        # Regression: the cost-ceiling fallback in evolve_tool historically
        # built run_inputs without `enable_confusable_bucket`, which broke
        # `TestGateDecisionSchemaOnDeploy::test_gate_decision_schema_on_deploy`
        # whenever the cost ceiling tripped on a deploy path.
        config = _fake_config()
        config.enable_confusable_bucket = True
        result = build_run_inputs(
            config=config,
            iterations=10,
            optimizer_model="openai/gpt-4.1",
            quality_gate_preset="default",
            eval_source="synthetic",
            fitness_profile="balanced",
            enable_confusable_bucket=config.enable_confusable_bucket,
        )
        assert "enable_confusable_bucket" in result
        assert result["enable_confusable_bucket"] is True
