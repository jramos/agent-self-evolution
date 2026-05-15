"""Tests for the cheap-model cost advisor.

Covers:
  * Lookup strategy: direct keys + strip-prefix fallback
  * Same-provider enumeration with cost + context-window filters
  * Graceful skip for off-catalog providers (Bedrock, Codex, local)
  * Suggestion model string preserves the original prefix shape
  * Rich panel renders with the right ratio + paste-ready CLI flag
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from evolution.core.cost_advisor import (
    CheaperAlternative,
    find_cheaper_alternative,
    render_suggestion_panel,
)


# A deterministic fake catalog that exercises the same shape variants as
# the real litellm.model_cost dict. Keeps tests independent of LiteLLM
# pricing updates and of any specific provider's current model lineup.
_FAKE_CATALOG = {
    # Anthropic-style: bare keys, no prefix
    "claude-opus-test": {
        "input_cost_per_token": 5e-6,
        "output_cost_per_token": 25e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "anthropic-test",
    },
    "claude-sonnet-test": {
        "input_cost_per_token": 3e-6,
        "output_cost_per_token": 15e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "anthropic-test",
    },
    "claude-haiku-test": {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 5e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "anthropic-test",
    },
    "claude-haiku-tiny-ctx": {
        # Cheaper than opus but smaller context — must be filtered out.
        "input_cost_per_token": 0.5e-6,
        "output_cost_per_token": 2e-6,
        "max_input_tokens": 8_000,
        "litellm_provider": "anthropic-test",
    },
    # OpenAI-style: bare keys, no prefix
    "gpt-test": {
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 10e-6,
        "max_input_tokens": 128_000,
        "litellm_provider": "openai-test",
    },
    "gpt-mini-test": {
        "input_cost_per_token": 0.5e-6,
        "output_cost_per_token": 2e-6,
        "max_input_tokens": 128_000,
        "litellm_provider": "openai-test",
    },
    # OpenRouter-style: prefixed keys preserved verbatim
    "openrouter-test/openai/gpt-test": {
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 10e-6,
        "max_input_tokens": 128_000,
        "litellm_provider": "openrouter-test",
    },
    "openrouter-test/openai/gpt-mini-test": {
        "input_cost_per_token": 0.5e-6,
        "output_cost_per_token": 2e-6,
        "max_input_tokens": 128_000,
        "litellm_provider": "openrouter-test",
    },
    # Lone-cheap-no-cheaper-available: a single model with no siblings.
    "lonely-model": {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 5e-6,
        "max_input_tokens": 32_000,
        "litellm_provider": "lonely-test",
    },
    # Entry with missing pricing — must be filtered out.
    "no-pricing": {
        "input_cost_per_token": None,
        "output_cost_per_token": None,
        "max_input_tokens": 128_000,
        "litellm_provider": "anthropic-test",
    },
}


@pytest.fixture(autouse=True)
def _patch_catalog():
    with patch("evolution.core.cost_advisor.litellm.model_cost", _FAKE_CATALOG):
        yield


# ---------------------------------------------------------------------------
# Lookup strategies
# ---------------------------------------------------------------------------


class TestLookupStrategies:
    def test_strip_prefix_lookup_for_anthropic_style(self):
        # User passed "anthropic-test/claude-opus-test"; catalog has the
        # bare key. The advisor must strip and retry.
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        assert alt.provider == "anthropic-test"

    def test_direct_key_lookup_for_openrouter_style(self):
        # OpenRouter keys are full strings; no prefix-strip needed.
        alt = find_cheaper_alternative("openrouter-test/openai/gpt-test")
        assert alt is not None
        assert alt.provider == "openrouter-test"
        assert alt.suggested_model == "openrouter-test/openai/gpt-mini-test"

    def test_unknown_model_returns_none(self):
        # Bedrock / Codex / local-server endpoints land here.
        assert find_cheaper_alternative("bedrock/us.anthropic.claude-test") is None
        assert find_cheaper_alternative("openai/gpt-5-codex-fake-doesnt-exist") is None


# ---------------------------------------------------------------------------
# Suggestion logic
# ---------------------------------------------------------------------------


class TestSuggestionLogic:
    def test_suggests_cheapest_same_provider_with_sufficient_context(self):
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        # Cheapest qualifying candidate is haiku-test, NOT haiku-tiny-ctx
        # (which has only 8k context vs the opus's 200k requirement).
        assert alt.suggested_model == "anthropic-test/claude-haiku-test"
        assert alt.input_cost_ratio == pytest.approx(5.0)
        assert alt.output_cost_ratio == pytest.approx(5.0)
        assert alt.suggested_max_input_tokens == 200_000

    def test_filters_out_smaller_context_window_candidates(self):
        # Even though claude-haiku-tiny-ctx is the cheapest in the catalog,
        # its 8k context disqualifies it for an opus user with 200k.
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        assert alt.suggested_model != "anthropic-test/claude-haiku-tiny-ctx"

    def test_no_cheaper_alternative_returns_none(self):
        # claude-haiku-test is already the cheapest qualifying anthropic-test
        # entry. No suggestion to make.
        alt = find_cheaper_alternative("anthropic-test/claude-haiku-test")
        assert alt is None

    def test_provider_with_only_one_model_returns_none(self):
        alt = find_cheaper_alternative("lonely-model")
        assert alt is None

    def test_skips_catalog_entries_with_missing_pricing(self):
        # If the cheapest candidate has None pricing, the advisor must
        # skip it rather than crash on multiplication.
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        assert alt.suggested_model != "anthropic-test/no-pricing"

    def test_does_not_suggest_cross_provider_swaps(self):
        # gpt-mini-test is cheaper than claude-opus-test but on a different
        # provider. The advisor must not cross provider boundaries.
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        assert "gpt" not in alt.suggested_model

    def test_suggested_model_string_paste_ready_for_eval_model_flag(self):
        # The suggestion has to flow through `--eval-model <X>` directly,
        # which means the same prefix shape as the original.
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        # User passed prefixed; suggestion must be prefixed.
        assert alt.suggested_model.startswith("anthropic-test/")

    def test_cost_per_1m_uses_per_token_times_million(self):
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        # 5e-6 per token × 1M = $5/M. Watch for the implicit float math.
        assert alt.current_input_cost_per_1m == pytest.approx(5.0)
        assert alt.suggested_input_cost_per_1m == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------


class TestPanel:
    def test_panel_includes_cli_flag_and_ratio(self):
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        panel = render_suggestion_panel("eval", alt)

        # Render to a plain string for substring checks.
        from rich.console import Console
        from io import StringIO

        buf = StringIO()
        Console(file=buf, width=120, color_system=None).print(panel)
        out = buf.getvalue()

        assert "--eval-model anthropic-test/claude-haiku-test" in out
        assert "5.0× cheaper" in out
        assert "$5.00/M input" in out
        assert "$1.00/M input" in out
        # Tradeoff caveat must surface so users don't blindly downgrade
        # the optimizer/reflection roles too.
        assert "reasoning quality" in out
