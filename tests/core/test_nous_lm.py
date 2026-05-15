"""Tests for the Nous Portal LM subclass.

Covers:
  * Construction wiring (inference_base_url, agent_key)
  * Initial mint when agent_key missing or expiring
  * OAuth refresh when access_token expiring
  * Two-stage refresh-then-mint when both expiring
  * Mint 401 → refresh + retry mint (Hermes pattern)
  * Inference 401 → force re-mint and retry once
  * Cross-instance state sharing (4 workers, 1 mint)
  * Async path (aforward)
  * Error classification: invalid_grant, refresh_token_reused
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import httpx
import litellm
import pytest

from evolution.core.hermes_provider import HermesProviderError
from evolution.core.nous_lm import (
    AGENT_KEY_MIN_TTL_SECONDS,
    NOUS_OAUTH_CLIENT_ID,
    NousLM,
    _format_mint_error,
    _format_oauth_error,
    _reset_state_for_tests,
)
from evolution.core.oauth_helpers import parse_iso_or_epoch


@pytest.fixture(autouse=True)
def _clean_nous_state():
    _reset_state_for_tests()
    yield
    _reset_state_for_tests()


def _mock_response(*, status_code: int = 200, json_body: dict | None = None) -> MagicMock:
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    if json_body is not None:
        mock.json = MagicMock(return_value=json_body)
    else:
        mock.json = MagicMock(side_effect=ValueError("no body"))
    return mock


def _mock_httpx_post(responses: list):
    """Build an httpx.Client mock that returns responses in order across
    calls to .post(). Lets us script multi-step flows (refresh-then-mint,
    mint-401-refresh-retry).
    """
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.side_effect = responses
    return client


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestNousLMConstruction:
    def test_wires_inference_base_url_and_initial_agent_key(self):
        # Pre-supplying a fresh agent_key should NOT trigger initial mint.
        with patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            lm = NousLM(
                model="openai/test-model",
                access_token="oauth-tok",
                refresh_token="refresh-tok",
                oauth_expires_at=time.time() + 86400,  # not expiring
                agent_key="initial-agent-key",
                agent_key_expires_at=time.time() + 1800,  # not expiring
                inference_base_url="https://test-inference/v1",
            )
            assert lm.kwargs["api_base"] == "https://test-inference/v1"
            assert lm.kwargs["api_key"] == "initial-agent-key"
            mock_cls.assert_not_called()

    def test_falls_back_to_default_inference_base(self):
        with patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_post(
                [_mock_response(json_body={"api_key": "minted", "expires_in": 1800})]
            )
            lm = NousLM(
                model="openai/test-model",
                access_token="oauth-tok",
                refresh_token="refresh-tok",
                oauth_expires_at=time.time() + 86400,
            )
            assert "inference-api.nousresearch.com" in lm.kwargs["api_base"]


# ---------------------------------------------------------------------------
# Initial mint behavior
# ---------------------------------------------------------------------------


class TestInitialMint:
    def test_mints_when_agent_key_missing(self):
        with patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_post(
                [_mock_response(json_body={"api_key": "fresh-mint", "expires_in": 1800})]
            )
            lm = NousLM(
                model="openai/test-model",
                access_token="oauth-tok",
                refresh_token="refresh-tok",
                oauth_expires_at=time.time() + 86400,
                agent_key=None,
            )
            assert lm.kwargs["api_key"] == "fresh-mint"
            # Verify the mint POST shape
            client = mock_cls.return_value
            assert client.post.call_count == 1
            call = client.post.call_args
            assert "/api/oauth/agent-key" in call.args[0]
            assert call.kwargs["headers"]["Authorization"] == "Bearer oauth-tok"
            assert call.kwargs["json"]["min_ttl_seconds"] == AGENT_KEY_MIN_TTL_SECONDS

    def test_mints_when_agent_key_within_skew_window(self):
        with patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_post(
                [_mock_response(json_body={"api_key": "fresh-mint", "expires_in": 1800})]
            )
            lm = NousLM(
                model="openai/test-model",
                access_token="oauth-tok",
                refresh_token="refresh-tok",
                oauth_expires_at=time.time() + 86400,
                agent_key="stale-key",
                agent_key_expires_at=time.time() + 60,  # inside 120s skew
            )
            assert lm.kwargs["api_key"] == "fresh-mint"

    def test_skips_mint_when_agent_key_fresh(self):
        with patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            lm = NousLM(
                model="openai/test-model",
                access_token="oauth-tok",
                refresh_token="refresh-tok",
                oauth_expires_at=time.time() + 86400,
                agent_key="fresh-key",
                agent_key_expires_at=time.time() + 1800,  # well outside skew
            )
            assert lm.kwargs["api_key"] == "fresh-key"
            mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Two-stage refresh + mint
# ---------------------------------------------------------------------------


class TestTwoStageRefreshMint:
    def test_oauth_expiring_refreshes_then_mints(self):
        with patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_post(
                [
                    _mock_response(
                        json_body={
                            "access_token": "refreshed-oauth",
                            "expires_in": 86400,
                        }
                    ),
                    _mock_response(
                        json_body={"api_key": "post-refresh-mint", "expires_in": 1800}
                    ),
                ]
            )
            lm = NousLM(
                model="openai/test-model",
                access_token="stale-oauth",
                refresh_token="refresh-tok",
                oauth_expires_at=time.time() + 30,  # within 120s skew
                agent_key=None,  # also needs mint
            )
            client = mock_cls.return_value
            assert client.post.call_count == 2
            # First call: OAuth refresh
            first = client.post.call_args_list[0]
            assert "/api/oauth/token" in first.args[0]
            assert first.kwargs["data"]["grant_type"] == "refresh_token"
            assert first.kwargs["data"]["client_id"] == NOUS_OAUTH_CLIENT_ID
            # Second call: mint with the REFRESHED access_token
            second = client.post.call_args_list[1]
            assert "/api/oauth/agent-key" in second.args[0]
            assert second.kwargs["headers"]["Authorization"] == "Bearer refreshed-oauth"
            assert lm.kwargs["api_key"] == "post-refresh-mint"

    def test_oauth_response_rotated_refresh_token_persisted(self):
        with patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_post(
                [
                    _mock_response(
                        json_body={
                            "access_token": "new-oauth",
                            "refresh_token": "rotated-refresh",
                            "expires_in": 86400,
                        }
                    ),
                    _mock_response(
                        json_body={"api_key": "minted", "expires_in": 1800}
                    ),
                ]
            )
            lm = NousLM(
                model="openai/test-model",
                access_token="stale-oauth",
                refresh_token="original-refresh",
                oauth_expires_at=time.time() + 30,
            )
            assert lm._shared_state.refresh_token == "rotated-refresh"


# ---------------------------------------------------------------------------
# Mint 401 → refresh OAuth + retry mint (Hermes pattern)
# ---------------------------------------------------------------------------


class TestMint401TriggersRefreshRetry:
    def test_mint_401_refreshes_and_retries(self):
        with patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_post(
                [
                    # First mint attempt: 401
                    _mock_response(status_code=401, json_body={"error": "invalid_token"}),
                    # OAuth refresh succeeds
                    _mock_response(
                        json_body={"access_token": "refreshed", "expires_in": 86400}
                    ),
                    # Second mint attempt with refreshed access_token: success
                    _mock_response(json_body={"api_key": "post-retry-mint", "expires_in": 1800}),
                ]
            )
            lm = NousLM(
                model="openai/test-model",
                access_token="stale",
                refresh_token="refresh-tok",
                oauth_expires_at=time.time() + 86400,  # OAuth says "still valid"
            )
            client = mock_cls.return_value
            assert client.post.call_count == 3
            assert lm.kwargs["api_key"] == "post-retry-mint"

    def test_mint_401_retry_also_fails_propagates(self):
        with patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_post(
                [
                    _mock_response(status_code=401, json_body={"error": "invalid_token"}),
                    _mock_response(
                        json_body={"access_token": "refreshed", "expires_in": 86400}
                    ),
                    # Retry mint also 401 — give up.
                    _mock_response(status_code=401, json_body={"error": "invalid_token"}),
                ]
            )
            with pytest.raises(HermesProviderError, match="hermes model"):
                NousLM(
                    model="openai/test-model",
                    access_token="stale",
                    refresh_token="refresh-tok",
                    oauth_expires_at=time.time() + 86400,
                )


# ---------------------------------------------------------------------------
# Inference 401 → force re-mint + retry once
# ---------------------------------------------------------------------------


class TestInferenceForceRemint:
    def _build_lm_with_initial_mint(self):
        with patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_post(
                [_mock_response(json_body={"api_key": "first-mint", "expires_in": 1800})]
            )
            return NousLM(
                model="openai/test-model",
                access_token="oauth-tok",
                refresh_token="refresh-tok",
                oauth_expires_at=time.time() + 86400,
            )

    def test_forward_recovers_from_401_with_remint_and_retry(self):
        lm = self._build_lm_with_initial_mint()
        # Now inference 401s once, then succeeds after re-mint.
        with patch("dspy.LM.forward", autospec=True) as mock_super, \
             patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            mock_super.side_effect = [
                litellm.AuthenticationError(
                    message="401 Unauthorized",
                    llm_provider="openai",
                    model="openai/test-model",
                ),
                "ok",
            ]
            mock_cls.return_value = _mock_httpx_post(
                [_mock_response(json_body={"api_key": "post-401-mint", "expires_in": 1800})]
            )
            result = lm.forward(messages=[{"role": "user", "content": "hi"}])
            assert result == "ok"
            assert mock_super.call_count == 2
            # The cached agent_key was refreshed before the retry.
            assert lm.kwargs["api_key"] == "post-401-mint"

    def test_forward_propagates_second_401(self):
        lm = self._build_lm_with_initial_mint()
        with patch("dspy.LM.forward", autospec=True) as mock_super, \
             patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            err = litellm.AuthenticationError(
                message="401",
                llm_provider="openai",
                model="openai/test-model",
            )
            mock_super.side_effect = [err, err]
            mock_cls.return_value = _mock_httpx_post(
                [_mock_response(json_body={"api_key": "remint", "expires_in": 1800})]
            )
            with pytest.raises(litellm.AuthenticationError):
                lm.forward(messages=[{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Cross-instance state sharing — concurrent mint race
# ---------------------------------------------------------------------------


class TestCrossInstanceSharing:
    def test_concurrent_mint_only_posts_once(self):
        """Four NousLM instances sharing the same refresh_token must
        coordinate so only ONE actually POSTs to /api/oauth/agent-key.
        Without shared state, three would race the portal.
        """
        post_count = {"n": 0}
        post_lock = threading.Lock()

        def slow_post(*args, **kwargs):
            with post_lock:
                post_count["n"] += 1
                time.sleep(0.05)
            return _mock_response(
                json_body={"api_key": "concurrent-mint", "expires_in": 1800}
            )

        with patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            client = MagicMock()
            client.__enter__.return_value = client
            client.post.side_effect = slow_post
            mock_cls.return_value = client

            shared_args = dict(
                model="openai/test-model",
                access_token="oauth-tok",
                refresh_token="shared-refresh",
                oauth_expires_at=time.time() + 86400,
                agent_key="stale-key",
                agent_key_expires_at=time.time() + 60,
            )
            instances = [NousLM(**shared_args) for _ in range(4)]

            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = [pool.submit(lm._ensure_credentials) for lm in instances]
                for f in futs:
                    f.result(timeout=10)

            # Initial construction triggers one mint (skew check positive
            # because agent_key_expires_at < now+120). _ensure_credentials
            # called again on each of 4 threads should observe shared state
            # and NOT mint again. So total POSTs = 1 (the constructor mint).
            assert post_count["n"] == 1
            for lm in instances:
                assert lm.kwargs["api_key"] == "concurrent-mint"


# ---------------------------------------------------------------------------
# Async path
# ---------------------------------------------------------------------------


class TestAsyncPath:
    def test_aforward_invokes_ensure_credentials(self):
        with patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_post(
                [_mock_response(json_body={"api_key": "minted", "expires_in": 1800})]
            )
            lm = NousLM(
                model="openai/test-model",
                access_token="oauth",
                refresh_token="refresh",
                oauth_expires_at=time.time() + 86400,
            )

            with patch.object(NousLM, "_ensure_credentials") as mock_ensure, \
                 patch("dspy.LM.aforward", autospec=True) as mock_super_aforward:
                mock_super_aforward.return_value = asyncio.sleep(0, result="ok")

                asyncio.run(
                    lm.aforward(messages=[{"role": "user", "content": "hi"}])
                )

                mock_ensure.assert_called_once()
                mock_super_aforward.assert_called_once()


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class TestErrorClassification:
    def test_invalid_grant_surfaces_relogin_message(self):
        with patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_post(
                [
                    # OAuth refresh fails with invalid_grant.
                    _mock_response(
                        status_code=400,
                        json_body={"error": "invalid_grant", "error_description": "bad"},
                    ),
                ]
            )
            with pytest.raises(HermesProviderError, match="hermes model"):
                NousLM(
                    model="openai/test-model",
                    access_token="stale",
                    refresh_token="bad-refresh",
                    oauth_expires_at=time.time() + 30,  # forces refresh path
                )

    def test_refresh_token_reused_special_message(self):
        with patch("evolution.core.nous_lm.httpx.Client") as mock_cls:
            mock_cls.return_value = _mock_httpx_post(
                [
                    _mock_response(
                        status_code=400,
                        json_body={
                            "error": {
                                "code": "refresh_token_reused",
                                "message": "Already consumed",
                            }
                        },
                    )
                ]
            )
            with pytest.raises(HermesProviderError) as excinfo:
                NousLM(
                    model="openai/test-model",
                    access_token="stale",
                    refresh_token="reused",
                    oauth_expires_at=time.time() + 30,
                )
            msg = str(excinfo.value)
            assert "another client" in msg
            assert "hermes model" in msg

    def test_format_oauth_error_handles_no_body(self):
        resp = _mock_response(status_code=500, json_body=None)
        msg = _format_oauth_error(resp)
        assert "status 500" in msg

    def test_format_mint_error_extracts_openai_shape(self):
        resp = _mock_response(
            status_code=403,
            json_body={"error": {"code": "rate_limited", "message": "slow down"}},
        )
        msg = _format_mint_error(resp)
        assert "rate_limited" in msg
        assert "slow down" in msg
        assert "hermes model" in msg


# ---------------------------------------------------------------------------
# parse_iso_or_epoch
# ---------------------------------------------------------------------------


class TestParseIsoOrEpoch:
    def test_iso8601_with_offset(self):
        result = parse_iso_or_epoch("2026-05-15T10:30:00+00:00")
        assert result == 1778841000.0

    def test_iso8601_with_z_suffix(self):
        # OpenAI-shaped: trailing Z is shorthand for +00:00
        result = parse_iso_or_epoch("2026-05-15T10:30:00Z")
        assert result == 1778841000.0

    def test_unix_epoch_float(self):
        assert parse_iso_or_epoch(1779179400.0) == 1779179400.0

    def test_unix_epoch_int(self):
        assert parse_iso_or_epoch(1779179400) == 1779179400.0

    def test_numeric_string(self):
        assert parse_iso_or_epoch("1779179400") == 1779179400.0

    def test_none_returns_none(self):
        assert parse_iso_or_epoch(None) is None

    def test_empty_string_returns_none(self):
        assert parse_iso_or_epoch("") is None
        assert parse_iso_or_epoch("   ") is None

    def test_garbage_returns_none(self):
        assert parse_iso_or_epoch("not-a-timestamp") is None
