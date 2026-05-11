"""Tests for evolution.tools.session_mining — Hermes session log mining."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evolution.core.external_importers import HermesSessionImporter
from evolution.tools.session_mining import HermesToolImporter, _extract_tool_name
from evolution.tools.tool_source import ToolManifest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tool_manifests"


@pytest.fixture
def manifest():
    """Manifest containing search_files, grep_in_terminal, read_file,
    cat_in_terminal, list_directory, ls_in_terminal, compute_sha256."""
    return ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")


@pytest.fixture
def session_dir(tmp_path):
    """Patches HermesSessionImporter.SESSION_DIR to tmp_path for the test."""
    with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path):
        yield tmp_path


def _write_session(dir_path: Path, name: str, messages: list[dict], session_id: str = "") -> None:
    payload = {"messages": messages}
    if session_id:
        payload["session_id"] = session_id
    (dir_path / f"{name}.json").write_text(json.dumps(payload))


class TestExtractToolName:
    """The shape-tolerant tool-name extractor."""

    def test_handles_openai_nested_shape(self):
        assert _extract_tool_name({"function": {"name": "search_files"}}) == "search_files"

    def test_handles_flat_shape(self):
        assert _extract_tool_name({"name": "search_files"}) == "search_files"

    def test_prefers_nested_when_both_present(self):
        # The nested form is the canonical OpenAI shape; prefer it.
        result = _extract_tool_name({"function": {"name": "real"}, "name": "stale"})
        assert result == "real"

    def test_returns_none_when_neither_shape_present(self):
        assert _extract_tool_name({"id": "abc"}) is None

    def test_falls_back_when_nested_missing_name(self):
        assert _extract_tool_name({"function": {}, "name": "fallback"}) == "fallback"


class TestHermesToolImporterExtraction:
    """Core extraction logic — tool_calls parsing, manifest filter, drop counters."""

    def test_extracts_single_tool_call(self, manifest, session_dir):
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Find Python test files"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "search_files"}}]},
            {"role": "tool", "content": "results"},
        ], session_id="abc")

        candidates, drops = HermesToolImporter.extract_candidates(manifest)

        assert len(candidates) == 1
        assert candidates[0]["task_input"] == "Find Python test files"
        assert candidates[0]["invoked_tool"] == "search_files"
        assert candidates[0]["session_id"] == "abc"
        assert candidates[0]["source"] == "hermes"

    def test_picks_last_tool_call_in_chain(self, manifest, session_dir):
        # First call is "get oriented"; last is the one that resolved intent.
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Show me the failing test output"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "read_file"}},
                {"function": {"name": "search_files"}},
                {"function": {"name": "grep_in_terminal"}},
            ]},
        ])
        candidates, _ = HermesToolImporter.extract_candidates(manifest)
        assert len(candidates) == 1
        assert candidates[0]["invoked_tool"] == "grep_in_terminal"

    def test_handles_flat_tool_call_shape(self, manifest, session_dir):
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Look for the bug in auth.py"},
            {"role": "assistant", "tool_calls": [{"name": "read_file"}]},
        ])
        candidates, _ = HermesToolImporter.extract_candidates(manifest)
        assert candidates[0]["invoked_tool"] == "read_file"

    def test_drops_non_manifest_invocations(self, manifest, session_dir):
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Do something with terraform"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "tf_apply"}}]},
        ])
        candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert candidates == []
        assert drops["non_manifest"] == 1

    def test_drops_when_no_tool_calls(self, manifest, session_dir):
        # Assistant answered with text only — no selection signal.
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "What does this code do?"},
            {"role": "assistant", "content": "This iterates over the list."},
        ])
        candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert candidates == []
        assert drops["no_tool_calls"] == 1

    def test_drops_short_tasks(self, manifest, session_dir):
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "search_files"}}]},
        ])
        candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert candidates == []
        assert drops["short_task"] == 1

    def test_drops_slash_commands(self, manifest, session_dir):
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "/plugin install foo-bar"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "search_files"}}]},
        ])
        candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert candidates == []
        assert drops["slash_command"] == 1

    def test_drops_tasks_with_secrets(self, manifest, session_dir):
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Use ANTHROPIC_API_KEY=sk-ant-api03-xyz and read main.py"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]},
        ])
        candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert candidates == []
        assert drops["secret"] == 1

    def test_stops_at_next_user_message(self, manifest, session_dir):
        # User 1's tool call would be missed if we kept scanning into User 2's turn.
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Initial question without a tool"},
            {"role": "assistant", "content": "Sure, I'll need more info."},
            {"role": "user", "content": "Please search for foo across the repo"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "search_files"}}]},
        ])
        candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert len(candidates) == 1
        assert candidates[0]["task_input"] == "Please search for foo across the repo"
        # First user message gets the "no_tool_calls" drop.
        assert drops["no_tool_calls"] == 1

    def test_handles_intervening_tool_result(self, manifest, session_dir):
        # tool_result messages between user and assistant shouldn't terminate the scan.
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Read the config file and tell me the port"},
            {"role": "tool", "content": "earlier result"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]},
        ])
        candidates, _ = HermesToolImporter.extract_candidates(manifest)
        assert len(candidates) == 1
        assert candidates[0]["invoked_tool"] == "read_file"

    def test_skips_text_only_assistant_turns_before_tool_call(self, manifest, session_dir):
        # Assistant talks first, then makes a tool call — still attributable to the user.
        _write_session(session_dir, "s1", [
            {"role": "user", "content": "Help me find a config issue"},
            {"role": "assistant", "content": "Let me look."},
            {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]},
        ])
        candidates, _ = HermesToolImporter.extract_candidates(manifest)
        assert len(candidates) == 1
        assert candidates[0]["invoked_tool"] == "read_file"

    def test_respects_limit(self, manifest, session_dir):
        _write_session(session_dir, "s1", [
            msg
            for i in range(5)
            for msg in (
                {"role": "user", "content": f"Read file number {i} please"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]},
            )
        ])
        candidates, _ = HermesToolImporter.extract_candidates(manifest, limit=3)
        assert len(candidates) == 3

    def test_empty_session_dir_returns_empty(self, manifest, session_dir):
        candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert candidates == []
        assert all(v == 0 for v in drops.values())

    def test_handles_missing_session_dir(self, manifest, tmp_path):
        with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path / "nope"):
            candidates, drops = HermesToolImporter.extract_candidates(manifest)
        assert candidates == []
        assert all(v == 0 for v in drops.values())
