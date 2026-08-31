"""Tests for external session importers.

Tests cover:
  - Secret detection and filtering (true positives + false positives)
  - Skill relevance heuristics
  - Claude Code history parsing + edge cases
  - Copilot events.jsonl parsing + edge cases
  - LLM scoring JSON parser
  - RelevanceFilter with mocked DSPy
  - build_dataset_from_external orchestration
  - Input validation and normalization
  - _load_skill_text skill loader
  - CLI entry point via CliRunner
  - EvalExample serialization roundtrip
"""

import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from evolution.core.external_importers import (
    contains_secret,
    _is_relevant_to_skill,
    parse_scoring_json,
    _parse_copilot_events,
    _read_copilot_workspace,
    _load_skill_text,
    _validate_eval_example,
    build_dataset_from_external,
    iter_hermes_sessions,
    main,
    ClaudeCodeImporter,
    CopilotImporter,
    HermesSessionImporter,
    RelevanceFilter,
    VALID_DIFFICULTIES,
    MIN_DATASET_SIZE,
)
from evolution.core.dataset_builder import EvalExample




class TestSecretDetection:
    """Verify that known secret formats are caught and normal text is not."""

    def test_detects_anthropic_key(self):
        assert contains_secret("here is sk-ant-api03-abc123def456 my key")

    def test_detects_openrouter_key(self):
        assert contains_secret("sk-or-v1-abcdef1234567890abcdef")

    def test_detects_github_pat(self):
        assert contains_secret("use ghp_abcdefghijklmnopqrstuvwxyz12")

    def test_detects_github_user_token(self):
        assert contains_secret("ghu_abcdef123456")

    def test_detects_slack_bot_token(self):
        assert contains_secret("SLACK_TOKEN is xoxb-123-456-abcdef")

    def test_detects_slack_app_token(self):
        assert contains_secret("xapp-1-ABC-123456-xyz")

    def test_detects_notion_token(self):
        assert contains_secret("ntn_37257299567bdiGRuYDIjhNH8uFribb461")

    def test_detects_bearer_token(self):
        assert contains_secret("Authorization: Bearer dsm_bYxe4QUvPsDjRThUu2Qb3z")

    def test_detects_env_var_anthropic(self):
        assert contains_secret("export ANTHROPIC_API_KEY=something")

    def test_detects_env_var_openai(self):
        assert contains_secret("OPENAI_API_KEY=sk-blah")

    def test_detects_long_sk_prefix(self):
        assert contains_secret("sk-proj-1234567890abcdefghij")

    def test_ignores_normal_text(self):
        assert not contains_secret("sort these messages by topic")

    def test_ignores_short_sk_prefix(self):
        # "sk" alone or "sk-foo" (short) should not trigger
        assert not contains_secret("I asked about sk")

    def test_ignores_prose_with_key_word(self):
        assert not contains_secret("the key insight is that we need to refactor")

    def test_ignores_word_bearer_in_prose(self):
        # "Bearer" followed by short token shouldn't match (< 20 chars)
        assert not contains_secret("the bearer of bad news")

    def test_ignores_code_token_discussion(self):
        # Discussing tokens in general shouldn't trigger
        assert not contains_secret("parse the JWT token from the header")

    def test_ignores_ask_substring(self):
        # "ask" contains "sk" — should not trigger
        assert not contains_secret("ask the user for their preferences")

    def test_detects_password_assignment(self):
        assert contains_secret("password=my_super_secret_123")
        assert contains_secret("password: hunter2abc")

    def test_detects_secret_assignment(self):
        assert contains_secret("secret=abc123def456")
        assert contains_secret("secret: my_api_secret_key")

    def test_detects_token_assignment_long_value(self):
        assert contains_secret("token=abcdef1234567890")
        assert contains_secret("token: eyJhbGciOiJIUzI1NiJ9")

    def test_ignores_token_assignment_short_value(self):
        # "token=abc" has < 10 chars after = so should NOT trigger
        assert not contains_secret("token=abc")

    def test_ignores_password_in_prose(self):
        # "password" without assignment operator shouldn't trigger
        assert not contains_secret("the password field should be validated")
        assert not contains_secret("reset your password using the link")

    def test_ignores_secret_in_prose(self):
        assert not contains_secret("the secret to success is consistency")
        assert not contains_secret("it's no secret that we need to refactor")

    def test_detects_openrouter_env_var(self):
        assert contains_secret("export OPENROUTER_API_KEY=sk-or-xyz")

    def test_detects_slack_bot_token_env_var(self):
        assert contains_secret("SLACK_BOT_TOKEN=xoxb-abc")

    def test_detects_github_token_env_var(self):
        assert contains_secret("GITHUB_TOKEN=ghp_abc123")

    def test_detects_aws_access_key(self):
        assert contains_secret("AKIAIOSFODNN7EXAMPLE")

    def test_detects_pem_private_key(self):
        assert contains_secret("-----BEGIN RSA PRIVATE KEY-----\nMIIEow...")
        assert contains_secret("-----BEGIN PRIVATE KEY-----\nMIIEvQ...")

    def test_detects_aws_secret_env_var(self):
        assert contains_secret("export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG")

    def test_detects_database_url_env_var(self):
        assert contains_secret("DATABASE_URL=postgres://user:pass@host/db")




class TestRelevanceHeuristics:
    """Verify cheap pre-filter catches obvious matches and rejects non-matches."""

    SKILL_TEXT = (
        "Sort any batch of text (messages, notes, emails, transcripts) "
        "into topics. Works with natural language. Detects topics automatically "
        "if none provided. Supports categorization of content by theme."
    )

    def test_matches_skill_name_keyword(self):
        assert _is_relevant_to_skill(
            "categorize these messages", "tim-categorize", self.SKILL_TEXT
        )

    def test_matches_skill_content_keywords(self):
        assert _is_relevant_to_skill(
            "sort these transcripts into topics for me",
            "tim-categorize",
            self.SKILL_TEXT,
        )

    def test_rejects_unrelated(self):
        assert not _is_relevant_to_skill(
            "deploy the app to production",
            "tim-categorize",
            self.SKILL_TEXT,
        )

    def test_short_skill_name_words_ignored(self):
        # Words <= 3 chars from skill name shouldn't trigger false matches
        assert not _is_relevant_to_skill(
            "run the test suite", "tim-tdd", "Test driven development enforcement"
        )

    def test_single_keyword_not_enough(self):
        # Requires >= 2 keyword matches from skill text
        assert not _is_relevant_to_skill(
            "send an email to the team",
            "tim-categorize",
            self.SKILL_TEXT,
        )




class TestScoringJsonParser:
    """Verify _parse_scoring_json handles various LLM output formats."""

    def test_clean_json(self):
        result = parse_scoring_json('{"relevant": true, "difficulty": "easy"}')
        assert result["relevant"] is True

    def test_json_in_markdown(self):
        text = 'Here is my assessment:\n```json\n{"relevant": false}\n```'
        result = parse_scoring_json(text)
        assert result is not None
        assert result["relevant"] is False

    def test_json_with_surrounding_text(self):
        text = 'I think this is relevant. {"relevant": true, "category": "sorting"} That is my answer.'
        result = parse_scoring_json(text)
        assert result["category"] == "sorting"

    def test_no_json_returns_none(self):
        assert parse_scoring_json("This is just plain text with no JSON") is None

    def test_malformed_json_returns_none(self):
        assert parse_scoring_json("{broken json: ???}") is None

    def test_direct_parse_fast_path(self):
        """Clean JSON should be parsed directly without regex fallback."""
        result = parse_scoring_json('{"relevant": true, "nested": {"key": "val"}}')
        assert result is not None
        assert result["relevant"] is True
        # Nested objects work via direct parse but would fail regex
        assert result["nested"]["key"] == "val"

    def test_non_dict_json_returns_none(self):
        """A JSON array or string should return None (we need a dict)."""
        assert parse_scoring_json('[1, 2, 3]') is None
        assert parse_scoring_json('"just a string"') is None

    def test_nested_braces_in_values(self):
        """JSON with braces inside string values must parse correctly."""
        text = 'Here: {"relevant": true, "expected_behavior": "handle {edge} cases"}'
        result = parse_scoring_json(text)
        assert result is not None
        assert result["relevant"] is True
        assert "{edge}" in result["expected_behavior"]

    def test_empty_string_returns_none(self):
        assert parse_scoring_json("") is None




class TestClaudeCodeImporter:
    def test_parses_history_jsonl(self, tmp_path):
        history = tmp_path / "history.jsonl"
        history.write_text(
            json.dumps({"display": "sort my slack messages by topic", "timestamp": 1700000000000, "project": "/test", "sessionId": "abc"}) + "\n"
            + json.dumps({"display": "yes go", "timestamp": 1700000001000, "project": "/test", "sessionId": "abc"}) + "\n"
            + json.dumps({"display": "here is sk-ant-api03-SECRETKEY123456789012345678 the key", "timestamp": 1700000002000, "project": "/test", "sessionId": "abc"}) + "\n"
        )

        with patch.object(ClaudeCodeImporter, "HISTORY_PATH", history):
            messages = ClaudeCodeImporter.extract_messages()

        # Second message is too short, third contains a secret.
        assert len(messages) == 1
        assert messages[0]["task_input"] == "sort my slack messages by topic"
        assert messages[0]["source"] == "claude-code"
        assert messages[0]["project"] == "/test"

    def test_handles_missing_file(self, tmp_path):
        with patch.object(ClaudeCodeImporter, "HISTORY_PATH", tmp_path / "nonexistent.jsonl"):
            messages = ClaudeCodeImporter.extract_messages()
        assert messages == []

    def test_respects_limit(self, tmp_path):
        history = tmp_path / "history.jsonl"
        lines = [
            json.dumps({"display": f"message number {i} with enough length to pass", "timestamp": i, "project": "/test", "sessionId": "s"})
            for i in range(100)
        ]
        history.write_text("\n".join(lines) + "\n")

        with patch.object(ClaudeCodeImporter, "HISTORY_PATH", history):
            messages = ClaudeCodeImporter.extract_messages(limit=5)

        assert len(messages) == 5

    def test_skips_malformed_json_lines(self, tmp_path):
        history = tmp_path / "history.jsonl"
        history.write_text(
            "this is not json\n"
            + json.dumps({"display": "valid message with sufficient length", "timestamp": 1, "project": "/test", "sessionId": "s"}) + "\n"
            + "{broken\n"
        )

        with patch.object(ClaudeCodeImporter, "HISTORY_PATH", history):
            messages = ClaudeCodeImporter.extract_messages()

        assert len(messages) == 1

    def test_skips_empty_lines(self, tmp_path):
        history = tmp_path / "history.jsonl"
        history.write_text(
            "\n\n"
            + json.dumps({"display": "valid message with enough length", "timestamp": 1, "project": "/test", "sessionId": "s"}) + "\n"
            + "\n"
        )

        with patch.object(ClaudeCodeImporter, "HISTORY_PATH", history):
            messages = ClaudeCodeImporter.extract_messages()

        assert len(messages) == 1




class TestCopilotImporter:
    def test_parses_events_jsonl(self, tmp_path):
        session_dir = tmp_path / "session-state" / "test-session-1"
        session_dir.mkdir(parents=True)
        (session_dir / "workspace.yaml").write_text("id: test-session-1\ncwd: /Users/test/project\n")

        events = [
            {"type": "session.start", "data": {"sessionId": "test-session-1"}},
            {"type": "user.message", "data": {"content": "sort these emails into categories for the team"}},
            {"type": "assistant.message", "data": {"content": "I'll categorize your emails into the following topics..."}},
            {"type": "user.message", "data": {"content": "now do the second batch"}},
            {"type": "assistant.message", "data": {"content": "Here are the categories for batch 2..."}},
        ]
        (session_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )

        with patch.object(CopilotImporter, "SESSION_DIR", tmp_path / "session-state"):
            messages = CopilotImporter.extract_messages()

        assert len(messages) == 2
        assert messages[0]["task_input"] == "sort these emails into categories for the team"
        assert messages[0]["assistant_response"] == "I'll categorize your emails into the following topics..."
        assert messages[0]["source"] == "copilot"
        assert messages[0]["project"] == "/Users/test/project"

    def test_filters_secrets_from_copilot(self, tmp_path):
        session_dir = tmp_path / "session-state" / "test-session-2"
        session_dir.mkdir(parents=True)
        (session_dir / "workspace.yaml").write_text("id: test-2\ncwd: /test\n")

        events = [
            {"type": "user.message", "data": {"content": "here is my key sk-ant-api03-SECRET123456789012345678901234"}},
            {"type": "assistant.message", "data": {"content": "I see your API key"}},
        ]
        (session_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )

        with patch.object(CopilotImporter, "SESSION_DIR", tmp_path / "session-state"):
            messages = CopilotImporter.extract_messages()

        assert len(messages) == 0

    def test_handles_missing_dir(self, tmp_path):
        with patch.object(CopilotImporter, "SESSION_DIR", tmp_path / "nonexistent"):
            messages = CopilotImporter.extract_messages()
        assert messages == []

    def test_unpaired_user_message_dropped(self, tmp_path):
        """A user message with no following assistant response is dropped."""
        session_dir = tmp_path / "session-state" / "s1"
        session_dir.mkdir(parents=True)
        (session_dir / "workspace.yaml").write_text("cwd: /test\n")

        events = [
            {"type": "user.message", "data": {"content": "hello this is a long enough message"}},
        ]
        (session_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )

        with patch.object(CopilotImporter, "SESSION_DIR", tmp_path / "session-state"):
            messages = CopilotImporter.extract_messages()

        assert len(messages) == 0

    def test_multiline_assistant_response(self, tmp_path):
        """Multiple assistant.message events concatenate into one response."""
        session_dir = tmp_path / "session-state" / "s1"
        session_dir.mkdir(parents=True)
        (session_dir / "workspace.yaml").write_text("cwd: /test\n")

        events = [
            {"type": "user.message", "data": {"content": "explain this code in detail please"}},
            {"type": "assistant.message", "data": {"content": "First, the function validates input."}},
            {"type": "assistant.message", "data": {"content": "Then it processes the data in chunks."}},
        ]
        (session_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )

        with patch.object(CopilotImporter, "SESSION_DIR", tmp_path / "session-state"):
            messages = CopilotImporter.extract_messages()

        assert len(messages) == 1
        assert "\n" in messages[0]["assistant_response"]
        assert "First" in messages[0]["assistant_response"]
        assert "chunks" in messages[0]["assistant_response"]

    def test_empty_events_file(self, tmp_path):
        session_dir = tmp_path / "session-state" / "s1"
        session_dir.mkdir(parents=True)
        (session_dir / "workspace.yaml").write_text("cwd: /test\n")
        (session_dir / "events.jsonl").write_text("")

        with patch.object(CopilotImporter, "SESSION_DIR", tmp_path / "session-state"):
            messages = CopilotImporter.extract_messages()

        assert messages == []


class TestCopilotHelpers:
    def test_read_workspace_with_cwd(self, tmp_path):
        ws = tmp_path / "workspace.yaml"
        ws.write_text("id: s1\ncwd: /Users/dev/myproject\nother: stuff\n")
        assert _read_copilot_workspace(ws) == "/Users/dev/myproject"

    def test_read_workspace_missing_file(self, tmp_path):
        assert _read_copilot_workspace(tmp_path / "nope.yaml") == ""

    def test_read_workspace_no_cwd(self, tmp_path):
        ws = tmp_path / "workspace.yaml"
        ws.write_text("id: s1\nother: stuff\n")
        assert _read_copilot_workspace(ws) == ""

    def test_parse_copilot_events_malformed_json(self, tmp_path):
        events_path = tmp_path / "events.jsonl"
        events_path.write_text(
            "not json\n"
            + json.dumps({"type": "user.message", "data": {"content": "hello this is a message"}}) + "\n"
            + json.dumps({"type": "assistant.message", "data": {"content": "hi there"}}) + "\n"
        )
        pairs = _parse_copilot_events(events_path, "s1", "/test")
        assert len(pairs) == 1

    def test_parse_copilot_events_file_not_found(self, tmp_path):
        """Outer try-catch: file doesn't exist -> returns empty list, doesn't crash."""
        pairs = _parse_copilot_events(tmp_path / "nope.jsonl", "s1", "/test")
        assert pairs == []

    def test_parse_copilot_events_permission_error(self, tmp_path):
        """Outer try-catch: unreadable file -> returns empty list, doesn't crash."""
        events_path = tmp_path / "events.jsonl"
        events_path.write_text("data")
        events_path.chmod(0o000)
        try:
            pairs = _parse_copilot_events(events_path, "s1", "/test")
            assert pairs == []
        finally:
            events_path.chmod(0o644)  # Restore for cleanup




class TestHermesSessionImporter:
    def test_parses_session_json(self, tmp_path):
        session = {
            "session_id": "test-session",
            "messages": [
                {"role": "user", "content": "Fix the bug in auth.py"},
                {"role": "assistant", "content": None, "tool_calls": [{"name": "read_file"}]},
                {"role": "tool", "content": "file contents here"},
                {"role": "assistant", "content": "I found the issue and fixed it."},
                {"role": "user", "content": "Now run the tests"},
                {"role": "assistant", "content": "All 42 tests passed."},
            ],
        }
        (tmp_path / "session_001.json").write_text(json.dumps(session))

        with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path):
            msgs = HermesSessionImporter.extract_messages()

        assert len(msgs) == 2
        assert msgs[0]["task_input"] == "Fix the bug in auth.py"
        assert msgs[0]["assistant_response"] == "I found the issue and fixed it."
        assert msgs[0]["source"] == "hermes"
        assert msgs[1]["task_input"] == "Now run the tests"
        assert msgs[1]["assistant_response"] == "All 42 tests passed."

    def test_skips_short_messages(self, tmp_path):
        session = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        }
        (tmp_path / "s.json").write_text(json.dumps(session))

        with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path):
            msgs = HermesSessionImporter.extract_messages()
        assert len(msgs) == 0

    def test_filters_secrets(self, tmp_path):
        session = {
            "messages": [
                {"role": "user", "content": "Set ANTHROPIC_API_KEY=sk-ant-api03-xyz in the env"},
                {"role": "assistant", "content": "Done."},
            ],
        }
        (tmp_path / "s.json").write_text(json.dumps(session))

        with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path):
            msgs = HermesSessionImporter.extract_messages()
        assert len(msgs) == 0

    def test_handles_missing_dir(self, tmp_path):
        with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path / "nonexistent"):
            msgs = HermesSessionImporter.extract_messages()
        assert msgs == []

    def test_handles_malformed_json(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not valid json")

        with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path):
            msgs = HermesSessionImporter.extract_messages()
        assert msgs == []

    def test_handles_no_assistant_response(self, tmp_path):
        session = {
            "messages": [
                {"role": "user", "content": "Do something interesting please"},
            ],
        }
        (tmp_path / "s.json").write_text(json.dumps(session))

        with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path):
            msgs = HermesSessionImporter.extract_messages()
        assert len(msgs) == 1
        assert msgs[0]["assistant_response"] == ""

    def test_respects_limit(self, tmp_path):
        session = {
            "messages": [
                {"role": "user", "content": f"Message number {i} with enough text"} for i in range(10)
            ],
        }
        (tmp_path / "s.json").write_text(json.dumps(session))

        with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path):
            msgs = HermesSessionImporter.extract_messages(limit=3)
        assert len(msgs) == 3


class TestIterHermesSessions:
    """The shared session iterator that both skill-path and tool-path miners use."""

    def test_yields_session_id_and_messages(self, tmp_path):
        session = {
            "session_id": "abc-123",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        }
        (tmp_path / "s.json").write_text(json.dumps(session))

        with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path):
            results = list(iter_hermes_sessions())

        assert results == [("abc-123", session["messages"])]

    def test_skips_malformed_json(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not valid")
        with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path):
            assert list(iter_hermes_sessions()) == []

    def test_skips_sessions_without_messages(self, tmp_path):
        (tmp_path / "empty.json").write_text(json.dumps({"session_id": "x"}))
        with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path):
            assert list(iter_hermes_sessions()) == []

    def test_falls_back_to_filename_stem_when_session_id_missing(self, tmp_path):
        (tmp_path / "session_42.json").write_text(json.dumps({
            "messages": [{"role": "user", "content": "Anything"}],
        }))
        with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path):
            results = list(iter_hermes_sessions())
        assert results[0][0] == "session_42"

    def test_returns_empty_when_dir_missing(self, tmp_path):
        with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path / "nope"):
            assert list(iter_hermes_sessions()) == []

    def test_orders_newest_first_by_mtime(self, tmp_path):
        import os
        older = tmp_path / "old.json"
        newer = tmp_path / "new.json"
        older.write_text(json.dumps({
            "session_id": "old",
            "messages": [{"role": "user", "content": "older message"}],
        }))
        newer.write_text(json.dumps({
            "session_id": "new",
            "messages": [{"role": "user", "content": "newer message"}],
        }))
        # Backdate `older` so mtime ordering is unambiguous regardless of write order.
        os.utime(older, (1000, 1000))
        os.utime(newer, (2000, 2000))

        with patch.object(HermesSessionImporter, "SESSION_DIR", tmp_path):
            results = list(iter_hermes_sessions())

        assert [sid for sid, _ in results] == ["new", "old"]


# --- state.db sourcing (triage #102) -------------------------------------------------

def _make_hermes_state_db(path, sessions):
    """Write a minimal hermes-shaped ``state.db``.

    ``sessions``: list of ``(session_id, source, started_at, messages)``. Each message
    is ``{"role", "content"?, "tool_calls"?, "tool_calls_raw"?}`` — ``tool_calls`` is
    JSON-serialized; ``tool_calls_raw`` is inserted verbatim (to exercise malformed text).
    """
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, model TEXT, started_at REAL);"
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, "
        "role TEXT, content TEXT, tool_calls TEXT);"
    )
    for session_id, source, started_at, messages in sessions:
        conn.execute(
            "INSERT INTO sessions (id, source, model, started_at) VALUES (?, ?, ?, ?)",
            (session_id, source, "m", started_at),
        )
        for m in messages:
            if "tool_calls_raw" in m:
                tc = m["tool_calls_raw"]
            elif m.get("tool_calls") is not None:
                tc = json.dumps(m["tool_calls"])
            else:
                tc = None
            conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls) VALUES (?, ?, ?, ?)",
                (session_id, m["role"], m.get("content"), tc),
            )
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def _isolate_hermes_state_db(tmp_path, monkeypatch):
    """``iter_hermes_sessions`` now reads ``~/.hermes/state.db`` first; a real db on the
    dev/CI box would hijack the JSON-fixture tests. Default ``STATE_DB`` to an absent
    path — the state.db tests below override it explicitly."""
    monkeypatch.setattr(HermesSessionImporter, "STATE_DB", tmp_path / "no_state.db")


class TestIterHermesSessionsFromDb:
    """state.db is the canonical store; iter_hermes_sessions reads it first."""

    def test_pairs_user_assistant_from_db(self, tmp_path, monkeypatch):
        db = tmp_path / "state.db"
        _make_hermes_state_db(db, [
            ("s1", "cli", 1.0, [
                {"role": "user", "content": "Refactor the auth module please"},
                {"role": "assistant", "content": "Done — extracted a helper."},
            ]),
        ])
        monkeypatch.setattr(HermesSessionImporter, "STATE_DB", db)
        msgs = HermesSessionImporter.extract_messages()
        assert len(msgs) == 1
        assert msgs[0]["source"] == "hermes"
        assert msgs[0]["task_input"] == "Refactor the auth module please"
        assert msgs[0]["assistant_response"] == "Done — extracted a helper."
        assert msgs[0]["session_id"] == "s1"

    def test_mines_all_sessions_source_agnostic(self, tmp_path, monkeypatch):
        db = tmp_path / "state.db"
        _make_hermes_state_db(db, [
            ("s1", "cli", 1.0, [
                {"role": "user", "content": "First task with enough length"},
                {"role": "assistant", "content": "First answer."},
            ]),
            ("s2", "cli", 2.0, [
                {"role": "user", "content": "Second task with enough length"},
                {"role": "assistant", "content": "Second answer."},
            ]),
        ])
        monkeypatch.setattr(HermesSessionImporter, "STATE_DB", db)
        tasks = {m["task_input"] for m in HermesSessionImporter.extract_messages()}
        assert tasks == {"First task with enough length", "Second task with enough length"}

    def test_strips_model_switch_note_prefix(self, tmp_path, monkeypatch):
        note = ("[Note: model was just switched from openai/gpt-5.4 to claude-opus-4-7 "
                "via Anthropic. Adjust your self-identification accordingly.]")
        db = tmp_path / "state.db"
        _make_hermes_state_db(db, [
            ("s1", "cli", 1.0, [
                {"role": "user", "content": f"{note}\n\nPlease summarize the changelog"},
                {"role": "assistant", "content": "Here is the summary."},
            ]),
        ])
        monkeypatch.setattr(HermesSessionImporter, "STATE_DB", db)
        msgs = HermesSessionImporter.extract_messages()
        assert len(msgs) == 1
        # Note stripped, genuine instruction kept (NOT dropped).
        assert msgs[0]["task_input"] == "Please summarize the changelog"

    def test_note_only_message_is_dropped(self, tmp_path, monkeypatch):
        note = ("[Note: model was just switched from a to b via c. "
                "Adjust your self-identification accordingly.]")
        db = tmp_path / "state.db"
        _make_hermes_state_db(db, [
            ("s1", "cli", 1.0, [
                {"role": "user", "content": note},  # nothing real after stripping
                {"role": "assistant", "content": "ok"},
            ]),
        ])
        monkeypatch.setattr(HermesSessionImporter, "STATE_DB", db)
        assert HermesSessionImporter.extract_messages() == []

    def test_tool_calls_decoded_as_list(self, tmp_path, monkeypatch):
        db = tmp_path / "state.db"
        _make_hermes_state_db(db, [
            ("s1", "cli", 1.0, [
                {"role": "user", "content": "Find the config file for me"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"function": {"name": "search_files"}}]},
            ]),
        ])
        monkeypatch.setattr(HermesSessionImporter, "STATE_DB", db)
        _, msg_list = list(iter_hermes_sessions())[0]
        tc = [m["tool_calls"] for m in msg_list if m["role"] == "assistant"][0]
        assert isinstance(tc, list) and tc[0]["function"]["name"] == "search_files"

    def test_malformed_tool_calls_become_none(self, tmp_path, monkeypatch):
        db = tmp_path / "state.db"
        _make_hermes_state_db(db, [
            ("s1", "cli", 1.0, [
                {"role": "user", "content": "Do the thing with enough length"},
                {"role": "assistant", "content": "x", "tool_calls_raw": "{not json"},
            ]),
        ])
        monkeypatch.setattr(HermesSessionImporter, "STATE_DB", db)
        _, msg_list = list(iter_hermes_sessions())[0]
        assert [m["tool_calls"] for m in msg_list if m["role"] == "assistant"] == [None]

    def test_falls_back_to_json_when_db_empty(self, tmp_path, monkeypatch):
        db = tmp_path / "state.db"
        _make_hermes_state_db(db, [])  # tables exist, no rows
        monkeypatch.setattr(HermesSessionImporter, "STATE_DB", db)
        json_dir = tmp_path / "sessions"
        json_dir.mkdir()
        (json_dir / "s.json").write_text(json.dumps({
            "session_id": "from-json",
            "messages": [
                {"role": "user", "content": "JSON fallback task with length"},
                {"role": "assistant", "content": "from json"},
            ],
        }))
        monkeypatch.setattr(HermesSessionImporter, "SESSION_DIR", json_dir)
        msgs = HermesSessionImporter.extract_messages()
        assert len(msgs) == 1 and msgs[0]["session_id"] == "from-json"


class TestSkillNameMatching:
    """Verify that short skill names match via exact full-name check."""

    def test_short_skill_name_matches_exact(self):
        assert _is_relevant_to_skill(
            "configure the mcp server settings",
            "mcp",
            "Model Context Protocol server management",
        )

    def test_short_skill_name_tdd(self):
        assert _is_relevant_to_skill(
            "set up tdd workflow for the project",
            "tdd",
            "Test driven development enforcement",
        )

    def test_hyphenated_skill_name_matches(self):
        assert _is_relevant_to_skill(
            "I need to do a code review on this PR",
            "code-review",
            "Review pull requests and provide feedback",
        )




class TestRelevanceFilter:
    """Test RelevanceFilter with mocked LLM calls."""

    @pytest.fixture
    def mock_dspy(self):
        """Mock dspy.LM and dspy.context to avoid real LLM calls."""
        with patch("evolution.core.external_importers.dspy") as mock:
            mock.context.return_value.__enter__ = MagicMock(return_value=None)
            mock.context.return_value.__exit__ = MagicMock(return_value=False)
            yield mock

    def test_relevant_messages_become_examples(self, mock_dspy):
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock()
        rf.scorer.return_value = SimpleNamespace(
            scoring='{"relevant": true, "expected_behavior": "group by topic", "difficulty": "easy", "category": "sorting"}'
        )

        messages = [
            {"task_input": "sort these messages by topic", "source": "claude-code"},
            {"task_input": "categorize my emails please", "source": "copilot", "assistant_response": "Sure!"},
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics. Categorize content.", max_examples=10)

        assert len(examples) == 2
        inputs = {ex.task_input for ex in examples}
        assert "sort these messages by topic" in inputs
        assert "categorize my emails please" in inputs
        for ex in examples:
            assert ex.expected_behavior == "group by topic"
            assert ex.difficulty == "easy"

    def test_irrelevant_messages_filtered_out(self, mock_dspy):
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock()
        rf.scorer.return_value = SimpleNamespace(
            scoring='{"relevant": false}'
        )

        messages = [
            {"task_input": "deploy the app to production", "source": "claude-code"},
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics.", max_examples=10)
        assert len(examples) == 0

    def test_malformed_llm_output_counted_as_error(self, mock_dspy):
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock()
        rf.scorer.return_value = SimpleNamespace(scoring="I cannot determine relevance right now")

        messages = [
            {"task_input": "sort these messages by topic please", "source": "claude-code"},
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics.", max_examples=10)
        assert len(examples) == 0

    def test_max_examples_cap_respected(self, mock_dspy):
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock()
        rf.scorer.return_value = SimpleNamespace(
            scoring='{"relevant": true, "expected_behavior": "test", "difficulty": "easy", "category": "test"}'
        )

        messages = [
            {"task_input": f"categorize message number {i} into topics", "source": "claude-code"}
            for i in range(20)
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics. Categorize content.", max_examples=3)
        assert len(examples) == 3

    def test_scorer_exception_counted_as_error(self, mock_dspy):
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock(side_effect=RuntimeError("API timeout"))

        messages = [
            {"task_input": "sort these messages by topic please", "source": "claude-code"},
        ]

        # Should not raise — errors are caught and counted
        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics.", max_examples=10)
        assert len(examples) == 0




class TestBuildDataset:
    """Test the main orchestration function."""

    def test_builds_dataset_with_splits(self, tmp_path):
        """Verify end-to-end: import -> filter -> split -> save."""
        mock_messages = [
            {"task_input": f"categorize batch {i} into topics", "source": "claude-code"}
            for i in range(10)
        ]

        mock_examples = [
            EvalExample(
                task_input=f"categorize batch {i} into topics",
                expected_behavior="group by topic",
                difficulty="easy",
                category="sorting",
                source="claude-code",
            )
            for i in range(10)
        ]

        output = tmp_path / "output"

        with patch.object(ClaudeCodeImporter, "extract_messages", return_value=mock_messages), \
             patch.object(RelevanceFilter, "filter_and_score", return_value=mock_examples):
            dataset = build_dataset_from_external(
                skill_name="categorize",
                skill_text="Sort text into topics.",
                sources=["claude-code"],
                output_path=output,
                model="test-model",
                max_examples=10,
            )

        assert len(dataset.train) > 0
        assert len(dataset.val) > 0
        assert len(dataset.all_examples) == 10

        assert (output / "train.jsonl").exists()
        assert (output / "val.jsonl").exists()
        assert (output / "holdout.jsonl").exists()

    def test_uses_configured_ratios_not_hardcoded_50_25(self, tmp_path):
        """Sessiondb path was hardcoded 50/25/25; must now match the synthetic
        path's normalized 0.5/0.40/0.50 → 36/29/36 of N=60. Locking this
        prevents the regression where someone tunes EvolutionConfig defaults
        and only the synthetic path picks it up.
        """
        from evolution.core.config import EvolutionConfig
        from evolution.core.dataset_builder import split_examples

        n = 60
        mock_messages = [
            {"task_input": f"categorize batch {i} into topics", "source": "claude-code"}
            for i in range(n)
        ]
        mock_examples = [
            EvalExample(
                task_input=f"categorize batch {i} into topics",
                expected_behavior="group by topic",
                difficulty="easy",
                category="sorting",
                source="claude-code",
            )
            for i in range(n)
        ]

        with patch.object(ClaudeCodeImporter, "extract_messages", return_value=mock_messages), \
             patch.object(RelevanceFilter, "filter_and_score", return_value=mock_examples):
            sessiondb_ds = build_dataset_from_external(
                skill_name="categorize",
                skill_text="Sort text into topics.",
                sources=["claude-code"],
                output_path=tmp_path / "out",
                model="test-model",
                max_examples=n,
                seed=42,
            )

        # Compute the expected split via the same helper the function uses.
        cfg = EvolutionConfig()
        expected = split_examples(
            mock_examples, seed=42,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            holdout_ratio=cfg.holdout_ratio,
        )

        assert len(sessiondb_ds.train) == len(expected.train)
        assert len(sessiondb_ds.val) == len(expected.val)
        assert len(sessiondb_ds.holdout) == len(expected.holdout)
        # And confirm the new sizes are NOT the old hardcoded 50/25/25.
        assert (len(sessiondb_ds.train), len(sessiondb_ds.val)) != (30, 15)

    def test_no_messages_returns_empty_dataset(self, tmp_path):
        with patch.object(ClaudeCodeImporter, "extract_messages", return_value=[]):
            dataset = build_dataset_from_external(
                skill_name="categorize",
                skill_text="Sort text into topics.",
                sources=["claude-code"],
                output_path=tmp_path / "out",
                model="test-model",
            )

        assert len(dataset.all_examples) == 0

    def test_no_relevant_examples_returns_empty_dataset(self, tmp_path):
        mock_messages = [{"task_input": "deploy the app", "source": "claude-code"}]

        with patch.object(ClaudeCodeImporter, "extract_messages", return_value=mock_messages), \
             patch.object(RelevanceFilter, "filter_and_score", return_value=[]):
            dataset = build_dataset_from_external(
                skill_name="categorize",
                skill_text="Sort text into topics.",
                sources=["claude-code"],
                output_path=tmp_path / "out",
                model="test-model",
            )

        assert len(dataset.all_examples) == 0

    def test_multiple_sources(self, tmp_path):
        cc_msgs = [{"task_input": "sort from claude code session", "source": "claude-code"}]
        cp_msgs = [{"task_input": "sort from copilot session", "source": "copilot", "assistant_response": "ok"}]

        all_examples = [
            EvalExample(task_input="sort from claude code session", expected_behavior="test", source="claude-code"),
            EvalExample(task_input="sort from copilot session", expected_behavior="test", source="copilot"),
        ]

        with patch.object(ClaudeCodeImporter, "extract_messages", return_value=cc_msgs), \
             patch.object(CopilotImporter, "extract_messages", return_value=cp_msgs), \
             patch.object(RelevanceFilter, "filter_and_score", return_value=all_examples):
            dataset = build_dataset_from_external(
                skill_name="categorize",
                skill_text="Sort text.",
                sources=["claude-code", "copilot"],
                output_path=tmp_path / "out",
                model="test-model",
            )

        sources = {ex.source for ex in dataset.all_examples}
        assert "claude-code" in sources
        assert "copilot" in sources

    def test_unknown_source_ignored(self, tmp_path):
        """An unrecognized source name is silently skipped."""
        with patch.object(ClaudeCodeImporter, "extract_messages", return_value=[]):
            dataset = build_dataset_from_external(
                skill_name="test",
                skill_text="Test.",
                sources=["claude-code", "nonexistent-tool"],
                output_path=tmp_path / "out",
                model="test-model",
            )

        assert len(dataset.all_examples) == 0




class TestEndToEndRoundtrip:
    """Verify the full pipeline: fake files -> import -> filter -> save -> reload.

    This is the most important test. It proves that output from
    build_dataset_from_external is loadable by GoldenDatasetLoader,
    which is the actual consumer in evolve_skill.py.
    """

    def test_output_loadable_by_golden_loader(self, tmp_path):
        """Write fake Claude Code history, run pipeline, load with GoldenDatasetLoader."""
        from evolution.core.dataset_builder import GoldenDatasetLoader

        # Create fake Claude Code history
        history = tmp_path / "history.jsonl"
        lines = [
            json.dumps({"display": f"categorize these {i} messages into topics", "timestamp": i, "project": "/test", "sessionId": "s1"})
            for i in range(20)
        ]
        history.write_text("\n".join(lines) + "\n")

        mock_examples = [
            EvalExample(
                task_input=f"categorize these {i} messages into topics",
                expected_behavior="group by theme",
                difficulty="easy",
                category="sorting",
                source="claude-code",
            )
            for i in range(8)
        ]

        output = tmp_path / "dataset"

        with patch.object(ClaudeCodeImporter, "HISTORY_PATH", history), \
             patch.object(RelevanceFilter, "filter_and_score", return_value=mock_examples):
            dataset = build_dataset_from_external(
                skill_name="categorize",
                skill_text="Sort text into topics.",
                sources=["claude-code"],
                output_path=output,
                model="test-model",
            )

        assert (output / "train.jsonl").exists()
        assert (output / "val.jsonl").exists()
        assert (output / "holdout.jsonl").exists()

        reloaded = GoldenDatasetLoader.load(output)
        assert len(reloaded.all_examples) == len(dataset.all_examples)

        for ex in reloaded.all_examples:
            assert ex.task_input.startswith("categorize these")
            assert ex.expected_behavior == "group by theme"
            assert ex.source == "claude-code"

    def test_full_pipeline_with_copilot_events(self, tmp_path):
        """Create real Copilot events, import, filter (mocked), save, reload."""
        from evolution.core.dataset_builder import EvalDataset

        session_dir = tmp_path / "session-state" / "test-session"
        session_dir.mkdir(parents=True)
        (session_dir / "workspace.yaml").write_text("cwd: /Users/test/project\n")

        events = [
            {"type": "user.message", "data": {"content": "sort these messages into categories for me"}},
            {"type": "assistant.message", "data": {"content": "I grouped them into 3 categories"}},
        ]
        (session_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )

        mock_examples = [
            EvalExample(
                task_input="sort these messages into categories for me",
                expected_behavior="group into categories",
                difficulty="easy",
                category="sorting",
                source="copilot",
            ),
        ]

        output = tmp_path / "dataset"

        with patch.object(CopilotImporter, "SESSION_DIR", tmp_path / "session-state"), \
             patch.object(RelevanceFilter, "filter_and_score", return_value=mock_examples):
            dataset = build_dataset_from_external(
                skill_name="categorize",
                skill_text="Sort text.",
                sources=["copilot"],
                output_path=output,
                model="test-model",
            )

        assert len(dataset.all_examples) == 1

        reloaded = EvalDataset.load(output)
        assert len(reloaded.all_examples) == 1
        assert reloaded.all_examples[0].task_input == "sort these messages into categories for me"




class TestLoadSkillText:
    def test_loads_skill_from_directory(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nDo stuff.")

        name, text = _load_skill_text("my-skill", skills_dir=tmp_path)
        assert name == "my-skill"
        assert "Do stuff." in text

    def test_loads_skill_from_subdirectory(self, tmp_path):
        sub = tmp_path / "custom" / "my-skill"
        sub.mkdir(parents=True)
        (sub / "SKILL.md").write_text("Nested skill content.")

        name, text = _load_skill_text("my-skill", skills_dir=tmp_path)
        assert name == "my-skill"
        assert "Nested skill content." in text

    def test_raises_on_missing_skill(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            _load_skill_text("nonexistent-skill", skills_dir=tmp_path)

    def test_ignores_dir_without_skill_md(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "README.md").write_text("Not a SKILL.md file")

        with pytest.raises(FileNotFoundError):
            _load_skill_text("my-skill", skills_dir=tmp_path)




class TestValidateEvalExample:
    """Verify _validate_eval_example normalizes, rejects, and caps fields."""

    def test_valid_input_passes_through(self):
        result = _validate_eval_example(
            task_input="sort these messages by topic",
            expected_behavior="group by theme",
            difficulty="easy",
            category="sorting",
        )
        assert result is not None
        assert result["task_input"] == "sort these messages by topic"
        assert result["expected_behavior"] == "group by theme"
        assert result["difficulty"] == "easy"
        assert result["category"] == "sorting"

    def test_empty_task_input_returns_none(self):
        assert _validate_eval_example("", "behavior", "easy", "cat") is None

    def test_whitespace_task_input_returns_none(self):
        assert _validate_eval_example("   ", "behavior", "easy", "cat") is None

    def test_none_task_input_returns_none(self):
        assert _validate_eval_example(None, "behavior", "easy", "cat") is None

    def test_empty_expected_behavior_returns_none(self):
        assert _validate_eval_example("task", "", "easy", "cat") is None

    def test_whitespace_expected_behavior_returns_none(self):
        assert _validate_eval_example("task", "  \n  ", "easy", "cat") is None

    def test_none_expected_behavior_returns_none(self):
        assert _validate_eval_example("task", None, "easy", "cat") is None

    def test_invalid_difficulty_defaults_to_medium(self):
        result = _validate_eval_example("task", "behavior", "impossible", "cat")
        assert result["difficulty"] == "medium"

    def test_empty_difficulty_defaults_to_medium(self):
        result = _validate_eval_example("task", "behavior", "", "cat")
        assert result["difficulty"] == "medium"

    def test_none_difficulty_defaults_to_medium(self):
        result = _validate_eval_example("task", "behavior", None, "cat")
        assert result["difficulty"] == "medium"

    def test_difficulty_case_insensitive(self):
        result = _validate_eval_example("task", "behavior", "HARD", "cat")
        assert result["difficulty"] == "hard"

    def test_difficulty_stripped(self):
        result = _validate_eval_example("task", "behavior", "  easy  ", "cat")
        assert result["difficulty"] == "easy"

    def test_all_valid_difficulties_accepted(self):
        for diff in VALID_DIFFICULTIES:
            result = _validate_eval_example("task", "behavior", diff, "cat")
            assert result["difficulty"] == diff

    def test_empty_category_defaults_to_general(self):
        result = _validate_eval_example("task", "behavior", "easy", "")
        assert result["category"] == "general"

    def test_none_category_defaults_to_general(self):
        result = _validate_eval_example("task", "behavior", "easy", None)
        assert result["category"] == "general"

    def test_whitespace_category_defaults_to_general(self):
        result = _validate_eval_example("task", "behavior", "easy", "   ")
        assert result["category"] == "general"

    def test_task_input_capped_at_2000_chars(self):
        long_input = "x" * 5000
        result = _validate_eval_example(long_input, "behavior", "easy", "cat")
        assert result is not None
        assert len(result["task_input"]) == 2000

    def test_expected_behavior_stripped(self):
        result = _validate_eval_example("task", "  behavior with spaces  ", "easy", "cat")
        assert result["expected_behavior"] == "behavior with spaces"

    def test_category_stripped(self):
        result = _validate_eval_example("task", "behavior", "easy", "  sorting  ")
        assert result["category"] == "sorting"


class TestValidationIntegration:
    """Verify validation is wired correctly into RelevanceFilter."""

    @pytest.fixture
    def mock_dspy(self):
        with patch("evolution.core.external_importers.dspy") as mock:
            mock.context.return_value.__enter__ = MagicMock(return_value=None)
            mock.context.return_value.__exit__ = MagicMock(return_value=False)
            yield mock

    def test_empty_expected_behavior_drops_example(self, mock_dspy):
        """LLM returns relevant=True but empty expected_behavior -> example dropped."""
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock()
        rf.scorer.return_value = SimpleNamespace(
            scoring='{"relevant": true, "expected_behavior": "", "difficulty": "easy", "category": "sorting"}'
        )

        messages = [
            {"task_input": "sort these messages by topic", "source": "claude-code"},
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics. Categorize content.", max_examples=10)
        assert len(examples) == 0

    def test_invalid_difficulty_normalized(self, mock_dspy):
        """LLM returns invalid difficulty -> normalized to medium."""
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock()
        rf.scorer.return_value = SimpleNamespace(
            scoring='{"relevant": true, "expected_behavior": "group", "difficulty": "ultra-hard", "category": "sorting"}'
        )

        messages = [
            {"task_input": "sort these messages by topic", "source": "claude-code"},
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics. Categorize content.", max_examples=10)
        assert len(examples) == 1
        assert examples[0].difficulty == "medium"

    def test_missing_source_field_drops_message(self, mock_dspy):
        """Messages missing 'source' key are dropped before scoring."""
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock()
        rf.scorer.return_value = SimpleNamespace(
            scoring='{"relevant": true, "expected_behavior": "test", "difficulty": "easy", "category": "test"}'
        )

        messages = [
            {"task_input": "sort these messages by topic"},  # missing source
            {"task_input": "categorize emails", "source": "claude-code"},
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics. Categorize content.", max_examples=10)
        assert len(examples) == 1
        assert examples[0].source == "claude-code"

    def test_missing_task_input_drops_message(self, mock_dspy):
        """Messages missing 'task_input' are dropped before scoring."""
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock()

        messages = [
            {"source": "claude-code"},  # missing task_input
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics.", max_examples=10)
        assert len(examples) == 0
        # scorer should never be called for invalid messages
        rf.scorer.assert_not_called()


class TestMinDatasetSizeWarning:
    """Verify MIN_DATASET_SIZE warning in build_dataset_from_external."""

    def test_small_dataset_still_returned(self, tmp_path):
        """Even with < MIN_DATASET_SIZE examples, a dataset is still returned."""
        mock_messages = [{"task_input": "sort stuff", "source": "claude-code"}]
        mock_examples = [
            EvalExample(
                task_input="sort stuff",
                expected_behavior="group by topic",
                difficulty="easy",
                category="sorting",
                source="claude-code",
            ),
        ]

        output = tmp_path / "output"

        with patch.object(ClaudeCodeImporter, "extract_messages", return_value=mock_messages), \
             patch.object(RelevanceFilter, "filter_and_score", return_value=mock_examples):
            dataset = build_dataset_from_external(
                skill_name="categorize",
                skill_text="Sort text.",
                sources=["claude-code"],
                output_path=output,
                model="test-model",
            )

        # Still returns the dataset even though size < MIN_DATASET_SIZE
        assert len(dataset.all_examples) == 1
        assert MIN_DATASET_SIZE > 1  # Confirm the constant is meaningful




class TestCLI:
    """Test the Click CLI entry point using CliRunner."""

    def test_dry_run(self, tmp_path):
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: test-skill\n---\nTest skill.")

        with patch.object(ClaudeCodeImporter, "extract_messages", return_value=[
            {"task_input": "hello with enough length", "source": "claude-code"},
        ]):
            runner = CliRunner()
            result = runner.invoke(main, [
                "--skill", "test-skill",
                "--source", "claude-code",
                "--dry-run",
            ], catch_exceptions=False, env={"HOME": str(tmp_path.parent)})
        with patch("evolution.core.external_importers._load_skill_text", return_value=("test-skill", "Test skill.")), \
             patch.object(ClaudeCodeImporter, "extract_messages", return_value=[
                 {"task_input": "hello with enough length", "source": "claude-code"},
             ]):
            runner = CliRunner()
            result = runner.invoke(main, [
                "--skill", "test-skill",
                "--source", "claude-code",
                "--dry-run",
            ], catch_exceptions=False)

        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_missing_skill_exits_with_error(self):
        with patch("evolution.core.external_importers._load_skill_text", side_effect=FileNotFoundError("not found")):
            runner = CliRunner()
            result = runner.invoke(main, ["--skill", "nonexistent"])

        assert result.exit_code != 0

    def test_help_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Import external session data" in result.output




class TestEvalExampleFormat:
    def test_roundtrip_serialization(self):
        ex = EvalExample(
            task_input="sort messages by topic",
            expected_behavior="group by theme, list topic names",
            difficulty="medium",
            category="categorize",
            source="copilot",
        )
        d = ex.to_dict()
        restored = EvalExample.from_dict(d)
        assert restored.task_input == ex.task_input
        assert restored.expected_behavior == ex.expected_behavior
        assert restored.source == "copilot"

    def test_golden_jsonl_format(self, tmp_path):
        """Verify output matches GoldenDatasetLoader expected format."""
        ex = EvalExample(
            task_input="categorize these notes",
            expected_behavior="detect topics, group by theme",
            difficulty="easy",
            category="sorting",
            source="claude-code",
        )
        jsonl_path = tmp_path / "golden.jsonl"
        jsonl_path.write_text(json.dumps(ex.to_dict()) + "\n")

        with open(jsonl_path) as f:
            data = json.loads(f.readline())

        assert set(data.keys()) == {"task_input", "expected_behavior", "difficulty", "category", "source"}


class TestRelevanceRanking:
    """Candidate ordering must put the strongest matches ahead of the caps.

    Two caps exist between the heuristic pre-filter and the produced eval set:
    the ``max_examples * 3`` candidate truncation and the scoring loop's
    ``len(examples) >= max_examples`` break. Both consume candidates in list
    order, so ordering decides which messages ever reach the LLM scorer.
    """

    SKILL_NAME = "categorize"
    SKILL_TEXT = "Sort text into topics. Categorize content by theme."

    # Qualifies only via keyword overlap (>= 2 of topics/categorize/content/theme),
    # and deliberately avoids the skill name so it cannot reach a higher tier.
    WEAK = "arrange the content by theme please"
    # Qualifies via the strongest signal: the full skill name as a substring.
    STRONG = "categorize these messages for me"

    @pytest.fixture
    def mock_dspy(self):
        with patch("evolution.core.external_importers.dspy") as mock:
            mock.context.return_value.__enter__ = MagicMock(return_value=None)
            mock.context.return_value.__exit__ = MagicMock(return_value=False)
            yield mock

    def _filter(self):
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"
        rf.scorer = MagicMock()
        rf.scorer.return_value = SimpleNamespace(
            scoring='{"relevant": true, "expected_behavior": "b", "difficulty": "easy", "category": "c"}'
        )
        return rf

    @staticmethod
    def _msg(text):
        return {"task_input": text, "source": "claude-code"}

    def test_strongest_survives_the_candidate_cap(self, mock_dspy):
        """A tier-A match past ``max_examples * 3`` must not be truncated away."""
        rf = self._filter()
        # 12 weak qualifiers overflow the cap of 9; the strong match is last.
        messages = [self._msg(f"{self.WEAK} {i}") for i in range(12)]
        messages.append(self._msg(self.STRONG))

        examples = rf.filter_and_score(
            messages, self.SKILL_NAME, self.SKILL_TEXT, max_examples=3
        )

        assert self.STRONG in {ex.task_input for ex in examples}

    def test_strongest_survives_the_scoring_break(self, mock_dspy):
        """Ordering also decides the output when the cap never engages.

        Six qualifiers sit under the cap of 9, so truncation is not involved --
        the scoring loop's early break alone excludes everything after the
        first ``max_examples`` candidates.
        """
        rf = self._filter()
        messages = [self._msg(f"{self.WEAK} {i}") for i in range(5)]
        messages.append(self._msg(self.STRONG))

        examples = rf.filter_and_score(
            messages, self.SKILL_NAME, self.SKILL_TEXT, max_examples=3
        )

        assert self.STRONG in {ex.task_input for ex in examples}

    def test_scoring_order_is_strongest_first(self, mock_dspy):
        """The scorer sees candidates strongest-first, backfill last."""
        rf = self._filter()
        messages = [
            self._msg(self.WEAK),
            self._msg("deploy the app to production"),  # non-matching -> backfill
            self._msg(self.STRONG),
        ]

        rf.filter_and_score(messages, self.SKILL_NAME, self.SKILL_TEXT, max_examples=10)

        scored = [c.kwargs["user_message"] for c in rf.scorer.call_args_list]
        assert scored.index(self.STRONG) < scored.index(self.WEAK), (
            "tier-A match must be scored before a keyword-overlap-only match"
        )
        assert scored.index(self.WEAK) < scored.index("deploy the app to production"), (
            "every qualifier must be scored before any backfilled message"
        )

    def test_equal_scores_keep_import_order(self, mock_dspy):
        """Ties must resolve to import order, and do so reproducibly.

        Every message here scores identically, so the sort is pure tie-breaking.
        Comparing two runs only proves the implementation is *deterministic* --
        an unstable-but-deterministic order would pass that. Pinning the
        expected order is what actually verifies the stable sort the caller
        relies on to keep source-priority ordering intact among equals.
        """
        texts = [f"categorize batch {i} of messages" for i in range(6)]
        messages = [self._msg(text) for text in texts]

        orders = []
        for _ in range(2):
            rf = self._filter()
            rf.filter_and_score(
                messages, self.SKILL_NAME, self.SKILL_TEXT, max_examples=10
            )
            orders.append([c.kwargs["user_message"] for c in rf.scorer.call_args_list])

        assert orders[0] == texts, "tied messages must stay in import order"
        assert orders[0] == orders[1]

    def test_score_preserves_the_qualifying_set(self):
        """``_relevance_score`` must qualify exactly what the predicate did.

        The score exists to *order* candidates, never to widen or narrow which
        ones qualify. This oracle is a verbatim copy of the boolean predicate;
        any disagreement means the change altered eval-set membership.
        """
        import random as _random
        import re as _re

        from evolution.core.external_importers import _relevance_score

        def oracle(text, skill_name, skill_text):
            text_lower = text.lower()
            skill_lower = skill_name.lower().replace("-", " ").replace("_", " ")
            if skill_lower in text_lower:
                return True
            for word in skill_lower.split():
                if len(word) > 3 and word in text_lower:
                    return True
            skill_keywords = set()
            for word in skill_text[:500].lower().split():
                word = _re.sub(r"[^a-z]", "", word)
                if len(word) > 4:
                    skill_keywords.add(word)
            message_words = set(_re.sub(r"[^a-z\s]", "", text_lower).split())
            return len(message_words & skill_keywords) >= 2

        cases = [
            # Edges that have bitten this predicate before.
            ("", "categorize", self.SKILL_TEXT),
            ("anything at all", "", self.SKILL_TEXT),          # "" is in every string
            ("anything at all", "   ", self.SKILL_TEXT),
            ("run the test suite", "tim-tdd", "Test driven development"),  # <=3-char words
            ("send an email to the team", "categorize", self.SKILL_TEXT),  # overlap == 1
            ("CATEGORIZE THIS", "categorize", self.SKILL_TEXT),            # case
            ("sort content by theme", "categorize", ""),                   # empty skill text
        ]
        # Vocabulary deliberately spans the character classes the scorer's
        # normalisation touches: case folding, the two punctuation strippers, and
        # non-ASCII. A fuzz over lowercase ASCII alone leaves .lower() and both
        # re.sub calls as no-ops on every case, so it cannot exercise them.
        vocab = [
            "categorize", "Categorize", "CATEGORIZE", "content", "content,",
            "theme.", "topics!", "sort", "deploy", "email", "test", "suite",
            "messages", "tim", "tdd", "the", "a", "topic5", "42", "co-ntent",
            "th_eme", "réview", "naïve", "日本語", "  ", "\ttabbed", "categorize's",
        ]
        # Varying skill_text matters: it drives the keyword set, the 500-char
        # truncation boundary, and the len(word) > 4 filter.
        long_text = ("alpha bravo charlie delta echo foxtrot golf hotel india " * 12)
        skill_texts = [
            self.SKILL_TEXT,
            "",
            "Sort.",
            "TOPICS CONTENT THEME",
            "content théme tópics naïve",
            long_text,                      # straddles the 500-char slice
            long_text[:498] + " zulu",      # keyword split across the boundary
        ]
        rng = _random.Random(1234)
        for _ in range(2000):
            text = " ".join(rng.choices(vocab, k=rng.randint(0, 8)))
            name = rng.choice([
                "categorize", "tim-categorize", "tim_tdd", "sort", "x",
                "Fix-Python-Bugs", "", "   ", "sort content by theme", "日本語-skill",
            ])
            cases.append((text, name, rng.choice(skill_texts)))

        for text, name, skill_text in cases:
            assert bool(any(_relevance_score(text, name, skill_text))) == oracle(
                text, name, skill_text
            ), f"qualifying set changed for {text!r} / {name!r}"

    def test_single_keyword_overlap_scores_zero(self):
        """Overlap of exactly 1 must contribute nothing.

        The predicate requires >= 2 overlapping keywords. Awarding partial
        credit for a single overlap would silently widen the qualifying set,
        which is the one way this change could alter behavior.
        """
        from evolution.core.external_importers import _relevance_score

        score = _relevance_score(
            "send an email to the team", self.SKILL_NAME, self.SKILL_TEXT
        )
        assert not any(score), f"expected an all-zero score, got {score}"

    # A multi-word skill name is required to separate the middle tier from the
    # outer two: with a single-word name, "full name matched" and "one name word
    # matched" fire on identical conditions and no ordering test can tell the
    # tiers apart.
    MULTI_NAME = "fix-python-bugs"
    MULTI_TEXT = (
        "Repair broken python modules, report failing behaviour, analyse stack "
        "traces, inspect regression suites, summarise coverage metrics, and "
        "validate exception handling."
    )
    MULTI_TIER_A = "please fix python bugs in this module"
    MULTI_TIER_B = "rewrite it in python"
    MULTI_TIER_C = (
        "repair broken modules report failing behaviour analyse traces inspect "
        "regression suites summarise coverage metrics"
    )

    def test_name_word_tier_outranks_a_larger_keyword_overlap(self, mock_dspy):
        """The middle tier must beat the bottom tier regardless of magnitude.

        This is the test that distinguishes a lexicographic tuple key from any
        weighted sum. The bottom-tier message here overlaps on ~14 keywords
        while the middle-tier message matches a single name word, so a scheme
        like ``name_words * 10 + keyword_overlap`` ranks them backwards while a
        tuple ranks by tier first. Without this case, dropping the middle tier
        or swapping it with the keyword tier would pass every other test.
        """
        rf = self._filter()
        messages = [
            self._msg(self.MULTI_TIER_C),
            self._msg(self.MULTI_TIER_B),
            self._msg(self.MULTI_TIER_A),
        ]

        rf.filter_and_score(messages, self.MULTI_NAME, self.MULTI_TEXT, max_examples=10)

        scored = [c.kwargs["user_message"] for c in rf.scorer.call_args_list]
        assert scored.index(self.MULTI_TIER_A) < scored.index(self.MULTI_TIER_B), (
            "a full-name match must outrank a single name-word match"
        )
        assert scored.index(self.MULTI_TIER_B) < scored.index(self.MULTI_TIER_C), (
            "a name-word match must outrank keyword overlap, however large"
        )

    def test_tiers_are_distinguishable_for_a_multi_word_name(self):
        """Guards the premise of the ordering test above.

        If these three messages ever collapse to the same tier, the ordering
        test would still pass while proving nothing.
        """
        from evolution.core.external_importers import _relevance_score

        a = _relevance_score(self.MULTI_TIER_A, self.MULTI_NAME, self.MULTI_TEXT)
        b = _relevance_score(self.MULTI_TIER_B, self.MULTI_NAME, self.MULTI_TEXT)
        c = _relevance_score(self.MULTI_TIER_C, self.MULTI_NAME, self.MULTI_TEXT)

        assert a[0] == 1 and b[0] == 0 and c[0] == 0, (a, b, c)
        assert b[1] > 0 and c[1] == 0, (b, c)
        assert c[2] > b[1], f"bottom tier must be numerically larger to be a real test: {c} vs {b}"

    def test_ties_beyond_the_cap_keep_import_order(self, mock_dspy):
        """When every candidate ties, truncation must take the earliest.

        The stable sort is what preserves source-priority ordering among equal
        scores; asserting it here means the claim is tested rather than only
        stated in a comment.
        """
        rf = self._filter()
        texts = [f"categorize group {i}" for i in range(12)]
        messages = [self._msg(text) for text in texts]

        rf.filter_and_score(messages, self.SKILL_NAME, self.SKILL_TEXT, max_examples=3)

        scored = [c.kwargs["user_message"] for c in rf.scorer.call_args_list]
        assert scored == texts[:len(scored)]

    def test_backfill_only_corpus_still_produces_candidates(self, mock_dspy):
        """Nothing qualifying is the one path where the sort sees an empty list.

        The seeded backfill must still pad the candidate list so the LLM gets a
        usable sample rather than the run silently yielding nothing.
        """
        rf = self._filter()
        messages = [self._msg(f"deploy release {i} to production") for i in range(4)]

        examples = rf.filter_and_score(
            messages, self.SKILL_NAME, self.SKILL_TEXT, max_examples=2
        )

        assert rf.scorer.call_args_list, "backfill should still be scored"
        assert examples
