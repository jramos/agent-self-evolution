"""Mine Hermes session logs for tool-selection evaluation data.

The tool-path analog of ``evolution.core.external_importers``. Walks
Hermes session JSON via the shared ``iter_hermes_sessions`` helper,
extracts ``(user_task, invoked_tool)`` tuples from
``role: user → role: assistant.tool_calls`` sequences, and (in a follow-up
commit) re-judges each tuple against the *current* manifest with a
confidence-banded LLM judge.

Claude Code (``~/.claude/history.jsonl``) and Copilot
(``~/.copilot/session-state/*/events.jsonl``) are not mined: their logs
carry only user/assistant text, no ``tool_use`` blocks.

Tests monkeypatch the session directory via
``evolution.core.external_importers.HermesSessionImporter.SESSION_DIR``.
"""

from __future__ import annotations

import logging
from typing import Optional

from evolution.core.external_importers import contains_secret, iter_hermes_sessions
from evolution.tools.tool_source import ToolManifest

logger = logging.getLogger(__name__)


# Mirrors the skill-path threshold in `HermesSessionImporter.extract_messages`.
MIN_TASK_LENGTH = 10


def _extract_tool_name(tool_call: dict) -> Optional[str]:
    """Pull a tool name from a tool_call object across the two shapes
    Hermes sessions emit in practice: OpenAI-style nested
    ``{"function": {"name": ...}}`` and the flat ``{"name": ...}``.
    """
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if name:
            return name
    return tool_call.get("name")


class HermesToolImporter:
    """Extract ``(task, invoked_tool)`` candidates from Hermes session logs.

    Stateless. Returns ``(candidates, drop_counts)`` so the orchestrator
    can surface the drop breakdown in ``gate_decision.json``.

    Per-user-message rule: scan forward to the next ``role: assistant``
    message *before* the next ``role: user``; if that assistant emitted
    ``tool_calls``, take the **last** call's function name. The last call
    is more often the one that resolved the user's intent — earlier
    calls in a chain tend to be "get oriented" reads.
    """

    @staticmethod
    def extract_candidates(
        manifest: ToolManifest,
        limit: int = 0,
    ) -> tuple[list[dict], dict[str, int]]:
        """Walk all Hermes sessions and return candidate (task, tool) pairs.

        Args:
            manifest: Current manifest. Used to filter out invocations of
                tools that aren't being evolved (and don't exist in the
                manifest under evolution).
            limit: Cap on emitted candidates. ``0`` means no cap.

        Returns:
            ``(candidates, drops)`` where ``candidates`` is a list of
            ``{source, task_input, invoked_tool, session_id}`` dicts and
            ``drops`` is a per-reason counter. The orchestrator passes
            both into the next pipeline stage.
        """
        manifest_names = {t.name for t in manifest.tools}
        candidates: list[dict] = []
        drops = {
            "short_task": 0,
            "slash_command": 0,
            "secret": 0,
            "no_tool_calls": 0,
            "non_manifest": 0,
        }

        for session_id, msg_list in iter_hermes_sessions():
            for i, msg in enumerate(msg_list):
                if msg.get("role") != "user":
                    continue

                user_text = msg.get("content", "") or ""
                if len(user_text) < MIN_TASK_LENGTH:
                    drops["short_task"] += 1
                    continue
                if user_text.lstrip().startswith("/"):
                    drops["slash_command"] += 1
                    continue
                if contains_secret(user_text):
                    drops["secret"] += 1
                    continue

                invoked = _find_invoked_tool(msg_list, start=i + 1)
                if invoked is None:
                    drops["no_tool_calls"] += 1
                    continue
                if invoked not in manifest_names:
                    drops["non_manifest"] += 1
                    continue

                candidates.append({
                    "source": "hermes",
                    "task_input": user_text,
                    "invoked_tool": invoked,
                    "session_id": session_id,
                })

                if limit and len(candidates) >= limit:
                    return candidates, drops

        return candidates, drops


def _find_invoked_tool(messages: list[dict], start: int) -> Optional[str]:
    """Scan forward for the next assistant tool_call, stopping at the
    next user message. Returns the last tool_call's name on that
    assistant turn, or None if no tool call appeared before the next
    user message.
    """
    for j in range(start, len(messages)):
        role = messages[j].get("role")
        if role == "user":
            return None
        if role != "assistant":
            continue
        tool_calls = messages[j].get("tool_calls") or []
        if not tool_calls:
            continue
        last = tool_calls[-1]
        if isinstance(last, dict):
            name = _extract_tool_name(last)
            if name:
                return name
    return None
