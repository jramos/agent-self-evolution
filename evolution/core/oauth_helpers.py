"""Shared OAuth helpers used by Codex and Nous LM wrappers.

Kept as a small standalone module so the next OAuth provider that needs
in-memory refresh has somewhere obvious to drop shared utilities without
bloating either provider's LM file.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional


def parse_iso_or_epoch(value: Any) -> Optional[float]:
    """Coerce an expires_at value into Unix epoch seconds.

    Different OAuth providers serialize token expiry in different shapes:

      * Nous Portal stores ISO 8601 strings ("2026-05-15T10:30:00+00:00")
        in ``~/.hermes/auth.json``.
      * Codex stores Unix epoch floats (or decodes from a JWT ``exp`` claim).
      * Older or hand-edited entries may omit it entirely.

    Returns the equivalent Unix epoch float, or None when the value is
    missing, malformed, or has no parseable shape. Callers treat None as
    "unknown" — typically meaning "trigger a refresh" defensively.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # ISO 8601 — Python's fromisoformat handles "+00:00" but not the
        # bare "Z" suffix common in OpenAI-shaped responses.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s).timestamp()
        except ValueError:
            pass
        # Numeric-looking string ("1747299600") — treat as epoch seconds.
        try:
            return float(s)
        except ValueError:
            return None
    return None
