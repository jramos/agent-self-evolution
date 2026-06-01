"""Tests for HermesPromptSource — AST-based read/write/list."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from evolution.prompts.hermes_prompt_source import HermesPromptSource


@pytest.fixture
def fake_hermes_repo(tmp_path: Path) -> Path:
    """A tmp hermes-agent-like checkout with a stub prompt_builder.py."""
    (tmp_path / "agent").mkdir()
    pb = tmp_path / "agent" / "prompt_builder.py"
    pb.write_text(textwrap.dedent('''\
        """Stub prompt_builder for tests."""
        import os

        MEMORY_GUIDANCE = (
            "You have persistent memory across sessions. "
            "Save durable facts."
        )

        SKILLS_GUIDANCE = "After completing a complex task, save the approach."

        PLATFORM_HINTS = {
            "cli": "You are a CLI AI Agent.",
        }

        def _not_a_constant():
            return "ignored"
    '''))
    return tmp_path


def test_read_concatenated_string_constant(fake_hermes_repo: Path):
    source = HermesPromptSource(hermes_repo=fake_hermes_repo)
    text = source.read("MEMORY_GUIDANCE")
    assert "persistent memory" in text
    assert "Save durable facts." in text


def test_read_simple_string_constant(fake_hermes_repo: Path):
    source = HermesPromptSource(hermes_repo=fake_hermes_repo)
    text = source.read("SKILLS_GUIDANCE")
    assert text == "After completing a complex task, save the approach."


def test_read_skips_dict_constants(fake_hermes_repo: Path):
    """PLATFORM_HINTS is a dict; v1 doesn't support dict-shape sections."""
    source = HermesPromptSource(hermes_repo=fake_hermes_repo)
    with pytest.raises(KeyError, match="PLATFORM_HINTS"):
        source.read("PLATFORM_HINTS")


def test_read_unknown_constant_raises(fake_hermes_repo: Path):
    source = HermesPromptSource(hermes_repo=fake_hermes_repo)
    with pytest.raises(KeyError, match="NONEXISTENT"):
        source.read("NONEXISTENT")


def test_missing_prompt_builder_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="prompt_builder.py"):
        HermesPromptSource(hermes_repo=tmp_path)
