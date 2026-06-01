"""Tests for the PromptSource protocol contract."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from evolution.prompts.prompt_source import PromptSource, SectionDescriptor


def test_section_descriptor_is_frozen():
    descriptor = SectionDescriptor(
        name="MEMORY_GUIDANCE",
        current_text="baseline text",
        source_path=Path("/tmp/fake.py"),
    )
    assert dataclasses.is_dataclass(descriptor)
    try:
        descriptor.name = "OTHER"
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("SectionDescriptor must be frozen")


def test_prompt_source_protocol_runtime_checkable():
    """A concrete class implementing the three methods satisfies isinstance()."""

    class StubSource:
        name = "stub"

        def read(self, section_name: str) -> str:
            return "stub"

        def write(self, section_name: str, new_text: str) -> None:
            return None

        def list_sections(self) -> list[SectionDescriptor]:
            return []

    assert isinstance(StubSource(), PromptSource)


def test_prompt_source_protocol_rejects_incomplete():
    """Missing a required method => not a PromptSource."""

    class MissingWrite:
        name = "incomplete"

        def read(self, section_name: str) -> str:
            return "x"

        def list_sections(self) -> list[SectionDescriptor]:
            return []

    assert not isinstance(MissingWrite(), PromptSource)
