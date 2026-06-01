"""PromptSource Protocol — adapters that read, write, and enumerate named prompt sections.

Phase 3 integrates via in-place splice-and-restore (see
``HermesPromptSectionInstaller``), so the runtime override seam lives in
the installer, not here. A PromptSource only needs to read the baseline,
persist an evolved value, and enumerate what's targetable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SectionDescriptor:
    """Metadata about an evolvable prompt section.

    ``applicability`` is informational at design time; it's not used for
    runtime filtering in v1, but downstream joint-optimization work will
    consume it (e.g., model-family-targeted sections only get evaluated
    against that family).
    """

    name: str
    current_text: str
    source_path: Path
    applicability: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class PromptSource(Protocol):
    """Adapter contract for prompt-section evolution targets."""

    name: str

    def read(self, section_name: str) -> str:
        """Return the canonical baseline text of the named section."""
        ...

    def write(self, section_name: str, new_text: str) -> None:
        """Persist evolved text to the canonical source.

        Used both at deploy time and as the splice primitive the
        closed-loop installer drives during validation (the installer
        owns the backup/restore around the mutation).
        """
        ...

    def list_sections(self) -> list[SectionDescriptor]:
        """Enumerate all evolvable sections this source can target."""
        ...
