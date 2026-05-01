"""Pluggable skill discovery across multiple agent frameworks.

The optimizer is agent-agnostic — it operates on `(skill_text, eval_examples)`
where skill_text is opaque markdown. Different agent frameworks lay out
their SKILL.md files differently:

- Hermes Agent: ``<root>/skills/<category>/<name>/SKILL.md``
- Claude Code:  ``~/.claude/plugins/cache/<vendor>/<plugin>/<version>/skills/<name>/SKILL.md``
- Codex / openclaw / custom: ``<dir>/<name>/SKILL.md`` (escape hatch)

This module abstracts the "find the SKILL.md given a skill name" step
behind a `SkillSource` protocol. `discover_skill_sources()` builds the
default multi-source list by sniffing the environment.

Frontmatter format is identical across all three (`name:` and
`description:` between `---` markers), so `load_skill()` in
`evolution/skills/skill_module.py` doesn't change — only discovery does.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class SkillSource(Protocol):
    """Discovers SKILL.md files matching a given skill name."""

    name: str

    def find_skill(self, skill_name: str) -> Optional[Path]: ...

    def list_skills(self) -> list[str]: ...


class HermesSkillSource:
    """Hermes Agent layout: ``<root>/skills/<category>/<name>/SKILL.md``.

    `<category>` is one or more directory levels (e.g. ``research/arxiv``,
    ``apple/imessage``). Discovery walks recursively under ``<root>/skills``
    and matches by directory name first, then falls back to parsing
    frontmatter ``name:`` for skills whose directory name differs from the
    canonical name.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.name = "hermes"

    def _skills_dir(self) -> Optional[Path]:
        d = self.root / "skills"
        return d if d.exists() and d.is_dir() else None

    def find_skill(self, skill_name: str) -> Optional[Path]:
        skills_dir = self._skills_dir()
        if skills_dir is None:
            return None
        for skill_md in skills_dir.rglob("SKILL.md"):
            if skill_md.parent.name == skill_name:
                return skill_md
        # Fall back to frontmatter `name:` for skills whose dir name differs.
        for skill_md in skills_dir.rglob("SKILL.md"):
            try:
                head = skill_md.read_text()[:500]
            except OSError:
                continue
            if f"name: {skill_name}" in head or f'name: "{skill_name}"' in head:
                return skill_md
        return None

    def list_skills(self) -> list[str]:
        skills_dir = self._skills_dir()
        if skills_dir is None:
            return []
        return sorted({p.parent.name for p in skills_dir.rglob("SKILL.md")})


class ClaudeCodeSkillSource:
    """Claude Code plugin cache layout.

    Skills live at:
        ``<plugins_cache>/<vendor>/<plugin>/<version>/skills/<name>/SKILL.md``

    Multiple plugin packages may install simultaneously, and a single
    plugin may have multiple versions in the cache. On collision (same
    skill name across plugins or versions), the highest-versioned entry
    wins via lexicographic sort of version strings (works for SemVer).
    """

    def __init__(
        self,
        plugins_cache: Path = Path.home() / ".claude" / "plugins" / "cache",
    ) -> None:
        self.plugins_cache = Path(plugins_cache)
        self.name = "claude-code"

    def _skill_dirs(self) -> list[Path]:
        """All ``<vendor>/<plugin>/<version>/skills`` directories on disk.

        On per-plugin version collisions the highest version (lex-sorted, works
        for SemVer) wins.
        """
        if not self.plugins_cache.exists():
            return []
        roots: list[Path] = []
        for vendor in self.plugins_cache.iterdir():
            if not vendor.is_dir():
                continue
            for plugin in vendor.iterdir():
                if not plugin.is_dir():
                    continue
                versions = sorted(
                    (v for v in plugin.iterdir() if v.is_dir()),
                    key=lambda p: p.name,
                )
                if not versions:
                    continue
                latest_skills = versions[-1] / "skills"
                if latest_skills.is_dir():
                    roots.append(latest_skills)
        return roots

    def find_skill(self, skill_name: str) -> Optional[Path]:
        for skills_dir in self._skill_dirs():
            candidate = skills_dir / skill_name / "SKILL.md"
            if candidate.is_file():
                return candidate
        return None

    def list_skills(self) -> list[str]:
        names: set[str] = set()
        for skills_dir in self._skill_dirs():
            for entry in skills_dir.iterdir():
                if entry.is_dir() and (entry / "SKILL.md").is_file():
                    names.add(entry.name)
        return sorted(names)


class LocalDirSkillSource:
    """Generic flat layout: ``<root>/<name>/SKILL.md``.

    Escape hatch for Codex, openclaw, or any custom agent framework that
    emits SKILL.md files in a flat directory of skill folders. Pointed at
    via the repeatable ``--skill-source-dir`` CLI flag.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.name = f"local-dir:{self.root}"

    def find_skill(self, skill_name: str) -> Optional[Path]:
        if not self.root.is_dir():
            return None
        candidate = self.root / skill_name / "SKILL.md"
        return candidate if candidate.is_file() else None

    def list_skills(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            entry.name
            for entry in self.root.iterdir()
            if entry.is_dir() and (entry / "SKILL.md").is_file()
        )


def _hermes_root_from_env() -> Optional[Path]:
    """Resolve the Hermes skill repo from env or the default install dir."""
    env_path = os.getenv("SKILL_SOURCES_HERMES_REPO")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_dir():
            return p
    fallback = Path.home() / ".hermes" / "hermes-agent"
    if fallback.is_dir():
        return fallback
    sibling = Path(__file__).resolve().parent.parent.parent.parent / "hermes-agent"
    if sibling.is_dir():
        return sibling
    return None


def discover_skill_sources(
    explicit_dirs: Optional[list[Path]] = None,
) -> list[SkillSource]:
    """Build the default skill-source list by sniffing the environment.

    Priority (first match wins in `find_skill`):

    1. Each ``--skill-source-dir`` from the CLI as a `LocalDirSkillSource`
    2. `HermesSkillSource` if `SKILL_SOURCES_HERMES_REPO` is set or
       ``~/.hermes/hermes-agent`` exists
    3. `ClaudeCodeSkillSource` if ``~/.claude/plugins/cache`` exists

    Sources whose roots don't exist are omitted so `find_skill` doesn't
    waste rglob calls on missing directories.
    """
    sources: list[SkillSource] = []
    for d in explicit_dirs or []:
        sources.append(LocalDirSkillSource(d))
    hermes_root = _hermes_root_from_env()
    if hermes_root is not None:
        sources.append(HermesSkillSource(hermes_root))
    cc_cache = Path.home() / ".claude" / "plugins" / "cache"
    if cc_cache.exists():
        sources.append(ClaudeCodeSkillSource(cc_cache))
    return sources
