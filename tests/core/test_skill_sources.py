"""Tests for the SkillSource abstraction.

Pure-Python; no LM calls. Each test fakes the relevant on-disk layout
under tmp_path and exercises one source class.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evolution.core.skill_sources import (
    ClaudeCodeSkillSource,
    HermesSkillSource,
    LocalDirSkillSource,
    discover_skill_sources,
)


def _write_skill(path: Path, name: str, body: str = "# Body\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: test skill\n---\n\n{body}")
    return path


def _symlink_or_skip(target: Path, link: Path, *, dir_ok: bool = True) -> None:
    """Create ``link`` -> ``target`` or skip the test if the platform/user
    can't make symlinks (e.g. Windows without developer mode)."""
    try:
        link.symlink_to(target, target_is_directory=dir_ok)
    except (OSError, NotImplementedError) as e:  # pragma: no cover - platform-gated
        pytest.skip(f"symlinks unavailable here: {e}")




class TestHermesSkillSource:
    def test_finds_skill_by_dir_name(self, tmp_path: Path):
        skill = _write_skill(
            tmp_path / "skills" / "research" / "arxiv" / "SKILL.md",
            name="arxiv",
        )
        source = HermesSkillSource(tmp_path)
        assert source.find_skill("arxiv") == skill

    def test_falls_back_to_frontmatter_name_when_dir_differs(self, tmp_path: Path):
        # Directory named "arxiv-alt" but frontmatter says "arxiv"
        skill = _write_skill(
            tmp_path / "skills" / "research" / "arxiv-alt" / "SKILL.md",
            name="arxiv",
        )
        source = HermesSkillSource(tmp_path)
        assert source.find_skill("arxiv") == skill

    def test_returns_none_when_skill_missing(self, tmp_path: Path):
        _write_skill(
            tmp_path / "skills" / "research" / "arxiv" / "SKILL.md", "arxiv",
        )
        source = HermesSkillSource(tmp_path)
        assert source.find_skill("missing") is None

    def test_returns_none_when_skills_dir_missing(self, tmp_path: Path):
        # Root exists but no skills/ subdir.
        source = HermesSkillSource(tmp_path)
        assert source.find_skill("arxiv") is None
        assert source.list_skills() == []

    def test_list_skills_returns_unique_directory_names(self, tmp_path: Path):
        _write_skill(tmp_path / "skills" / "a" / "alpha" / "SKILL.md", "alpha")
        _write_skill(tmp_path / "skills" / "b" / "beta" / "SKILL.md", "beta")
        source = HermesSkillSource(tmp_path)
        assert source.list_skills() == ["alpha", "beta"]


class TestHermesSkillSourceSymlinks:
    """The Hermes layout symlinks user-installed skills into the framework
    tree; `Path.rglob` refuses to descend into symlinked dirs on Python <3.13,
    so discovery must use a symlink-following, cycle-safe walk.
    """

    # --- Correctness: symlinked artifacts are discovered ---

    def test_finds_skill_through_symlinked_dir(self, tmp_path: Path):
        # C1: the skill's own directory is a symlink into the skills/ tree.
        real = _write_skill(tmp_path / "store" / "arxiv" / "SKILL.md", "arxiv")
        (tmp_path / "skills" / "research").mkdir(parents=True)
        _symlink_or_skip(
            tmp_path / "store" / "arxiv",
            tmp_path / "skills" / "research" / "arxiv",
        )
        source = HermesSkillSource(tmp_path)
        found = source.find_skill("arxiv")
        assert found is not None
        assert found.read_text() == real.read_text()
        assert "arxiv" in source.list_skills()

    def test_traverses_symlinked_intermediate_category(self, tmp_path: Path):
        # C2: an intermediate category directory is a symlink.
        _write_skill(tmp_path / "store" / "arxiv" / "SKILL.md", "arxiv")
        (tmp_path / "skills").mkdir(parents=True)
        _symlink_or_skip(tmp_path / "store", tmp_path / "skills" / "research")
        source = HermesSkillSource(tmp_path)
        assert source.find_skill("arxiv") is not None
        assert "arxiv" in source.list_skills()

    def test_finds_symlinked_skill_md_file(self, tmp_path: Path):
        # C3: the SKILL.md file itself is a symlink.
        _write_skill(tmp_path / "store" / "SKILL.md", "writing")
        (tmp_path / "skills" / "cat" / "writing").mkdir(parents=True)
        link = tmp_path / "skills" / "cat" / "writing" / "SKILL.md"
        _symlink_or_skip(tmp_path / "store" / "SKILL.md", link, dir_ok=False)
        source = HermesSkillSource(tmp_path)
        assert source.find_skill("writing") == link
        assert "writing" in source.list_skills()

    def test_frontmatter_fallback_through_symlink(self, tmp_path: Path):
        # C4: dir name differs from frontmatter `name:`, reached via symlink.
        _write_skill(tmp_path / "store" / "arxiv-alt" / "SKILL.md", "arxiv")
        (tmp_path / "skills" / "research").mkdir(parents=True)
        _symlink_or_skip(
            tmp_path / "store" / "arxiv-alt",
            tmp_path / "skills" / "research" / "arxiv-alt",
        )
        source = HermesSkillSource(tmp_path)
        assert source.find_skill("arxiv") is not None

    def test_lists_both_real_and_symlinked_skills(self, tmp_path: Path):
        # C5: a mixed tree returns the complete set.
        _write_skill(tmp_path / "skills" / "a" / "alpha" / "SKILL.md", "alpha")
        _write_skill(tmp_path / "store" / "beta" / "SKILL.md", "beta")
        (tmp_path / "skills" / "b").mkdir(parents=True)
        _symlink_or_skip(tmp_path / "store" / "beta", tmp_path / "skills" / "b" / "beta")
        source = HermesSkillSource(tmp_path)
        assert source.list_skills() == ["alpha", "beta"]

    # --- Safety: cycles, permissions, dangling links ---

    def test_symlink_cycle_terminates(self, tmp_path: Path):
        # S1: a symlink pointing back to an ancestor must not hang.
        _write_skill(tmp_path / "skills" / "a" / "SKILL.md", "a")
        _symlink_or_skip(tmp_path / "skills", tmp_path / "skills" / "a" / "loop")
        source = HermesSkillSource(tmp_path)
        assert source.list_skills() == ["a"]
        assert source.find_skill("a") is not None

    @pytest.mark.skipif(
        hasattr(os, "getuid") and os.getuid() == 0,
        reason="root bypasses directory permission bits",
    )
    def test_unreadable_dir_is_skipped(self, tmp_path: Path):
        # S2: an unreadable directory is skipped, traversal continues.
        _write_skill(tmp_path / "skills" / "a" / "alpha" / "SKILL.md", "alpha")
        noperm = tmp_path / "skills" / "b"
        _write_skill(noperm / "beta" / "SKILL.md", "beta")
        os.chmod(noperm, 0o000)
        try:
            source = HermesSkillSource(tmp_path)
            assert source.list_skills() == ["alpha"]  # no raise; beta unreachable
        finally:
            os.chmod(noperm, 0o755)

    def test_dangling_symlink_does_not_raise(self, tmp_path: Path):
        # S3: a broken symlink must not raise.
        _write_skill(tmp_path / "skills" / "a" / "alpha" / "SKILL.md", "alpha")
        (tmp_path / "skills" / "b").mkdir(parents=True)
        _symlink_or_skip(
            tmp_path / "does-not-exist", tmp_path / "skills" / "b" / "ghost"
        )
        source = HermesSkillSource(tmp_path)
        assert "alpha" in source.list_skills()
        assert source.find_skill("alpha") is not None

    # --- Determinism ---

    def test_duplicate_skill_name_resolves_deterministically(self, tmp_path: Path):
        # D1: two dirs share a skill name → the lexicographically-first branch
        # wins, stably across repeated calls.
        a = _write_skill(tmp_path / "skills" / "a" / "dup" / "SKILL.md", "dup", "A\n")
        _write_skill(tmp_path / "skills" / "z" / "dup" / "SKILL.md", "dup", "Z\n")
        source = HermesSkillSource(tmp_path)
        first = source.find_skill("dup")
        assert first == a
        assert all(source.find_skill("dup") == first for _ in range(3))


class TestClaudeCodeSkillSource:
    def test_finds_skill_in_plugin_cache(self, tmp_path: Path):
        skill = _write_skill(
            tmp_path
            / "vendor-a" / "plugin-x" / "1.0.0" / "skills" / "writing" / "SKILL.md",
            name="writing",
        )
        source = ClaudeCodeSkillSource(plugins_cache=tmp_path)
        assert source.find_skill("writing") == skill

    def test_walks_multiple_plugins(self, tmp_path: Path):
        _write_skill(
            tmp_path
            / "vendor-a" / "plugin-x" / "1.0.0" / "skills" / "writing" / "SKILL.md",
            "writing",
        )
        debug = _write_skill(
            tmp_path
            / "vendor-b" / "plugin-y" / "2.0.0" / "skills" / "debugging" / "SKILL.md",
            "debugging",
        )
        source = ClaudeCodeSkillSource(plugins_cache=tmp_path)
        assert source.find_skill("debugging") == debug
        assert set(source.list_skills()) == {"writing", "debugging"}

    def test_picks_latest_version_on_per_plugin_collision(self, tmp_path: Path):
        old = _write_skill(
            tmp_path
            / "vendor-a" / "plugin-x" / "1.0.0" / "skills" / "writing" / "SKILL.md",
            "writing", body="old body\n",
        )
        new = _write_skill(
            tmp_path
            / "vendor-a" / "plugin-x" / "2.0.0" / "skills" / "writing" / "SKILL.md",
            "writing", body="new body\n",
        )
        source = ClaudeCodeSkillSource(plugins_cache=tmp_path)
        # Latest version wins; old version not returned.
        assert source.find_skill("writing") == new
        assert source.find_skill("writing") != old

    def test_returns_none_when_cache_missing(self, tmp_path: Path):
        source = ClaudeCodeSkillSource(plugins_cache=tmp_path / "nope")
        assert source.find_skill("x") is None
        assert source.list_skills() == []

    def test_returns_none_when_skill_not_in_any_plugin(self, tmp_path: Path):
        _write_skill(
            tmp_path
            / "vendor-a" / "plugin-x" / "1.0.0" / "skills" / "writing" / "SKILL.md",
            "writing",
        )
        source = ClaudeCodeSkillSource(plugins_cache=tmp_path)
        assert source.find_skill("missing") is None




class TestLocalDirSkillSource:
    def test_finds_skill_in_flat_layout(self, tmp_path: Path):
        skill = _write_skill(tmp_path / "myskill" / "SKILL.md", "myskill")
        source = LocalDirSkillSource(tmp_path)
        assert source.find_skill("myskill") == skill

    def test_returns_none_when_root_missing(self, tmp_path: Path):
        source = LocalDirSkillSource(tmp_path / "nonexistent")
        assert source.find_skill("anything") is None
        assert source.list_skills() == []

    def test_list_skills_only_includes_dirs_with_skill_md(self, tmp_path: Path):
        _write_skill(tmp_path / "with-skill" / "SKILL.md", "with-skill")
        (tmp_path / "without-skill").mkdir()  # has dir but no SKILL.md
        source = LocalDirSkillSource(tmp_path)
        assert source.list_skills() == ["with-skill"]




class TestDiscoverSkillSources:
    def test_explicit_dirs_appear_first(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("SKILL_SOURCES_HERMES_REPO", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        # Empty home → no Hermes fallback, no Claude Code cache.
        d1, d2 = tmp_path / "d1", tmp_path / "d2"
        d1.mkdir(); d2.mkdir()
        sources = discover_skill_sources(explicit_dirs=[d1, d2])
        assert [s.name for s in sources] == [f"local-dir:{d1}", f"local-dir:{d2}"]

    def test_hermes_added_when_env_var_set(self, tmp_path: Path, monkeypatch):
        repo = tmp_path / "hermes-repo"
        (repo / "skills").mkdir(parents=True)
        monkeypatch.setenv("SKILL_SOURCES_HERMES_REPO", str(repo))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        sources = discover_skill_sources()
        assert any(s.name == "hermes" for s in sources)

    def test_claude_code_added_when_cache_exists(self, tmp_path: Path, monkeypatch):
        home = tmp_path / "home"
        (home / ".claude" / "plugins" / "cache").mkdir(parents=True)
        monkeypatch.delenv("SKILL_SOURCES_HERMES_REPO", raising=False)
        monkeypatch.setattr(Path, "home", lambda: home)
        sources = discover_skill_sources()
        assert [s.name for s in sources] == ["claude-code"]

    def test_omits_sources_with_missing_roots(self, tmp_path: Path, monkeypatch):
        # No env var, no ~/.hermes, no ~/.claude/plugins/cache.
        monkeypatch.delenv("SKILL_SOURCES_HERMES_REPO", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "empty-home")
        sources = discover_skill_sources()
        assert sources == []

    def test_priority_explicit_then_hermes_then_claude_code(
        self, tmp_path: Path, monkeypatch,
    ):
        explicit = tmp_path / "explicit"; explicit.mkdir()
        repo = tmp_path / "hermes-repo"
        (repo / "skills").mkdir(parents=True)
        home = tmp_path / "home"
        (home / ".claude" / "plugins" / "cache").mkdir(parents=True)

        monkeypatch.setenv("SKILL_SOURCES_HERMES_REPO", str(repo))
        monkeypatch.setattr(Path, "home", lambda: home)
        sources = discover_skill_sources(explicit_dirs=[explicit])
        assert [s.name for s in sources] == [
            f"local-dir:{explicit}",
            "hermes",
            "claude-code",
        ]
