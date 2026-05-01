"""Tests for the SkillSource abstraction.

Pure-Python; no LM calls. Each test fakes the relevant on-disk layout
under tmp_path and exercises one source class.
"""

from __future__ import annotations

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
