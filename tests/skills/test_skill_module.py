"""Tests for skill module loading and parsing."""

import pytest
from pathlib import Path
from evolution.skills.skill_module import SkillModule, load_skill, reassemble_skill


SAMPLE_SKILL = """---
name: test-skill
description: A skill for testing things
version: 1.0.0
metadata:
  hermes:
    tags: [testing]
---

# Test Skill — Testing Things

## When to Use
Use this when you need to test things.

## Procedure
1. First, do the thing
2. Then, verify it worked
3. Report results

## Pitfalls
- Don't forget to check edge cases
"""


class TestLoadSkill:
    def test_parses_frontmatter(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        assert skill["name"] == "test-skill"
        assert skill["description"] == "A skill for testing things"
        assert "version: 1.0.0" in skill["frontmatter"]

    def test_parses_body(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        assert "# Test Skill" in skill["body"]
        assert "## Procedure" in skill["body"]
        assert "Don't forget" in skill["body"]

    def test_raw_contains_everything(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        assert skill["raw"] == SAMPLE_SKILL

    def test_path_is_stored(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        assert skill["path"] == skill_file


class TestReassembleSkill:
    def test_roundtrip(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        reassembled = reassemble_skill(skill["frontmatter"], skill["body"])
        assert "---" in reassembled
        assert "name: test-skill" in reassembled
        assert "# Test Skill" in reassembled

    def test_preserves_frontmatter(self):
        frontmatter = "name: my-skill\ndescription: Does stuff"
        body = "# My Skill\nDo the thing."
        result = reassemble_skill(frontmatter, body)

        assert result.startswith("---\n")
        assert "name: my-skill" in result
        assert "# My Skill" in result

    def test_evolved_body_replaces_original(self):
        frontmatter = "name: my-skill\ndescription: Does stuff"
        evolved_body = "# EVOLVED\nNew and improved procedure."
        result = reassemble_skill(frontmatter, evolved_body)

        assert "EVOLVED" in result
        assert "New and improved" in result

    def test_strips_accidental_frontmatter_block_from_body(self, caplog):
        # GEPA's reflection LM occasionally mimics YAML structure. If the
        # mutated body itself starts with a `---...---` block, naive
        # reassembly would produce a file with double frontmatter that
        # downstream YAML parsers will choke on.
        frontmatter = "name: my-skill\ndescription: real description"
        evolved_body_with_yaml = (
            "---\n"
            "name: gepa-imagined-skill\n"
            "description: hallucinated\n"
            "---\n\n"
            "# Real Body\nThe actual evolved content."
        )

        with caplog.at_level("WARNING", logger="evolution.skills.skill_module"):
            result = reassemble_skill(frontmatter, evolved_body_with_yaml)

        assert result.startswith("---\nname: my-skill\n")
        # Real frontmatter appears exactly once; the GEPA-imagined one is gone.
        assert result.count("---\n") == 2  # opening + closing of the real block
        assert "gepa-imagined-skill" not in result
        assert "hallucinated" not in result
        assert "# Real Body" in result
        # And we observed it.
        assert any(
            "stripped a leading frontmatter-like block" in rec.message
            for rec in caplog.records
        )


class TestSkillModuleGEPAContract:
    """Lock the contract that GEPA mutates the same place forward() reads.

    GEPA mutates Predict.signature.instructions, not arbitrary module attributes.
    If skill_text and signature.instructions diverge, GEPA's mutations are silently
    discarded and holdout deltas are meaningless.
    """

    def test_skill_text_is_signature_instructions(self):
        module = SkillModule("HELLO SKILL BODY")

        assert module.skill_text == "HELLO SKILL BODY"
        # ChainOfThought wraps a Predict named `predict`; that's the signature
        # GEPA mutates via named_predictors(). skill_text must point at the
        # same surface or mutations are silently lost.
        assert module.predictor.predict.signature.instructions == "HELLO SKILL BODY"

    def test_skill_text_mutation_round_trip(self):
        module = SkillModule("INITIAL")

        module.predictor.predict.signature = (
            module.predictor.predict.signature.with_instructions("MUTATED")
        )

        assert module.skill_text == "MUTATED"

    def test_named_predictors_exposes_skill_signature(self):
        # GEPA discovers what to mutate via named_predictors(); confirm our
        # skill body is reachable through that traversal.
        module = SkillModule("REACHABLE")

        names_and_instructions = [
            (name, p.signature.instructions) for name, p in module.named_predictors()
        ]
        assert any(
            instructions == "REACHABLE" for _, instructions in names_and_instructions
        ), names_and_instructions
