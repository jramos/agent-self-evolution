"""Tests for HermesToolSource — AST-based read path."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from evolution.tools.hermes_source import HermesToolSource
from evolution.tools.tool_source import ToolManifest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tool_manifests"
HERMES_SHAPE = FIXTURES / "hermes_shape"


class TestSupports:
    def test_supports_returns_true_for_hermes_shape_dir(self):
        assert HermesToolSource(HERMES_SHAPE).supports(HERMES_SHAPE) is True

    def test_supports_returns_false_for_empty_dir(self, tmp_path: Path):
        (tmp_path / "notes.txt").write_text("nothing to see here")
        assert HermesToolSource(tmp_path).supports(tmp_path) is False

    def test_supports_returns_false_for_json_file(self):
        json_path = FIXTURES / "multiple_tools.json"
        assert HermesToolSource(json_path).supports(json_path) is False

    def test_supports_returns_false_for_py_files_without_schema_names(self, tmp_path: Path):
        (tmp_path / "regular.py").write_text("CONSTANT = {'foo': 'bar'}\n")
        assert HermesToolSource(tmp_path).supports(tmp_path) is False


class TestFindManifestReadAll:
    @pytest.fixture
    def manifest(self) -> ToolManifest:
        return HermesToolSource(HERMES_SHAPE).find_manifest(HERMES_SHAPE)

    def test_find_manifest_reads_all_extractable_tools(self, manifest: ToolManifest):
        names = {t.name for t in manifest.tools}
        expected = {
            "simple_tool",
            "list_tool_a",
            "list_tool_b",
            "list_tool_c",
            "nonliteral_sibling",
            "name_ref_tool",
            "f_string_tool",
            "multi_line_concat",
        }
        assert names == expected
        assert len(manifest.tools) == 8

    def test_tools_are_sorted_by_name(self, manifest: ToolManifest):
        names = [t.name for t in manifest.tools]
        assert names == sorted(names)

    def test_find_manifest_drops_function_built_tools(self, manifest: ToolManifest):
        assert manifest.dropped_tools, "expected at least one dropped tool"
        hints = {hint for hint, _reason in manifest.dropped_tools}
        # The function-built schema is dropped under its variable name since
        # the name field isn't statically reachable through ast.Call.
        assert any("FUNCTION_BUILT_SCHEMA" in hint for hint in hints) or "function_built_tool" in hints
        # Every dropped entry has a non-empty reason string.
        for _, reason in manifest.dropped_tools:
            assert reason and isinstance(reason, str)


class TestSourceKind:
    @pytest.fixture
    def manifest(self) -> ToolManifest:
        return HermesToolSource(HERMES_SHAPE).find_manifest(HERMES_SHAPE)

    @pytest.mark.parametrize(
        "tool_name,expected_kind",
        [
            ("simple_tool", "literal"),
            ("list_tool_a", "literal"),
            ("list_tool_b", "literal"),
            ("list_tool_c", "literal"),
            ("nonliteral_sibling", "literal"),
            ("multi_line_concat", "literal"),
            ("name_ref_tool", "name_ref"),
            ("f_string_tool", "joined_str"),
        ],
    )
    def test_source_kind_per_shape(self, manifest: ToolManifest, tool_name: str, expected_kind: str):
        tool = manifest.find_tool(tool_name)
        assert tool.source_kind == expected_kind

    def test_source_location_is_set_for_every_tool(self, manifest: ToolManifest):
        for tool in manifest.tools:
            assert tool.source_location is not None, f"{tool.name} missing source_location"
            file_path, lineno, col_offset, end_lineno, end_col_offset = tool.source_location
            assert isinstance(file_path, Path)
            assert file_path.exists()
            assert lineno >= 1
            assert end_lineno >= lineno
            assert col_offset >= 0
            assert end_col_offset >= 0


class TestNameRef:
    def test_name_ref_resolves_to_constant_text(self):
        manifest = HermesToolSource(HERMES_SHAPE).find_manifest(HERMES_SHAPE)
        tool = manifest.find_tool("name_ref_tool")
        assert tool.description == "Description lives in a separate constant at module top level."
        assert tool.source_kind == "name_ref"
        file_path, lineno, *_ = tool.source_location
        assert file_path.name == "name_ref_tool.py"
        # NAME_REF_DESCRIPTION is on line 1 of the fixture; the source_location
        # should point at the *constant*, not the SCHEMA's description field.
        assert lineno == 1


class TestMultiLineConcat:
    def test_multi_line_concat_description_is_single_string(self):
        manifest = HermesToolSource(HERMES_SHAPE).find_manifest(HERMES_SHAPE)
        tool = manifest.find_tool("multi_line_concat")
        # Parser folds parenthesized concat into one Constant.
        assert tool.description == (
            "First line of the description.\n\n"
            "Second paragraph with **markdown**.\n\n"
            "Third paragraph that wraps."
        )
        assert tool.source_kind == "literal"
        _, lineno, _, end_lineno, _ = tool.source_location
        # Spans multiple source lines, from opening to closing paren.
        assert end_lineno > lineno


class TestFStringDescription:
    def test_f_string_description_carries_rendered_text(self):
        manifest = HermesToolSource(HERMES_SHAPE).find_manifest(HERMES_SHAPE)
        tool = manifest.find_tool("f_string_tool")
        assert tool.source_kind == "joined_str"
        # The description value is the ast.unparse-rendered f-string source —
        # i.e., starts with f' or f" and contains the {__name__} placeholder.
        assert tool.description.startswith("f'") or tool.description.startswith('f"')
        assert "{__name__}" in tool.description


class TestApplyEvolvedStub:
    def test_apply_evolved_raises_not_implemented(self):
        source = HermesToolSource(HERMES_SHAPE)
        manifest = source.find_manifest(HERMES_SHAPE)
        with pytest.raises(NotImplementedError, match="next commit"):
            source.apply_evolved(
                source_path=HERMES_SHAPE,
                evolved_manifest=manifest,
                target_tool="simple_tool",
                new_description="updated",
            )


class TestSidecarMetadata:
    def test_sidecar_metadata_loads_confusable_neighbors(self, tmp_path: Path):
        # Copy the fixture so we can drop a sidecar without polluting the
        # checked-in tree.
        copied = tmp_path / "hermes_shape"
        shutil.copytree(HERMES_SHAPE, copied)
        sidecar = copied / "_evolution_metadata.json"
        sidecar.write_text(json.dumps({"confusable_neighbors": {"simple_tool": "list_tool_a"}}))

        manifest = HermesToolSource(copied).find_manifest(copied)
        assert manifest.confusable_neighbor_for("simple_tool") == "list_tool_a"

    def test_missing_sidecar_yields_empty_confusable_map(self):
        manifest = HermesToolSource(HERMES_SHAPE).find_manifest(HERMES_SHAPE)
        assert manifest.confusable_neighbors == {}
