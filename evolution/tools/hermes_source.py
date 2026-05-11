"""ToolSource adapter for tool schemas defined in Python source.

Reads Hermes-Agent-shaped tool descriptions: top-level ``*_SCHEMA`` /
``*_SCHEMAS`` assignments in ``.py`` files. Description extraction is
purely AST-based — no module execution — and tolerates schemas whose
sibling fields use non-literal values (Name refs, function calls,
lists). Schemas whose description itself is not extractable via AST
(e.g., built from a function call) are dropped and surfaced through
``ToolManifest.dropped_tools`` so callers can see what was skipped.

The write path (``apply_evolved``) lands in the next commit.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path

from evolution.tools.tool_source import ToolEntry, ToolManifest

logger = logging.getLogger(__name__)

# Matches Hermes-style schema-variable names. The leading underscore is
# optional (Hermes has ``_STATIC_CORE_SCHEMA`` as a module-private
# canonical form). ``_SCHEMA`` is a single tool; ``_SCHEMAS`` is a list.
_TOOL_NAME_PATTERN = re.compile(r"^_?[A-Z][A-Z0-9_]*_(SCHEMA|SCHEMAS)$")


class HermesToolSource:
    """Reads tool descriptions out of Python source via AST."""

    name = "hermes_source"

    def __init__(self, root: Path):
        self.root = Path(root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def supports(self, path: Path) -> bool:
        """True iff ``path`` is a directory containing at least one ``.py``
        file with a top-level Assign whose target id matches
        ``_TOOL_NAME_PATTERN``.
        """
        candidate = Path(path)
        if not candidate.is_dir():
            return False
        for py_file in self._iter_py_files(candidate):
            try:
                tree = ast.parse(py_file.read_text())
            except (SyntaxError, OSError):
                continue
            for node in tree.body:
                if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                    continue
                target = node.targets[0]
                if isinstance(target, ast.Name) and _TOOL_NAME_PATTERN.match(target.id):
                    return True
        return False

    def find_manifest(self, path: Path) -> ToolManifest | None:
        candidate = Path(path)
        if not self.supports(candidate):
            return None

        tools: list[ToolEntry] = []
        dropped: list[tuple[str, str]] = []

        for py_file in sorted(self._iter_py_files(candidate)):
            try:
                source = py_file.read_text()
                tree = ast.parse(source)
            except (SyntaxError, OSError) as exc:
                logger.warning("skipping %s: parse failed: %s", py_file, exc)
                continue

            # Build a one-shot lookup of top-level name -> Constant(str) value
            # for one-hop Name-ref resolution in the description field.
            name_consts = _collect_name_constants(tree)

            for node in tree.body:
                if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                    continue
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                match = _TOOL_NAME_PATTERN.match(target.id)
                if not match:
                    continue

                kind_suffix = match.group(1)  # "SCHEMA" or "SCHEMAS"
                if kind_suffix == "SCHEMA":
                    self._consume_schema_node(
                        py_file=py_file,
                        var_name=target.id,
                        value=node.value,
                        name_consts=name_consts,
                        tools=tools,
                        dropped=dropped,
                    )
                else:  # SCHEMAS
                    if not isinstance(node.value, ast.List):
                        dropped.append(
                            (
                                f"<{target.id}>",
                                f"{py_file.name}: expected list literal for *_SCHEMAS, "
                                f"got {type(node.value).__name__}",
                            )
                        )
                        continue
                    for idx, elt in enumerate(node.value.elts):
                        self._consume_schema_node(
                            py_file=py_file,
                            var_name=f"{target.id}[{idx}]",
                            value=elt,
                            name_consts=name_consts,
                            tools=tools,
                            dropped=dropped,
                        )

        tools.sort(key=lambda t: t.name)
        confusable = _load_sidecar_confusable_neighbors(candidate)
        return ToolManifest(
            tools=tuple(tools),
            confusable_neighbors=confusable,
            dropped_tools=tuple(dropped),
        )

    def apply_evolved(
        self,
        source_path: Path,
        evolved_manifest: ToolManifest,
        target_tool: str,
        new_description: str,
    ) -> None:
        """Write the evolved description back to source.

        Byte-precise rewriting lands in the next commit.
        """
        raise NotImplementedError("apply_evolved lands in the next commit")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _iter_py_files(self, root: Path):
        for py_file in root.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            yield py_file

    def _consume_schema_node(
        self,
        *,
        py_file: Path,
        var_name: str,
        value: ast.AST,
        name_consts: dict[str, ast.Constant],
        tools: list[ToolEntry],
        dropped: list[tuple[str, str]],
    ) -> None:
        """Extract one tool entry from a dict-shaped schema node, or drop."""
        if not isinstance(value, ast.Dict):
            dropped.append(
                (
                    f"<{var_name}>",
                    f"{py_file.name}: schema value is {type(value).__name__}, "
                    "not a dict literal — unreachable via pure AST",
                )
            )
            return

        name_value = _extract_name(value)
        if name_value is None:
            dropped.append(
                (
                    "<unparseable_name>",
                    f"{py_file.name}: {var_name} 'name' field is missing or not a string literal",
                )
            )
            return

        desc_result = _extract_description(value, name_consts)
        if desc_result is None:
            dropped.append(
                (
                    name_value,
                    f"{py_file.name}: {var_name} 'description' field is missing or "
                    "not extractable via pure AST",
                )
            )
            return

        description, source_kind, source_location_node = desc_result
        source_location = (
            py_file,
            source_location_node.lineno,
            source_location_node.col_offset,
            source_location_node.end_lineno,
            source_location_node.end_col_offset,
        )

        tools.append(
            ToolEntry(
                name=name_value,
                description=description,
                input_schema={},
                source_kind=source_kind,
                source_location=source_location,
            )
        )


def _collect_name_constants(tree: ast.Module) -> dict[str, ast.Constant]:
    """Return {var_name: Constant} for every top-level ``X = "..."`` Assign."""
    out: dict[str, ast.Constant] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            out[target.id] = node.value
    return out


def _extract_name(dict_node: ast.Dict) -> str | None:
    """Return the literal string value of the dict's ``"name"`` key, or None."""
    for key, val in zip(dict_node.keys, dict_node.values):
        if isinstance(key, ast.Constant) and key.value == "name":
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                return val.value
            return None
    return None


def _extract_description(
    dict_node: ast.Dict,
    name_consts: dict[str, ast.Constant],
) -> tuple[str, str, ast.AST] | None:
    """Return (description, source_kind, location_node) or None if not extractable."""
    for key, val in zip(dict_node.keys, dict_node.values):
        if not (isinstance(key, ast.Constant) and key.value == "description"):
            continue
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            return val.value, "literal", val
        if isinstance(val, ast.Name):
            resolved = name_consts.get(val.id)
            if resolved is not None:
                return resolved.value, "name_ref", resolved
            return None
        if isinstance(val, ast.JoinedStr):
            return ast.unparse(val), "joined_str", val
        # ast.Call, list, etc. — not statically extractable.
        return None
    return None


def _load_sidecar_confusable_neighbors(root: Path) -> dict[str, str]:
    """Load ``<root>/_evolution_metadata.json`` if present and return its
    ``confusable_neighbors`` mapping. Returns an empty dict if the sidecar
    is missing or malformed (and logs a warning).
    """
    sidecar = root / "_evolution_metadata.json"
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("failed to read sidecar %s: %s", sidecar, exc)
        return {}
    neighbors = data.get("confusable_neighbors", {})
    if not isinstance(neighbors, dict):
        logger.warning(
            "sidecar %s has confusable_neighbors of type %s, expected dict; ignoring",
            sidecar,
            type(neighbors).__name__,
        )
        return {}
    return neighbors
