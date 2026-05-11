"""ToolSource adapter for tool schemas defined in Python source.

Reads Hermes-Agent-shaped tool descriptions: top-level ``*_SCHEMA`` /
``*_SCHEMAS`` assignments in ``.py`` files. Description extraction is
purely AST-based — no module execution — and tolerates schemas whose
sibling fields use non-literal values (Name refs, function calls,
lists). Schemas whose description itself is not extractable via AST
(e.g., built from a function call) are dropped and surfaced through
``ToolManifest.dropped_tools`` so callers can see what was skipped.

The write path (``apply_evolved``) splices the evolved description into
the source file's bytes at the AST-derived span, then atomically
replaces the file. Multi-line spans collapse to a single triple-quoted
string; name_ref descriptions modify the resolved constant's assignment
rather than the schema dict's "description" key.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import tempfile
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

        Splices ``new_description`` into the bytes of the file recorded in
        the target tool's ``source_location``. Other tools' bytes are left
        verbatim. Refuses to rewrite f-string-described tools — the caller
        must convert the description to a literal first.
        """
        try:
            entry = evolved_manifest.find_tool(target_tool)
        except KeyError as exc:
            raise KeyError(f"target tool {target_tool!r} not in manifest") from exc

        if entry.source_kind == "joined_str":
            raise ValueError(
                f"cannot apply evolved description to tool {target_tool!r}: "
                f"its description is an f-string. Rewrite the tool to use a "
                f"literal description and try again."
            )
        if entry.source_location is None:
            raise ValueError(
                f"tool {target_tool!r} has no source_location; adapter cannot write it back"
            )

        file_path, lineno, col_offset, end_lineno, end_col_offset = entry.source_location
        text = file_path.read_text(encoding="utf-8")

        # AST line/col offsets address bytes in the encoded source. For pure
        # ASCII source — the common case for tool description literals —
        # byte offsets and str-index offsets coincide. Multi-byte chars in
        # the description body itself would skew col_offset; we accept that
        # limitation rather than re-decoding around every splice.
        start_offset = _compute_byte_offset(text, lineno, col_offset)
        end_offset = _compute_byte_offset(text, end_lineno, end_col_offset)

        multi_line = lineno != end_lineno
        replacement = _format_replacement(
            new_description,
            multi_line=multi_line,
            indent=col_offset,
        )
        new_text = text[:start_offset] + replacement + text[end_offset:]

        # Atomic write: temp file in the same dir, then os.replace.
        fd, tmp_name = tempfile.mkstemp(dir=file_path.parent, suffix=file_path.suffix)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(new_text)
            os.replace(tmp_path, file_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

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


def _compute_byte_offset(text: str, lineno: int, col_offset: int) -> int:
    """Convert AST ``(lineno, col_offset)`` (1-based line, 0-based col) to a
    str-index offset into ``text``.

    AST col_offset is technically in bytes per CPython, but for Python source
    that is pure ASCII (or has no multi-byte chars in the columns we splice
    around), the byte count equals the str length. The description body
    itself may contain non-ASCII, but the *boundaries* — the quote chars at
    start/end — are ASCII, so the resulting offsets are accurate.
    """
    lines = text.splitlines(keepends=True)
    if lineno < 1 or lineno > len(lines):
        raise ValueError(f"lineno {lineno} out of range [1, {len(lines)}]")
    return sum(len(line) for line in lines[: lineno - 1]) + col_offset


def _format_replacement(new_description: str, *, multi_line: bool, indent: int) -> str:
    """Build the source-literal replacement for ``new_description``.

    For single-line spans we emit ``repr(new_description)`` — Python's repr
    always produces a valid string literal with correct escaping, at the
    cost of losing the original quote style. For multi-line spans we emit a
    triple-double-quoted string and re-indent continuation lines so the
    closing ``\"\"\"`` lines up with the original column.
    """
    if not multi_line:
        return repr(new_description)
    # Triple-quoted block. Escape any embedded triple-double-quote.
    body = new_description.replace('"""', '\\"\\"\\"')
    if indent > 0:
        pad = " " * indent
        # The opening triple-quote begins at col_offset; subsequent lines of
        # the body need pad so the block reads naturally at that indent.
        lines = body.split("\n")
        body = ("\n" + pad).join(lines)
        return f'"""{body}"""'
    return f'"""{body}"""'


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
