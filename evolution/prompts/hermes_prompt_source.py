"""HermesPromptSource — read/write named string constants in Hermes prompt_builder.py.

Walks ``agent/prompt_builder.py`` for top-level ``NAME = "..."`` (or
concatenated-string) assignments. v1 supports string-typed constants
only; dict-typed constants (like ``PLATFORM_HINTS``) raise KeyError on
read.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from evolution.prompts.prompt_source import SectionDescriptor

logger = logging.getLogger(__name__)


class HermesPromptSource:
    """Read/write named string constants in Hermes prompt_builder.py."""

    name = "hermes_prompt_source"

    def __init__(self, hermes_repo: Path) -> None:
        self.hermes_repo = Path(hermes_repo)
        self.prompt_builder_path = self.hermes_repo / "agent" / "prompt_builder.py"
        if not self.prompt_builder_path.is_file():
            raise FileNotFoundError(
                f"prompt_builder.py not found at {self.prompt_builder_path}"
            )

    def read(self, section_name: str) -> str:
        constants = self._parse_string_constants()
        if section_name not in constants:
            raise KeyError(
                f"section {section_name!r} not found in {self.prompt_builder_path} "
                f"(v1 only supports top-level string-typed constants). "
                f"Available: {sorted(constants)}"
            )
        return constants[section_name][0]

    def _parse_string_constants(self) -> dict[str, tuple[str, ast.Constant]]:
        """Return ``{name: (value, value_ast_node)}`` for every top-level
        string-typed assignment in prompt_builder.py.

        Concatenated-string forms like ``X = ("a" "b" "c")`` are folded to
        a single ``ast.Constant`` by the parser, so they read back as one
        string. The AST node is retained so ``write`` can splice by byte
        offset.
        """
        source = self.prompt_builder_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(self.prompt_builder_path))
        out: dict[str, tuple[str, ast.Constant]] = {}
        for node in tree.body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                out[target.id] = (value.value, value)
        return out
