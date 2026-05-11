"""Tool description evolution — Phase 2 of the framework."""

from evolution.tools.tool_module import (
    ToolModule,
    ToolSelectionSignature,
)
from evolution.tools.tool_proposer import (
    BudgetAwareToolProposer,
    extract_and_rebuild,
)
from evolution.tools.tool_source import (
    MCPManifestSource,
    SentinelParseError,
    ToolEntry,
    ToolManifest,
    ToolSource,
    discover_tool_sources,
)

__all__ = [
    "BudgetAwareToolProposer",
    "MCPManifestSource",
    "SentinelParseError",
    "ToolEntry",
    "ToolManifest",
    "ToolModule",
    "ToolSelectionSignature",
    "ToolSource",
    "discover_tool_sources",
    "extract_and_rebuild",
]
