"""ArtifactInstaller Protocol — how the validator gets an artifact onto disk.

v1 only ships ``HermesToolDescriptionInstaller`` (splice into an
existing tool's description via ``HermesToolSource``). v2 will add
skill installers (drop a SKILL.md into the sandboxed HERMES_HOME's
skills dir) without changing ``ClosedLoopValidator``.
"""

from __future__ import annotations

import ast
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Protocol

from evolution.tools.hermes_source import HermesToolSource
from evolution.tools.tool_source import ToolManifest


class ArtifactInstaller(Protocol):
    """Install (and uninstall) a candidate artifact in the agent's environment."""

    target_path: Path
    """The single file the installer mutates. Used by the validator for
    backup + flock + checksum book-keeping."""

    def install(self, artifact_source: Path) -> str:
        """Apply ``artifact_source`` to ``target_path``. Returns the sha256
        of ``target_path``'s on-disk bytes after installation so the
        validator can verify the file wasn't mutated between tasks."""
        ...


class HermesToolDescriptionInstaller:
    """Splice an evolved tool description into a Hermes ``*_SCHEMA`` file.

    The artifact source is the full evolved tool-module file (same
    layout as the baseline). We reuse ``HermesToolSource`` to do the
    AST parse + byte-precise splice; this class only manages the
    target_path bookkeeping and the post-install checksum.
    """

    def __init__(self, hermes_repo: Path, tool_name: str) -> None:
        self.hermes_repo = hermes_repo
        self.tool_name = tool_name
        self._source = HermesToolSource(hermes_repo / "tools")
        self.target_path = self._locate_target()

    def _locate_target(self) -> Path:
        manifest = self._source.find_manifest(self.hermes_repo / "tools")
        if manifest is None:
            raise FileNotFoundError(
                f"No Hermes tools manifest found under {self.hermes_repo / 'tools'}"
            )
        entry = manifest.find_tool(self.tool_name)
        if entry.source_location is None:
            raise ValueError(
                f"Tool {self.tool_name!r} has no statically-resolved source location"
            )
        return Path(entry.source_location[0])

    def install(self, artifact_source: Path) -> str:
        """Read ``artifact_source`` as the new full module file content,
        splice the target tool's description from it into the installed
        manifest's source. Returns sha256 of ``target_path``."""
        evolved_manifest = self._read_manifest_from_file(artifact_source)
        evolved_entry = evolved_manifest.find_tool(self.tool_name)
        # Re-read the live manifest each install so we always splice into
        # the current on-disk state (the validator may have just restored
        # the baseline before this call).
        live_manifest = self._source.find_manifest(self.hermes_repo / "tools")
        if live_manifest is None:
            raise FileNotFoundError(
                "live Hermes manifest disappeared between init and install"
            )
        self._source.apply_evolved(
            source_path=self.target_path,
            evolved_manifest=evolved_manifest,
            target_tool=self.tool_name,
            new_description=evolved_entry.description,
        )
        return sha256_of(self.target_path)

    def _read_manifest_from_file(self, artifact_source: Path) -> ToolManifest:
        # The artifact source is itself a Hermes tool-module file. Parse
        # it as a one-file manifest by giving the source a tmp root.
        if artifact_source.parent != self.hermes_repo / "tools":
            # Same parse logic but rooted at a tmp dir holding only the
            # candidate file. HermesToolSource's discovery walks .py files
            # in its root, so we materialize a single-file root.
            with tempfile.TemporaryDirectory(prefix="cl_install_") as tmp:
                tmp_root = Path(tmp)
                staged = tmp_root / artifact_source.name
                staged.write_bytes(artifact_source.read_bytes())
                manifest = HermesToolSource(tmp_root).find_manifest(tmp_root)
                if manifest is None:
                    raise ValueError(
                        f"Could not parse {artifact_source} as a Hermes tool module"
                    )
                return manifest
        return self._source.find_manifest(artifact_source.parent)  # type: ignore[return-value]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomic file write — same primitive HermesToolSource uses internally.
    Crash mid-write leaves the original file intact (or absent), never
    half-written.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def verify_python_parses(path: Path) -> None:
    """Raise SyntaxError if ``path`` doesn't parse as Python.

    Used to validate ``.cl_backup`` before trusting it for restore — a
    truncated backup from a SIGKILL-during-backup-write must not be
    silently restored over the original.
    """
    ast.parse(path.read_bytes())
