"""OS-level filesystem confinement for subprocesses we do not fully trust.

Two callers need this, for the same reason from opposite directions: the agent
runner invokes ``claude -p`` with its own Write/Edit tools, and the code
evolution path runs ``pytest`` against LLM-modified source. Both are "execute
this, but do not let it write outside a known root".

What the profile guarantees is narrower than isolation, and callers that record a
posture should describe it in these terms: **writes outside the named roots and
the temp roots are denied; reads, process-exec and network are unrestricted.**

Confinement is macOS-only. The *policy* when it is unavailable is deliberately
left to each caller — the agent runner refuses to run, while the code evolution
path proceeds and records an unconfined posture so its evidence never overstates
what happened.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Temp roots a confined process legitimately writes to, beyond the roots its
# caller names. macOS canonicalizes /tmp and /var/folders under /private.
_SANDBOX_TEMP_ROOTS = ("/private/tmp", "/private/var/folders", "/dev")


class SandboxUnavailableError(RuntimeError):
    """OS-level filesystem sandboxing is not available and was not waived."""


def sandbox_available() -> bool:
    """Whether OS filesystem confinement can be applied on this machine."""
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def macos_write_sandbox_profile(write_roots: list[Path]) -> str:
    """SBPL profile: allow everything, then deny all writes, then re-allow writes
    only under the given roots + temp dirs. Confines Write/Edit/Bash alike."""
    roots = list(write_roots) + [Path(p) for p in _SANDBOX_TEMP_ROOTS]
    allows = "\n".join(f'    (subpath "{r}")' for r in roots)
    return (
        "(version 1)\n"
        "(allow default)\n"
        "(deny file-write*)\n"
        f"(allow file-write*\n{allows})\n"
    )


def wrap_argv(
    argv: list[str],
    *,
    write_roots: list[Path],
    require: bool,
    available: bool | None = None,
) -> tuple[list[str], bool]:
    """Return ``(argv_to_exec, sandboxed)``, confining writes when the OS allows.

    Raises :class:`SandboxUnavailableError` when confinement is unavailable and
    ``require`` is set, so a caller that demands it never silently runs
    unconfined — the failure mode this module exists to prevent.

    Pass ``available`` to supply a posture the caller has already resolved (and
    likely already reported). Letting this function re-decide per call would make
    the recorded posture and the actual behavior two independent judgements, free
    to disagree.
    """
    if sandbox_available() if available is None else available:
        profile = macos_write_sandbox_profile(list(write_roots))
        return ["sandbox-exec", "-p", profile, *argv], True
    if require:
        raise SandboxUnavailableError(
            "No OS filesystem sandbox available on this platform "
            f"({sys.platform}); refusing to run unconfined."
        )
    return list(argv), False
