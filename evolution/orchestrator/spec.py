"""Run-spec loader + validator for the cross-phase orchestrator.

A run-spec is a YAML file naming an ordered list of phases, each with a target
``name`` and a dict of phase-specific ``args`` (keys are the phase CLI's own flag
names, snake_case), plus an optional shared ``defaults`` block merged into every
phase (a phase's own args win on conflict).

All structural problems raise ``SpecError`` at load time — the orchestrator fails
fast before launching any subprocess, mirroring monitor's ``--attempt-top`` guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Canonical phase order. monitor is intentionally excluded — it is the
# propose-only discovery sentinel, not a candidate-improving evolver.
PHASES = ("skills", "tools", "prompts", "code")

_ALLOWED_TOP = {"defaults", "phases"}
_ALLOWED_PHASE_KEYS = {"phase", "name", "args", "create_pr"}


class SpecError(ValueError):
    """A run-spec is structurally invalid (caught at load, before any launch)."""


@dataclass(frozen=True)
class PhaseSpec:
    phase: str
    name: str
    args: dict
    create_pr: bool = False


@dataclass(frozen=True)
class RunSpec:
    phases: tuple[PhaseSpec, ...]
    defaults: dict


def load_spec(path: Path) -> RunSpec:
    """Parse + validate a YAML run-spec into a ``RunSpec``. Raises ``SpecError``."""
    # Imported here (not at module scope) to keep the spec layer free of any
    # per-phase knowledge except what the adapters declare.
    from evolution.orchestrator.adapters import PHASE_ADAPTERS

    try:
        data = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as exc:
        raise SpecError(f"could not parse YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise SpecError("run-spec must be a mapping with a 'phases' list")
    unknown = set(data) - _ALLOWED_TOP
    if unknown:
        raise SpecError(f"unknown top-level key(s): {sorted(unknown)}")

    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise SpecError("'defaults' must be a mapping")
    if "create_pr" in defaults or "create-pr" in defaults:
        raise SpecError("'create_pr' may not appear in 'defaults'; set it per phase")

    raw_phases = data.get("phases")
    if not isinstance(raw_phases, list) or not raw_phases:
        raise SpecError("'phases' must be a non-empty list")

    phases = tuple(
        _validate_phase(raw, index=i, adapters=PHASE_ADAPTERS)
        for i, raw in enumerate(raw_phases)
    )
    return RunSpec(phases=phases, defaults=defaults)


def _validate_phase(raw, *, index: int, adapters) -> PhaseSpec:
    where = f"phases[{index}]"
    if not isinstance(raw, dict):
        raise SpecError(f"{where}: must be a mapping")
    unknown = set(raw) - _ALLOWED_PHASE_KEYS
    if unknown:
        raise SpecError(f"{where}: unknown key(s) {sorted(unknown)}")

    phase = raw.get("phase")
    if phase not in PHASES:
        raise SpecError(f"{where}: 'phase' must be one of {list(PHASES)}, got {phase!r}")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SpecError(f"{where}: 'name' must be a non-empty string")

    args = raw.get("args", {}) or {}
    if not isinstance(args, dict):
        raise SpecError(f"{where}: 'args' must be a mapping")
    if "create_pr" in args or "create-pr" in args:
        raise SpecError(
            f"{where}: 'create_pr' is a top-level phase field, not an 'args' key"
        )
    if "output_dir" in args or "output-dir" in args:
        raise SpecError(
            f"{where}: 'output_dir' is set by the orchestrator, not an 'args' key"
        )

    create_pr = raw.get("create_pr", False)
    if not isinstance(create_pr, bool):
        raise SpecError(f"{where}: 'create_pr' must be a boolean")

    adapter = adapters[phase]
    if create_pr and not adapter.supports_create_pr:
        raise SpecError(
            f"{where}: phase '{phase}' does not support create_pr "
            "(its PR automation is not available)"
        )
    missing = [f for f in adapter.required_fields() if f != "name" and f not in args]
    if missing:
        raise SpecError(f"{where}: phase '{phase}' requires args {sorted(missing)}")

    return PhaseSpec(phase=phase, name=name, args=dict(args), create_pr=create_pr)
