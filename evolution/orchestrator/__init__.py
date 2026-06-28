"""Cross-phase evolution orchestrator.

A thin, propose-only dispatch shell that sequences the per-subsystem evolvers
(skills → tools → prompts → code) from a YAML run-spec, isolates each phase as a
subprocess, captures each phase's existing ``gate_decision.json`` verdict, and
writes a JSONL run history + summary. It never deploys and opens no PRs unless a
phase opts in *and* ``--allow-pr`` is passed. "Shape only": no DAG, no
data-passing between phases, no new gate logic.
"""

from evolution.orchestrator.spec import PHASES, PhaseSpec, RunSpec, SpecError, load_spec

__all__ = ["PHASES", "PhaseSpec", "RunSpec", "SpecError", "load_spec"]
