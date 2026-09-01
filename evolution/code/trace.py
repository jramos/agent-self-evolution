"""Thin repair trace for a code-evolution run.

A single-proposer repair loop does not need the GEPA-shaped lineage/dossier
machinery (Pareto frontiers, per-candidate ancestry); that would be ceremony
over a linear sequence of rounds. This records exactly what a human reviewer
needs to judge the PR: how many rounds it took, what each round saw, and the
final diff that ships.
"""

from __future__ import annotations

import json
from pathlib import Path

from evolution.code.repair import RepairResult

_OUTPUT_TAIL = 1200


def build_repair_trace(
    *,
    tool: str,
    visible_test: str,
    holdout_test: str,
    result: RepairResult,
    final_diff: str,
    containment: dict | None = None,
) -> dict:
    """Assemble the repair trace payload (see :func:`write_repair_trace`)."""
    return {
        "tool": tool,
        "visible_test": visible_test,
        "holdout_test": holdout_test,
        "fixed": result.fixed,
        "fixed_round": result.fixed_round,
        "rounds_used": len(result.rounds),
        "rounds": [
            {
                "round": r.round,
                "proposed": r.proposed,
                "freeze_violations": r.freeze_violations,
                "test_passed": r.test_passed,
                "output_tail": r.output_tail[-_OUTPUT_TAIL:],
            }
            for r in result.rounds
        ],
        "final_diff": final_diff,
        # How confined the test execution actually was. None means the env did
        # not report a posture, which is deliberately distinct from False -- an
        # unknown posture must never read as a known-unconfined one.
        "containment": containment,
    }


def containment_of(env) -> dict | None:
    """Read a run environment's containment posture, if it reports one.

    Not every env is a :class:`WorktreeEnv`: the SWE-bench env and the test
    fakes duck-type ``run_test`` alone, so the attribute may be absent.
    """
    reporter = getattr(env, "containment", None)
    return reporter() if callable(reporter) else None


def write_repair_trace(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "repair_trace.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
