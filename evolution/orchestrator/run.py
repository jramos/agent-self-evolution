"""The sequencer: run each phase as an isolated subprocess, capture its verdict.

Control flow per phase: resolve effective args → build argv → clear any stale
run dir → run via the injected ``phase_runner`` (default: ``subprocess.run`` —
the process boundary is the fault isolation) → read the phase's
``gate_decision.json`` at the deterministic ``--output-dir`` → reconcile to a
(status, decision) → append a ledger row → continue, or halt under
``--stop-on-error``.

Status (did the phase produce a verdict cleanly) is distinct from decision (what
the gate said): a clean run whose gate rejected the candidate is
``status=passed, decision=reject`` — not an orchestrator failure.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from evolution.orchestrator.adapters import PHASE_ADAPTERS
from evolution.orchestrator.history import (
    LEDGER_NAME,
    HISTORY_SCHEMA_VERSION,
    append_row,
    build_summary,
    load_done,
    render_summary_md,
    row_key,
)
from evolution.orchestrator.spec import PhaseSpec, RunSpec

logger = logging.getLogger(__name__)

_HALT_STATUSES = {"failed", "aborted"}
# A phase is "done" for --resume only if it produced a real verdict. failed /
# aborted phases (and dry-run "skipped" rows) are re-run — resuming a crashed
# run should retry the phase that crashed, not skip it.
_RESUME_DONE_STATUSES = {"passed", "denied"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def default_phase_runner(argv: list[str], *, env: dict, cwd: Path, timeout: float | None = None) -> int:
    """Run a phase as a subprocess and return its exit code. A phase crash,
    SystemExit, or cost-ceiling abort is just a non-zero code here — it cannot
    take down the orchestrator. A wedged phase is killed at ``timeout`` seconds
    and reported as exit 124 (no gate written → reconciles to ``failed``)."""
    try:
        return subprocess.run(argv, env=env, cwd=str(cwd), timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return 124  # subprocess.run kills the child before raising


def read_gate(run_dir: Path) -> dict | None:
    """Read ``gate_decision.json`` from a phase run dir, or None if absent/unreadable."""
    path = Path(run_dir) / "gate_decision.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def reconcile(gate: dict | None, exit_code: int) -> tuple[str, str]:
    """Map a captured gate (+ exit code) to (status, decision), grounded in the
    gate file — phase exit codes are inconsistent across evolvers, so they are
    recorded for forensics but never drive status."""
    if gate is None:
        return "failed", "missing"  # no verdict produced → halts under --stop-on-error
    decision = gate.get("decision", "unknown")
    if decision == "aborted":
        return "aborted", decision  # e.g. cost-ceiling → halts
    if decision == "denied":
        return "denied", decision  # saturation no-headroom → does NOT halt
    return "passed", decision  # deploy | reject | dry_run


def _row(*, run_id, spec_index, ps: PhaseSpec, status, decision, exit_code,
         run_dir, argv, started_at, ended_at, error) -> dict:
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "run_id": run_id,
        "spec_index": spec_index,
        "phase": ps.phase,
        "name": ps.name,
        "status": status,
        "decision": decision,
        "exit_code": exit_code,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "create_pr": ps.create_pr,
        "argv": argv,
        "started_at": started_at,
        "ended_at": ended_at,
        "error": error,
    }


def run_pipeline(
    spec: RunSpec,
    *,
    run_root: Path,
    only: tuple[str, ...] | None = None,
    stop_on_error: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    phase_timeout: float | None = None,
    phase_runner=default_phase_runner,
    clock=_utcnow,
    cwd: Path = Path("."),
) -> dict:
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    ledger = run_root / LEDGER_NAME
    run_id = clock().strftime("%Y%m%d_%H%M%S")

    done: dict[str, dict] = {}
    rows: list[dict] = []
    if resume:
        done = {k: v for k, v in load_done(ledger).items()
                if v.get("status") in _RESUME_DONE_STATUSES}
        rows = sorted(done.values(), key=lambda r: r["spec_index"])
    stopped_early = False

    for spec_index, ps in enumerate(spec.phases):
        if only is not None and ps.phase not in only:
            continue
        adapter = PHASE_ADAPTERS[ps.phase]
        eff = PhaseSpec(ps.phase, ps.name, {**spec.defaults, **ps.args}, ps.create_pr)
        key = row_key({"spec_index": spec_index, "phase": ps.phase, "name": ps.name})
        if resume and key in done:
            continue

        run_dir = adapter.output_dir(eff, run_root)
        argv = adapter.build_argv(eff, run_root)

        if dry_run:
            row = _row(run_id=run_id, spec_index=spec_index, ps=eff, status="skipped",
                       decision="dry_run", exit_code=None, run_dir=run_dir, argv=argv,
                       started_at=None, ended_at=None, error=None)
            rows.append(row)
            append_row(ledger, row)
            continue

        # Clear any stale artifacts so read_gate can only ever see THIS run's
        # verdict — otherwise a phase that dies before rewriting its gate would
        # be reconciled against a prior run's (possibly "deploy") gate file.
        if run_dir.exists():
            shutil.rmtree(run_dir)

        started_at = clock().isoformat()
        error = None
        try:
            exit_code = phase_runner(argv, env=os.environ.copy(), cwd=cwd, timeout=phase_timeout)
        except Exception as exc:  # the runner itself failing != the phase failing
            exit_code, error = -1, repr(exc)
            logger.warning("orchestrator: phase_runner raised for %s/%s: %s",
                           ps.phase, ps.name, exc)
        gate = read_gate(run_dir)
        status, decision = reconcile(gate, exit_code)
        if error is not None:
            status = "aborted"
        if status != "passed":
            logger.info("orchestrator: %s/%s → %s (%s, exit=%s)",
                        ps.phase, ps.name, status, decision, exit_code)

        row = _row(run_id=run_id, spec_index=spec_index, ps=eff, status=status,
                   decision=decision, exit_code=exit_code, run_dir=run_dir, argv=argv,
                   started_at=started_at, ended_at=clock().isoformat(), error=error)
        rows.append(row)
        append_row(ledger, row)

        if stop_on_error and status in _HALT_STATUSES:
            stopped_early = True
            break

    summary = build_summary(rows, run_id=run_id, stopped_early=stopped_early)
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_root / "summary.md").write_text(render_summary_md(summary))
    return summary
