"""``python -m evolution.orchestrator`` — the propose-only cross-phase driver.

Sequences the per-subsystem evolvers from a YAML run-spec, isolating each as a
subprocess and capturing its gate verdict into a JSONL run history + summary.
Propose-only by default: every phase's create-pr is stripped unless ``--allow-pr``
is passed, and a surviving PR phase must carry a cost ceiling.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from evolution.orchestrator.adapters import PHASE_ADAPTERS
from evolution.orchestrator.run import run_pipeline
from evolution.orchestrator.spec import PHASES, RunSpec, load_spec
from evolution.orchestrator.history import render_summary_md

console = Console()


def _enforce_propose_only(spec: RunSpec, *, allow_pr: bool) -> RunSpec:
    """Strip every create_pr unless --allow-pr; with --allow-pr, require a cost
    ceiling on each surviving PR phase (mirrors monitor's --attempt-top guard)."""
    if not allow_pr:
        return replace(spec, phases=tuple(replace(p, create_pr=False) for p in spec.phases))
    for p in spec.phases:
        if not p.create_pr:
            continue
        cost_flag = PHASE_ADAPTERS[p.phase].cost_flag
        eff = {**spec.defaults, **p.args}
        # `0` is a valid (abort-on-first-call) ceiling; only absent/null is rejected.
        if cost_flag is not None and eff.get(cost_flag) is None:
            raise click.UsageError(
                f"phase '{p.phase}' (name {p.name!r}) sets create_pr with --allow-pr "
                f"but no spend ceiling; set '{cost_flag}' in its args."
            )
    return spec


@click.command()
@click.option("--spec", "spec_path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="YAML run-spec (ordered phases + per-phase args).")
@click.option("--only", multiple=True, type=click.Choice(PHASES),
              help="Run only these phases (repeatable). Default: all in spec order.")
@click.option("--resume", is_flag=True, default=False,
              help="Skip phases already recorded in the run root's run_history.jsonl.")
@click.option("--stop-on-error/--continue-on-error", default=False,
              help="Halt on a failed/aborted phase. Default: continue (phases are independent).")
@click.option("--base-output", default=None, type=click.Path(file_okay=False, path_type=Path),
              help="Orchestrator run root (ledger + summary + per-phase dirs). "
                   "Default: output/orchestrator/<timestamp>/.")
@click.option("--phase-timeout", default=None, type=click.FloatRange(min=1.0),
              help="Per-phase wall-clock timeout (seconds). A phase exceeding it is killed "
                   "and recorded as failed. Default: no timeout.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Resolve + record each phase's argv without launching subprocesses.")
@click.option("--allow-pr", is_flag=True, default=False,
              help="Honor create_pr in the spec. Without it, ALL phases are forced to "
                   "no-PR regardless of the spec (propose-only default).")
def main(spec_path, only, resume, stop_on_error, base_output, phase_timeout, dry_run, allow_pr):
    spec = load_spec(spec_path)
    spec = _enforce_propose_only(spec, allow_pr=allow_pr)

    run_root = Path(base_output) if base_output else (
        Path("output") / "orchestrator" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    console.print(f"[bold]orchestrator[/bold] — {len(spec.phases)} phase(s) → "
                  f"[dim]{run_root}[/dim]" + (" [yellow](dry-run)[/yellow]" if dry_run else ""))
    summary = run_pipeline(
        spec, run_root=run_root, only=only or None, stop_on_error=stop_on_error,
        resume=resume, dry_run=dry_run, phase_timeout=phase_timeout,
    )
    console.print(render_summary_md(summary))
    console.print(f"\n  run root: [dim]{run_root}[/dim]")


if __name__ == "__main__":
    main()
