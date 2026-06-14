"""``evolve_code_campaign`` — the code-evolution measurement campaign.

Harvests real historical tool bugs, repairs each in an isolated worktree, and
verifies the repair against the upstream-fix oracle (no held-out split — the
oracle is the ground truth). Runs staged (N=8→20→50) with an organism-level
futility stop, so spend escalates only while the GREEN holds. Reports
cluster-honest organism-level estimands. This validates the loop at scale; the
novel-bug-repair product (held-out split, live feed) is deferred.

Run: ``python -m evolution.code.campaign --repo <hermes> --max-organisms 8``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from evolution.code.campaign_report import OrganismResult, build_report, wilson_lower
from evolution.code.gate import run_code_oracle_gate
from evolution.code.harvest import (
    Candidate,
    _failures,
    harvest_candidates,
    stratify,
)
from evolution.code.repair import RepairEngine, build_dspy_proposer
from evolution.code.worktree import WorktreeEnv, WorktreeError
from evolution.core.hermes_provider import resolve_default_lm

console = Console()


def _git_show(repo: Path, ref: str) -> Optional[str]:
    import subprocess

    r = subprocess.run(["git", "-C", str(repo), "show", ref],
                       capture_output=True, text=True, timeout=120)
    return r.stdout if r.returncode == 0 else None


def run_organism(
    repo: Path, c: Candidate, engine: RepairEngine, *, seeds: int, base_python: str | None,
) -> Optional[OrganismResult]:
    """Repair one harvested bug across ``seeds`` seeds in a single worktree and
    verify each against the oracle. Returns None if the candidate isn't a valid
    organism (parent doesn't fail anything the fix passes, or setup failed)."""
    try:
        env = WorktreeEnv.create(repo, base_ref=c.fix_sha, base_python=base_python)
    except WorktreeError:
        return None
    try:
        env.assert_authoritative(c.tool_path.split("/")[0])
        # Oracle (fix) is on disk at fix_sha → its own failures are env-flaky tests.
        oracle_failures = frozenset(_failures(env, c.test_path))
        parent_src = _git_show(repo, f"{c.parent_sha}:{c.tool_path}")
        if parent_src is None:
            return None
        env.write_tool(c.tool_path, parent_src)
        bug_tests = tuple(sorted(set(_failures(env, c.test_path)) - oracle_failures))
        if not bug_tests:
            return None
        seed_results: list[bool] = []
        for _ in range(seeds):
            env.write_tool(c.tool_path, parent_src)  # reset to buggy
            repair = engine.repair(env, c.tool_path, bug_tests)
            gate = run_code_oracle_gate(
                env, tool_relpath=c.tool_path, test_relpath=c.test_path,
                bug_tests=bug_tests, oracle_failures=oracle_failures,
                base_src=parent_src, repair_result=repair,
            )
            seed_results.append(bool(gate.deploy))
        return OrganismResult(tool=c.tool_path, fix_sha=c.fix_sha, seeds=seed_results)
    except WorktreeError:
        return None
    finally:
        env.destroy()


def run_campaign(
    repo: Path,
    *,
    output_dir: Path,
    max_organisms: int = 50,
    stages: tuple[int, ...] = (8, 20, 50),
    seeds: int = 3,
    max_rounds: int = 5,
    max_per_tool: int = 3,
    max_cost_usd: Optional[float] = None,
    proposer_model: Optional[str] = None,
    base_python: Optional[str] = None,
    organism_runner=None,
    candidates: Optional[list[Candidate]] = None,
) -> dict:
    """Drive the staged campaign. Resumable: organisms already in the ledger are
    skipped. Futility-stops at a stage boundary if organism-level deploy-reachable
    Wilson-lower < 0.10.

    ``organism_runner`` / ``candidates`` are injection seams for testing the
    orchestration (ledger, resume, futility, stratification) without LM spend or
    worktrees; in production both default to the real harvest + repair+gate path.
    """
    from evolution.core.lm_timing_callback import (  # noqa: PLC0415
        COST_LEDGER,
        CostCeilingExceeded,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "campaign_ledger.jsonl"
    done: set[str] = set()
    organisms: list[OrganismResult] = []
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            row = json.loads(line)
            done.add(row["fix_sha"])
            if row.get("status") == "organism":
                organisms.append(OrganismResult(row["tool"], row["fix_sha"], row["seeds"]))

    cost_summary = dict
    if organism_runner is None:
        import dspy  # noqa: PLC0415
        from evolution.core.lm_timing_callback import LMTimingCallback  # noqa: PLC0415

        rlm = resolve_default_lm(role="optimizer", explicit_model=proposer_model)
        lm = dspy.LM(rlm.model, **rlm.lm_kwargs, temperature=0.7, max_tokens=8000)
        dspy.configure(callbacks=[LMTimingCallback()])
        COST_LEDGER.reset()
        if max_cost_usd is not None:
            COST_LEDGER.set_ceiling(max_cost_usd)
        engine = RepairEngine(build_dspy_proposer(lm), max_rounds=max_rounds)
        cost_summary = COST_LEDGER.summary
        console.print(f"[bold]evolve_code_campaign[/bold] — proposer [dim]{rlm.model}[/dim], "
                      f"target {max_organisms} organisms (stages {stages})")

        def organism_runner(c: Candidate):  # noqa: F811
            return run_organism(repo, c, engine, seeds=seeds, base_python=base_python)

    if candidates is None:
        candidates = stratify(harvest_candidates(repo), max_per_tool=max_per_tool)
    console.print(f"  {len(candidates)} stratified candidates "
                  f"({len({c.tool_path for c in candidates})} tools)")

    def _append(row: dict) -> None:
        with ledger_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    aborted = False
    for c in candidates:
        if len(organisms) >= max_organisms:
            break
        if c.fix_sha in done:
            continue
        try:
            org = organism_runner(c)
        except CostCeilingExceeded:
            console.print("[yellow]cost ceiling reached — stopping with partial ledger[/yellow]")
            aborted = True
            break
        done.add(c.fix_sha)
        if org is None:
            _append({"fix_sha": c.fix_sha, "tool": c.tool_path, "status": "not_valid"})
            continue
        organisms.append(org)
        _append({"status": "organism", "tool": org.tool, "fix_sha": org.fix_sha,
                 "seeds": org.seeds, "deploy_reachable": org.deploy_reachable})
        n = len(organisms)
        n_dr = sum(1 for o in organisms if o.deploy_reachable)
        console.print(f"  [{n}/{max_organisms}] {org.tool.split('/')[-1]:<22} "
                      f"correct {org.n_correct}/{seeds}  "
                      f"(deploy-reachable {n_dr}/{n}, Wilson-lo {wilson_lower(n_dr, n):.2f})")
        if n in stages:
            lo = wilson_lower(n_dr, n)
            console.print(f"  [bold]stage N={n}[/bold]: deploy-reachable {n_dr}/{n}, "
                          f"Wilson-lower {lo:.3f}")
            if lo < 0.10:
                console.print("  [red]FUTILITY STOP[/red] — Wilson-lower < 0.10; "
                              "GREEN not supported at this N.")
                break

    report = build_report(organisms)
    report["aborted_on_cost"] = aborted
    report["cost_summary"] = cost_summary()
    (output_dir / "campaign_report.json").write_text(json.dumps(report, indent=2))
    console.print(f"\n[bold]verdict: {report['verdict']}[/bold] — deploy-reachable "
                  f"{report['deploy_reachable']['k']}/{report['deploy_reachable']['n']} "
                  f"Wilson {tuple(round(x, 2) for x in report['deploy_reachable']['wilson'])}")
    console.print(f"  artifacts: [dim]{output_dir}[/dim]")
    return report


@click.command()
@click.option("--repo", "repo_root", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Target git repo to harvest + repair (e.g. the Hermes checkout).")
@click.option("--max-organisms", default=8, type=click.IntRange(min=1),
              help="Stop after this many valid organisms (default 8 = the pilot).")
@click.option("--seeds", default=3, type=click.IntRange(min=1),
              help="Repair attempts per organism (default 3; the organism is the honest unit).")
@click.option("--repair-rounds", default=5, type=click.IntRange(min=1))
@click.option("--max-per-tool", default=3, type=click.IntRange(min=1),
              help="Cap organisms per tool so the sample stays diverse (default 3).")
@click.option("--max-cost-usd", default=None, type=click.FloatRange(min=0.0),
              help="Abort cleanly when cumulative LM cost exceeds this.")
@click.option("--proposer-model", default=None)
@click.option("--base-python", default=None,
              help="Interpreter for the isolated venv (default: the repo's venv/.venv).")
@click.option("--output-dir", default=None, type=click.Path(file_okay=False, path_type=Path))
def main(repo_root, max_organisms, seeds, repair_rounds, max_per_tool, max_cost_usd,
         proposer_model, base_python, output_dir):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output") / "code_campaign" / ts
    run_campaign(repo_root, output_dir=Path(output_dir), max_organisms=max_organisms,
                 seeds=seeds, max_rounds=repair_rounds, max_per_tool=max_per_tool,
                 max_cost_usd=max_cost_usd, proposer_model=proposer_model,
                 base_python=base_python)


if __name__ == "__main__":
    main()
