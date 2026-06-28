"""Per-phase adapters — the one place phase-specific CLI knowledge lives.

Each adapter knows how to turn a ``PhaseSpec`` into a subprocess argv, where that
phase writes its run dir (a deterministic ``--output-dir`` under the orchestrator's
own run root), which fields are required, whether it has a ``--create-pr`` flag
pair, and the name of its cost-ceiling arg key (which the propose-only guard uses
to require a spend cap). Verdict reconciliation is uniform and lives in ``run.py``
— it reads the captured ``gate_decision.json``, so adapters carry no exit-code logic.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from evolution.orchestrator.spec import PhaseSpec


def _flagify(args: dict) -> list[str]:
    """Map a dict of CLI args to argv. ``{a_b: True}`` → ``--a-b``; ``False``/``None``
    omitted; lists → repeated ``--flag v``; scalars → ``--flag str(v)``."""
    out: list[str] = []
    for key, value in args.items():
        flag = "--" + str(key).replace("_", "-")
        if value is True:
            out.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, (list, tuple)):
            for item in value:
                out += [flag, str(item)]
        else:
            out += [flag, str(value)]
    return out


def _slug(name: str) -> str:
    """Filesystem-safe label for a phase target (code names are relpaths)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "phase"


@dataclass(frozen=True)
class PhaseAdapter:
    phase: str
    module: str
    name_flag: str
    extra_required: tuple[str, ...]
    create_pr_style: str  # "pair" (emits --create-pr/--no-create-pr) | "none" (no PR path)
    cost_flag: str | None  # the arg key that caps spend, or None if the phase has no such knob

    @property
    def supports_create_pr(self) -> bool:
        return self.create_pr_style == "pair"

    def required_fields(self) -> frozenset[str]:
        return frozenset({"name", *self.extra_required})

    def output_dir(self, ps: PhaseSpec, run_root: Path) -> Path:
        return Path(run_root) / "phases" / f"{ps.phase}-{_slug(ps.name)}"

    def build_argv(self, ps: PhaseSpec, run_root: Path) -> list[str]:
        argv = [sys.executable, "-m", self.module, self.name_flag, ps.name]
        argv += _flagify(ps.args)
        argv += ["--output-dir", str(self.output_dir(ps, run_root))]
        argv += self._create_pr_flags(ps)
        return argv

    def _create_pr_flags(self, ps: PhaseSpec) -> list[str]:
        # Emit the explicit off-switch on the strip path so propose-only never
        # relies on a phase's CLI defaulting create_pr to False.
        if self.create_pr_style == "pair":
            return ["--create-pr"] if ps.create_pr else ["--no-create-pr"]
        if self.create_pr_style == "none":
            return []  # phase has no PR path
        raise ValueError(f"unknown create_pr_style {self.create_pr_style!r}")


PHASE_ADAPTERS: dict[str, PhaseAdapter] = {
    "skills": PhaseAdapter(
        phase="skills", module="evolution.skills.evolve_skill", name_flag="--skill",
        extra_required=(), create_pr_style="pair", cost_flag="max_total_cost_usd",
    ),
    "tools": PhaseAdapter(
        phase="tools", module="evolution.tools.evolve_tool", name_flag="--tool",
        extra_required=("manifest",), create_pr_style="pair", cost_flag="max_total_cost_usd",
    ),
    "prompts": PhaseAdapter(
        phase="prompts", module="evolution.prompts.evolve_prompt_section", name_flag="--section",
        extra_required=("tasks",), create_pr_style="none", cost_flag="max_cost_usd",
    ),
    "code": PhaseAdapter(
        phase="code", module="evolution.code.evolve_code", name_flag="--tool",
        extra_required=("repo", "visible_test", "holdout_test"),
        # cost_flag=None: evolve_code has no spend ceiling, so the --allow-pr cost
        # guard is a no-op for code (its spend is bounded by repair_rounds).
        create_pr_style="pair", cost_flag=None,
    ),
}
