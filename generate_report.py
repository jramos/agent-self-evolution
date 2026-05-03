"""Render a phase-N validation report PDF from a prose YAML + a run dir.

Usage:
    python generate_report.py \\
        --run output/<skill>/<timestamp>/ \\
        --prose reports/phase1_prose.yaml \\
        [--out reports/phase1_validation_report.pdf]

Numbers (sizes, scores, bootstrap CI, decision, knee-point pick) are
extracted from the run dir's gate_decision.json + metrics.json + run.log
(LM call counts grep'd from the timing-callback lines). Editorial prose +
table contents come from the YAML. Every text block in the YAML may use
``{placeholder}``-style ``str.format`` substitutions; available keys are
documented in ``_extract_run_data``.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOGO = REPO_ROOT / "assets" / "dna.png"


def _extract_run_data(run_dir: Path) -> dict[str, Any]:
    """Pull all numbers the renderer needs from a run dir.

    Reads gate_decision.json (always present) + metrics.json (deploy only) +
    run.log (LM call counts grep'd from timing-callback lines).
    """
    gate = json.loads((run_dir / "gate_decision.json").read_text())
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}

    bootstrap = gate["bootstrap"]
    knee = gate.get("knee_point", {})
    dataset = gate["dataset"]

    # run.log only captures the lm_timing_callback logger (LM #N start/end);
    # dspy.teleprompt.gepa "Iteration N: ..." lines go to stdout and aren't
    # in the per-run artifact. So we count LM calls but not GEPA iterations.
    log = (run_dir / "run.log").read_text() if (run_dir / "run.log").is_file() else ""
    lm_calls_judge = len(re.findall(r"LM #\d+ start.*model=openai/gpt-4\.1-mini", log))
    lm_calls_reflection = len(re.findall(r"LM #\d+ start.*model=openai/gpt-5-mini", log))

    skill_name = metrics.get("skill_name") or run_dir.parent.name

    avg_baseline = float(gate["avg_baseline"])
    avg_evolved = float(gate["avg_evolved"])
    improvement = avg_evolved - avg_baseline
    growth_pct = float(gate["growth_pct"])

    decision = gate["decision"]
    decision_rule = gate.get("decision_rule_used", "no_regression_only")

    bootstrap_lower = float(bootstrap["lower_bound"])
    bootstrap_upper = float(bootstrap["upper_bound"])
    if bootstrap_lower > 0 and bootstrap_upper > 0:
        ci_zero_phrase = "excluding zero"
        bootstrap_interpretation = (
            "both ends positive, so we have ≥90% confidence the improvement is real, "
            "not sampling noise"
        )
    elif bootstrap_lower < 0 and bootstrap_upper > 0:
        ci_zero_phrase = "straddling zero"
        bootstrap_interpretation = (
            "the CI straddles zero, so we cannot statistically distinguish the "
            "evolved skill from baseline"
        )
    else:
        ci_zero_phrase = "below zero"
        bootstrap_interpretation = (
            "both ends negative, so we have ≥90% confidence the evolved skill "
            "regresses on holdout"
        )

    if decision == "deploy":
        decision_phrase = (
            "non-inferiority gate; bootstrap CI excludes zero"
            if decision_rule == "non_inferiority"
            else "passed regression-free check"
        )
    else:
        decision_phrase = "rejected by paired-bootstrap regression check"

    knee_picked_idx = knee.get("picked_idx")
    knee_default_idx = knee.get("gepa_default_idx")
    if knee_picked_idx is not None and knee_picked_idx == knee_default_idx:
        knee_default_match_phrase = (
            " and is also the candidate GEPA's own default selector would have chosen"
            " — the val-best and GEPA-default converged here"
        )
    else:
        knee_default_match_phrase = ""

    return {
        "skill_name": skill_name,
        "baseline_chars": int(gate["baseline_chars"]),
        "evolved_chars": int(gate["evolved_chars"]),
        "growth_pct": growth_pct,
        "growth_abs_pct": abs(growth_pct),
        "avg_baseline": avg_baseline,
        "avg_evolved": avg_evolved,
        "improvement": improvement,
        "improvement_pp": improvement * 100,
        "bootstrap_mean": float(bootstrap["mean"]),
        "bootstrap_lower": bootstrap_lower,
        "bootstrap_upper": bootstrap_upper,
        "n_holdout": int(dataset["size_holdout"]),
        "n_train": int(dataset["size_train"]),
        "n_val": int(dataset["size_val"]),
        "n_examples": int(dataset["size_total"]),
        "decision": decision,
        "decision_upper": "DEPLOYED" if decision == "deploy" else "REJECTED",
        # Internal rule name uses underscores ("non_inferiority"); the
        # user-facing flag name uses hyphens ("non-inferiority"). Prose
        # references the latter.
        "decision_rule": decision_rule.replace("_", "-"),
        "decision_phrase": decision_phrase,
        "ci_zero_phrase": ci_zero_phrase,
        "bootstrap_interpretation": bootstrap_interpretation,
        "elapsed_seconds": int(metrics.get("elapsed_seconds", 0)),
        "elapsed_minutes": int(metrics.get("elapsed_seconds", 0) // 60),
        "lm_calls_judge": lm_calls_judge,
        "lm_calls_reflection": lm_calls_reflection,
        "lm_calls_total": lm_calls_judge + lm_calls_reflection,
        "knee_picked_idx": knee_picked_idx,
        "knee_picked_val_score": float(knee.get("picked_val_score", 0.0)),
        "knee_picked_rank": int(knee.get("picked_val_rank_in_band", 0)),
        "knee_picked_body_chars": int(knee.get("picked_body_chars", 0)),
        "knee_band_size": int(knee.get("band_size", 0)),
        "knee_default_idx": knee_default_idx,
        "knee_default_match_phrase": knee_default_match_phrase,
    }


def _load_eval_examples(run_dir: Path, skill_name: str, n: int = 3) -> list[tuple[str, str]]:
    """Pull ``n`` example (task_input, expected_behavior) pairs from the
    run's train.jsonl. Falls back to an empty list if the dataset isn't on
    disk (e.g., gitignored output dir on a fresh checkout).
    """
    candidates = [
        REPO_ROOT / "datasets" / "skills" / skill_name / "train.jsonl",
        run_dir / "train.jsonl",
    ]
    for path in candidates:
        if path.is_file():
            examples: list[tuple[str, str]] = []
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                ex = json.loads(line)
                examples.append((ex["task_input"], ex["expected_behavior"]))
                if len(examples) >= n:
                    break
            return examples
    return []


def _wrap(text: str, width: int = 42) -> str:
    """Newline-wrap a short string at ~width chars for table-cell display."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def _fmt(template: str, ctx: dict[str, Any]) -> str:
    """Apply ``str.format`` substitution but tolerate missing keys gracefully.

    A missing key keeps the literal ``{placeholder}`` instead of raising —
    useful when prose YAML mentions a field a particular run dir doesn't
    expose (e.g., metrics.json absent on a rejection-path render).
    """
    class _Safe(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_Safe(ctx))


def _styles() -> Any:
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name='Title2', parent=base['Title'],
        fontSize=24, spaceAfter=6, textColor=HexColor('#1a1a2e'),
    ))
    base.add(ParagraphStyle(
        name='Subtitle', parent=base['Normal'],
        fontSize=14, textColor=HexColor('#555555'),
        alignment=TA_CENTER, spaceAfter=20,
    ))
    base.add(ParagraphStyle(
        name='SectionHead', parent=base['Heading1'],
        fontSize=16, spaceBefore=24, spaceAfter=10,
        textColor=HexColor('#1a1a2e'),
    ))
    base.add(ParagraphStyle(
        name='SubSection', parent=base['Heading2'],
        fontSize=13, spaceBefore=16, spaceAfter=8,
        textColor=HexColor('#2d2d44'),
    ))
    base.add(ParagraphStyle(
        name='BodyJust', parent=base['Normal'],
        fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        name='Metric', parent=base['Normal'],
        fontSize=11, leading=16, leftIndent=20, spaceAfter=4,
    ))
    base.add(ParagraphStyle(
        name='Footer', parent=base['Normal'],
        fontSize=8, textColor=HexColor('#999999'), alignment=TA_CENTER,
    ))
    return base


def _title_page(prose: dict, styles, logo_path: Path) -> list:
    meta = prose["meta"]
    flow: list = [Spacer(1, 1.5 * inch)]

    title_para = Paragraph(meta["title"], styles['Title2'])
    if logo_path.is_file():
        # Two-cell row: logo on the left, title on the right; centered as a unit.
        title_row = [[Image(str(logo_path), width=0.45 * inch, height=0.45 * inch), title_para]]
        title_table = Table(title_row, colWidths=[0.6 * inch, 4.5 * inch])
        title_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('ALIGN', (0, 0), (0, 0), 'RIGHT'),
        ]))
        wrapper = Table([[title_table]], colWidths=[6.5 * inch])
        wrapper.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))
        flow.append(wrapper)
    else:
        flow.append(title_para)

    flow.append(Paragraph(meta["subtitle"], styles['Subtitle']))
    flow.append(Spacer(1, 0.3 * inch))
    flow.append(HRFlowable(width="60%", thickness=1, color=HexColor('#cccccc')))
    flow.append(Spacer(1, 0.3 * inch))
    flow.append(Paragraph(
        f"Date: {datetime.now().strftime('%B %d, %Y')}",
        ParagraphStyle('DateStyle', parent=styles['Normal'], alignment=TA_CENTER,
                       fontSize=11, textColor=HexColor('#777777')),
    ))
    if meta.get("organization"):
        flow.append(Paragraph(
            f"Organization: {meta['organization']}",
            ParagraphStyle('OrgStyle', parent=styles['Normal'], alignment=TA_CENTER,
                           fontSize=11, textColor=HexColor('#777777')),
        ))
    flow.append(Paragraph(
        f"Repository: {meta['repository']}",
        ParagraphStyle('RepoStyle', parent=styles['Normal'], alignment=TA_CENTER,
                       fontSize=10, textColor=HexColor('#999999')),
    ))
    flow.append(PageBreak())
    return flow


def _key_result_box(prose: dict, ctx: dict) -> Table:
    box_cfg = prose["key_result_box"]
    rows = [[_fmt(box_cfg["title_template"], ctx)]]
    rows += [[_fmt(r, ctx)] for r in box_cfg["rows"]]
    table = Table(rows, colWidths=[5.5 * inch])

    if ctx["decision"] == "deploy":
        body_bg = HexColor('#e8f5e9')
        body_fg = HexColor('#2e7d32')
    else:
        body_bg = HexColor('#fff8e1')
        body_fg = HexColor('#5d4037')

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND', (0, 1), (-1, -1), body_bg),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 1), (-1, -1), body_fg),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 1, HexColor('#1a1a2e')),
    ]))
    return table


def _executive_summary(prose: dict, ctx: dict, styles) -> list:
    es = prose["executive_summary"]
    return [
        Paragraph("Executive Summary", styles['SectionHead']),
        Paragraph(_fmt(es["framework_intro"], ctx), styles['BodyJust']),
        Paragraph(_fmt(es["run_summary"], ctx), styles['BodyJust']),
        Spacer(1, 0.2 * inch),
        _key_result_box(prose, ctx),
        Spacer(1, 0.3 * inch),
    ]


def _highlight_table(
    header: list[str],
    rows: list[list[str]],
    col_widths: list[float],
    highlight_row: int | None = None,
    highlight_color: str = '#fff9c4',
) -> Table:
    data = [header] + rows
    table = Table(data, colWidths=col_widths)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]
    if highlight_row is not None:
        # +1 because the header is row 0.
        idx = highlight_row + 1
        style.append(('BACKGROUND', (0, idx), (-1, idx), HexColor(highlight_color)))
    table.setStyle(TableStyle(style))
    return table


def _background(prose: dict, ctx: dict, styles) -> list:
    bg = prose["background"]
    layers = bg["layers"]
    return [
        Paragraph("Background", styles['SectionHead']),
        Paragraph(_fmt(bg["intro"], ctx), styles['BodyJust']),
        _highlight_table(
            header=layers["header"],
            rows=layers["rows"],
            col_widths=[1.2 * inch, 2.3 * inch, 2.5 * inch],
            highlight_row=layers.get("highlight_row"),
        ),
        Spacer(1, 0.15 * inch),
        Paragraph(_fmt(bg["closing"], ctx), styles['BodyJust']),
    ]


def _approach(prose: dict, ctx: dict, styles) -> list:
    ap = prose["approach"]
    engines = ap["engines"]
    flow = [
        Paragraph("Approach: Evolutionary Skill Optimization", styles['SectionHead']),
        Paragraph("Three Optimization Engines", styles['SubSection']),
        _highlight_table(
            header=engines["header"],
            rows=engines["rows"],
            col_widths=[1.4 * inch, 2.0 * inch, 0.8 * inch, 1.8 * inch],
        ),
        Paragraph(_fmt(ap["gepa_narrative"], ctx), styles['BodyJust']),
        Paragraph("The Optimization Pipeline", styles['SubSection']),
    ]
    for i, step in enumerate(ap["pipeline_steps"], start=1):
        flow.append(Paragraph(f"{i}. {_fmt(step, ctx)}", styles['Metric']))
    flow += [
        Spacer(1, 0.1 * inch),
        Paragraph(_fmt(ap["cost_paragraph"], ctx), styles['BodyJust']),
    ]
    return flow


def _experiment(prose: dict, ctx: dict, styles, examples: list[tuple[str, str]]) -> list:
    exp = prose["experiment"]
    overrides = exp["config_overrides"]

    config_rows = [
        ['Target Skill', _fmt(overrides["target_skill_label"], ctx)],
        ['Baseline Size', f'{ctx["baseline_chars"]:,} characters'],
        ['Optimizer LM', overrides["optimizer_lm"]],
        ['Reflection LM (GEPA)', overrides["reflection_lm"]],
        ['Eval / Judge LM', overrides["eval_judge_lm"]],
        ['Optimizer', overrides["optimizer_label"]],
        ['Synthetic Eval Set',
         f'{ctx["n_examples"]} examples ({ctx["n_train"]} train / {ctx["n_val"]} val / {ctx["n_holdout"]} holdout)'],
        ['Total Optimization Time',
         f'{ctx["elapsed_seconds"]:,} seconds (~{ctx["elapsed_minutes"]} minutes)'],
        ['Total LM Calls',
         f'~{ctx["lm_calls_total"]:,} ({ctx["lm_calls_judge"]:,} gpt-4.1-mini + {ctx["lm_calls_reflection"]} gpt-5-mini)'],
        ['Quality Gate', overrides["quality_gate_label"]],
        ['Knee-point Strategy', overrides["knee_point_strategy_label"]],
    ]
    config_table = Table([['Parameter', 'Value']] + config_rows, colWidths=[2.2 * inch, 3.8 * inch])
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9.5),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
    ]))

    examples_rows = (
        [[_wrap(t, 38), _wrap(b, 38)] for t, b in examples]
        or [["(no train.jsonl found)", ""]]
    )
    examples_table = Table(
        [['Task Input', 'Expected Behavior (Rubric)']] + examples_rows,
        colWidths=[2.5 * inch, 3.5 * inch],
    )
    examples_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))

    return [
        Paragraph("Phase 1 Experiment", styles['SectionHead']),
        Paragraph("Configuration", styles['SubSection']),
        config_table,
        Paragraph("Evaluation Dataset", styles['SubSection']),
        Paragraph(_fmt(exp["dataset_intro"], ctx), styles['BodyJust']),
        examples_table,
        Paragraph("Fitness Function", styles['SubSection']),
        Paragraph(_fmt(exp["fitness_intro"], ctx), styles['BodyJust']),
        Paragraph(
            f"<font face='Courier' size=9>{exp['fitness_formula']}</font>",
            ParagraphStyle('Formula', parent=styles['Normal'], alignment=TA_CENTER,
                           spaceBefore=8, spaceAfter=8, fontSize=10),
        ),
        Paragraph(_fmt(exp["fitness_closing"], ctx), styles['BodyJust']),
    ]


def _results(prose: dict, ctx: dict, styles) -> list:
    res = prose["results"]
    if ctx["decision"] == "deploy":
        decision_cell = "DEPLOYED"
        decision_note = "CI excludes 0" if ctx["bootstrap_lower"] > 0 else "non-inferiority"
        accent_bg = HexColor('#e8f5e9')
        accent_fg = HexColor('#2e7d32')
    else:
        decision_cell = "REJECTED"
        decision_note = "regression check"
        accent_bg = HexColor('#fff8e1')
        accent_fg = HexColor('#5d4037')

    results_rows = [
        ['Metric', 'Baseline', 'Evolved (knee-point pick)', 'Δ'],
        ['Body size (chars)', f'{ctx["baseline_chars"]:,}', f'{ctx["evolved_chars"]:,}', f'{ctx["growth_pct"]:+.1%}'],
        [f'Avg holdout score (n={ctx["n_holdout"]})',
         f'{ctx["avg_baseline"]:.3f}', f'{ctx["avg_evolved"]:.3f}', f'{ctx["improvement"]:+.3f}'],
        ['Bootstrap mean diff', '—', f'{ctx["bootstrap_mean"]:+.3f}', '—'],
        ['Bootstrap 90% CI lower', '—', f'{ctx["bootstrap_lower"]:+.3f}', '—'],
        ['Bootstrap 90% CI upper', '—', f'{ctx["bootstrap_upper"]:+.3f}', '—'],
        ['Decision', '—', decision_cell, decision_note],
    ]
    results_table = Table(results_rows, colWidths=[1.9 * inch, 1.3 * inch, 1.7 * inch, 1.1 * inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (2, 1), (2, 1), accent_bg),
        ('TEXTCOLOR', (2, 1), (2, 1), accent_fg),
        ('FONTNAME', (2, 1), (2, 1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), accent_bg),
        ('TEXTCOLOR', (-1, -1), (-1, -1), accent_fg),
        ('FONTNAME', (-1, -1), (-1, -1), 'Helvetica-Bold'),
    ]))

    flow = [
        Paragraph("Results", styles['SectionHead']),
        results_table,
        Spacer(1, 0.15 * inch),
        Paragraph(_fmt(res["narrative"], ctx), styles['BodyJust']),
        Paragraph("How the Result Was Produced", styles['SubSection']),
        Paragraph(_fmt(res["how_produced_intro"], ctx), styles['BodyJust']),
    ]
    for i, step in enumerate(res["how_produced_steps"], start=1):
        flow.append(Paragraph(f"{i}. {_fmt(step, ctx)}", styles['Metric']))
    flow += [
        Spacer(1, 0.1 * inch),
        Paragraph(_fmt(res["how_produced_closing"], ctx), styles['BodyJust']),
    ]
    return flow


def _safety(prose: dict, ctx: dict, styles) -> list:
    sf = prose["safety"]
    table = sf["table"]
    return [
        Paragraph("Safety and Guardrails", styles['SectionHead']),
        Paragraph(_fmt(sf["intro"], ctx), styles['BodyJust']),
        _highlight_table(
            header=table["header"],
            rows=table["rows"],
            col_widths=[1.6 * inch, 2.8 * inch, 1.1 * inch],
        ),
        Spacer(1, 0.1 * inch),
        Paragraph(_fmt(sf["closing"], ctx), styles['BodyJust']),
    ]


def _roadmap(prose: dict, ctx: dict, styles) -> list:
    rm = prose["roadmap"]
    table = rm["table"]
    return [
        Paragraph("Roadmap", styles['SectionHead']),
        _highlight_table(
            header=table["header"],
            rows=table["rows"],
            col_widths=[0.9 * inch, 1.6 * inch, 1.3 * inch, 1.0 * inch, 1.0 * inch],
            highlight_row=table.get("highlight_row"),
            highlight_color='#e8f5e9',
        ),
        Spacer(1, 0.15 * inch),
        Paragraph(_fmt(rm["closing"], ctx), styles['BodyJust']),
    ]


def _next_steps(prose: dict, ctx: dict, styles) -> list:
    flow = [Paragraph("Immediate Next Steps", styles['SectionHead'])]
    for i, step in enumerate(prose["next_steps"], start=1):
        flow.append(Paragraph(f"{i}. {_fmt(step, ctx)}", styles['Metric']))
    return flow


def _footer(prose: dict, styles) -> list:
    meta = prose["meta"]
    parts = [meta['title'], meta['subtitle'], datetime.now().strftime('%B %d, %Y')]
    if meta.get("organization"):
        parts.append(meta["organization"])
    return [
        Spacer(1, 0.5 * inch),
        HRFlowable(width="100%", thickness=0.5, color=HexColor('#cccccc')),
        Spacer(1, 0.1 * inch),
        Paragraph(" — ".join(parts), styles['Footer']),
        Paragraph(meta['repository'], styles['Footer']),
    ]


def build_report(
    *,
    run_dir: Path,
    prose_path: Path,
    output_path: Path,
    logo_path: Path = DEFAULT_LOGO,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prose = yaml.safe_load(prose_path.read_text())
    ctx = _extract_run_data(run_dir)
    examples = _load_eval_examples(run_dir, ctx["skill_name"], n=3)
    styles = _styles()

    doc = SimpleDocTemplate(
        str(output_path), pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=1 * inch, rightMargin=1 * inch,
    )
    flow: list = []
    flow += _title_page(prose, styles, logo_path)
    flow += _executive_summary(prose, ctx, styles)
    flow += _background(prose, ctx, styles)
    flow += _approach(prose, ctx, styles)
    flow += _experiment(prose, ctx, styles, examples)
    flow += _results(prose, ctx, styles)
    flow += _safety(prose, ctx, styles)
    flow += _roadmap(prose, ctx, styles)
    flow += _next_steps(prose, ctx, styles)
    flow += _footer(prose, styles)

    doc.build(flow)
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run", type=Path,
        default=Path("output/arxiv/20260503_105337"),
        help="Path to the run dir (must contain gate_decision.json + metrics.json + run.log)",
    )
    parser.add_argument(
        "--prose", type=Path,
        default=Path("reports/phase1_prose.yaml"),
        help="Path to the prose YAML file",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("reports/phase1_validation_report.pdf"),
        help="Where to write the PDF",
    )
    parser.add_argument(
        "--logo", type=Path,
        default=DEFAULT_LOGO,
        help="PNG asset for the title-page logo (set to a non-existent path to omit)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    path = build_report(
        run_dir=args.run, prose_path=args.prose,
        output_path=args.out, logo_path=args.logo,
    )
    print(f"Report generated: {path}")
