"""Generate the Phase 1 validation report as PDF."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.lib import colors
from datetime import datetime


def build_report(output_path: str = "reports/phase1_validation_report.pdf"):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(
        name='Title2',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=6,
        textColor=HexColor('#1a1a2e'),
    ))
    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=HexColor('#555555'),
        alignment=TA_CENTER,
        spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        name='SectionHead',
        parent=styles['Heading1'],
        fontSize=16,
        spaceBefore=24,
        spaceAfter=10,
        textColor=HexColor('#1a1a2e'),
    ))
    styles.add(ParagraphStyle(
        name='SubSection',
        parent=styles['Heading2'],
        fontSize=13,
        spaceBefore=16,
        spaceAfter=8,
        textColor=HexColor('#2d2d44'),
    ))
    styles.add(ParagraphStyle(
        name='BodyJust',
        parent=styles['Normal'],
        fontSize=10.5,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name='CodeBlock',
        parent=styles['Code'],
        fontSize=9,
        leading=12,
        backColor=HexColor('#f5f5f5'),
        borderColor=HexColor('#dddddd'),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name='Metric',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        leftIndent=20,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name='Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=HexColor('#999999'),
        alignment=TA_CENTER,
    ))

    story = []

    # ── TITLE PAGE ──────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("🧬 Agent Self-Evolution", styles['Title2']))
    story.append(Paragraph("Phase 1 Validation Report", styles['Subtitle']))
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="60%", thickness=1, color=HexColor('#cccccc')))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        f"Date: {datetime.now().strftime('%B %d, %Y')}",
        ParagraphStyle('DateStyle', parent=styles['Normal'], alignment=TA_CENTER,
                       fontSize=11, textColor=HexColor('#777777'))
    ))
    story.append(Paragraph(
        "Organization: Nous Research",
        ParagraphStyle('OrgStyle', parent=styles['Normal'], alignment=TA_CENTER,
                       fontSize=11, textColor=HexColor('#777777'))
    ))
    story.append(Paragraph(
        "Repository: github.com/jramos/agent-self-evolution",
        ParagraphStyle('RepoStyle', parent=styles['Normal'], alignment=TA_CENTER,
                       fontSize=10, textColor=HexColor('#999999'))
    ))

    story.append(PageBreak())

    # ── EXECUTIVE SUMMARY ───────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", styles['SectionHead']))
    story.append(Paragraph(
        "Agent Self-Evolution is a standalone optimization pipeline that uses DSPy and GEPA "
        "(Genetic-Pareto Prompt Evolution) to automatically improve an agent's skills, "
        "tool descriptions, system prompts, and code through evolutionary search — all via "
        "API calls with no GPU training required. Originally built for Hermes Agent; now "
        "works for any agent framework that emits SKILL.md files.",
        styles['BodyJust']
    ))
    story.append(Paragraph(
        "This report documents the Phase 1 validation of the skill evolution pipeline as it stands "
        "today, after the rebrand from Hermes-only to a multi-framework optimizer. Using DSPy GEPA "
        "with OpenAI's gpt-4.1 / gpt-5-mini / gpt-4.1-mini stack, we re-evolved the <b>arxiv</b> skill "
        "(the same target as the original Phase 1 experiment) end-to-end through the current pipeline. "
        "The evolution achieved a <b>40.0% size reduction</b> AND a <b>+5.4 point holdout improvement</b> "
        "(0.908 → 0.962) — the paired-bootstrap 90% CI on the per-example diff is "
        "[+0.017, +0.095], excluding zero. The deploy gate accepted the candidate under the new "
        "<i>non-inferiority</i> rule, and the knee-point selector picked the highest-val candidate "
        "within the ε-band rather than the smallest, after a prior parsimony-first bias was "
        "corrected this cycle.",
        styles['BodyJust']
    ))

    # Key result box
    result_data = [
        ['KEY RESULT — arxiv (Hermes Agent), GEPA light budget'],
        ['Size:   10,036 → 6,017 chars   (−40.0%)'],
        ['Holdout score:   0.908 → 0.962   (Δ +0.054, 90% CI [+0.017, +0.095], n=65)'],
        ['Decision:   DEPLOYED   (non-inferiority gate; bootstrap CI excludes zero)'],
    ]
    result_table = Table(result_data, colWidths=[5.5 * inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#e8f5e9')),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#2e7d32')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 1, HexColor('#1a1a2e')),
    ]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(result_table)
    story.append(Spacer(1, 0.3 * inch))

    # ── BACKGROUND ──────────────────────────────────────────────────────
    story.append(Paragraph("Background", styles['SectionHead']))
    story.append(Paragraph(
        "Hermes Agent is a general-purpose AI agent built by Nous Research that uses tool-calling "
        "LLMs to complete tasks via terminal commands, file operations, web search, code execution, "
        "and more. The Self-Evolution framework was originally built for it but has since been "
        "generalized: a pluggable <font face='Courier'>SkillSource</font> protocol now discovers skills "
        "in the Hermes Agent layout, the Claude Code plugin cache, or any flat local directory — "
        "the same optimizer can target any of them. Hermes Agent's behavior is governed by three "
        "layers:",
        styles['BodyJust']
    ))

    layers_data = [
        ['Layer', 'What It Is', 'How It\'s Currently Improved'],
        ['Model Weights', 'The underlying LLM (Claude, GPT, etc.)', 'RL training (Tinker-Atropos)'],
        ['Instructions', 'Skills, system prompts, tool descriptions', 'Manual authoring (static)'],
        ['Tool Code', 'Python implementations of each tool', 'Manual development'],
    ]
    layers_table = Table(layers_data, colWidths=[1.2 * inch, 2.3 * inch, 2.5 * inch])
    layers_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 2), (-1, 2), HexColor('#fff9c4')),
    ]))
    story.append(layers_table)
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "The <b>instructions layer</b> (highlighted) is the sweet spot for automated optimization: "
        "it's pure text that LLMs can meaningfully mutate, changes are immediately deployable, and "
        "results are directly measurable. Agent Self-Evolution targets this layer.",
        styles['BodyJust']
    ))

    # ── APPROACH ────────────────────────────────────────────────────────
    story.append(Paragraph("Approach: Evolutionary Skill Optimization", styles['SectionHead']))

    story.append(Paragraph("Three Optimization Engines", styles['SubSection']))
    engines_data = [
        ['Engine', 'What It Optimizes', 'License', 'Role'],
        ['DSPy + GEPA', 'Skills, prompts, tool descriptions', 'MIT', 'Primary (validated)'],
        ['DSPy MIPROv2', 'Few-shot examples, instruction text', 'MIT', 'Fallback optimizer'],
        ['Darwinian Evolver', 'Code files, algorithms', 'AGPL v3', 'Code evolution (Phase 4)'],
    ]
    engines_table = Table(engines_data, colWidths=[1.4 * inch, 2.0 * inch, 0.8 * inch, 1.8 * inch])
    engines_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(engines_table)

    story.append(Paragraph(
        "<b>GEPA</b> (Genetic-Pareto Prompt Evolution) is the star engine — an ICLR 2026 Oral paper "
        "from Stanford/UC Berkeley. Unlike traditional evolutionary search that only sees pass/fail "
        "scores, GEPA reads full execution traces to understand <i>why</i> things failed, then proposes "
        "targeted mutations. It outperforms reinforcement learning (GRPO) by +6% with 35x fewer "
        "rollouts, and outperforms DSPy's previous best optimizer (MIPROv2) by +10%. It works with "
        "as few as 3 training examples.",
        styles['BodyJust']
    ))

    story.append(Paragraph("The Optimization Pipeline", styles['SubSection']))
    pipeline_steps = [
        "1. <b>Discover and load skill</b> — Resolve the skill via the SkillSource protocol "
        "(Hermes / Claude Code / local-dir), parse YAML frontmatter and body",
        "2. <b>Generate eval dataset</b> — An LLM (gpt-4.1-mini) reads the skill and synthesizes "
        "60+ (task, expected_behavior) pairs, then splits into ~36% train / ~29% val / ~36% holdout",
        "3. <b>Wrap as DSPy module</b> — The skill text becomes a parameterized DSPy module "
        "where the instructions are the optimizable parameter",
        "4. <b>Run optimizer</b> — DSPy GEPA evolves the skill instructions, scored by an "
        "LLM-as-judge with a structured rubric (correctness, procedure-following, conciseness)",
        "5. <b>Knee-point Pareto selection</b> — Among candidates within ε of the val-best, pick the "
        "smallest one (parsimony principle for deployable, cache-friendly skills)",
        "6. <b>Validate constraints</b> — Static size limits, structural integrity, growth-quality "
        "gate backed by a paired-bootstrap CI on the held-out set",
        "7. <b>Report</b> — Structured gate decision JSON, before/after artifacts, full LM trace log",
    ]
    for step in pipeline_steps:
        story.append(Paragraph(step, styles['Metric']))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "Critically, <b>no GPU training is involved</b>. The entire pipeline operates via LLM API calls. "
        "The arxiv run reported here issued ~1,570 LM calls (1,526 to gpt-4.1-mini for synthesis / "
        "evaluation / judging, 42 to gpt-5-mini for GEPA reflection); typical end-to-end cost is "
        "$5–8 in OpenAI API credits per skill on the light GEPA budget at the current "
        "<font face='Courier'>eval_dataset_size=150</font> default.",
        styles['BodyJust']
    ))

    # ── EXPERIMENT ──────────────────────────────────────────────────────
    story.append(Paragraph("Phase 1 Experiment", styles['SectionHead']))

    story.append(Paragraph("Configuration", styles['SubSection']))
    config_data = [
        ['Parameter', 'Value'],
        ['Target Skill', 'arxiv (Hermes Agent — arXiv paper search and retrieval)'],
        ['Baseline Size', '10,036 characters'],
        ['Optimizer LM', 'openai/gpt-4.1'],
        ['Reflection LM (GEPA)', 'openai/gpt-5-mini'],
        ['Eval / Judge LM', 'openai/gpt-4.1-mini'],
        ['Optimizer', 'DSPy GEPA (light budget — ~440 metric calls)'],
        ['Synthetic Eval Set', '178 examples (63 train / 50 val / 65 holdout)'],
        ['Total Optimization Time', '3,353 seconds (~56 minutes)'],
        ['Total LM Calls', '~1,570 (1,526 gpt-4.1-mini + 42 gpt-5-mini)'],
        ['Quality Gate', 'non-inferiority (tolerance=0.02; regression-only branch)'],
        ['Knee-point Strategy', 'val-best (default, May 2026+)'],
    ]
    config_table = Table(config_data, colWidths=[2.2 * inch, 3.8 * inch])
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
    story.append(config_table)

    story.append(Paragraph("Evaluation Dataset", styles['SubSection']))
    story.append(Paragraph(
        "The evaluation dataset was synthetically generated by openai/gpt-4.1-mini. Given the full "
        "arxiv SKILL.md text, the model produced 178 realistic test cases with rubric-based "
        "expected behaviors, then split them into train / val / holdout per the framework's "
        "configured ratios (the default <font face='Courier'>eval_dataset_size</font> was bumped from "
        "60 to 150 this cycle to tighten the holdout bootstrap CI; the LM produced ~10% more than "
        "requested). Examples drawn from the generated set:",
        styles['BodyJust']
    ))

    examples_data = [
        ['Task Input', 'Expected Behavior (Rubric)'],
        ['Fetch paper metadata with multiple\ncategories and verify primary category',
         'Primary category is correctly identified\nand displayed separately'],
        ['Extract and parse XML output from arXiv\nAPI using provided python snippet',
         'Parsed output includes numbered list with\npaper IDs, titles, authors, dates, categories'],
        ["Search papers with comment containing\n'accepted NeurIPS' for accepted submissions",
         'Returns papers with comments mentioning\nacceptance at NeurIPS, metadata included'],
    ]
    examples_table = Table(examples_data, colWidths=[2.5 * inch, 3.5 * inch])
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
    story.append(examples_table)

    story.append(Paragraph("Fitness Function", styles['SubSection']))
    story.append(Paragraph(
        "Fitness is now measured by an LLM-as-judge (gpt-4.1-mini) that scores each candidate "
        "output along three independent rubric dimensions. The composite score is a weighted "
        "combination, with a length-penalty term that discourages runaway expansion:",
        styles['BodyJust']
    ))
    story.append(Paragraph(
        "<font face='Courier' size=9>composite = 0.5·correctness + 0.3·procedure_following + 0.2·conciseness − length_penalty</font>",
        ParagraphStyle('Formula', parent=styles['Normal'], alignment=TA_CENTER,
                       spaceBefore=8, spaceAfter=8, fontSize=10)
    ))
    story.append(Paragraph(
        "The judge also returns a free-text feedback string that GEPA's reflection LM consumes to "
        "propose targeted instruction-text mutations on the next iteration — this trace-aware loop "
        "is the core of GEPA's sample efficiency. The keyword-overlap heuristic used in the original "
        "Phase 1 experiment was retired in favor of this rubric scorer.",
        styles['BodyJust']
    ))

    # ── RESULTS ─────────────────────────────────────────────────────────
    story.append(Paragraph("Results", styles['SectionHead']))

    results_data = [
        ['Metric', 'Baseline', 'Evolved (knee-point pick)', 'Δ'],
        ['Body size (chars)', '10,036', '6,017', '−40.0%'],
        ['Avg holdout score (n=65)', '0.908', '0.962', '+0.054'],
        ['Bootstrap mean diff', '—', '+0.054', '—'],
        ['Bootstrap 90% CI lower', '—', '+0.017', '—'],
        ['Bootstrap 90% CI upper', '—', '+0.095', '—'],
        ['Decision', '—', 'DEPLOYED', 'CI excludes 0'],
    ]
    results_table = Table(results_data, colWidths=[1.9 * inch, 1.3 * inch, 1.7 * inch, 1.1 * inch])
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
        ('BACKGROUND', (2, 1), (2, 1), HexColor('#e8f5e9')),
        ('TEXTCOLOR', (2, 1), (2, 1), HexColor('#2e7d32')),
        ('FONTNAME', (2, 1), (2, 1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), HexColor('#e8f5e9')),
        ('TEXTCOLOR', (-1, -1), (-1, -1), HexColor('#2e7d32')),
        ('FONTNAME', (-1, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph(
        "The evolved arxiv skill is <b>40.0% smaller</b> than baseline (6,017 vs 10,036 characters) "
        "AND scores <b>+0.054 higher</b> on the 65-example holdout (0.962 vs 0.908). The paired "
        "90% bootstrap CI on the per-example difference is [+0.017, +0.095] — both ends positive, "
        "so we have ≥90% confidence the improvement is real, not sampling noise. The deploy gate "
        "passes under the new <i>non-inferiority</i> rule (introduced this cycle) and would have "
        "passed the older <i>no-regression-only</i> rule too because the bootstrap mean is positive. "
        "The candidate is written to <font face='Courier'>evolved_skill.md</font> alongside the "
        "verbatim baseline for diffing.",
        styles['BodyJust']
    ))

    story.append(Paragraph("How the Result Was Produced", styles['SubSection']))
    story.append(Paragraph(
        "GEPA evolves skill instructions through a reflective loop:",
        styles['BodyJust']
    ))
    improve_steps = [
        "1. Run candidate skill instruction text on training examples; the judge scores each output "
        "and emits free-text feedback",
        "2. Reflection LM (gpt-5-mini) reads the execution traces + feedback and proposes a "
        "targeted mutation of the instruction text",
        "3. Score the mutated candidate on the validation set (50 examples); track every "
        "candidate's per-example Pareto front",
        "4. After ~30 iterations / ~440 metric calls, freeze the candidate population",
        "5. Knee-point selection: among all candidates within ε = 1/n_val of the val-best, pick "
        "the highest-val candidate (smallest body as tiebreak). On this run candidate 7 won the "
        "band (val=0.980, rank 1 of 3, 5,754 body chars) and is also the candidate GEPA's own "
        "default selector would have chosen — the val-best and GEPA-default converged here.",
    ]
    for step in improve_steps:
        story.append(Paragraph(step, styles['Metric']))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "Three changes from the prior validation cycle made this run deployable: (a) the default "
        "<font face='Courier'>eval_dataset_size</font> bumped 60 → 150 yielded a 65-example "
        "holdout (vs. 23 prior), tightening the bootstrap CI from [−0.135, +0.094] down to "
        "[+0.017, +0.095]; (b) the new <i>non-inferiority</i> deploy gate (rather than a "
        "hardcoded <font face='Courier'>mean ≥ 0</font> floor) ships compression-without-regression "
        "candidates explicitly — though this run cleared the older rule too; (c) knee-point "
        "selection now defaults to picking the val-best candidate within the ε-band rather than "
        "the smallest, removing a parsimony bias that had cost val score on the prior run.",
        styles['BodyJust']
    ))

    # ── SAFETY ──────────────────────────────────────────────────────────
    story.append(Paragraph("Safety and Guardrails", styles['SectionHead']))
    story.append(Paragraph(
        "Every evolved variant must pass all of the following constraints before deployment:",
        styles['BodyJust']
    ))
    safety_data = [
        ['Constraint', 'Enforcement', 'Status'],
        ['Self-evolution test suite', '268 pytest tests pass on the optimizer itself', 'Implemented'],
        ['Static size limits', 'Skills ≤15KB, tool descs ≤500 chars (configurable)', 'Implemented'],
        ['Absolute char ceiling', 'Hard cap on evolved artifact size (default 5,000)', 'Implemented'],
        ['Growth-quality gate', 'Required improvement scales linearly with growth %', 'Implemented'],
        ['Paired-bootstrap CI', '90% CI on per-example holdout diffs gates deploy', 'Implemented'],
        ['Knee-point selection', 'Smallest candidate within ε of val-best', 'Implemented'],
        ['Structural integrity', 'Valid YAML frontmatter required', 'Implemented'],
        ['Deployment via PR', 'Human review required, never auto-merge', 'By design'],
        ['Benchmark regression', 'TBLite / skill-specific harness must hold', 'Planned'],
    ]
    safety_table = Table(safety_data, colWidths=[1.6 * inch, 2.8 * inch, 1.1 * inch])
    safety_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(safety_table)
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "Source skill repositories are never modified directly. All evolution output (evolved "
        "artifacts, gate decisions, run logs) is written under the framework's local "
        "<font face='Courier'>output/</font> directory, and improvements are proposed as pull requests "
        "against the source repo for human review.",
        styles['BodyJust']
    ))

    # ── ROADMAP ─────────────────────────────────────────────────────────
    story.append(Paragraph("Roadmap", styles['SectionHead']))
    roadmap_data = [
        ['Phase', 'Target', 'Engine', 'Timeline', 'Status'],
        ['Phase 1', 'Skill files (SKILL.md)', 'DSPy + GEPA', '3-4 weeks', 'Validated ✓'],
        ['Phase 2', 'Tool descriptions', 'DSPy + GEPA', '2-3 weeks', 'Planned'],
        ['Phase 3', 'System prompt sections', 'DSPy + GEPA', '2-3 weeks', 'Planned'],
        ['Phase 4', 'Tool implementation code', 'Darwinian Evolver', '3-4 weeks', 'Planned'],
        ['Phase 5', 'Continuous improvement', 'Automated pipeline', '2 weeks', 'Planned'],
    ]
    roadmap_table = Table(roadmap_data, colWidths=[0.9 * inch, 1.6 * inch, 1.3 * inch, 1.0 * inch, 1.0 * inch])
    roadmap_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, 1), HexColor('#e8f5e9')),
    ]))
    story.append(roadmap_table)
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "Each phase must demonstrate measurable improvement and pass benchmark regression gates "
        "before proceeding. If a phase does not produce meaningful gains, we reassess before continuing. "
        "The full plan is documented in PLAN.md within the repository.",
        styles['BodyJust']
    ))

    # ── NEXT STEPS ──────────────────────────────────────────────────────
    story.append(Paragraph("Immediate Next Steps", styles['SectionHead']))
    next_steps = [
        "1. <b>Evolve more skills</b> — Run the same pipeline against additional Hermes and Claude "
        "Code skills to measure how often the regression-free deployment criterion is actually met "
        "and where the framework's defaults need calibration.",
        "2. <b>Calibrate the deploy gate</b> — Use the bootstrap CIs from a portfolio of runs to "
        "set evidence-based defaults for <font face='Courier'>growth_free_threshold</font> and "
        "<font face='Courier'>growth_quality_slope</font>, and revisit the knee-point ε choice "
        "(this run picked a candidate that was only 5% smaller than the val-best at meaningful val cost).",
        "3. <b>Larger holdout sets</b> — n=23 left a wide bootstrap CI; raising "
        "<font face='Courier'>eval_dataset_size</font> or <font face='Courier'>holdout_ratio</font> "
        "would tighten the deploy / reject signal.",
        "4. <b>Benchmark gating</b> — Add TBLite or skill-specific regression harnesses for skills "
        "where the bootstrap CI alone is too noisy.",
        "5. <b>PR automation</b> — Auto-generate pull requests against the source skill repository "
        "with the evolved artifact, gate decision JSON, and full metrics.",
        "6. <b>Tier 2: tool descriptions</b> — Extend the optimizer beyond skill files to tool "
        "description text (currently a stub).",
    ]
    for step in next_steps:
        story.append(Paragraph(step, styles['Metric']))

    # ── FOOTER ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#cccccc')))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        f"Agent Self-Evolution — Phase 1 Validation Report — {datetime.now().strftime('%B %d, %Y')} — Nous Research",
        styles['Footer']
    ))
    story.append(Paragraph(
        "github.com/jramos/agent-self-evolution",
        styles['Footer']
    ))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    path = build_report()
    print(f"Report generated: {path}")
