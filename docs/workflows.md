# Workflows

Step-by-step traces of the framework's main flows.

## Workflow 1: Evolve a skill (synthetic dataset, deploy path)

The standard happy path.

```bash
python -m evolution.skills.evolve_skill \
    --skill obsidian \
    --budget light \
    --eval-source synthetic
```

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant CLI as evolve_skill.main
    participant Cfg as EvolutionConfig
    participant Src as SkillSource
    participant Log as FileHandler
    participant Cb as LMTimingCallback
    participant LL as litellm.failure_callback
    participant Gen as SyntheticDatasetBuilder
    participant Val as ConstraintValidator
    participant LM as dspy.LM (judge)
    participant J as LLMJudge
    participant SM as SkillModule
    participant Prop as BudgetAwareProposer
    participant GEPA as dspy.GEPA
    participant Knee as select_knee_point
    participant Eval as dspy.Evaluate
    participant Boot as paired_bootstrap

    U->>CLI: --skill obsidian --budget light
    CLI->>Cfg: EvolutionConfig(...)
    Cfg->>Src: discover_skill_sources()
    Src-->>Cfg: [Hermes, ClaudeCode]
    CLI->>Src: find_skill("obsidian")
    Src-->>CLI: Path to SKILL.md
    CLI->>CLI: load_skill(path) → {name, description, body, frontmatter}

    Note over CLI,Log: §1.5 Per-run output dir + run.log + litellm hook
    CLI->>Log: addHandler(FileHandler(output/obsidian/<ts>/run.log))
    CLI->>LL: register_litellm_failure_callback()

    Note over CLI,Gen: §2 Build dataset
    CLI->>Gen: generate(skill_text, "skill", n=60)
    Gen->>LM: judge_model.generate(GenerateTestCases prompt)
    LM-->>Gen: JSON array of test cases
    Gen-->>CLI: EvalDataset(train=21, val=17, holdout=22)
    CLI->>CLI: dataset.save(datasets/skills/obsidian/)

    Note over CLI,Val: §3 Baseline static checks (warn-only)
    CLI->>Val: validate_static(skill.raw, "skill")
    Val-->>CLI: [size_limit ✓, non_empty ✓, skill_structure ✓]

    Note over CLI,Cb: §4 Configure DSPy + judge + metric
    CLI->>Cb: dspy.configure(lm, callbacks=[LMTimingCallback()])
    CLI->>SM: baseline_module = SkillModule(skill.body)
    CLI->>J: judge = LLMJudge(config)
    CLI->>CLI: metric = make_skill_fitness_metric(judge, baseline_text, max_growth=free_threshold)
    CLI->>Prop: BudgetAwareProposer(baseline_chars, max_growth=free_threshold)

    Note over CLI,GEPA: §5 GEPA optimization
    CLI->>GEPA: compile(baseline_module, trainset, valset)
    loop per iteration
        GEPA->>Prop: __call__(candidate, reflective_dataset, ["self"])
        Prop->>LM: reflection_lm.propose(current, examples_with_feedback)
        LM-->>Prop: improved_instruction
        Prop-->>GEPA: {self: new_text}
        GEPA->>SM: forward each train example
        SM->>LM: predict
        LM-->>SM: output
        SM-->>GEPA: prediction
        GEPA->>J: metric(example, prediction)
        J->>LM: score (judge call)
        LM-->>J: scores
        J-->>GEPA: dspy.Prediction(score, feedback)
    end
    GEPA-->>CLI: optimized_module (with detailed_results)

    Note over CLI,Knee: §5b Knee-point Pareto selection
    CLI->>Knee: select_knee_point(candidates, val_scores, n_val, validator)
    Knee->>Val: validate_static for each band candidate (ascending body chars)
    Knee-->>CLI: CandidatePick (picked_idx, body_chars, fallback="knee")
    CLI->>SM: optimized_module = SkillModule(knee_pick.skill_text)

    Note over CLI,Val: §6+§7 Reassemble + static checks on evolved
    CLI->>CLI: evolved_full = reassemble_skill(frontmatter, optimized_module.skill_text)
    CLI->>Val: validate_static(evolved_full, "skill")
    Val-->>CLI: [size_limit ✓, non_empty ✓, skill_structure ✓]

    Note over CLI,Eval: §8 Holdout evaluation (≈ 2 × |holdout| judge calls)
    CLI->>Eval: evaluate(baseline_module, holdout)
    Eval->>J: per-example metric → score
    Eval-->>CLI: avg_baseline, baseline_per_example
    CLI->>Eval: evaluate(optimized_module, holdout)
    Eval-->>CLI: avg_evolved, evolved_per_example

    Note over CLI,Boot: §9 Paired bootstrap + growth-quality gate
    CLI->>Boot: paired_bootstrap(baseline_per_ex, evolved_per_ex)
    Boot-->>CLI: {mean, lower_bound, upper_bound, ...}
    CLI->>Val: validate_growth_with_quality(evolved, baseline, bootstrap)
    Val-->>CLI: [growth_quality_gate ✓, absolute_char_ceiling ✓]

    CLI->>CLI: write gate_decision.json (decision="deploy")
    CLI->>CLI: write evolved_skill.md, baseline_skill.md, metrics.json
    CLI-->>U: ✓ Evolution improved skill by +0.054 (+6.1%)
```

## Workflow 2: Evolve a skill (rejected on quality gate)

Same as Workflow 1 through §8. Diverges at §9.

```mermaid
sequenceDiagram
    autonumber
    participant CLI as evolve_skill
    participant Boot as paired_bootstrap
    participant Val as ConstraintValidator
    participant FS as filesystem

    Note over CLI: §9 Bootstrap + gate
    CLI->>Boot: paired_bootstrap(baseline, evolved)
    Boot-->>CLI: {mean=-0.025, lower_bound=-0.095, ...}
    CLI->>Val: validate_growth_with_quality(evolved, baseline, bootstrap)
    Val-->>CLI: [growth_quality_gate ✗ "regression — mean -0.025 < 0"]

    CLI->>FS: write gate_decision.json (decision="reject", reason="growth_quality_gate")
    CLI->>FS: write evolved_FAILED.md
    CLI-->>CLI: print red banner; return (no metrics.json, no evolved_skill.md)
```

The reject path is deliberately quiet — it returns instead of raising, so callers (including pytest harnesses) can treat reject as a normal outcome.

## Workflow 3: Evolve a skill (rejected on static check)

Triggered when GEPA produces an artifact that fails size/structure/non-empty. Short-circuits *before* spending judge calls on the holdout.

```mermaid
sequenceDiagram
    autonumber
    participant CLI as evolve_skill
    participant Val as ConstraintValidator
    participant FS as filesystem

    Note over CLI: §6+§7 Reassemble + static
    CLI->>CLI: evolved_full = reassemble_skill(frontmatter, evolved_body)
    CLI->>Val: validate_static(evolved_full, "skill")
    Val-->>CLI: [size_limit ✗ "Size exceeded: 16500/15000 chars"]

    CLI->>FS: write evolved_FAILED.md
    CLI->>FS: write gate_decision.json (decision="reject", reason="static_constraint_failure", knee_point + dataset blocks)
    CLI-->>CLI: return — holdout never runs
```

This is the cost-savings shortcut: ~2N judge calls (where N = holdout size) saved per static-failed run.

## Workflow 4: GEPA → MIPROv2 fallback

Triggered when GEPA raises any exception (including `TimeoutError` from a stuck reflection LM). `--no-fallback` re-raises instead.

```mermaid
sequenceDiagram
    autonumber
    participant CLI as _build_optimizer_and_compile
    participant GEPA as _default_gepa_runner
    participant MIPRO as _default_mipro_runner
    participant FS as filesystem
    participant U as User

    CLI->>GEPA: try compile
    GEPA-->>CLI: TimeoutError (reflection LM exceeded 300s × 2 retries)
    CLI->>FS: write output/<skill>/gepa_failure.log
    CLI->>U: print fallback banner
    CLI->>MIPRO: compile(baseline, trainset, metric)
    alt optuna installed
        MIPRO-->>CLI: optimized_module (no detailed_results)
    else optuna missing
        MIPRO-->>CLI: ImportError
        CLI->>U: print "install agent-self-evolution[miprov2]"
        CLI-->>CLI: raise ImportError from gepa_exc
    end
```

After MIPROv2 fallback, knee-point selection is **skipped** (the optimized module has no `detailed_results`). `gate_decision.json.knee_point.applied` will be `false` with `reason="no_detailed_results"`.

## Workflow 5: Build dataset from sessiondb

Triggered by `--eval-source sessiondb`.

```mermaid
sequenceDiagram
    autonumber
    participant CLI as evolve_skill
    participant Build as build_dataset_from_external
    participant CC as ClaudeCodeImporter
    participant Cop as CopilotImporter
    participant H as HermesSessionImporter
    participant Filter as RelevanceFilter
    participant LM as relevance LM
    participant FS as filesystem

    CLI->>Build: build_dataset_from_external(skill_name, skill_text, ["claude-code","copilot","hermes"], output_path, model)
    Build->>CC: extract_messages()
    CC->>FS: read ~/.claude/history.jsonl
    CC-->>Build: list of dicts (filtered through SECRET_PATTERNS)
    Build->>Cop: extract_messages()
    Cop->>FS: read ~/.copilot/session-state/*/events.jsonl
    Cop-->>Build: list of (user, assistant) pairs
    Build->>H: extract_messages()
    H->>FS: read ~/.hermes/sessions/*.json
    H-->>Build: list of (user, assistant) pairs

    Build->>Filter: filter_and_score(all_messages, skill_name, skill_text)
    Filter->>Filter: heuristic pre-filter (_is_relevant_to_skill)
    loop per candidate
        Filter->>LM: ScoreRelevance(skill, msg, response)
        LM-->>Filter: JSON {relevant, expected_behavior, difficulty, category}
        alt relevant
            Filter->>Filter: validate + accumulate EvalExample
        end
    end
    Filter-->>Build: list[EvalExample]

    alt examples >= MIN_DATASET_SIZE (3)
        Build->>Build: shuffle + split 50/25/25 train/val/holdout
        Build->>FS: dataset.save(output_path)
        Build-->>CLI: EvalDataset
    else
        Build-->>CLI: EvalDataset() (empty)
        CLI-->>CLI: sys.exit(1) "no relevant examples"
    end
```

Note: the sessiondb path uses a hardcoded **50/25/25 split**, not the `EvolutionConfig` ratios. This is a known minor inconsistency — the synthetic path normalizes the configured ratios; the sessiondb path doesn't.

## Workflow 6: Standalone session importer (preview mode)

```bash
python -m evolution.core.external_importers --source all --skill obsidian --dry-run
```

Goes through the same `*.extract_messages()` path but skips `RelevanceFilter` and just prints message counts per source. Useful for confirming session data exists before paying for LLM relevance scoring.

## Workflow 7: Loading a previously-generated dataset

```bash
python -m evolution.skills.evolve_skill \
    --skill obsidian \
    --eval-source golden \
    --dataset-path datasets/skills/obsidian/
```

`GoldenDatasetLoader.load(path, seed)`:
1. If `path/train.jsonl` exists, load each split file directly via `EvalDataset.load(path)`.
2. Else, look for `path/golden.jsonl` (or `path` itself if it ends in `.jsonl`), shuffle + auto-split 50/25/25.

This path is also how the sessiondb-mined datasets are reused — once `datasets/skills/<skill>/` has split files, you can re-run with `--eval-source golden` to skip re-mining.

## Workflow 8: Test the framework

```bash
pytest tests/ -q
```

Tests are organized:
- `tests/core/` — `test_constraints.py`, `test_dataset_builder.py`, `test_external_importers.py`, `test_fitness.py`, `test_lm_timing_callback.py`, `test_skill_sources.py`, `test_stats.py`
- `tests/skills/` — `test_budget_aware_proposer.py`, `test_evolve_skill_helpers.py`, `test_evolve_skill_validation_flow.py`, `test_knee_point.py`, `test_skill_module.py`

All tests use mocks for LM calls — no real API keys required. The `_skill_source_env` autouse fixture (in tests that touch `EvolutionConfig`) sets `SKILL_SOURCES_HERMES_REPO` to a `tmp_path` fake repo so discovery doesn't pick up the developer's real `~/.hermes` install.

## Failure-mode summary

| Trigger | Outcome | Where to look |
|---|---|---|
| Skill not found | `sys.exit(1)`, prints available skills per source | console only |
| Holdout < `min_holdout_size` | `sys.exit(1)` early | console only |
| Static fail on baseline | warns, proceeds | console only |
| Static fail on evolved | reject, no holdout run | `evolved_FAILED.md` + `gate_decision.json` |
| Quality gate reject | reject after holdout | `evolved_FAILED.md` + `gate_decision.json` |
| GEPA exception | MIPROv2 fallback (unless `--no-fallback`) | `output/<skill>/gepa_failure.log` |
| Reflection LM stall | `TimeoutError` after `300s × 2` retries → MIPROv2 fallback | `run.log` (heartbeats + `[litellm RETRY/FAIL]`) |
| Judge LM stall | `TimeoutError` after `60s × 5` retries → propagates up to GEPA → fallback | `run.log` |
| Dataset gen JSON truncation | already fixed (`max_tokens=16000`); legacy: `JSONDecodeError` | `run.log` |
| MIPROv2 missing optuna | `ImportError` re-raised with GEPA failure as `__cause__` | console |
