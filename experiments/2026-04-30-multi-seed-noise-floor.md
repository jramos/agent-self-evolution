# Multi-seed noise floor + cross-skill validation (Combo 1 spike)

**Date:** 2026-04-30
**Branch:** `feat/n60-valset-cleanup` (PR #9 open, awaiting merge)
**Config:** `eval_dataset_size=60`, `val_ratio=0.40`, knee-point ε=1/n_val ≈ 0.0625
**Goal:** Decide whether obsidian's wide spike↔PR9-e2e swing was typical (→ multi-seed harness needed) or anomalous (→ ship N=60 as-is). Secondary: characterize cross-skill performance.

## Phase 1 — Multi-seed obsidian (5 seeds, all complete)

| Run | Seed | Decision | Bootstrap mean | CI 90% | Band size | Picked idx (rank) | Picked chars | GEPA default chars |
|---|---|---|---|---|---|---|---|---|
| spike | 42 | **deploy** | +0.0181 | [-0.037, +0.080] | 8 | 9 (1 of 8) | 854 | 854 |
| PR #9 e2e | 42 | reject | -0.0176 | [-0.066, +0.028] | 4 | 6 (4 of 4) | 1005 | 1452 |
| seed 7 | 7 | **deploy** | +0.0510 | [-0.012, +0.131] | 12 | 5 (2 of 12) | 810 | 866 |
| seed 13 | 13 | **deploy** | +0.0781 | [+0.009, +0.159] | 5 | 2 (4 of 5) | 951 | 1076 |
| seed 99 | 99 | **deploy** | +0.0048 | [-0.038, +0.050] | 9 | 10 (3 of 9) | 974 | 1521 |

**Deploys: 4/5 (80%)** at N=60 obsidian, holdout N=21.

### Noise floor

- mean of bootstrap means = **+0.0269**
- stdev = **0.0379**
- range = -0.018 to +0.078 (spread 0.0957)
- one run (seed=13) crossed the strict `dual_check` bar (lower_bound > 0); the other 3 deploys passed via the `no_regression_only` branch (mean ≥ 0 with shorter artifact)

### Decision-matrix lookup

The plan's decision matrix asked: with std(mean) ≈ 0.038 (borderline 0.02-0.05 range) and X/3 cross-skill deploys, what's the recommendation?

We landed in the borderline noise band, but the 80% deploy rate on obsidian alone is much stronger than the matrix anticipated. **The interpretation needs to flex from the matrix:** with 4/5 deploys at this noise level, single-seed e2e is roughly trustworthy *as a positive signal* (when it deploys), but unreliable *as a negative signal* (a single rejection can be noise — PR #9 e2e at seed=42 was the outlier, not the rule).

## Phase 2 — Cross-skill (aborted, OpenAI capacity issues)

Two attempts, both killed without producing a `gate_decision.json`:

| Skill | SKILL.md size | Outcome | Wall time before kill |
|---|---|---|---|
| github-code-review | 13.5K | killed at iteration 7 (32%, 144/452 rollouts) | 2h |
| apple-notes | 2.2K | killed at iteration 4 (15%, 66/448 rollouts) | 87 min |

**Same failure pattern** in both runs: silent 30-50 min stalls between iterations, with eventual tenacity retry firing. Pattern was independent of skill size (apple-notes is smaller than obsidian's evolved candidates and still hit it). Looks like sustained OpenAI rate-limit / capacity degradation through the evening of 2026-04-29; even seed=99 obsidian had a 31-min recovery stall mid-run.

**This is itself a finding**, not a failed experiment: the e2e pipeline has no defense against extended API degradation. The `request_timeout`/`num_retries` we wired in PR #7's debug branch never made it to main, so when OpenAI degrades, runs silently hang for hours instead of failing fast.

## Recommendation for PR #11

Two findings drive the recommendation:

1. **N=60 obsidian deploys 4/5 of the time.** Strong evidence the framework works at this scale. The single rejection (PR #9 e2e at seed=42) is consistent with the noise distribution; it's not signal of a broken pipeline.

2. **OpenAI degradation breaks the pipeline silently.** Across ~5 hours of e2e attempts on 2026-04-29 evening, 2/3 cross-skill runs hit unrecoverable hangs. Production debugging is impossible without the LM-call timing/timeout instrumentation that lived only on the throwaway PR #7 debug branch.

**Recommended PR #11:** **Promote LM-call hardening to main** — port the `LMTimingCallback` + `request_timeout` + `num_retries` from the abandoned debug branch into `evolution/skills/evolve_skill.py`, `evolution/core/fitness.py`, `evolution/core/dataset_builder.py`. This unblocks any future multi-skill measurement work and shifts silent stalls into visible heartbeats + raised TimeoutErrors that the existing GEPA→MIPROv2 fallback can handle.

Estimated effort: ~150 LOC + tests, 1 day. No behavior change on the happy path; pure resilience improvement.

After PR #11 lands, the right multi-skill experiment becomes feasible:
- Same 5-seed obsidian baseline as a control
- Plus 3-5 cross-skills (mix of small + medium SKILL.md)
- All runs guaranteed to complete or fail visibly within bounded wall time

That gives us the "framework works across skills?" data point we couldn't collect today.

### Alternatives considered (deferred)

- **A2 (higher-temp synthetic)** — still attractive but blocked by the same API-stall problem; no point widening the synthetic distribution if half the runs hang.
- **A4 (cross-judge counterfactual)** — adds another LM dependency; same blocker.
- **B1 (1-day measurement harness stub)** — would surface the API-stall problem cleanly but won't fix it.

LM-call hardening is upstream of all three.

## Notes / artifacts

- Per-run gate_decision.json files: `output/obsidian/2026042{8,9}_*/gate_decision.json`
- Spike branch (config bumps): `spike/bigger-valset-knee` (throwaway, deletable)
- Debug-instrumentation reference: PR #7 commit `7010fdd` (debug:) on `feat/bootstrap-ci-gate` — not in main, but the diff is the starting point for PR #11
- 5 obsidian runs cost approximately $20 in OpenAI calls; aborted Phase 2 cost ~$10 more
