# Recent changes (since the last knowledge-base refresh, PR #13)

This file records changes to the codebase that postdate the last full
documentation refresh (PR #13, commit `4037831`). It is intended as a
quick orientation aid for AI assistants reading the docs out of order.
For deep dives, the per-area docs (`architecture.md`, `components.md`,
`interfaces.md`, `data_models.md`) have been updated in-place.

## PR #14 — `Cleanup: numpy dep + unify dataset splits + drop dead flag` (commit `d009899`)

- **numpy** promoted from transitive to direct dep; declared with floor
  `>=1.24` (no upper bound) in `pyproject.toml`.
- **`split_examples()` helper** extracted in `evolution/core/dataset_builder.py`;
  the synthetic, sessiondb, and golden paths now route through one shuffle+split
  helper (was three hardcoded copies). Tests in
  `tests/core/test_dataset_builder.py:TestSplitExamples` and
  `TestSplitConsistencyAcrossPaths` lock the contract.
- **`length_penalty_weight`** removed from `EvolutionConfig`, the `evolve()`
  signature, and the `--length-penalty-weight` CLI option (it was a forward-wired
  no-op nobody had wired up).

## PR #15 — `Unblock deploys: non-inferiority gate + larger holdout + val-best knee-point` (commit `755f5b4`)

Three independent changes that together unblock deploy-rate on
previously-unevolved skills. All three are reflected in
`docs/data_models.md` (EvolutionConfig + gate_decision.json schema),
`docs/architecture.md` (deploy-gate + presets section), and
`docs/interfaces.md` (CLI flag tables).

- **R1 — non-inferiority deploy gate.** Adds `--quality-gate non-inferiority
  --inferiority-tolerance 0.02`. The `bootstrap.mean ≥ 0` regression floor
  was previously hardcoded; the new mode passes when `bootstrap.lower_bound
  > -tolerance` (Decagon-style; recommended for compression runs at small
  N where the bootstrap CI swamps tiny effects). New `EvolutionConfig`
  fields `gate_mode: str` and `inferiority_tolerance: float`.
  `gate_decision.json` schema bumped 3 → 4; new fields `gate_mode` and
  `inferiority_tolerance` added to the schema lock. `--quality-gate off`
  is preserved for back-compat but emits a one-line WARNING noting it
  doesn't actually disable the regression check.
- **R3 — default `eval_dataset_size` 60 → 150.** Yields a ~53-example
  holdout (vs. ~22 prior) so the bootstrap CI is tight enough to detect
  small effects (paper uses n=300; we were 5× under). Eval cost grows
  ~2.5× (linear in N); GEPA optimizer cost is unchanged. Historical
  `experiments/2026-04-30-multi-seed-noise-floor.md` was footnoted as
  "N=60 baseline only — re-run at N=150 to re-establish noise floor."
- **R4 — knee-point picks val-best by default.** Was greedy parsimony
  (smallest body in band, regardless of val cost); now picks the
  highest-val candidate within the ε-band, smallest body as tiebreak.
  New `--knee-point-strategy {val-best, smallest}` flag; `smallest`
  reproduces the prior behavior for users explicitly chasing compression.

Test count grew 262 → 282 (added `TestNonInferiorityGate`,
`TestResolveDecisionRule`, `TestValBestStrategy`, `TestSmallestStrategy`,
`TestStrategyValidation` plus `TestSplitExamples` from PR #14).

## Branch `feat/phase1-report-regen` — Phase 1 report refactor

- **`generate_report.py` is now a renderer, not a content-store.** New
  CLI: `--run output/<skill>/<ts>/ --prose reports/<phase>_prose.yaml
  --out reports/<phase>_validation_report.pdf`. Numbers (sizes, scores,
  bootstrap CI, decision, knee-point pick, dataset splits, LM call
  counts) are auto-extracted from the run dir's
  `gate_decision.json` + `metrics.json` + `run.log`. Editorial prose
  + table contents + config-row labels live in YAML and may use
  `{placeholder}`-style format substitutions filled from the extracted
  data. Decision-dependent styling (deploy = green, reject = amber)
  is picked automatically from `gate_decision.decision`.
- **`reports/phase1_prose.yaml`** is the new editorial source. Schema
  documented inline at the top of the file.
- **`assets/dna.png`** (Twemoji 14.0, MIT-licensed) is the title-page
  logo; replaces the missing-glyph "■" Helvetica was producing for the
  unicode 🧬 emoji.
- **`pyyaml`** promoted from transitive to direct dep in `pyproject.toml`.
- **`pyproject.toml` authors** changed from "Nous Research" to "jramos";
  cover-page Organization line is opt-in via `meta.organization` in
  YAML and is empty by default in `phase1_prose.yaml`.
