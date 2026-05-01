# Documentation Review Notes

A consistency + completeness pass over the docs in this directory, performed 2026-04-30 against the codebase at branch `feature/docs`.

## Verified accurate

- Module / package layout matches `find evolution -type f -name "*.py"`.
- `EvolutionConfig` field defaults match `evolution/core/config.py:18-99`.
- `gate_decision.json` schema_version `"3"` matches `evolution/skills/evolve_skill.py:713` and the test fixture in `tests/skills/test_evolve_skill_validation_flow.py:208`.
- `_HEARTBEAT_TIERS` table matches `evolution/core/lm_timing_callback.py:37-42`.
- LM `request_timeout` / `num_retries` values per surface verified:
  - judge LM (`fitness.py:83`): `request_timeout=60, num_retries=5`
  - dataset gen LM (`dataset_builder.py:134`): `request_timeout=120, num_retries=5`
  - reflection LM (`evolve_skill.py:227-228`): `request_timeout=300, num_retries=2`
- Pinned dep ranges verified against `pyproject.toml`.
- 262 tests collected (`pytest --collect-only` inside venv).

## Minor inconsistencies in the codebase (worth tracking, not blockers)

### 4. Module-import-time `logging.basicConfig`
`evolution/skills/evolve_skill.py:30-34` calls `logging.basicConfig` at import. This is *idempotent* in stdlib (only first call wins) but means importing `evolve_skill` from another script silently configures the root logger. Documented in `interfaces.md` (Logging conventions) — flag if a future user wants to import `evolve()` from a notebook without the side effect.

### 5. `HermesSkillSource` env var name has changed
The `external_importers._load_skill_text` standalone CLI uses `~/.hermes/skills/`, but the `HermesSkillSource` adapter uses `~/.hermes/hermes-agent/skills/` (or `$SKILL_SOURCES_HERMES_REPO/skills/`). Different path under the same `~/.hermes/` prefix; could confuse a user who deletes one and expects both surfaces to break together.

### 6. CLI flag naming inconsistency
- `--bootstrap-resamples` (CLI) maps to `bootstrap_n_resamples` (Python) — note the `n_` prefix difference.
- All other CLI flags map straightforwardly.

### 7. Tier-2/3/4/5 packages are empty stubs
`evolution/{tools,prompts,code,monitor}/` contain only `__init__.py`. They anchor the planned architecture but currently do nothing. Documented in `codebase_info.md` (implementation status table). Could confuse a new contributor expecting working code there.

## Gaps that warrant future docs

### 1. No deployment / release docs
No `release.md`, `CONTRIBUTING.md`, `RELEASE.md`. Project is currently single-author with PR-based merges; if it scales, these would be needed.

### 2. No example `gate_decision.json` walkthrough
`data_models.md` shows the schema; a worked example narrating "the bootstrap CI lower bound was -0.06 so dual-check rejected" would help users reading their own decisions for the first time. Could be added if rejection diagnostics become a frequent user task.

### 3. No "how to add a new constraint" guide
`ConstraintValidator` is closed over a hardcoded set of checks. Adding a new one requires editing both the validator and (for the gate-payload integration) `evolve_skill.py`. Pattern is straightforward but undocumented; would be useful when Tier 2/3 lands and tool-description-specific constraints are added.

### 4. No GEPA-vs-MIPROv2 comparison
The fallback chain is implemented but the "when does GEPA underperform / when does MIPROv2" narrative isn't documented. The MIPROv2 path is a degraded mode (no knee-point, no `detailed_results`); user-facing implications are not surfaced beyond "knee_point.applied=false."

### 5. PLAN.md is canonical for Tier 2-5, but very long
`PLAN.md` is the source of truth for the planned tiers but it's 40K+ chars and mixes design with retrospective notes. A short "what's next" pointer in `codebase_info.md` would help — currently we just say "see PLAN.md."

## Recommended documentation maintenance

1. **Re-verify defaults on every release.** `EvolutionConfig` defaults are tuned often; doc table in `data_models.md` will drift.
2. **Re-collect test count when refactoring.** Currently 262; bump if tests are added/removed.
3. **Update `gate_decision.json` schema docs on every schema bump.** When `schema_version` increments, both `data_models.md` and `interfaces.md` (test surfaces) need to mention the new fields.
4. **Verify mermaid diagrams render.** GitHub renders mermaid in markdown; if a diagram breaks during edits, the rest of the page still renders, so silent breakage is possible. Spot-check on github.com after pushing.

## What's NOT documented (intentionally)

- **Per-PR rationale.** That's `git log --oneline` + PR descriptions.
- **Bug-fix recipes.** The fix is in the code; the commit message has the context.
- **Debugging output samples.** Run logs and `gate_decision.json` snapshots are user-specific and rot fast.
- **Style preferences.** Lives in `AGENTS.md`.
