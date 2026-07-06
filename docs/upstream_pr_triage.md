# Upstream PR Triage

Living record of our periodic review of open pull requests on the upstream parent
(`NousResearch/hermes-agent-self-evolution`) and what, if anything, we incorporate into
this fork (`jramos/agent-self-evolution`).

This fork has diverged substantially from upstream — we run our own evolution stack
(closed-loop behavioral validation, held-out anti-gaming deploy gates, statistical
significance gating, shipped+measured code evolution, a propose-only sentinel). As a
result **most open upstream PRs are already addressed or strictly superseded here.** The
job of this review is to dedup the backlog, decide the few items worth adopting, and
record dispositions so we don't re-litigate the same clusters every cycle.

## How to run a review

1. **Pull the open PRs** with metadata:
   `gh pr list --repo NousResearch/hermes-agent-self-evolution --state open --limit 200 --json number,title,author,changedFiles,additions,deletions`
2. **Group by theme** from titles. Expect heavy duplication — the same early bug is
   often re-fixed by dozens of contributors.
3. **Deep-dive each group with the right lens.** For every PR: read it
   (`gh pr view N --repo … --json title,body` + `gh pr diff N --repo …`), understand the
   root problem, then **grep/read our actual code** to decide our current state. Our fork
   may have fixed the bug differently, replaced the code path entirely, or never had it.
   Many upstream diffs won't even apply (they patch call sites we refactored away).
4. **Classify**: `ADOPT` / `CHERRY-PICK <what>` / `INVESTIGATE` / `SKIP (<why>)`.
5. **Don't re-review settled clusters** — see "Already covered" below; only reconsider if
   our code in that area changes materially.
6. **Flag sensitive PRs as do-not-merge** — several upstream PRs commit client identifiers,
   API keys, or local paths. Never import these; note them so a future cycle skips them.

## Review log

- **2026-06-28** — First full review. 70 open PRs → 9 thematic groups → ~7 actionable
  items after dedup. No upstream PR was incorporated wholesale; the value is in *gaps the
  upstream crowd built that we skipped*, not fixes to anything we got wrong.
- **2026-06-28** — #132 (parameter-description evolution) **investigated → null**. A
  measure-first kill-gate compared precise vs. name-only parameter descriptions on the case
  most favorable to signal (the hermes `terminal` background/notify *convention*, carried
  entirely in prose), scoring the agent's supplied argument values: **capable model 10/10 vs
  10/10; weak `gpt-5.4-nano` 23/26 vs 27/29** (reps=3). The non-saturated weak model had room
  to benefit from the guidance and didn't — the description text does not move value-selection;
  the agent infers it from the task + param name. Did not build the `evolution/parameters/`
  subsystem; ~$1 spent vs. a six-file build. (Probe + results are a local spike under
  `spikes/param_probe/`, uncommitted per convention; reproducible via `probe.py`.)
- **2026-06-28** — #102 (skill/tool importer sourced stale JSON) **fixed**.
  `iter_hermes_sessions` read `~/.hermes/sessions/*.json` — request-error dumps with no
  `messages` key — so the skill *and* tool mining paths silently got zero sessions. Now reads
  the canonical `state.db` first (JSON fallback), mines all sessions regardless of
  `sessions.source`, and strips the prepended model-switch note while keeping the real
  instruction. End-to-end: the importer went 0 → real pairs; full non-slow suite green.
  The #26 `RelevanceFilter` recall improvement remains a follow-up.
- **2026-06-28** — #106 (Dependabot + pre-commit) **added**. `.github/dependabot.yml` (weekly
  `uv` + `github-actions`) and `.pre-commit-config.yaml` (hygiene hooks: trailing-whitespace,
  end-of-file-fixer, check-yaml/toml, check-merge-conflict, check-added-large-files with
  `uv.lock` excluded); `pre-commit` added to the dev group. Dropped upstream's `ruff`/`ruff-format`
  (the codebase isn't ruff-clean — needs a line-length policy first) and kept `gitleaks` separate
  (#107). Hooks enforce on changed files; a one-time `--all-files` cleanup (~18 EOF/whitespace
  nits, mostly report JSONs) is a deferred follow-up.
- **2026-06-28** — #134 (graduated skill-size cap) **investigated → not applicable**. The
  premise — a pre-cap length-penalty "cliff" in `LLMJudge.score` that docks under-cap skills — is
  dead code on our fork: no caller passes the `artifact_size`/`max_size` the penalty needs, so it
  is always 0.0 and docks nothing. Length pressure is handled on purpose by the proposer's length
  budget and two hard deploy ceilings (`_check_size`/`max_skill_size` and the baseline-scaled
  `effective_absolute_char_ceiling`); building the graduated ramp would re-introduce a
  deliberately-removed mechanism. Removed the vestigial `length_penalty` instead (the field, the
  ratio/penalty computation, and the unused `artifact_size`/`max_size` params) — behavior-preserving,
  since the penalty was always 0.0, so `composite` values are unchanged; full non-slow suite green.
- **2026-06-28** — #133 (cross-phase orchestrator) **built native**. Upstream's PR is a real
  `evolution/loop/` orchestrator (history + scheduler + orchestrator + `evolve_all` CLI), but it
  couples phases in-process, targets a phase set our fork has diverged from (it has `params` — which
  we killed as NULL in #132 — and lacks our `code` phase), and bundles unrelated changes (a Jaccard
  "semantic-preservation" constraint, a model-alias map, a judge cache, length penalties — which we
  just removed in #134). Built our own `evolution/orchestrator/` instead: a propose-only
  `python -m evolution.orchestrator` that sequences skills→tools→prompts→code from a YAML run-spec,
  isolates each phase as a **subprocess** (true fault containment + handles the CLI-only code phase),
  captures each phase's existing `gate_decision.json` at a deterministic `--output-dir`, and writes a
  JSONL run history + summary. Verdict status is grounded in the gate file (not the evolvers'
  inconsistent exit codes); `--allow-pr` is required to honor any PR opt-in (default strips it). Also
  added the missing `--output-dir` CLI option to the skills and tools evolvers so capture is uniform.
- **2026-06-28** — #127 (broad-benchmark-regression-as-a-gate, on the **skill** path) **investigated →
  already covered**. The premise — "we have the regression-floor/oracle analogue for code only" — does
  not hold up: the broad-benchmark deploy gate is the `--benchmark-cmd` hook, and it is symmetric across
  evolvers. `evolve_skill.py` stages `evolved_skill.md`/`baseline_skill.md`, passes `$EVOLVED_PATH`/
  `$BASELINE_PATH`/`$RUN_DIR`, fails closed on nonzero exit, and records the outcome in `gate_decision.json`
  — the *same* `run_benchmark_hook` (`evolution/core/quality_gate.py`) the code evolver uses for its
  full-suite tier (`evolve_code.py`). Upstream's `benchmark_gate.py` (a hard-coded TBLite command → JSON
  `{score|pass_rate}` → 2% absolute floor) is a strict special case of our BYO hook: the user runs the broad
  benchmark on both arms and applies any floor inside the command (a 2% floor reproduces upstream's
  threshold). The one thing the **code** path uniquely has — the automatic per-candidate Tier-5 `tests/tools`
  regression floor (`evolution/code/gate.py`, baseline-vs-repaired diff) — has **no cheap analogue for
  skills**: a skill is a scoped, opt-in instruction artifact installed in isolation per task
  (`SkillFileInstaller`), so it can't deterministically break a sibling the way a code edit to tool A breaks
  tool B's unit test. Its only broad-regression failure mode (a widened trigger that mis-fires on unrelated
  tasks) is stochastic agentic behavior detectable solely by expensive rollout — exactly what the opt-in hook
  provides — so no automatic skill backstop is warranted. No code change; added a skill broad-regression
  `--benchmark-cmd` recipe to `docs/usage.md` to retire the perception gap. (Note: the standalone
  `evolution.validation.closed_loop` CLI is **not** the skill's broad gate — it is tool-only, requires
  `--tool`/`--hermes-repo`, and even inline scores the skill's *own* task suite, not a broad benchmark.)
- **2026-06-28** — #85 (Claude Code subscription backend — FastAPI OpenAI-compatible shim over
  `claude-agent-sdk`) **investigated → SKIP (ToS-prohibited + server-side-blocked)**. The proposed
  design re-exposes Pro/Max subscription OAuth as a `provider:custom` + `base_url` endpoint for the
  framework's LM roles — exactly the pattern Anthropic banned. Anthropic's compliance docs (updated
  2026-02-19/20) state: *"Using OAuth tokens obtained through Claude Free, Pro, or Max accounts in any
  other product, tool, or service — including the Agent SDK — is not permitted and constitutes a
  violation of the Consumer Terms of Service"*; subscription OAuth is exclusive to Claude Code +
  claude.ai. It is also enforced at the wire: since 2026-01-09 subscription OAuth tokens are rejected
  outside the official CLI (the OpenClaw/OpenCode/Roo Code/Goose sweep), and subscriptions stopped
  covering third-party tools on 2026-04-04 — so the shim is neither permitted nor technically viable.
  The legitimate path is unaffected and already present: our closed-loop agent backend drives the
  **real `claude -p` CLI** with `CLAUDE_CODE_OAUTH_TOKEN` (the sanctioned scripts/CI use of the official
  product, not a re-exposing shim). What it does *not* buy is cheap subscription inference for the GEPA
  optimizer/judge roles — the part #85 wanted and the part that is prohibited. Claude inference for
  evolution stays on API-key billing via `resolve_default_lm`. No code change. (Sources: Claude Code
  Authentication docs, code.claude.com/docs/en/authentication; The Register, 2026-02-20.)
- **2026-06-28** — **ruff linter adopted** (the #106 deferred follow-up). Measured first: at ruff's
  default rule set (`E4/E7/E9/F`) the tree had 134 findings, 79 auto-fixable. Adopted the linter only —
  `[tool.ruff.lint] select = ["E4","E7","E9","F"]` in `pyproject.toml`, ruff pinned in the dev group, a
  `ruff` pre-commit hook (`--fix`), and a `lint` CI job (`uv run ruff check evolution tests`). The
  **formatter stays deferred**: 169/225 files would reflow (+10.8k/−4.5k lines) and the deferral blocker
  from #106 is real — the tree is written to ~100 chars (1208 lines > 88 but only 318 > 100), so `E501`
  is intentionally **not** selected and a one-shot `line-length = 100` reformat (logged in
  `.git-blame-ignore-revs`) remains the separate, optional path. Shipped code (`evolution/`) is fully
  clean under the rules; `tests/**` gets a scoped ignore for three style-only rules (`E402`/`E702`/`E741`)
  while every pyflakes (F) real-bug rule stays enforced there. The linter earned its keep immediately —
  it surfaced **two genuine latent issues**: an undefined `Optional` in `tool_module.py` (F821, masked by
  `from __future__ import annotations`) and dead test setup in `test_validator.py` (a `_ScriptedRunner`
  built then overridden, F841). Full non-slow suite green (1754 passed). Making the `lint` job a *required*
  status check is a one-line branch-protection change for the maintainer; gitleaks (#107) and the one-time
  pre-commit `--all-files` hygiene pass remain deferred.
- **2026-06-30** — Incremental sweep. Backlog 70 → **72 open**; **two new PRs since the 2026-06-28 cycle**,
  both **already covered → SKIP** (no action):
  - **#139** (MaxFreedomPollard, "emit a pull request for an evolved skill") — its central premise, *"there is
    no git or `gh` code anywhere in the tree; only a `create_pr` flag and a dry-run print exist,"* is true of
    **upstream**, not our fork. We already ship `evolution/core/pr_automation.py` (`create_pr`) wired into **all
    four** evolvers (skills `evolve_skill.py:1737`, tools, prompts, code), gated behind `--create-pr` flags and
    governed centrally by the orchestrator's propose-only boundary (`--allow-pr`, `__main__.py:26`
    `_enforce_propose_only`). #139 adds the last-mile PR step for the **skill path only** — a strict subset of
    what we have, minus the cross-phase propose-only enforcement. The dry-run line it cites as "the only trace"
    (`evolve_skill.py:808`) is just the `--dry-run` message; the real path is at line 1737. Nothing to adopt.
  - **#140** (aranya-chatterjee, "Fix evolve_skill validation and assembly bugs (fixes #119)") — all three code
    bugs land in settled clusters and are already fixed here: Bug 1 (validate `evolved_full` not `evolved_body`)
    is the constraint-validator cluster — we validate the reassembled full text at `evolve_skill.py:1261`; Bug 2
    (nested frontmatter on reassembly) is **#104**, already done — `reassemble_skill` strips a leading
    frontmatter-like block (`skill_module.py:118`); Bug 3 (declare `optuna`) is **#41/#105**, already done — it
    ships via `dspy[optuna]` in the `miprov2` extra. Re-fix of already-covered ground.
- **2026-07-06** — Incremental sweep. Backlog steady at **72 open** (6 opened, 6 closed since 2026-06-30);
  **six new PRs #142–#147**. Five already covered → SKIP; **one genuinely-new latent bug surfaced → new
  action item** (symlink-aware skill resolver, see table):
  - **#142** (bob0x-ai, "GEPA 3.2.x compat + symlink resolver + validator false-positive") — 6 fixes, 5 already
    covered: three GEPA/DSPy-3.2 compat items (explicit `reflection_lm`, `max_full_evals`-only, 5-arg metric)
    are the GEPA-compat cluster; `validate_all(body)` → `raw` is the constraint-validator cluster; one is
    provider-specific. **The 6th is real and new here:** `find_skill` traversal via `Path.rglob("SKILL.md")`
    does not descend into **symlinked** skill dirs on Python <3.13 — our `skill_sources.py:63/67/80` use exactly
    that. Promoted to an action item (CHERRY-PICK, recommended).
  - **#143** (diegokolling, "read Hermes session history from state.db, not dead JSON") — **our #102**, already
    done; `iter_hermes_sessions`/`_iter_state_db_sessions` read `state.db` **read-only** (`mode=ro`,
    `external_importers.py:414`) with JSON fallback. #143's only extras are cross-platform DB discovery
    (`HERMES_STATE_DB` env override + Windows `%LOCALAPPDATA%`) — marginal for our macOS/Linux posture; the
    one-line env override is a trivial optional pickup, not worth a cycle. SKIP.
  - **#144 / #145** (Sunwo0u, "HSE … sanitized evidence packet") — report-bundle-only PRs (9 and 13 files, all
    under `reports/…`), no reusable mechanism — the same status-token evidence pattern already skipped for
    #108/#117/#120. SKIP.
  - **#146** (mgandal, "extract evolved instruction with overfit/collapse guard") — premise (*"`skill_text` is
    an input field never mutated, so evolved == baseline"*) is **upstream-only**: our `skill_text` is a
    `@property` over `signature.instructions` (`skill_module.py:113`), the surface GEPA mutates — the
    extraction cluster (#5/#24/#49). Its metric caveat (keyword-overlap prefers boilerplate) is moot — we
    default to `LLMJudge` + closed-loop behavioral scoring. SKIP.
  - **#147** (andreransom58-coder, "DSPy compat + report material skill diffs honestly") — overlaps #142
    (GEPA compat), #140 (optuna, validate full file), #146 (extraction). Its one distinct idea — track
    `material_diff` separately so a run never claims a deployable win when the saved `SKILL.md` is
    byte-identical to baseline — is largely subsumed here: an identical artifact scores ~0 improvement and our
    behavioral deploy gate won't deploy it. Low-value belt-and-suspenders; parked, not adopted. SKIP.
- **2026-07-06** — #142 (partial, symlink-aware skill resolver) **fixed**. `HermesSkillSource` discovery used
  `Path.rglob("SKILL.md")` at three sites; `rglob` refuses to descend into **symlinked** directories on Python
  <3.13 (the `recurse_symlinks` kwarg is 3.13-only), so a Hermes layout that symlinks user-installed skills into
  the framework tree resolved "not found" (reproduced red on 3.13: the symlinked-*directory* discovery tests
  failed while the symlinked-*file* case already worked, since `rglob` matches symlinked files). Replaced all
  three sites with one private `_iter_skill_files(root)` helper: `os.walk(followlinks=True)` + a `(st_dev,
  st_ino)` visited-set cycle guard (prunes a symlink pointing back to an ancestor — provably terminating) +
  per-level `sort()` for deterministic first-match-wins; `onerror`/`stat` failures are debug-logged and skipped
  (parity with the old silent skip). `find_skill` now walks once and runs both passes over the materialized
  list. Scope held to `HermesSkillSource` — the flat `ClaudeCode`/`LocalDir` sources already follow symlinks via
  `is_file()`/`iterdir()` (noted inline). 9 new symlink tests (correctness C1–C5, cycle/permission/dangling
  safety S1–S3, determinism D1) with an `os.symlink`-unavailable skip-guard; full non-slow suite green (1763
  passed, +9), ruff clean. The other five #142 fixes are already-covered clusters (GEPA/DSPy compat,
  validate-full) — not adopted.

## Action items (open)

Disposition lens: against our diverged tree, these are "cherry-pick the idea/mechanism,"
not "merge the PR." Our-code anchors point at where the change would land.

| # | What | Why we'd benefit | Our-code anchor | Disposition | Status |
|---|---|---|---|---|---|
| #132 | **Parameter-description evolution** — AST reader/writer for `parameters.properties.{name}.description`; dot-labeled `[[tool.param]]` GEPA targets | A genuinely missing evolution axis. We have the constraint stub but no evolver touches param descriptions. | `evolution/core/constraints.py:112` (`max_param_desc_size` stub), `evolution/tools/tool_source.py` (treats `input_schema` read-only), `tool_module.py` | **Investigated → NULL** (axis saturated — param-description text doesn't move agent value-selection; see review log, 2026-06-28) | ✅ |
| #102 (+ #26) | Skill importer reads Hermes **`state.db`** (SQLite) + filters machine-generated user messages; #26 adds a 3-stage relevance filter (LLM keyword expansion + full-corpus scan) | Our skill importer reads stale `~/.hermes/sessions/*.json`; our own validation path proves `state.db` is canonical. The skill path lags the tool path on data quality + recall. | `evolution/core/external_importers.py` (`HermesSessionImporter`, `RelevanceFilter`); cf. `evolution/validation/hermes_runner.py` (`parse_session_from_db`) | **DONE** — `iter_hermes_sessions` now reads `state.db` first; both skill + tool paths fixed (importer 0 → real pairs). #26 recall improvement deferred. | ✅ |
| #134 | Graduated / class-aware skill-size cap (soft target + hard ceiling + ramp) | Premise was a **pre-cap length-penalty cliff** in fitness that docks under-cap skills — but on our fork that penalty is dead code (no caller passes the size it needs, so it is always 0.0). Length pressure is handled deliberately by the proposer's length budget + two hard deploy ceilings. | `evolution/core/fitness.py` (the dead `length_penalty`), `evolution/core/constraints.py` (`_check_size`, `effective_absolute_char_ceiling`) | **Investigated → not applicable** (dead-code premise; removed the vestigial penalty rather than build the ramp; see review log, 2026-06-28) | ✅ |
| #106 | `.github/dependabot.yml` + `.pre-commit-config.yaml` | Missing infra hygiene; we have neither. | `.github/`, repo root; reconcile with `.github/workflows/tests.yml` (py3.10–3.13 matrix) | **DONE** — added `dependabot.yml` (uv + github-actions) + `.pre-commit-config.yaml` (hygiene hooks); ruff **linter** later adopted as a follow-up (see review log, 2026-06-28); ruff-format + gitleaks (#107) deferred | ✅ |
| #133 | Cross-phase orchestrator + unified `evolve_all` CLI — **shape only** (dependency-ordered phases, fault isolation, JSONL run history) | We had no unified driver sequencing skills→tools→prompts→code; only per-subsystem. | `evolution/orchestrator/` (new); borrows upstream's shape, keeps our gated evolvers + propose-only boundary | **DONE** — built native `python -m evolution.orchestrator` (subprocess-isolated phases, gate-grounded verdicts, JSONL history); rejected upstream's in-process coupling + bundled cruft + auto-scheduler | ✅ |
| #127 | Broad-benchmark-regression-as-a-gate, applied to the **skill** path | We have the regression-floor/oracle analogue for **code** only. | `evolution/code/gate.py` (the code analogue); skill deploy gate in `evolution/skills/evolve_skill.py` | **Investigated → already covered** (the broad-benchmark gate is the `--benchmark-cmd` hook, symmetric with the code evolver's full-suite tier via the shared `run_benchmark_hook`; code's automatic `tests/tools` floor has no cheap analogue for scoped skill artifacts — see review log, 2026-06-28) | ✅ |
| #85 | Claude Code **subscription** backend — FastAPI OpenAI-compatible shim over `claude-agent-sdk` | A new capability: our OAuth backends cover OpenAI-Codex + Nous, not Claude-subscription. Plugs in as `provider: custom` + `base_url`, no code-layer change → cheaper evolution. | `evolution/core/hermes_provider.py` (`resolve_default_lm`); standalone `scripts/` proxy | **Investigated → SKIP (ToS-prohibited)** — re-exposing Pro/Max subscription OAuth via an OpenAI shim / the Agent SDK violates Anthropic's Consumer Terms (clarified 2026-02-19/20) and is server-side-blocked since 2026-01-09; the sanctioned `claude -p` + `CLAUDE_CODE_OAUTH_TOKEN` backend is already present, but it doesn't give cheap subscription inference for the GEPA roles (see review log, 2026-06-28) | ✅ |
| #142 (partial) | **Symlink-aware skill resolver** — `find_skill` traversal follows symlinked skill directories | `Path.rglob("SKILL.md")` doesn't descend into symlinked dirs on Python <3.13; a Hermes layout that symlinks user-installed skills into the framework tree would silently resolve "not found." Real latent bug in a path we own. | `evolution/core/skill_sources.py` (`HermesSkillSource`, was 3 `rglob("SKILL.md")` sites) | **DONE** — replaced the three `rglob` sites with one cycle-safe `_iter_skill_files` helper (`os.walk(followlinks=True)` + `(st_dev, st_ino)` visited-set + sorted deterministic order); only `HermesSkillSource` touched (flat ClaudeCode/LocalDir sources already follow symlinks via `is_file()`). 9 new symlink tests, full non-slow suite green (1763) — see review log, 2026-07-06 | ✅ |

## Optional / low-value (parked)

- **#136** — lift the **BCa interval + exact paired sign-flip test** as an optional
  *diagnostic* into `evolution/core/stats.py` (our code even notes "BCa is the upgrade
  path once N≥20"). **Not** as the deploy gate — our CI-lower-bound non-inferiority gate
  is deliberately tuned for the small-N holdout regime and stays.
- **#51** — one-line `load_skill` dir-path guard (`evolution/skills/skill_module.py`).
- **#137** — a `_check_no_secrets` *skill* constraint (we scan secrets on the import path
  but not in `ConstraintValidator`).
- **#69** — its only salvageable idea is a ~50-line TF-IDF/embedding "purpose-drift" gate.
  Reimplement against our `quality_gate.py`/`constraints.py` if ever wanted —
  **do not merge the PR** (see do-not-merge below).

## Already covered — do not re-review (unless our code in that area changes)

- **Constraint-validator "validate `evolved_full` not `evolved_body`" cluster**
  (#7, #23, #50, #53, #95, #97, #104, #113, #114, #140): we validate the reassembled full skill
  at every gate; upstream's patched call site (`validate_all(evolved_body, …)`) doesn't
  exist in our tree. #140 (fixes #119) also re-fixes #104 (frontmatter strip) + #41/#105 (optuna
  dep), both already done here — see review log, 2026-06-30.
- **PR emission / last-mile automation** (#139): we already have `evolution/core/pr_automation.py`
  (`create_pr`) wired into all four evolvers behind `--create-pr`, with the orchestrator's
  propose-only `--allow-pr` boundary on top. #139 adds a skill-only subset of this — see review
  log, 2026-06-30.
- **GEPA / DSPy 3.x compat** (#13, #14, #35, #46, #48, #73, #91, #109, #137-core): moot —
  we pin `dspy>=3.2.0,<3.3` + a gepa git override and call the modern API
  (`max_full_evals`, `reflection_lm`, 5-arg metric); the SkillModule "GEPA mutates
  `signature.instructions`" fix is already in (`skill_module.py`).
- **Skill-text extraction** (#5, #24, #49): done — `skill_text` is a property over
  `signature.instructions`.
- **Fitness / judge** (#25, #28): superseded by our `LLMJudge` + closed-loop behavioral
  scoring; #28's deterministic text-similarity proxy is a step backward.
- **Model providers** (#8, #15, #19, #22, #92, #112): subsumed by `resolve_default_lm`
  (~20 providers, OpenAI-wire/local, two OAuth backends with token refresh the PRs lack).
  #85 (Claude subscription) looked genuinely new but is **SKIP** — re-exposing subscription
  OAuth is ToS-prohibited and server-side-blocked (see review log, 2026-06-28).
- **#126 local-first code evolver** — *hard skip*. Its fitness is
  `0.25·AST-complexity + 0.25·keyword-coverage + 0.5·self-LLM-judge` with no held-out
  split, no oracle, no regression floor, no surface freeze, and it writes to disk ungated —
  exactly the circular/gameable eval `evolution/code/gate.py` exists to prevent.
- **Phase / HSE mega-PRs** (#30, #42, #86, #98, #108, #117, #120) and **eksays #129/#130/#131**:
  parallel reconstructions of our roadmap with *synthetic-judge* gates; we use real
  `hermes -z` / `claude -p` behavioral gates + a held-out anti-gaming split they lack.
  #108 is 206 files of status-token evidence, no reusable mechanism.
- **Misc**: #101 (skill-body-as-artifact) already live via `reassemble_skill`; #100
  (registry monitor) overlaps our richer `evolution/monitor/`; #41/#105 (optuna) done;
  #76 (live smoke) duplicates our manual smoke; #20/#21 (MAD) subsumed by
  `stats.paired_bootstrap` + the A/A noise floor; #45 (SSoT) weak fit; #77–#80 (bot cases)
  have no harness here.

## Do not merge — sensitive data

These commit material that violates our non-exposure posture; never import them:

- **#94** — embeds real client Telegram group IDs and chat names plus ~50 log/state files.
- **#17** — local repo paths and expired API-key dumps in the PR body/scripts.
- **#69** — a fork-of-a-fork dump (parallel `*_v2` architecture + an unrelated kanban
  side-project + run-output bloat + local plan docs); ~85% noise around one small idea.

## Snapshot — 70 open PRs by group (2026-06-28)

For traceability of this cycle. Disposition key: A=adopt, C=cherry-pick, I=investigate,
S=skip (already covered / superseded), X=do-not-merge.

- **Constraint-validator / skill-assembly**: #7 S, #23 S, #50 S, #51 (C, dir guard), #53 S, #95 S, #97 S, #104 S, #113 S, #114 S, #134 **n/a (investigated)**
- **GEPA / DSPy 3.x compat**: #13 S, #14 S, #35 S, #46 S, #48 S, #73 S, #91 S, #109 S, #137 S (parked: secret-scan constraint)
- **Fitness / judge / significance + skill extraction**: #5 S, #24 S, #49 S, #25 S, #28 S, #136 (parked: BCa diagnostic)
- **Model providers / backends**: #8 S, #15 S, #19 S, #22 S, #85 **SKIP (ToS, investigated)**, #92 S, #112 S
- **Session importers / discovery / guardrails**: #26 **C** (deferred follow-up), #40 S (subset of #102), #102 **done**, #94 X
- **Reliability / real-mutation / gating + code evolver**: #16 S, #17 X, #75 S, #89 S, #126 S, #127 **covered (investigated)**
- **Phase / HSE mega-PRs**: #30 S, #42 S, #86 S, #98 S, #108 S, #117 S, #120 S
- **eksays Phase 1–5**: #129 S, #130 S, #131 S, #132 **null (investigated)**, #133 **done (built native)**
- **Misc / infra / tests**: #20 S, #21 S, #45 S, #69 X, #76 S, #77 S, #78 S, #79 S, #80 S, #100 S, #101 S, #105 S, #106 **done**, #107 (parked)

### Delta — 72 open PRs (2026-06-30)

Two new since the 2026-06-28 snapshot; both **S (already covered)**, no backlog regroup needed:

- **PR emission / automation**: #139 S (skill-only subset of our `pr_automation.create_pr` + orchestrator `--allow-pr`)
- **Constraint-validator / assembly**: #140 S (fixes #119 — re-fix of the `evolved_full` cluster + #104 frontmatter strip + #41/#105 optuna dep)

### Delta — 72 open PRs (2026-07-06)

Six new (#142–#147); 5 S, 1 new action item. No backlog regroup:

- **GEPA/DSPy compat + resolver**: #142 — 5 fixes S (compat cluster + validate-full), **1 → DONE** (symlink-aware skill resolver, `skill_sources.py`; see review log, 2026-07-06)
- **Session importers**: #143 S (our #102 — state.db read-only already shipped; only cross-platform discovery extras)
- **Phase / HSE mega-PRs**: #144 S, #145 S (Sunwo0u sanitized evidence packets, report-only, no mechanism)
- **Fitness / judge / skill extraction**: #146 S (extraction cluster — our `skill_text` is a property over instructions), #147 S (compat + `material_diff` reporting subsumed by our behavioral gate)
