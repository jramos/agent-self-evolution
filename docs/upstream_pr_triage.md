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

**Cadence: quarterly**, or event-driven when we touch a subsystem with a parked item.
Yield falls as the fork diverges — the first cycle mined ~7 actionable items from 70 PRs,
a later one 1 defect plus 2 liftable ideas from 26 new PRs — while upstream is unmaintained
and we are 0 commits behind it, so waiting accrues no merge risk.

1. **Pull the open PRs** with metadata:
   `gh pr list --repo NousResearch/hermes-agent-self-evolution --state open --limit 200 --json number,title,author,changedFiles,additions,deletions`
2. **Group by theme** from titles. Expect heavy duplication — the same early bug is
   often re-fixed by dozens of contributors.
3. **Pre-filter on titles before reading any diff.** Match each new title against the
   "Already covered" clusters below. A match is dispositioned `SKIP` on the title plus one
   confirming glance; only non-matching titles earn a deep-dive pass. Most of the backlog
   clears here, which is what keeps the expensive step affordable.
4. **Deep-dive each group with the right lens.** For every PR: read it
   (`gh pr view N --repo … --json title,body` + `gh pr diff N --repo …`), understand the
   root problem, then **grep/read our actual code** to decide our current state. Our fork
   may have fixed the bug differently, replaced the code path entirely, or never had it.
   Many upstream diffs won't even apply (they patch call sites we refactored away).
5. **Classify**: `ADOPT` / `REBUILD <what>` / `INVESTIGATE` / `SKIP (<why>)`.
6. **Don't re-review settled clusters** — see "Already covered" below; only reconsider if
   our code in that area changes materially.
7. **Flag sensitive PRs as do-not-merge** — several upstream PRs commit client identifiers,
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
    that. Promoted to an action item (REBUILD, recommended).
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

- **2026-08-31** — Quarterly sweep. Backlog **72 → 92 open**: 31 opened, 11 closed (5 of them
  opened *and* closed inside the window), leaving **26 new PRs** (#149–#183) and closing 6 of
  the 72 previously tracked. Of the 26: **24** SKIP (already covered, superseded, or otherwise
  not actionable), **1** a verified defect in our tree promoted to an action item, **1** hostile.
  Three of those skips nonetheless prompted native work here — #162, #154 and #179.
  Structural finding worth recording: upstream has merged **3 PRs ever** (most recent
  2026-06-17) against 52 closed-unmerged, and this fork is **156 commits ahead, 0 behind** —
  there is no upstream code to catch up on, only a contributor backlog to mine. The 26 new PRs
  come from 21 mostly one-shot authors. This is what moved the cadence to quarterly and added
  the title pre-filter (step 3).
  - **#149** (diegokolling, rank pre-filter candidates by relevance before the LLM-scoring cap) — the one
    **verified defect**, and the long-deferred #26 recall follow-up. `RelevanceFilter` picks
    candidates with a boolean predicate, then truncates at `max_examples * 3` in
    source-then-import order, so the strongest matches past the cap are dropped before scoring.
    Two aggravations found while confirming it: `build_dataset_from_external` concatenates per importer, so an
    overflowing source can crowd out a later one entirely; and the scoring loop's early break
    means candidate *order* decides the output set on every run, not only on overflow. Promoted
    to an action item (REBUILD).
  - **#179** (ideas24h, importer test isolation) — premise **falsified against our tree**: autouse fixtures
    already neutralize `STATE_DB` for the importer suites, verified by running them on a machine
    with a real populated `~/.hermes/state.db`: the importer suites pass with no leakage. One
    narrow residual — a non-UTF8 legacy session file escapes the
    `except (JSONDecodeError, OSError)` guard — is tracked as the hardening action item below. SKIP.
  - **#162** (MaxFreedomPollard, Phases 2–5) — SKIP wholesale: a parallel reconstruction of ground we already hold,
    with gates weaker than ours. Two narrow ideas **rebuilt natively** rather than adopted: OS-level
    confinement for the code-evolution test runner, and a minimum-detectable-effect diagnostic. Its
    per-tool pairwise interference gate is a real gap but costly; parked with the anchor recorded.
  - **#150** (MaxFreedomPollard, objective ground-truth verifier) — SKIP. The checker is genuinely deterministic and
    LLM-free, but its oracle is a 17-paper table of hardcoded bibliographic facts while the artifact GEPA mutates is
    the skill text itself, so the answer key is memorizable into the skill body and a held-out split
    cannot rescue a 17-item universe. Same circularity that got #126 hard-skipped, better disguised;
    it also targets a knowledge/recall axis our own campaigns found saturated. About 65% of its diff
    is machinery carried over from #162.
  - **#182** (numandev1, full self-evolution feature set) — SKIP. Its one genuinely absent mechanism is a
    post-deploy canary with auto-rollback, but that is infrastructure ahead of demand here: our
    deploy gate is a verified cold path with two deploys ever, both demos. Its code sidecar also
    shells out to a separately-packaged **AGPL** tool; the PR discloses and isolates that boundary
    carefully — a README section, module docstrings, and a test asserting nothing under
    `evolution/` imports it — but it remains a licensing entanglement we don't want.
  - **#154** (smfworks, constraint validator + JSON repair) — the validator half is the settled cluster; the
    JSON-robustness half is real and was rebuilt without taking the proposed dependency.
  - **#166** (trippyogi, add an MIT LICENSE) — surfaced that this fork has no `LICENSE` file. Deliberately not
    actioned here: upstream carries no license at all, which makes this a licensing decision rather
    than repo hygiene.
  - **#163, #165, #176** (RomanXSad, RomanXSad, enzo-adami — growth-limit waiver, skipped-holdout reporting, rejected-variant
    overwrites) — SKIP. Our gate architecture structurally avoids the first two bug classes, and all
    four evolvers already timestamp their default output dirs. One trace initially read as a defect
    in the rejection path turned out to be behavior the code deliberately documents.
  - **#157** (Baal-TehDriverman) — hostile; see do-not-merge below.
  - Remaining: #153, #155, #159, #161, #167, #168, #173, #174, #177, #178, #183 (GEPA/DSPy compat
    and validate-full clusters), #158, #160, #180, #181 — all SKIP, dispositioned by cluster.

- **2026-08-31** — #149 (rank pre-filter candidates by relevance) **fixed**. `RelevanceFilter`
  qualified candidates with a boolean predicate and then truncated at `max_examples * 3` in
  source-then-import order, so on any skill where heuristic matches overflow the cap the
  strongest matches were dropped before the LLM scorer ever saw them. Confirming it surfaced a
  second, wider path to the same loss: the scoring loop breaks at `len(examples) >= max_examples`,
  so candidate *order* decides the output set on every run, not only when the cap engages.
  Replaced the boolean with `_relevance_score`, returning a tiered tuple
  `(name_match, name_words, keyword_overlap)`; `_is_relevant_to_skill` is now `any()` over that
  tuple. A tuple rather than weighted integers because `skill_name` is caller-supplied and
  unbounded, so no fixed weight can stop a long name's word count from outranking a full-name
  match. The keyword tier contributes 0 below two overlaps — the one way this change could have
  silently *widened* the qualifying set instead of reordering it. Candidates now sort
  strongest-first before both caps; `sorted` is stable, so equally-scored messages keep import
  order and the seeded backfill is untouched. 6 new tests, including a 2000-case oracle
  comparison against a verbatim copy of the old predicate proving the qualifying set is
  unchanged; the pre-existing relevance tests were deliberately left unmodified as the
  equivalence surface (the boolean predicate now has no production callers and survives only
  to keep that guarantee tested — noted in its docstring so it isn't mistaken for live code
  or deleted). Review of the first pass found two things worth recording. First, every
  ordering test used a single-word skill name, which makes the middle tier indistinguishable
  from the top one, so a weighted-sum implementation would have passed all of them; added a
  multi-word-name case where the tiers genuinely disagree and confirmed by mutation (swapping
  the tuple for a weighted sum) that the new tests fail and the others do not. Second, the
  equivalence fuzz was weaker evidence than claimed — its charset left case folding and both
  punctuation strippers as no-ops on every case and never varied `skill_text`; widened to
  span case, punctuation, non-ASCII and the 500-char truncation boundary, though score-tuple
  diversity stays low, so it is an equivalence check over the normalisation paths rather than
  a broad exploration of the score space. 10 tests total. Full non-slow suite green
  (1773 passed, +10), ruff clean.
  Two consequences to carry forward. Scoring every tier removed the old short-circuit, so the
  keyword set is now built on the name-match path too; it is cached on `skill_text`, which
  cuts that cost from ~90x the old predicate to ~3x (0.024s per 20k messages), negligible
  ahead of a pipeline that then makes hundreds of LLM calls. And for the common single-word
  skill name the top two tiers coincide, leaving keyword overlap as the only intra-tier
  discriminator — a count that grows with message length, so eval-set composition now skews
  toward more verbose messages. That replaces one arbitrary intra-tier order (import order)
  with another, and is not obviously worse; normalising overlap by message length or capping
  the tier would remove the skew, and is parked rather than tuned on intuition. Note for future
  cycles: eval sets drawn after this change differ from earlier ones in both composition and
  train/val/holdout assignment, so don't compare them naively across the boundary.
- **2026-08-31** — #162 (partial, confine code-evolution test execution) **fixed**. `run_test`
  executed pytest against an LLM-modified worktree through a bare subprocess, while the agent
  runner two directories away refuses to run unconfined at all — an asymmetry in our own
  doctrine, and the code path where autonomously-generated source actually runs. Extracted the
  macOS profile builder, availability check and error type into `evolution/core/sandbox.py`, and
  gave `WorktreeEnv` a `require_sandbox` policy plus a `sandboxed` posture resolved once at
  create time. Reach was wider than first scoped: eleven `run_test` call sites rather than six,
  and three entry points drive the loop — the single-tool evolver, the campaign, and the gaming
  audit, whose proposer is deliberately built to game the gate and therefore has the strongest
  claim on confinement. All three take `--require-sandbox`; the bug harvester does not, since no
  LLM-authored code executes there.
  Two findings from review shaped the result. First, **a containment failure could have been
  certified as a passing gate**: `sandbox-exec` exits 65 without running the child when a profile
  fails to compile, the gate special-cases only pytest's exit 5, and its failure parser returns an
  empty set on unrecognised output — so a run where zero tests executed would have read as "no
  failures". A confined run whose exit code is not one of pytest's own now raises instead of
  returning a result. Second, the write-root argument is **a no-op on macOS**: the profile
  blanket-allows the temp roots and the run dir lives under one, so the real boundary is
  "non-temp writes denied; reads, process-exec and network unrestricted". The recorded posture
  says exactly that rather than implying isolation, and the containment test targets a path
  outside the temp roots — inside them it would have passed while proving nothing. Also fixed a
  design flaw the tests caught: the wrapper re-derived availability per call, so the posture we
  *recorded* and the posture we *applied* were independent judgements; the resolved value is now
  threaded through. 18 tests, including a real denied-write proof on macOS and a negative control
  that genuine pytest exit codes still return results.
  A second pass hardened the result further. The fuzz driver in the gaming audit executed the
  candidate through its own subprocess, so the flag promised confinement the harness's primary
  execution did not give — every in-worktree execution now goes through one `confine()` seam.
  Signal deaths (negative exit codes) were being misdiagnosed as containment failures, which on
  macOS alone would have dropped OOM-killed organisms from the campaign denominator; only
  positive non-pytest codes escalate now. Write roots are resolved before interpolation, since
  the kernel matches canonical paths and `mkdtemp` returns the uncanonical form — the named root
  was granting nothing. Paths containing quote, paren or backslash are rejected outright rather
  than interpolated, because SBPL cannot escape them and a crafted path could otherwise append
  allow rules of its own. Containment failures now raise a distinct `ContainmentError` so the
  campaign's skip handler cannot absorb a systemic failure as a fleet of skipped organisms, and
  each entry point checks the policy once at startup rather than discovering it per candidate
  (which also closed a tempdir leak per refusal). The posture is recorded for the campaign as
  well as the single-tool evolver, and surfaced in the human-review PR body, where it had been
  computed and then dropped. Also corrected there: the regression-floor line read a key that does
  not exist, so every PR body claimed the floor FAILED — including runs that deployed because it
  was green. 31 tests, full non-slow suite green (1794 passed, +31), ruff clean.

- **2026-08-31** — **Gate integrity: a pytest run that could not answer was scored as "nothing
  failed"** (found while reviewing the containment change; not an upstream item). `failing_tests`
  discarded the run's exit code and returned whatever the failure parser produced, and that parser
  yields an **empty set** for any output it does not recognise. `run_code_oracle_gate` consumes
  only that seam, so an empty set read as "nothing failed". Demonstrated against the real gate with
  a deliberately wrong repair: a **timed-out** run, a **usage-error (exit 4)** run whose
  `ERROR: not found:` line the parser does not match, and a **signal-killed** run each returned
  `decision='correct', deploy=True`. The timeout path is the most reachable of the three — the
  oracle scope runs the whole fix-commit test file while `fixed` was established on narrow bug
  tests, so an LLM-introduced hang or complexity blowup in a sibling path lands exactly here, and
  an introduced infinite loop is among the commonest repair defects.
  Fixed by making the seam refuse: only exit 0, 1 and 5 make a complete statement about what
  failed, and anything else raises rather than returning a set. Exit 5 stays authoritative because
  callers already reason about "no tests collected" explicitly. `harvest._failures` turned out to
  re-parse the output itself rather than use the seam, so the campaign's own bug-test computation
  bypassed the check entirely — it now delegates, which removes the duplicate parse as well. The
  refusal has its own type so the campaign ledger records `run_inconclusive` instead of blaming
  worktree setup for a candidate that simply could not be measured. Mirrors what the SWE-bench env
  already did: an id it cannot account for is treated as failing so it cannot certify as fixed.
  Review of that first pass corrected the rule and widened it. The rule was a proxy for the wrong
  invariant: exit 2 is *already conclusive* — an import failure exits 2 while naming the file it
  could not import — so refusing every code outside 0/1/5 would have dropped every candidate whose
  buggy parent fails at import, an unremarked change to the instrument's population in the
  direction that flatters the rate. The invariant is now **no failure evidence and no authoritative
  exit**: named failures count whatever the exit code, a clean exit 0 is a complete answer, and
  everything else refuses. That also closed a path where exit 5 could still certify, since the gate
  records `bug_tests_passed` as the negation of an empty set.
  The same defect turned out to live in the **regression floor**, on the product deploy path, where
  it is more reachable than anywhere else: the floor runs the whole `tests/tools` subset under a
  600s timeout on every deploy decision, and a floor that hung returned `deploy=True` with reason
  "regression floor green" while the guard recorded `repaired_failure_count: 0` next to
  `duration_seconds: 600.0`. The baseline half had it in the opposite direction — a hung baseline
  makes every pre-existing failure look introduced, rejecting a correct repair and naming it as the
  cause. Both are guarded, as is the held-out check, which was filing a hang as
  "teaching-to-the-test": an accusation of gaming recorded against a run that produced no verdict.
  Two further corrections. An inconclusive run *while scoring a repair* now counts as a failed seed
  rather than skipping the organism — by that point the repair has already passed its bug tests, so
  a hang over the wider oracle scope is the repair's own doing, and dropping it would inflate the
  rate exactly as the historical bias does. And the failure parser split node ids on the first
  `" - "`, so two parametrized ids containing that separator collapsed into one key, letting a
  newly introduced failure hide behind an unrelated pre-existing one — it now splits at bracket
  depth zero.
  19 tests, full non-slow suite green (1823 passed, +19), ruff clean.
  **Effect on earlier numbers — measured, not assumed.** The three Hermes code campaigns (156
  scored organisms, 468 seeds, 344 seeds true, 113/156 deploy-reachable, across 90 unique fix
  commits) were checked against every route by which this defect could have inflated them, including the
  node-id collapse fixed in the same change (none of the 1823 collected ids contain the separator,
  so that route is empty here too).
  *Two routes are structurally excluded.* The regression floor never ran: `run_code_oracle_gate`
  defaults `floor_paths=None`, the campaign passes none, and the floor body is gated on
  `if floor_paths:` — that hazard only ever applied to the held-out gate in the single-tool
  evolver. And a repair that hangs its **own** bug tests cannot reach the gate as fixed, since
  `fixed=True` requires `run.passed` — so it scores as a failed seed, not a false success.
  *The replayable route measured zero.* All **90/90** fix commits collect cleanly (exit 0) and
  execute authoritatively (86 exit-0, 4 exit-1) — that is a property of the repository, so it
  replays exactly and rules the class out for every oracle-side run. The seed-time runs are not
  covered by that replay, since a repair that breaks the tool's import changes collection; they are
  ruled out separately, because an import failure exits 2 while still printing an `ERROR` line, so
  the pre-fix parse returned a non-empty set and rejected rather than certified. The exit-2/4/5
  class therefore fired on **0 of 156 organisms and 0 of 468 seeds**.
  *One residual is unquantifiable from retained data.* A repair that passed its bug tests and then
  hung only the sibling tests in the same file would have been scored `correct`. The repaired
  sources were never retained — the campaigns kept only the ledger and the report, with no exit
  codes, durations, or per-seed gate decisions — so this cannot be replayed. It is narrow: baseline
  runtimes for those files are a median of 1.79s and a maximum of 10.98s, so reaching the 600s
  timeout needs a **55x to 334x** slowdown, i.e. a non-terminating loop rather than incidental
  slowness. Wall clock bounds it loosely from above: the campaigns ran 3.24h, 2.24h and 1.73h, and
  a single timeout consumes several times the average per-seed budget (5.8x, 7.1x and 20.1x by
  campaign), so at most **43 of 468 seeds (9.2%)**
  could have timed out even had the campaigns done nothing else.
  Net: the published deploy-reachable figures are better supported than the original caveat
  implied — two routes excluded by construction, the replayable one measured at zero, and the
  remainder capped at 9.2% of seeds by an argument that assumes the campaigns did nothing but hang.
  **Retention lesson:** per-seed exit codes are a few bytes and would have made this a grep instead
  of an inference; the repaired sources are gone for good.

- **2026-08-31** — **Minimum detectable effect** (from #162's statistics, rebuilt) **added**. A gate
  could certify a win, or enforce a regression floor, while its sample size was never capable of
  detecting the effect it claimed to police — the verdict said nothing about that, and neither did
  the run's evidence. One pure function in `stats.py` now states it, for the continuous regime
  (judge scores), surfaced as a `power_diagnostics.json` beside the decision plus a console line.
  A paired-binary companion was written and withdrawn before release (below).
  Three choices worth recording. α and sidedness are **derived from the bootstrap's own
  `confidence`** rather than hardcoded, because the gate consumes only the interval's lower bound —
  a one-sided decision; hardcoding a two-sided 0.05 against the default 0.90 interval would inflate
  the figure by about 13% (the z-terms differ by 19%, but the power term is common to both). The
  withdrawn binary version had been parameterised by **discordance rate, not marginal pass rate** —
  paired power
  depends on how often the arms disagree, and closed-loop pass counts are strongly correlated by
  construction, exactly where a marginal model reads optimistic. And the figure claims **no bound**: these use the normal approximation while the correct quantile at n of 8-20 comes
  from a noncentral t and is larger, and understating is the unsafe direction for a diagnostic whose
  purpose is admitting what a sample cannot see.
  Diagnostics only — nothing reads them, and that is checked rather than asserted: a structural test
  requires the diagnostic's values to reach nothing but the console line, mutation-verified by
  feeding one into a gate variable and watching it fail.
  Review then **withdrew half of it before release**. The paired-binary companion returned values
  above the algebraic maximum: `|p01 - p10| <= p01 + p10` bounds the effect, and the normal
  approximation violates that bound whenever `n * discordance < 6.18` — this project's entire
  operating range. Both the worked example in the docs and a test described as hand-verified pinned
  an impossible number, because the hand check recomputed the same wrong formula instead of checking
  the bound. Its lower-bound flag was wrong-signed as well (overstating by ~14% at n=8 rather
  than understating), and the discordance feeding it came from continuous judge differences, which
  are almost never exactly equal — so the rate drifted to 1.0 and the figure was `2.4865/√n`
  restated, carrying nothing about the run. Deferred with those reasons; doing it properly needs the
  Connor form and real pass/fail counts. The continuous half was verified against an exact
  noncentral-t computation, including the direction of its lower-bound claim (low by ~11% at n=8,
  ~5% at n=16).
  A second review round caught the sharper version of the same mistake in the half that shipped.
  The lower-bound direction had been verified against an exact paired **t-test** — but the gate does
  not run one. `paired_bootstrap` returns a *percentile* interval whose spread is the divisor-n
  resample sd with no t-correction, so rejecting on its lower bound is equivalent to requiring
  `t > z * sqrt((n-1)/n)`: 1.5386 at n=8, against a nominal 1.6449. That rule is **anti-conservative**
  (real one-sided error ≈0.08 where 0.05 is claimed), so its true detectable effect is *below* the
  reported figure and `is_lower_bound: True` was wrong-signed for the decision it sat beside — the
  very defect that got the binary regime withdrawn, surviving in the continuous half. The arithmetic
  had been right and the conclusion wrong, because the number was checked against a test this
  codebase never runs. The diagnostic now models the gate's own rule (reporting the effective
  critical multiplier) and claims no bound in either direction.
  Two further corrections from the same review: `ddof` was recorded but never used while the caller
  hardcoded the spread — a knob that looked like a parameter and changed nothing, letting the
  payload misreport its own provenance — so the function now takes the raw differences; and the
  first invariance test guarded the *code* evolver's payload, which this work never touches, proving
  something true and irrelevant. Exact paired tests stay deferred. 15 tests; full non-slow suite
  green (1828 passed), ruff clean.

- **2026-08-31** — **Importer and dataset-builder hardening** (from #154's JSON half and #179's
  residual) **fixed**. Three narrow places where a malformed input crossed a boundary the
  surrounding code did not expect.
  Two dataset-builder sites pulled a JSON substring out of prose and then parsed it unguarded, so a
  bracketed-but-invalid payload escaped as a raw decode error past the `ValueError` the surrounding
  code raises. Both now route through shared helpers rather than repeating the pattern, and the
  helpers also check **shape**: a payload can be valid JSON and still unusable, and a list of
  scalars would otherwise fail much later at a `.get()` far from the cause. The proposed
  JSON-repair dependency was declined — the fix needs no new package.
  Both decode sites were widened, not just the reported one. `UnicodeDecodeError` derives from
  `ValueError`, not `OSError`, so it escaped the legacy session read's guard; and the Claude Code
  history log decodes during line *iteration*, outside the per-line JSON guard, so a single bad
  byte sequence aborted the whole read. That log is long-lived and the likelier of the two sources
  to be mixed-encoding, so fixing only the rarer site would have over-claimed.
  The MIPROv2 fallback now receives the held-out split, which GEPA passes and it silently did not.
  Ours is a genuine named split rather than a slice of the trainset, so passing it replaces the
  optimizer's internal trainset-derived split rather than feeding training data back. Guarded with
  `valset or None`: the optimizer rejects a non-None empty valset, and passing one through would
  convert the path that exists to survive a GEPA failure into a hard crash. One existing test's fake
  optimizer was widened to accept the new keyword — the single existing test edited here, and
  called out rather than slipped in. Deploy integrity is unaffected either way; this changes the
  optimizer's internal candidate selection, not the deploy gate, which runs its own held-out
  behavioural validation downstream.
  8 tests, full non-slow suite green (1836 passed, +8), ruff clean.

## Action items (open)

Disposition lens: against our diverged tree, these are "rebuild the idea/mechanism
ourselves," never "merge the PR" — we do not apply upstream diffs. Our-code anchors point at where the change would land.

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
| #149 (+ #26) | **Rank importer pre-filter candidates by relevance** before the LLM-scoring cap — graded score replacing the boolean predicate, strongest-first ordering | `RelevanceFilter` qualifies candidates with a boolean predicate and then truncates at `max_examples * 3` in source-then-import order, so the strongest matches past the cap never reach the LLM scorer. The scoring loop's early break means order decides the output set on every run, not only on overflow. Closes the #26 recall follow-up. | `evolution/core/external_importers.py` (`_is_relevant_to_skill`, `RelevanceFilter.filter_and_score`) | **DONE** — `_relevance_score` returns a tiered tuple `(name_match, name_words, keyword_overlap)`; `_is_relevant_to_skill` is now `any(...)` over it, so the qualifying set is unchanged; candidates sort strongest-first (stable, ties keep import order) ahead of both caps — see review log, 2026-08-31 | ✅ |
| #162 (partial) | **Confine code-evolution test execution** and record the containment posture | `WorktreeEnv.run_test` executes pytest against an LLM-modified worktree through a bare subprocess, while the agent runner refuses to run unconfined at all — an asymmetry in our own doctrine. The adversarial gaming harness runs through the same path. | `evolution/code/worktree.py` (`run_test`), `evolution/validation/claude_runner.py` (the existing profile), `evolution/code/gate.py` (failure parsing) | **DONE** — shared `evolution/core/sandbox.py` (profile + availability + `wrap_argv`); `run_test` confines writes to the run root, records the posture in `repair_trace.json`, and raises rather than returning a non-pytest exit code from a confined run; `--require-sandbox` on all three LLM-loop entry points — see review log, 2026-08-31 | ✅ |
| #162 (partial), #136 | **Minimum detectable effect** as a gate-adjacent diagnostic | A gate can certify a win, or enforce a regression floor, without ever stating that its sample size could not detect the effect it claims to police. Absorbs the parked exact-test item. | `evolution/core/stats.py` (currently `paired_bootstrap` only) | **DONE (continuous only)** — `min_detectable_effect_paired` in `stats.py`, surfaced as `power_diagnostics.json` beside the decision; α and sidedness derived from the bootstrap's own confidence, the figure labelled a lower bound (verified against exact noncentral t). The paired-binary companion was written and **withdrawn** — it violated `|p01-p10| <= p01+p10` across our whole operating range; it needs the Connor form and real pass/fail counts. Exact paired tests still deferred — see review log, 2026-08-31 | ✅ |
| #154, #179 | **Importer and dataset-builder hardening** — malformed JSON raises our own error; non-UTF8 session files are skipped instead of crashing the importer | Two sites extract a JSON substring and then parse it unguarded, so a bracketed-but-malformed payload escapes as a raw decode error. Separately, `UnicodeDecodeError` is uncaught where legacy session files and the Claude Code history log are decoded — and the history log is the likelier of the two to be mixed-encoding. | `evolution/core/dataset_builder.py`, `evolution/core/external_importers.py` | **DONE** — both extract-then-parse sites route through shared guarded helpers that also check element shape; both decode sites widened (`UnicodeDecodeError` on the legacy read, `errors="replace"` on the history log, where decoding happens during iteration outside the per-line guard); no new dependency — see review log, 2026-08-31 | ✅ |
| — (found in review) | **Authoritative failure sets** — a pytest run that could not answer must not be scored as "nothing failed" | `failing_tests` discarded the exit code, and the failure parser returns an empty set on unrecognised output, so a timed-out, killed or uncollectable run certified a wrong repair as `correct` through the oracle gate. Demonstrated against the real gate. | `evolution/code/worktree.py` (`failing_tests`), `evolution/code/harvest.py` (`_failures`), `evolution/code/gate.py` (the parser) | **DONE** — the seam refuses a run that produced **no failure evidence and no authoritative exit**; the same guard added to both regression floors and the held-out check; `harvest._failures` delegates instead of re-parsing; a distinct error type keeps the ledger honest, and an inconclusive run while scoring counts as a failed seed rather than a dropped organism — see review log, 2026-08-31 | ✅ |
| #174 (partial) | **MIPROv2 fallback receives the held-out valset** | The fallback optimizer compiles without `valset` while the primary path passes it, so a fallback run loses its held-out set for internal candidate selection. Narrow: deploy integrity is unaffected, since the deploy gate runs its own held-out behavioral validation downstream. | `evolution/skills/evolve_skill.py` (`_default_mipro_runner`) | **DONE** — the fallback receives the held-out split, with `valset or None` so an empty split keeps the optimizer's internal behaviour rather than crashing the path that exists to survive a GEPA failure. `num_trials` stays excluded (mutually exclusive with the `auto` preset) — see review log, 2026-08-31 | ✅ |

## Optional / low-value (parked)

- **#136** — its exact-paired-test half is now tracked as the minimum-detectable-effect action
  item above; what remains parked here is the **BCa interval** as an optional
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
  (#7, #23, #50, #53, #95, #97, #104, #113, #114, #140, #154, #161, #173, #174, #177): we validate the reassembled full skill
  at every gate; upstream's patched call site (`validate_all(evolved_body, …)`) doesn't
  exist in our tree. #140 (fixes #119) also re-fixes #104 (frontmatter strip) + #41/#105 (optuna
  dep), both already done here — see review log, 2026-06-30.
- **PR emission / last-mile automation** (#139): we already have `evolution/core/pr_automation.py`
  (`create_pr`) wired into all four evolvers behind `--create-pr`, with the orchestrator's
  propose-only `--allow-pr` boundary on top. #139 adds a skill-only subset of this — see review
  log, 2026-06-30.
- **GEPA / DSPy 3.x compat** (#13, #14, #35, #46, #48, #73, #91, #109, #137-core, #142-core,
  #153, #155, #159, #167, #168, #173, #177, #178, #183): moot —
  we pin `dspy>=3.2.0,<3.3` + a gepa git override and call the modern API
  (`max_full_evals`, `reflection_lm`, 5-arg metric); the SkillModule "GEPA mutates
  `signature.instructions`" fix is already in (`skill_module.py`).
- **Skill-text extraction** (#5, #24, #49, #146, #174): done — `skill_text` is a property over
  `signature.instructions`.
- **Fitness / judge** (#25, #28): superseded by our `LLMJudge` + closed-loop behavioral
  scoring; #28's deterministic text-similarity proxy is a step backward.
- **Model providers** (#8, #15, #19, #22, #92, #112): subsumed by `resolve_default_lm`
  (~20 providers, OpenAI-wire/local, two OAuth backends with token refresh the PRs lack).
  #85 (Claude subscription) looked genuinely new but is **SKIP** — re-exposing subscription
  OAuth is ToS-prohibited and server-side-blocked (see review log, 2026-06-28).
- **Session-importer reconstructions** (#40, #143, #178, #179): the canonical fix is ours —
  `iter_hermes_sessions` reads `state.db` read-only with a JSON fallback, mines all sessions
  regardless of `sessions.source`, and strips the model-switch note. Later PRs re-propose it;
  their only extras are cross-platform DB discovery. #179's test-isolation premise does not hold
  here (autouse fixtures already neutralize `STATE_DB`) — see review log, 2026-08-31.
- **Prompt-section / MIPROv2 variant reconstructions** (#158, #180): #180 never writes the evolved
  text back into the agent's prompt builder at all — it is offline-only, strictly behind our
  shipped prompt path with its splice-and-restore integration. #158 commits generated artifacts
  plus an ungated auto-deploy on any positive improvement, which is a rigor regression against our
  deploy gate, not an improvement.
- **Launchers / entrypoints** (#181): moot — `uv run python -m evolution.<module>` already resolves
  the venv interpreter uniformly across every phase, which is what the proposed shell wrapper
  solves for one phase only.
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
- **#157** — hostile submission: deletes `evolution/core/constraints.py` outright, strips 26
  lines from `.gitignore` (removing the guardrail that keeps run output out of git), and adds
  ~1.6k lines of off-topic generated content. No credentials found, but reject on content and
  intent; do not mine it for salvage.
- **#69** — a fork-of-a-fork dump (parallel `*_v2` architecture + an unrelated kanban
  side-project + run-output bloat + local plan docs); ~85% noise around one small idea.

## Snapshot — 70 open PRs by group (2026-06-28)

For traceability of this cycle. Disposition key: A=adopt, R=rebuild, I=investigate,
S=skip (already covered / superseded), X=do-not-merge.

- **Constraint-validator / skill-assembly**: #7 S, #23 S, #50 S, #51 (R, dir guard), #53 S, #95 S, #97 S, #104 S, #113 S, #114 S, #134 **n/a (investigated)**
- **GEPA / DSPy 3.x compat**: #13 S, #14 S, #35 S, #46 S, #48 S, #73 S, #91 S, #109 S, #137 S (parked: secret-scan constraint)
- **Fitness / judge / significance + skill extraction**: #5 S, #24 S, #49 S, #25 S, #28 S, #136 (parked: BCa diagnostic)
- **Model providers / backends**: #8 S, #15 S, #19 S, #22 S, #85 **SKIP (ToS, investigated)**, #92 S, #112 S
- **Session importers / discovery / guardrails**: #26 **R** (deferred follow-up), #40 S (subset of #102), #102 **done**, #94 X
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

### Delta — 92 open PRs (2026-08-31)

Twenty-six new (#149–#183). Ledger: 31 opened, 11 closed (5 opened and closed inside the
window), 6 of the previously-tracked 72 closed. Disposition split: 21 S, 1 action item,
3 rebuilt-from, 1 X.

- **GEPA/DSPy compat + validate-full** (11): #153 S, #155 S, #159 S, #161 S, #167 S, #168 S,
  #173 S, #174 S, #177 S, #178 S, #183 S
- **Session importers** (2): #149 **R (action item — relevance ranking)**, #179 S (premise
  falsified against our tree)
- **Fitness / judge / verifier** (1): #150 S (memorizable fact-table oracle)
- **Reliability / gating** (3): #163 S, #165 S, #176 S
- **Phase / HSE mega-PRs** (2): #162 S wholesale — **two ideas rebuilt natively** (test
  containment, minimum detectable effect); #182 S (canary premature; AGPL entanglement)
- **Prompt-section / MIPROv2 reconstructions** (2): #180 S (offline-only, no write-back), #158 S
  (artifacts plus an ungated auto-deploy)
- **Misc / infra** (4): #154 S (JSON-robustness half rebuilt without the proposed dependency),
  #160 S (stray committed run artifact), #166 S (surfaced a licensing question, not repo
  hygiene), #181 S
- **Do not merge** (1): #157 X — see below

Group counts: 11 + 2 + 1 + 3 + 2 + 2 + 4 + 1 = **26**, matching the ledger above.
