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

## Action items (open)

Disposition lens: against our diverged tree, these are "cherry-pick the idea/mechanism,"
not "merge the PR." Our-code anchors point at where the change would land.

| # | What | Why we'd benefit | Our-code anchor | Disposition | Status |
|---|---|---|---|---|---|
| #132 | **Parameter-description evolution** — AST reader/writer for `parameters.properties.{name}.description`; dot-labeled `[[tool.param]]` GEPA targets | A genuinely missing evolution axis. We have the constraint stub but no evolver touches param descriptions. | `evolution/core/constraints.py:112` (`max_param_desc_size` stub), `evolution/tools/tool_source.py` (treats `input_schema` read-only), `tool_module.py` | **Investigated → NULL** (axis saturated — param-description text doesn't move agent value-selection; see review log, 2026-06-28) | ✅ |
| #102 (+ #26) | Skill importer reads Hermes **`state.db`** (SQLite) + filters machine-generated user messages; #26 adds a 3-stage relevance filter (LLM keyword expansion + full-corpus scan) | Our skill importer reads stale `~/.hermes/sessions/*.json`; our own validation path proves `state.db` is canonical. The skill path lags the tool path on data quality + recall. | `evolution/core/external_importers.py` (`HermesSessionImporter`, `RelevanceFilter`); cf. `evolution/validation/hermes_runner.py` (`parse_session_from_db`) | **DONE** — `iter_hermes_sessions` now reads `state.db` first; both skill + tool paths fixed (importer 0 → real pairs). #26 recall improvement deferred. | ✅ |
| #134 | Graduated / class-aware skill-size cap (soft target + hard ceiling + ramp) | Our fitness still has a **pre-cap length-penalty cliff** that docks skills already under the cap. | `evolution/core/fitness.py:111-115` (the cliff), reconcile with `evolution/core/constraints.py:241-267` (`effective_absolute_char_ceiling`) | ADOPT (adapted; drop the brittle keyword `is_reference_skill` heuristic) | ☐ |
| #106 | `.github/dependabot.yml` + `.pre-commit-config.yaml` | Missing infra hygiene; we have neither. | `.github/`, repo root; reconcile with `.github/workflows/tests.yml` (py3.10–3.13 matrix) | **DONE** — added `dependabot.yml` (uv + github-actions) + `.pre-commit-config.yaml` (hygiene hooks); ruff + gitleaks (#107) deferred | ✅ |
| #133 | Cross-phase orchestrator + unified `evolve_all` CLI — **shape only** (dependency-ordered phases, fault isolation, JSONL run history) | We have no unified driver sequencing skills→tools→prompts→params; only per-subsystem. Compounds with #132. | `evolution/monitor/` (sentinel/queue — keep the propose-only/human-in-loop boundary) | CHERRY-PICK (shell only; keep our gated evolvers) | ☐ |
| #127 | Broad-benchmark-regression-as-a-gate, applied to the **skill** path | We have the regression-floor/oracle analogue for **code** only. | `evolution/code/gate.py` (the code analogue); skill deploy gate in `evolution/skills/evolve_skill.py` | INVESTIGATE | ☐ |
| #85 | Claude Code **subscription** backend — FastAPI OpenAI-compatible shim over `claude-agent-sdk` | A new capability: our OAuth backends cover OpenAI-Codex + Nous, not Claude-subscription. Plugs in as `provider: custom` + `base_url`, no code-layer change → cheaper evolution. | `evolution/core/hermes_provider.py` (`resolve_default_lm`); standalone `scripts/` proxy | INVESTIGATE (verify `claude-agent-sdk` subscription-auth still viable) | ☐ |

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
  (#7, #23, #50, #53, #95, #97, #104, #113, #114): we validate the reassembled full skill
  at every gate; upstream's patched call site (`validate_all(evolved_body, …)`) doesn't
  exist in our tree.
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
  Only #85 (Claude subscription) is genuinely new.
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

- **Constraint-validator / skill-assembly**: #7 S, #23 S, #50 S, #51 (C, dir guard), #53 S, #95 S, #97 S, #104 S, #113 S, #114 S, #134 **A**
- **GEPA / DSPy 3.x compat**: #13 S, #14 S, #35 S, #46 S, #48 S, #73 S, #91 S, #109 S, #137 S (parked: secret-scan constraint)
- **Fitness / judge / significance + skill extraction**: #5 S, #24 S, #49 S, #25 S, #28 S, #136 (parked: BCa diagnostic)
- **Model providers / backends**: #8 S, #15 S, #19 S, #22 S, #85 **I**, #92 S, #112 S
- **Session importers / discovery / guardrails**: #26 **C** (deferred follow-up), #40 S (subset of #102), #102 **done**, #94 X
- **Reliability / real-mutation / gating + code evolver**: #16 S, #17 X, #75 S, #89 S, #126 S, #127 **I**
- **Phase / HSE mega-PRs**: #30 S, #42 S, #86 S, #98 S, #108 S, #117 S, #120 S
- **eksays Phase 1–5**: #129 S, #130 S, #131 S, #132 **null (investigated)**, #133 **C**
- **Misc / infra / tests**: #20 S, #21 S, #45 S, #69 X, #76 S, #77 S, #78 S, #79 S, #80 S, #100 S, #101 S, #105 S, #106 **done**, #107 (parked)
