# The oracle asymmetry: where self-evolution gets traction

This is the consolidated finding of the framework's exploratory campaign. It is one
claim, stated at the strength the evidence actually supports — with the dead ends
kept in, because they delimit the regime.

## The claim

> Evolutionary self-improvement of an agent produced **measurable, deploy-grade
> traction in exactly one regime: when a ground-truth *executable* oracle stands
> between the artifact and the verdict** — code under deterministic tests, with no
> agent in the loop to absorb the change. Where the fitness signal is instead an
> LLM judge or a capable agent's behavior, we measured **no detectable effect at our
> statistical power, on a single capable-agent class** — in both directions
> (improving and degrading the artifact). That is an upper bound on the effect we
> could detect, **not a proof of inertness.**

The asymmetry is the result. It says *where to spend*: on surfaces with an executable
oracle, not on polishing artifacts a capable agent reads past.

## Evidence class 1 — the positive: executable-oracle code re-derivation

We harvested real historical tool bugs from an active repository's git stream, each
carrying its upstream fix commit as a **ground-truth oracle**. The loop repairs the
buggy parent in an isolated worktree from failing-test feedback; a repair counts only
if it passes the full fix-commit test set and behaviorally matches the upstream fix.
The organism (one bug) is the unit of analysis; a bug is **deploy-reachable** when a
majority of independent seeds produce an oracle-matching fix.

**Result (N=20 organisms × 3 seeds, pre-registered kill line 0.10):**

| Estimand | Value |
|---|---|
| Deploy-reachable | **12 / 20 = 0.60** |
| Wilson 95% CI | **[0.387, 0.781]** |
| Cluster bootstrap (organism-level) | mean 0.602, CI **[0.40, 0.80]** |
| P(below 0.10 kill line) | **0.0** |
| ICC / design effect / effective-N | 0.326 / 1.65 / 36.3 |
| Verdict | **GREEN** |
| Cost | $3.10 (mini-class proposer, 106 calls) |

The honest unit is the organism. The pooled per-seed rate (41/60 = 0.68) is recorded
only *for contrast* and labelled dishonest in the artifact, because seed correlation
(ICC 0.33) inflates its apparent precision — averaging seeds within a bug is
pseudo-replication. An earlier, smaller pilot on an easier tool tier was consistent
(deploy-reachable 0.75 [0.41, 0.93], ICC 0.57, smaller N); the N=20 campaign above is
the canonical number because it is larger and pre-registered.

Why this is the regime that works: the test is deterministic and the verdict is
mechanical. Nothing between the candidate and "pass/fail" can quietly compensate for a
bad artifact — unlike a capable agent, which can.

## Evidence class 2 — the null: capable-agent artifact decoupling

On a capable agent (gpt-5-mini class), artifact quality/correctness was **decoupled
from behavior in both directions**:

- **Improving doesn't help.** Evolving skills, tool descriptions, and turn-level
  sequences produced null deploy results — including under a **judge-free
  `test_command` verdict**, so judge saturation is not the explanation.
- **Degrading doesn't hurt.** A strongly misdirecting `write_file` description, and
  even an **actively false** one ("write_file merges and preserves content, never
  deletes"), each produced **0/7 caught-losses against a 0/7 A/A false-alarm floor**.
  The agent routes from the tool name, the task, and its own prior — over the
  description text.

This is a real null, not a broken detector: a delivery **canary** (a description that
changes which file is written) separates cleanly at 1.00 vs 0.00, so the rig can see a
genuine behavior difference. There was simply nothing to catch.

**Scope, stated honestly.** This is a failure to reject the null on **one capable
agent class**, for tools whose behavior the agent can infer from their name. It bounds
the detectable effect; it does not prove universal inertness. The coupling is expected
to reappear where the agent *cannot* infer behavior from priors — a weaker/cheaper
tier, or a novel MCP tool with a non-obvious contract. Neither was a clean test here
(the weak tier abstains on the closed-loop path via a diagnosed delivery bug; no
MCP-bearing agent backend exists — that is a build, not a probe).

A related observation corroborates the judge leg: in the deploy-gate archive, the
synthetic LLM judge **saturates** in the high-baseline region (baseline ≥ 0.95: 15 of
338 runs, 0 with a bootstrap lower bound > 0), so above that band the judge cannot
distinguish candidates at all. Distinct from the decoupling (which holds even
judge-free), but the same direction: behavioral/judge signal is the binding
constraint, executable signal is not.

## Evidence class 3 — the dead ends, as boundary pilots

These are small-N negative pilots. They did not supply usable signal; they mark where
we looked and why the positive regime is narrow — not proofs of impossibility.

- **Metamorphic verification (no oracle): 0/8.** Generic invariants (idempotence,
  absent-input no-op, parse-shape) caught **0 of 8** real bugs. Real tool bugs are
  input-specific edge cases that don't violate broad invariants — so oracle-free
  *general* verification did not work at this scale.
- **Held-out-split independence: only 3/8.** For the held-out anti-gaming gate to be
  independent, a bug needs tests input-diverse enough to split; only **3 of 8** did.
  Real bugs are usually caught by the single test the fix authored, so the independent
  held-out product is deferred — the oracle (above) sidesteps it for measurement.
- **Dependency-regression supply: 0.** Over a year of one active repo's history, there
  were **zero** clean, tool-local dependency-version regressions: manifest-touching
  commits were broad migrations (dozens of files) or feature additions. The
  genuinely-novel-with-correct-oracle case we hoped to harvest does not occur often
  enough to build on.

## The unifying thesis

Put the three classes together and the boundary is sharp:

> Self-evolution gets traction in proportion to **how mechanical and agent-free the
> verdict is.** A deterministic test on executed code (oracle present, no agent) →
> deploy-grade gradient. An LLM judge or a capable agent's behavior (no executable
> oracle, agent absorbs the artifact) → no detectable effect at our power.

So the framework's evolution/validation pipeline is **high-value where an executable
oracle exists** (code; the shipped propose-only sentinel feeds exactly this case) and
**low-value for artifact quality on capable agents.** The remaining artifact-quality
items presuppose a discriminating signal this campaign shows is absent on capable
agents; pursue them only on a tier or surface where the coupling exists.

## Provenance

Every headline number traces to a checked-in artifact, so nothing here is cherry-picked.

| Claim | Source | N |
|---|---|---|
| Deploy-reachable 0.60 [0.387, 0.781]; bootstrap [0.40,0.80]; ICC 0.326 | `reports/asymmetry_campaign_report.json` (committed snapshot of the campaign run) | 20 organisms × 3 seeds |
| Pooled per-seed 0.68 (for-contrast, dishonest) | same | 60 seeds |
| Judge saturation ≥0.95: 15/338, 0 with LB>0; false-abort upper 20–49% | `reports/saturation_calibration_findings.md`, `reports/saturation_calibration.json` | 338 paired-vector runs |
| Capable-agent decoupling (both directions; canary 1.00 vs 0.00) | `PLAN.md` "Campaign conclusion — capable-agent artifact decoupling" | A/A floor 0/7 |
| Metamorphic 0/8; held-out 3/8; dep-supply 0 | `PLAN.md` deferred/kill records | 8-organism pilots / 1-year scan |

## Limitations

- The null results bound an effect on **one capable-agent class**; a different model
  tier or a novel-contract tool could couple. Untested here, by design.
- The positive result is N=20 organisms on one repo's bug stream; it is GREEN with a
  pre-registered kill line, but it is one corpus.
- The dead-end pilots are small-N (8 organisms; one year of one repo). They steer the
  search; they are not impossibility proofs.
