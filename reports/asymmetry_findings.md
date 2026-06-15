# The oracle asymmetry: where self-evolution gets traction

This is the consolidated finding of the framework's exploratory campaign. It is a
*spend-allocation* result — where to point evolutionary budget — stated at the
strength the evidence actually supports, with the dead ends and the validity threats
kept in.

## What this means for where we spend

- **Invest** where a **ground-truth executable oracle** stands between the artifact
  and the verdict, the baseline has **real headroom**, and the task is a
  **re-derivation** (code under deterministic tests). That regime produced a
  deploy-grade gradient.
- **Don't** spend on improving artifact *quality* (skills / tool descriptions /
  prompt sections) for a **capable agent** on tools whose behavior it can infer from
  their name — we measured no effect there, in either direction. Revisit only on a
  surface where the agent *can't* infer behavior (a weaker tier, or a novel-contract
  tool).

## The claim, stated carefully

> Evolutionary self-improvement produced measurable, deploy-grade traction under a
> **conjunction** of conditions — an executable oracle, real headroom, and a
> re-derivation task — and **no detectable effect** where the fitness signal is an
> LLM judge or a capable agent's behavior. The two arms are not one calibrated axis:
> the positive arm resolves small effects; the negative arm resolves only *large*
> ones (see power, below). So this is a direction-setting result, **not** a proof
> that behavioral signal is inert, and **not** a single "oracle vs. agent" law —
> oracle-presence co-varies with headroom and task type, and we did not separate them.

## Evidence class 1 — the positive: executable-oracle code re-derivation

We harvested real historical tool bugs from an active repository's git stream, each
carrying its upstream fix commit as a **ground-truth oracle**. The loop repairs the
buggy parent in an isolated worktree from failing-test feedback; a repair counts only
if it passes the full fix-commit test set and behaviorally matches the upstream fix.
The organism (one bug) is the unit; a bug is **deploy-reachable** when a majority of 3
independent seeds produce an oracle-matching fix.

**Result (N=20 organisms × 3 seeds, pre-registered futility floor 0.10):**

| Estimand | Value |
|---|---|
| Deploy-reachable | **12 / 20 = 0.60** |
| Wilson 95% CI | [0.387, 0.781] |
| Cluster bootstrap (organism-level) | mean 0.602, CI [0.40, 0.80] |
| Clears futility floor (0.10)? | yes — P(below 0.10) = 0.0 |
| ICC / design effect / effective-N | 0.326 / 1.65 / 36.3 |
| Cost | $3.10 (mini-class proposer, 106 calls) |

**Read the verdict honestly.** The pre-registered 0.10 line is a **futility floor**,
not a success bar: deploy-reachable would have to collapse to 4/20 before it failed,
so clearing it certifies "not catastrophic," not "validated deployment rate." The
honest headline is the point estimate **0.60 [0.39, 0.78] on one repository, one
proposer model** — promising, not settled.

**The number is fragile at the knee.** The per-seed distribution is bimodal —
{0 correct: 1 organism, 1: 7, 2: 2, 3: 10} — so 10 organisms are unanimous successes,
8 are clear failures, and **0.60 hinges on the 2 organisms sitting exactly at 2/3**.
Each is one seed-flip from flipping the verdict (as are the 7 organisms at 1/3, which
would flip the other way). The cluster bootstrap resamples the *already-dichotomized*
deploy flags, so its [0.40, 0.80] interval does **not** propagate the 3-seed sampling
noise on each organism — a two-stage bootstrap would be wider. Treat 0.60 as a coarse
estimate from 3 Bernoulli draws per organism, and note it is one operationalization:
"any-seed-reachable" and "all-3" give 12/20 and 10/20 respectively.

The pooled per-seed rate (41/60 = 0.68) is recorded only *for contrast* and labelled
dishonest in the source artifact — seed correlation (ICC 0.33) inflates its apparent
precision. Note 0.60 is **not** a de-biased 0.68; majority-of-3 is a different,
threshold-dependent estimand, not a correction of the marginal.

**Validity threat — teach-to-the-test (not closed).** The repair loop feeds the
proposer the failing test's full output, which for assertion failures includes the
expected values (`assert got == <expected>`); the measurement gate verifies an
oracle *match* with the held-out split **off**, and its own docstring notes it does
not catch input-hardcoding. A strong public-repo proposer may also recall the fix. So
0.60 mixes genuine re-derivation with possible transcription; a targeted check
(re-run with traceback feedback suppressed; inspect accepted rewrites for hard-coded
literals) is the right next step before treating 0.60 as a pure capability rate.

**Disposition (for an honest denominator).** Of 33 harvested candidates: 20 valid
organisms, 7 source-missing, 5 not-valid (parent didn't cleanly fail), 1 too-large
(whole-file-rewrite ceiling). The too-large exclusion is a single organism here, so it
does not materially skew the rate — but "0.60" is a rate over tools that fit a
whole-file rewrite and have a crisp test-pinned fix.

## Evidence class 2 — the null: capable-agent artifact decoupling

The sharp, defensible form of this result is about **inferability**, not behavior in
general:

> When a capable agent (gpt-5-mini class) can recover a tool's behavior from its
> **name plus the task**, the description text is redundant — and editing it (better
> *or* worse) does not move behavior.

- **Improving didn't help.** Evolving skills, tool descriptions, and turn-level
  sequences produced null deploy results — including under a **judge-free
  `test_command` verdict**, so judge saturation is not the cause. (These baselines
  were also largely already-saturated / hand-tuned, i.e. low headroom — part of why
  there was nothing to gain.)
- **Degrading didn't hurt.** A strongly misdirecting `write_file` description, and
  even an **actively false** one ("merges and preserves content, never deletes"),
  each produced 0/7 caught-losses against a 0/7 A/A false-alarm floor. `write_file`
  is a tool whose *name* states its function — the weakest possible test of
  description sensitivity, which is exactly why the agent routes past the text.

**Power — the binding caveat.** With n=7 per arm, 0/7-vs-0/7 resolves only **large**
couplings: a one-sided Fisher test cannot separate treatment from the floor until the
caught-loss rate reaches ~4/7 (≈57%); at 80% power the minimum detectable effect is
roughly 40%. So "no detectable effect" means **we can rule out only effects larger
than about half** — not "we had good power and saw nothing." A delivery **canary** (a
description that changes which file is written) separates cleanly at 1.00 vs 0.00,
which proves the harness *records routing*; it does not prove the harness could detect
a realistic, intermediate-magnitude quality effect — that control was not run.

So this is a failure to reject the null on **one capable-agent class**, for
**name-inferable** tools, on **low-headroom** artifacts. It bounds the detectable
effect; it does not prove inertness, and it predicts the coupling reappears where the
agent cannot infer behavior from priors (a weaker tier, or a novel-contract tool).

## Evidence class 3 — the dead ends (two honestly-distinct kinds)

**Supply-absent (a real boundary).** Over a year of one active repo's history there
were **zero** clean, tool-local dependency-version regressions: manifest-touching
commits were broad migrations or feature additions. This delimits regardless of effect
size — the instances simply don't exist in the corpus.

**Underpowered nulls (not boundaries).** These found no usable signal at small N, but
their confidence intervals are wide — they steer spend, they don't prove impossibility:
- **Metamorphic verification (no oracle): 0/8.** Generic invariants (idempotence,
  absent-input no-op, parse-shape) caught 0 of 8 real bugs (Wilson upper ≈ 0.37 — fully
  consistent with metamorphic working ~1-in-4). Real tool bugs are input-specific edge
  cases that don't violate broad invariants.
- **Held-out-split independence: 3/8.** Only 3 of 8 bugs had tests input-diverse enough
  for an independent held-out, so the independent held-out *product* is deferred — the
  oracle sidesteps it for measurement.

## The unifying thesis (and its limit)

Across the classes, traction tracked **how mechanical and agent-free the verdict is** —
a deterministic test on executed code yielded a gradient; a judge or a capable agent's
behavior did not. But the contrast is between **two instruments of unequal resolution**
(one resolves ≥0.10 effects, the other only ≥~0.5), and oracle-presence is confounded
with headroom and task-type, so the clean "oracle vs. agent" axis is a hypothesis the
campaign suggests but did not isolate. The disconfirming experiment never run: apply
the executable-oracle loop to an *already-correct* function (oracle present, no
headroom) — if the gradient vanishes, headroom, not the oracle, was load-bearing.

Practically, the implication stands: the evolution/validation pipeline is high-value
where an executable oracle and real headroom exist (code; fed by the shipped
propose-only triage sentinel), and low-value for artifact quality on capable,
name-inferable surfaces.

## Provenance

| Claim | Source | N |
|---|---|---|
| Deploy-reachable 0.60 [0.387, 0.781]; bootstrap [0.40,0.80]; ICC 0.326 | `reports/asymmetry_campaign_report.json` (committed summary) | 20 organisms × 3 seeds |
| Per-organism rows + per-seed flags (re-derives 12/20) | `reports/asymmetry_campaign_ledger.jsonl` (committed manifest) | 33 candidates → 20 organisms |
| Pooled per-seed 0.68 (for-contrast, dishonest) | same summary | 60 seeds |
| Judge saturation ≥0.95: 15/338, 0 with LB>0 — *but the source flags itself data-starved (Wilson upper 20–49%), not settled* | `reports/saturation_calibration_findings.md` | 338 paired-vector runs |
| Capable-agent decoupling (both directions; canary 1.00 vs 0.00) | `PLAN.md` "Campaign conclusion — capable-agent artifact decoupling" | A/A floor 0/7 (n=7) |
| Metamorphic 0/8; held-out 3/8; dep-supply 0 | `PLAN.md` deferred/kill records | 8-organism pilots / 1-year scan |
| Earlier smaller pilot directionally consistent (deploy-reachable ≈0.75, ICC ≈0.57) | working notes — *no committed artifact; cited as direction only* | smaller N |

## Limitations

- **Teach-to-the-test (top threat).** The proposer sees expected values in test
  tracebacks and the held-out split is off, so 0.60 may overstate genuine
  re-derivation until the leakage check is run.
- **Underpowered null.** n=7 resolves only ≥~50% couplings; the null is an upper bound
  on a large effect, scoped to one capable-agent class and name-inferable tools.
- **Confounded axis.** Oracle-presence co-varies with headroom and task type; the
  one-axis thesis is suggested, not isolated.
- **ICC instability.** ICC swung 0.33 (campaign) vs 0.57 (earlier pilot); effective-N
  (36.3) and thus the verdict's precision rest on a noisy estimate — don't read it to
  three significant figures.
- **One corpus, one proposer.** N=20 on one repo's bug stream with one mini-class model.
- **Dead-end pilots are small-N.** Metamorphic 0/8 and held-out 3/8 have wide CIs; only
  the dependency-supply absence is a true boundary.
