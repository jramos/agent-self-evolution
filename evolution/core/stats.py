"""Statistical helpers used by the deploy-gate decision logic."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def paired_bootstrap(
    baseline_scores: list[float],
    evolved_scores: list[float],
    *,
    confidence: float = 0.90,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap on per-example score differences.

    Resamples the per-example improvement vector (evolved - baseline)
    `n_resamples` times and returns mean + lower/upper percentile bounds
    at the given two-sided confidence level. Uses the basic (reverse-
    percentile) method, which is the literature-recommended choice when
    sample size is small (≤20). BCa is the upgrade path once N≥20.

    Args:
        baseline_scores: per-example scores of the baseline candidate.
        evolved_scores: per-example scores of the evolved candidate.
            Must be the same length as baseline_scores; element i in
            both arrays must come from the same example (paired).
        confidence: two-sided confidence level. Bounds are at the
            (1-confidence)/2 and (1+confidence)/2 percentiles of the
            bootstrap distribution.
        n_resamples: number of bootstrap iterations.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with mean (sample mean of improvements), lower_bound,
        upper_bound, n_examples, n_resamples, confidence.
    """
    if len(baseline_scores) != len(evolved_scores):
        raise ValueError(
            f"paired bootstrap requires equal-length score arrays; "
            f"got {len(baseline_scores)} baseline vs {len(evolved_scores)} evolved"
        )
    n = len(baseline_scores)
    if n == 0:
        raise ValueError("paired bootstrap requires non-empty score arrays")

    diffs = np.asarray(evolved_scores, dtype=float) - np.asarray(baseline_scores, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    resample_means = diffs[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(diffs.mean()),
        "lower_bound": float(np.quantile(resample_means, alpha)),
        "upper_bound": float(np.quantile(resample_means, 1.0 - alpha)),
        "n_examples": n,
        "n_resamples": n_resamples,
        "confidence": confidence,
    }


def _z(p: float) -> float:
    """Standard-normal quantile. Stdlib only — scipy is not a dependency here."""
    return NormalDist().inv_cdf(p)


def _one_sided_alpha(confidence: float) -> float:
    """The alpha the surrounding gate actually operates at.

    Derived rather than hardcoded so the diagnostic describes the decision it
    sits beside: :func:`paired_bootstrap` returns a two-sided interval at
    ``confidence``, and the gate consumes only its *lower* bound — a one-sided
    decision at (1 - confidence) / 2. Hardcoding a two-sided 0.05 against a 0.90
    interval would inflate the reported effect by about 13% — the z-terms
    differ by 19%, but the power term is common to both.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    return (1.0 - confidence) / 2.0


def min_detectable_effect_paired(
    diffs: list[float],
    *,
    confidence: float = 0.90,
    power: float = 0.80,
    ddof: int = 1,
) -> dict:
    """Smallest paired mean difference a sample of ``n`` could reliably detect.

    The honesty primitive. A gate can certify a win, or enforce a regression
    floor, while its sample size was never capable of detecting the effect it
    claims to police; this states that bound explicitly instead of leaving it
    implicit in a passing verdict.

    It models **the rule this gate actually runs**, which is not a textbook
    t-test. ``paired_bootstrap`` returns a *percentile* interval whose spread is
    the divisor-n resample sd with no t-correction, and the gate rejects on its
    lower bound alone — equivalent to requiring ``t > z_{1-alpha} * sqrt((n-1)/n)``.
    That threshold is below the nominal one, so the rule is anti-conservative at
    small n (its real one-sided error rate is about 0.08 at n=8, against a nominal
    0.05). Computing against the nominal quantile instead would describe a test
    that is not being run, and would report a detectable effect larger than the
    rule's own.

    Still a normal approximation, so it is approximate in both directions and
    claims no bound: against an exact paired t-test it understates by roughly 11%
    at n=8, while the gate's own rule sits on the other side. ``method`` records
    that rather than asserting a direction the number cannot support.

    Takes the raw per-example differences rather than a precomputed spread, so
    ``ddof`` genuinely applies: at n=8 the choice moves the spread by about 7%,
    the same order as the effects being judged, and a recorded-but-unused knob
    would let the payload misreport its own provenance.

    Deliberately covers the continuous regime only. A paired-binary companion was
    written and withdrawn: |p01 - p10| <= p01 + p10 is a hard algebraic bound, and
    the normal approximation violates it whenever n * discordance < 6.18 — this
    project's whole operating range. Note the obvious remedy does not work either:
    Connor's sample-size form is itself a normal approximation and breaks the same
    bound whenever the discordance falls below the squared effect, which covers
    most of that range again. Doing it properly needs an exact conditional
    binomial (exact McNemar) over real pass/fail counts, not thresholded score
    differences.
    """
    n = len(diffs)
    if n <= 1:
        raise ValueError(f"minimum detectable effect needs n > 1; got {n}")
    if ddof >= n:
        raise ValueError(f"ddof must be less than n; got ddof={ddof}, n={n}")
    mean = sum(diffs) / n
    sd_diff = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - ddof))
    alpha = _one_sided_alpha(confidence)
    # The percentile interval's spread uses the divisor-n resample sd, so the
    # gate's effective threshold is below the nominal quantile. Model that, not
    # the textbook test.
    critical = _z(1.0 - alpha) * math.sqrt((n - 1) / n)
    mde = (critical + _z(power)) * sd_diff / math.sqrt(n)
    return {
        "mde": float(mde),
        "n": n,
        "sd_diff": float(sd_diff),
        "alpha_one_sided": alpha,
        "critical_multiplier": float(critical),
        "power": power,
        "ddof": ddof,
        "method": "normal-approximation",
        "models_rule": "paired-bootstrap percentile lower bound (the gate's own rule)",
    }
