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

    The figure is a **lower bound** on the true MDE, and says so in its result.
    It uses the normal approximation, whereas the correct quantile at our
    operating sizes (n of roughly 8-20) comes from a noncentral t and is larger.
    Understating is the unsafe direction for a diagnostic whose entire purpose is
    admitting what the sample cannot see, so the caveat travels with the number.

    Takes the raw per-example differences rather than a precomputed spread, so
    ``ddof`` genuinely applies: at n=8 the choice moves the spread by about 7%,
    the same order as the effects being judged, and a recorded-but-unused knob
    would let the payload misreport its own provenance.

    Deliberately covers the continuous regime only. A paired-binary companion was
    written and withdrawn: |p01 - p10| <= p01 + p10 is a hard algebraic bound, and
    the normal approximation violates it whenever n * discordance < 6.18 — which
    is this project's whole operating range. Doing it properly needs the Connor
    form and real pass/fail counts rather than thresholded score differences.
    """
    n = len(diffs)
    if n <= 1:
        raise ValueError(f"minimum detectable effect needs n > 1; got {n}")
    if ddof >= n:
        raise ValueError(f"ddof must be less than n; got ddof={ddof}, n={n}")
    mean = sum(diffs) / n
    sd_diff = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - ddof))
    alpha = _one_sided_alpha(confidence)
    mde = (_z(1.0 - alpha) + _z(power)) * sd_diff / math.sqrt(n)
    return {
        "mde": float(mde),
        "n": n,
        "sd_diff": float(sd_diff),
        "alpha_one_sided": alpha,
        "power": power,
        "ddof": ddof,
        "method": "normal-approximation",
        "is_lower_bound": True,
    }
