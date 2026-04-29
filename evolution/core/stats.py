"""Statistical helpers used by the deploy-gate decision logic."""

from __future__ import annotations

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
