"""Tests for the paired-bootstrap helper."""

import pytest

from evolution.core.stats import paired_bootstrap


class TestPairedBootstrap:
    def test_returns_required_fields(self):
        result = paired_bootstrap([0.5, 0.6, 0.7], [0.6, 0.7, 0.8])
        for key in ("mean", "lower_bound", "upper_bound", "n_examples", "n_resamples", "confidence"):
            assert key in result, f"missing {key}"
        assert isinstance(result["mean"], float)
        assert isinstance(result["lower_bound"], float)
        assert isinstance(result["upper_bound"], float)
        assert result["n_examples"] == 3
        assert result["confidence"] == 0.90

    def test_lower_bound_below_mean_below_upper_bound(self):
        # Variable improvements; bounds should bracket the mean.
        result = paired_bootstrap([0.3, 0.5, 0.6, 0.4, 0.5], [0.4, 0.6, 0.5, 0.5, 0.7])
        assert result["lower_bound"] <= result["mean"] <= result["upper_bound"]

    def test_zero_difference_yields_zero_mean_and_zero_bounds(self):
        same = [0.5, 0.6, 0.7, 0.8]
        result = paired_bootstrap(same, same)
        assert result["mean"] == 0.0
        # Bounds also collapse to 0 because every resampled diff is 0.
        assert result["lower_bound"] == 0.0
        assert result["upper_bound"] == 0.0

    def test_uniformly_positive_diffs_yield_positive_lower_bound(self):
        # Every example improves by exactly +0.1 → resampling is invariant
        # and lower bound = upper bound = 0.1.
        baseline = [0.5, 0.6, 0.7, 0.8, 0.4]
        evolved = [b + 0.1 for b in baseline]
        result = paired_bootstrap(baseline, evolved)
        assert result["mean"] == pytest.approx(0.1)
        assert result["lower_bound"] == pytest.approx(0.1)
        assert result["upper_bound"] == pytest.approx(0.1)

    def test_seed_reproducibility(self):
        baseline = [0.5, 0.6, 0.7, 0.4, 0.5, 0.8]
        evolved = [0.55, 0.65, 0.5, 0.45, 0.7, 0.85]
        a = paired_bootstrap(baseline, evolved, seed=123)
        b = paired_bootstrap(baseline, evolved, seed=123)
        assert a == b
        c = paired_bootstrap(baseline, evolved, seed=999)
        assert c["lower_bound"] != a["lower_bound"]  # different seed → different bound

    def test_raises_on_unequal_lengths(self):
        with pytest.raises(ValueError, match="equal-length"):
            paired_bootstrap([0.5, 0.6], [0.6, 0.7, 0.8])

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            paired_bootstrap([], [])
