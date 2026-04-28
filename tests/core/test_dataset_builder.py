"""Tests for dataset construction reproducibility."""

import json
from pathlib import Path

from evolution.core.dataset_builder import GoldenDatasetLoader


def _write_golden(path: Path, n: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "task_input": f"task {i}",
                "expected_behavior": f"behavior {i}",
            }) + "\n")


class TestGoldenDatasetSeed:
    """A given seed must produce a stable train/val/holdout split.

    Without seed plumbing, every run reshuffles against the global RNG and
    holdout scores aren't comparable across runs.
    """

    def test_same_seed_same_split(self, tmp_path: Path):
        golden = tmp_path / "golden.jsonl"
        _write_golden(golden)

        d1 = GoldenDatasetLoader.load(golden, seed=42)
        d2 = GoldenDatasetLoader.load(golden, seed=42)

        assert [e.task_input for e in d1.train] == [e.task_input for e in d2.train]
        assert [e.task_input for e in d1.val] == [e.task_input for e in d2.val]
        assert [e.task_input for e in d1.holdout] == [e.task_input for e in d2.holdout]

    def test_different_seed_different_split(self, tmp_path: Path):
        golden = tmp_path / "golden.jsonl"
        _write_golden(golden)

        d1 = GoldenDatasetLoader.load(golden, seed=42)
        d2 = GoldenDatasetLoader.load(golden, seed=7)

        # With 30 examples, the probability that two distinct seeds produce
        # the exact same train slice is negligible.
        assert [e.task_input for e in d1.train] != [e.task_input for e in d2.train]
