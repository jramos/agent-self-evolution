"""Tests for dataset construction reproducibility."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from evolution.core.dataset_builder import GoldenDatasetLoader, SyntheticDatasetBuilder


@pytest.fixture(autouse=True, scope="session")
def _hermes_repo_env(tmp_path_factory):
    """SyntheticDatasetBuilder pulls EvolutionConfig which needs a repo path."""
    fake_repo = tmp_path_factory.mktemp("fake_hermes_repo")
    os.environ["HERMES_AGENT_REPO"] = str(fake_repo)
    yield


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


class TestSyntheticGeneratorLMConfig:
    """At eval_dataset_size=60 the synthetic generator's JSON output runs
    past the prior 4000-token ceiling and silently truncates → JSONDecodeError
    → process exit. Lock the bumped 16000-token budget.
    """

    def test_lm_constructed_with_bumped_max_tokens(self):
        from evolution.core.config import EvolutionConfig

        config = EvolutionConfig()
        builder = SyntheticDatasetBuilder(config)

        with patch("evolution.core.dataset_builder.dspy.LM") as lm_cls:
            lm_cls.return_value = MagicMock()
            # Stub the generator + parser so we don't hit a real LM.
            builder.generator = MagicMock(return_value=MagicMock(
                test_cases='[{"task_input": "x", "expected_behavior": "y"}]',
            ))
            builder.generate(artifact_text="skill", artifact_type="skill", num_cases=60)

        lm_cls.assert_called_once()
        _, kwargs = lm_cls.call_args
        assert kwargs["max_tokens"] == 16000, (
            f"max_tokens regressed from 16000 to {kwargs['max_tokens']}; "
            "JSON truncation will reappear at eval_dataset_size>=60"
        )
