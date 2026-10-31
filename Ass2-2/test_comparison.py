import json

import pytest

from compare_results import summarize


def test_partial_run_counts_completed_failure_but_excludes_interruption(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "model": "test", "profile": "baseline", "tasks": ["pickup"], "seeds": [42, 43, 44],
    }))
    for seed, success in ((42, 1), (43, 0), (44, None)):
        path = tmp_path / f"seed_{seed}/babyai/BabyAI-MixedTrainLocal-v0/pickup/BabyAI-MixedTrainLocal-v0/pickup_run_00.json"
        path.parent.mkdir(parents=True)
        path.with_suffix(".csv").write_text("Step,Action,Reasoning\n0,go forward,\n")
        if success is not None:
            path.write_text(json.dumps({"seed": seed, "task": "BabyAI-MixedTrainLocal-v0/pickup",
                "progression": success, "num_steps": 5, "failed_candidates": [],
                "input_tokens": 10, "output_tokens": 2}))
    overall = summarize(tmp_path)[0]
    assert overall["success_percent"] == 50
    assert overall["completed"] == 2
    assert overall["expected"] == 3
    assert overall["status"] == "partial"
    assert overall["actions"] == 10
    assert overall["agent"] == "naive"  # Legacy Exercise 1 manifests.


def test_rejects_mislabelled_episode(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "model": "test", "profile": "baseline", "tasks": ["goto"], "seeds": [42],
    }))
    path = tmp_path / "seed_42/babyai/BabyAI-MixedTrainLocal-v0/goto/BabyAI-MixedTrainLocal-v0/goto_run_00.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"seed": 43, "task": "BabyAI-MixedTrainLocal-v0/goto"}))
    with pytest.raises(ValueError, match="identity mismatch"):
        summarize(tmp_path)
