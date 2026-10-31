import json
import zipfile

import pytest

import package_submission
from compare_results import summarize


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    home = tmp_path / "Ass2-3"
    run = home / "results/example"
    run.mkdir(parents=True)
    (home / "experiments.json").write_text(json.dumps({"runs": ["results/example"]}))
    (home / "README.md").write_text("Test README")
    (home / "SECRETS").write_text("excluded-secret-value")
    (run / "manifest.json").write_text(json.dumps({"model": "test-model", "agent": "reflection",
        "status": "complete", "profile": "baseline", "tasks": ["goto"], "seeds": [42]}))
    ep = run / "seed_42/babyai/BabyAI-MixedTrainLocal-v0/goto/BabyAI-MixedTrainLocal-v0/goto_run_00.json"
    ep.parent.mkdir(parents=True)
    ep.write_text(json.dumps({"seed": 42, "task": "BabyAI-MixedTrainLocal-v0/goto", "progression": 1,
        "num_steps": 10, "failed_candidates": [], "input_tokens": 120, "output_tokens": 25,
        "reflection_stats": {"reviews": 1, "accepted_reviews": 1, "usage": {
            "actor": {"calls": 10, "input_tokens": 100, "output_tokens": 20},
            "critic": {"calls": 1, "input_tokens": 20, "output_tokens": 5}}}}))
    monkeypatch.setattr(package_submission, "HOME", home)
    monkeypatch.setattr("sys.argv", ["package_submission.py"])
    return home, run, ep


def test_completed_costs_and_archive(experiment):
    home, run, _ = experiment
    row = summarize(run)[0]
    assert row["success_percent"] == 100 and row["critic_calls"] == 1
    assert row["input_tokens"] == 120 and row["critic_input_tokens"] == 20
    package_submission.main()
    with zipfile.ZipFile(home.parent / "Ass2-3.zip") as archive:
        assert archive.testzip() is None
        assert b"Complete comparison" in archive.read("Ass2-3/REPORT_RESULTS.md")
        assert not any("SECRETS" in name for name in archive.namelist())


def test_partial_not_scored_as_failure_and_requires_flag(experiment, monkeypatch):
    home, run, ep = experiment
    ep.unlink()
    row = summarize(run)[0]
    assert row["completed"] == 0 and row["expected"] == 1 and row["success_percent"] == ""
    assert row["status"] == "partial"
    with pytest.raises(SystemExit):
        package_submission.main()
    monkeypatch.setattr("sys.argv", ["package_submission.py", "--allow-partial"])
    package_submission.main()
    assert "PARTIAL" in (home / "REPORT_RESULTS.md").read_text()


def test_running_run_and_secret_are_rejected(experiment):
    home, run, _ = experiment
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = "running"
    path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit):
        package_submission.main()
    manifest["status"] = "complete"
    path.write_text(json.dumps(manifest))
    (home / "bad.py").write_text("key = '" + "gsk_" + "x" * 35 + "'")
    with pytest.raises(ValueError, match="Possible API key"):
        package_submission.main()
    assert not (home.parent / "Ass2-3.zip").exists()
