import json
import zipfile

import pytest

import package_submission


def test_partial_archive_is_labelled_and_excludes_credentials(tmp_path, monkeypatch):
    home = tmp_path / "Ass2-2"
    (home / "agent").mkdir(parents=True)
    (home / "agent/custom.py").write_text("# Example agent\n")
    (home / "SECRETS").write_text("A_PRIVATE_VALUE=do-not-package\n")
    runs = []
    for agent, model in (("naive", "A"), ("naive", "B"), ("custom", "A"), ("custom", "B")):
        run = home / "results" / f"{agent}_{model}"
        run.mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps({
            "status": "failed_or_interrupted", "agent": agent, "model": model,
            "profile": "baseline", "tasks": ["goto"], "seeds": [42],
        }))
        runs.append(run)
    monkeypatch.setattr(package_submission, "HOME", home)
    monkeypatch.setattr(package_submission, "run_folders", lambda: runs)
    monkeypatch.setattr("sys.argv", ["package_submission.py"])
    with pytest.raises(SystemExit):
        package_submission.main()
    assert not (tmp_path / "Ass2-2.zip").exists()
    monkeypatch.setattr("sys.argv", ["package_submission.py", "--allow-partial"])
    package_submission.main()
    with zipfile.ZipFile(tmp_path / "Ass2-2.zip") as archive:
        assert archive.testzip() is None
        assert all("SECRETS" not in name for name in archive.namelist())
        assert b"PARTIAL" in archive.read("Ass2-2/REPORT_RESULTS.md")
        assert "Ass2-2/agent/custom.py" in archive.namelist()
    # Even an allowlisted source file must not contain an apparent API key.
    (home / "bad.py").write_text("key = '" + "gsk_" + "x" * 35 + "'\n")
    with pytest.raises(ValueError, match="Possible API key"):
        package_submission.main()
