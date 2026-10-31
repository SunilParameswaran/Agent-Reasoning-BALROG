"""Offline checks: real BabyAI/BALROG pipeline with simulated API responses."""
import io
import csv
import json
from pathlib import Path

import httpx
from openai import OpenAI
import pytest

import gemini_client
import run_exercise2 as run_local


def test_key_required(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Set GEMINI_API_KEY"):
        gemini_client.connect_gemini()


def test_gemini_endpoint_restricted(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["run_local.py", "--provider", "gemini",
                                     "--base-url", "https://example.com/v1"])
    with pytest.raises(SystemExit) as exc:
        run_local.main(output_home=tmp_path)
    assert exc.value.code == 2


def test_thinking_cannot_be_disabled_on_gemini3(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["run_local.py", "--provider", "gemini",
                                     "--reasoning-effort", "none"])
    with pytest.raises(SystemExit) as exc:
        run_local.main(output_home=tmp_path)
    assert exc.value.code == 2


def test_list_models_does_not_run_evaluation(monkeypatch, tmp_path, capsys):
    def respond(request):
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"object": "list", "data": [
            {"id": "models/gemini-3.5-flash-lite", "object": "model", "created": 0, "owned_by": "test"}
        ]})
    sdk = OpenAI(api_key="offline-test", base_url=gemini_client.GEMINI_BASE_URL,
                 http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(gemini_client, "connect_gemini", lambda: sdk)
    monkeypatch.setattr("sys.argv", ["run_local.py", "--provider", "gemini", "--list-models"])
    run_local.main(output_home=tmp_path)
    assert capsys.readouterr().out.strip() == "gemini-3.5-flash-lite"
    assert not (tmp_path / "results").exists()
    assert sdk.is_closed()


@pytest.mark.parametrize("provider", ["gemini", "ollama"])
@pytest.mark.parametrize("agent", ["naive", "custom"])
def test_recorded_episode_offline(provider, agent, monkeypatch, tmp_path):
    repo = Path(__file__).resolve().parent.parent / "BALROG"
    monkeypatch.syspath_prepend(str(repo))
    import balrog.client

    model = "gemini-3.5-flash-lite" if provider == "gemini" else "qwen2.5:1.5b"
    key = "test-key-offline-only"
    requests = []
    completion = "PLAN: Find the ball.\nSUBGOAL: Look for the ball.\nACTION: turn left" if agent == "custom" else "turn left"

    def respond(request):
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "models/" + model if provider == "gemini" else model,
                                   "object": "model", "created": 0, "owned_by": "test"}]})
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        assert body["model"] == model
        assert body["temperature"] == 1.0
        if provider == "gemini":
            assert request.url.host == "generativelanguage.googleapis.com"
            assert request.headers["authorization"] == f"Bearer {key}"
            assert body["reasoning_effort"] == "minimal"
            assert "reasoning_format" not in body
            assert body["max_tokens"] == 1024
        else:
            assert body["max_tokens"] == 128
            assert "reasoning_effort" not in body
        return httpx.Response(200, json={
            "id": "offline-test", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": completion},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        })

    def sdk(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    if provider == "gemini":
        monkeypatch.setenv("GEMINI_API_KEY", key)
        monkeypatch.setattr(gemini_client, "OpenAI", sdk)
    else:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(balrog.client, "OpenAI", sdk)
        monkeypatch.setattr(run_local, "urlopen", lambda *a, **kw: io.BytesIO(
            json.dumps({"data": [{"id": model}]}).encode()))
    monkeypatch.chdir(repo)
    monkeypatch.setattr("sys.argv", ["run_local.py", "--agent", agent, "--provider", provider, "--repo", str(repo),
                                     "--max-steps", "2", "--seeds", "42", "43", "--request-delay", "0"])
    run_local.main(output_home=tmp_path)
    run = next((tmp_path / "results").iterdir())
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["provider"] == provider
    calls = [json.loads(line) for line in (run / "prompts.jsonl").read_text().splitlines()]
    assert [(call["seed"], call["step"]) for call in calls] == [(42, 0), (42, 1), (43, 0), (43, 1)]
    assert all(call["response"]["completion"] == completion for call in calls)
    decisions = [json.loads(line) for line in (run / "decisions.jsonl").read_text().splitlines()]
    if agent == "custom":
        assert len(decisions) == 4
        assert decisions[0]["memory_before"]["plan"] is None
        assert decisions[2]["memory_before"]["plan"] is None
        assert all(d["action"] == "turn left" and not d["issues"] for d in decisions)
    else:
        assert not decisions
    assert (run / "matrix.csv").exists()
    for file in run.rglob("*"):
        if file.is_file():
            assert key not in file.read_text()
    assert sum(r.url.path.endswith("/chat/completions") for r in requests) == 4


def test_resume_preserves_completed_work_and_restarts_partial_episode(monkeypatch, tmp_path):
    repo = Path(__file__).resolve().parent.parent / "BALROG"
    monkeypatch.syspath_prepend(str(repo))
    monkeypatch.chdir(repo)
    monkeypatch.setenv("GEMINI_API_KEY", "offline-resume-test")
    state = {"calls": 0, "interrupt": True}

    def respond(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "gemini-3.5-flash-lite"}], "object": "list"})
        state["calls"] += 1
        if state["interrupt"] and state["calls"] == 4:
            raise RuntimeError("Simulated quota interruption")
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0, "model": "gemini-3.5-flash-lite",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "turn left"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        })

    monkeypatch.setattr(gemini_client, "OpenAI", lambda **kwargs: OpenAI(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr("sys.argv", ["run_local.py", "--agent", "naive", "--provider", "gemini", "--repo", str(repo),
                                     "--max-steps", "2", "--seeds", "42", "43", "--request-delay", "0"])
    with pytest.raises(RuntimeError, match="Simulated quota"):
        run_local.main(output_home=tmp_path)
    run = next((tmp_path / "results").iterdir())
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "failed_or_interrupted"
    assert manifest["completed_episodes"] == 1
    assert manifest["expected_episodes"] == 2
    with (run / "matrix.csv").open() as f:
        assert next(csv.DictReader(f))["status"] == "partial"
    done_path = next((run / "seed_42").rglob("*run_00.json"))
    done_bytes = done_path.read_bytes()
    partial_csv = next((run / "seed_43").rglob("*run_00.csv"))
    partial_bytes = partial_csv.read_bytes()
    prompt_prefix = (run / "prompts.jsonl").read_bytes()

    # Older saved manifests did not contain attempt history. They must also resume.
    manifest.pop("attempts")
    (run / "manifest.json").write_text(json.dumps(manifest))
    hashes = dict(manifest["source_hashes"])
    manifest["source_hashes"]["runner.py"] = "different-code"
    (run / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr("sys.argv", ["run_local.py", "--resume", str(run)])
    with pytest.raises(SystemExit) as exc:
        run_local.main(output_home=tmp_path)
    assert exc.value.code == 2 and state["calls"] == 4
    manifest["source_hashes"] = hashes
    (run / "manifest.json").write_text(json.dumps(manifest))
    state["interrupt"] = False
    monkeypatch.setattr("sys.argv", ["run_local.py", "--resume", str(run)])
    run_local.main(output_home=tmp_path)
    assert state["calls"] == 6  # Only two new model calls; seed 42 was skipped.
    assert done_path.read_bytes() == done_bytes
    assert (run / "prompts.jsonl").read_bytes().startswith(prompt_prefix)
    backup = next((run / "interrupted_attempts").rglob("*.csv"))
    assert backup.read_bytes() == partial_bytes
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["completed_episodes"] == manifest["expected_episodes"] == 2
    calls = [json.loads(line) for line in (run / "prompts.jsonl").read_text().splitlines()]
    assert len(calls) == 5  # Includes the preserved unsuccessful first attempt.
    assert calls[2]["attempt"] != calls[3]["attempt"]
    with (run / "matrix.csv").open() as f:
        row = next(csv.DictReader(f))
        assert row["status"] == "complete" and row["episodes"] == "2"
    run_local.main(output_home=tmp_path)  # Already-complete resume is a no-op, with no API calls.
    assert state["calls"] == 6


def test_resume_rejects_protocol_changes(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["run_local.py", "--resume", str(tmp_path), "--seeds", "45", "46"])
    with pytest.raises(SystemExit) as exc:
        run_local.main(output_home=tmp_path)
    assert exc.value.code == 2
