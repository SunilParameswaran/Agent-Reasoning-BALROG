"""Build an offline viewer from recorded episodes; replay actions without any LLM calls."""

import csv
import base64
import gzip
import html as html_module
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "BALROG"))
from balrog.environments import make_env
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT

ACTIONS = ("turn left", "turn right", "go forward", "pick up", "drop", "toggle")
TASKS = ("goto", "pickup", "open", "putnext", "pick_up_seq_go_to")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def candidate(raw, exercise):
    if exercise != 2:
        return re.sub(r"[^a-zA-Z\s:]", "", raw)
    fields = defaultdict(list)
    for line in raw.strip().splitlines():
        match = re.fullmatch(r"\s*(PLAN|SUBGOAL|ACTION)\s*:\s*(.*?)\s*", line, re.I)
        if match:
            fields[match[1].upper()].append(match[2])
    value = fields["ACTION"][0] if len(fields["ACTION"]) == 1 else raw if not any(fields.values()) else ""
    value = " ".join(value.strip().strip("`\"'").lower().split())
    return value if value in ACTIONS else ""


def strip_feedback(text):
    return text.split("\n\nObservation:\n", 1)[-1].strip()


def build():
    entries = json.loads((ROOT / "Ass2-3/experiments.json").read_text())["runs"]
    runs = [(ROOT / "Ass2-3" / entry).resolve() for entry in entries]
    data = {"runs": [], "episodes": [], "tasks": TASKS,
            "objects": IDX_TO_OBJECT, "colors": IDX_TO_COLOR,
            "note": "Completed main episodes only. Boards reconstructed from saved seeds/actions and verified against every logged post-action observation and reward. No model was called."}
    audit = {"episodes": [], "source_hashes": {}, "model_calls_made": 0}
    for ri, run in enumerate(runs):
        exercise = int(run.parent.parent.name[-1])
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["status"] == "complete", run
        grouped = defaultdict(lambda: defaultdict(list))
        for call in read_jsonl(run / "prompts.jsonl"):
            grouped[(call["seed"], call["task"])][call.get("attempt", "original")].append(call)
        run_meta = {"id": ri, "exercise": exercise, "model": manifest["model"],
                    "provider": manifest.get("provider", "ollama"),
                    "path": str(run.relative_to(ROOT)), "agent": manifest.get("agent", "naive"),
                    "temperature": manifest["temperature"], "max_tokens": manifest["max_tokens"],
                    "reasoning_effort": manifest.get("reasoning_effort"),
                    "attempts": len(manifest.get("attempts", [0])), "commit": manifest["commit"]}
        data["runs"].append(run_meta)
        for seed in manifest["seeds"]:
            cfg = OmegaConf.load(run / f"seed_{seed}/config.yaml")
            for task in TASKS:
                full_task = f"BabyAI-MixedTrainLocal-v0/{task}"
                path = run / f"seed_{seed}/babyai/{full_task}/{full_task}_run_00.json"
                source = json.loads(path.read_text())
                rows = list(csv.DictReader(path.with_suffix(".csv").open()))
                assert len(rows) == source["num_steps"]
                attempts = grouped[(seed, full_task)]
                chosen = None
                for attempt, calls in reversed(list(attempts.items())):
                    if source.get("completed_attempt") and attempt != source["completed_attempt"]:
                        continue
                    actors = [c for c in calls if c.get("role", "actor") == "actor"]
                    if len(actors) != len(rows):
                        continue
                    if all(c["step"] == i and candidate(c["response"]["completion"], exercise) == rows[i]["Action"] for i, c in enumerate(actors)):
                        chosen = (attempt, calls, actors)
                        break
                assert chosen is not None, f"Cannot match a completed attempt: {path}"
                attempt, calls, actors = chosen
                assert sum(c["response"]["input_tokens"] for c in calls) == source["input_tokens"], path
                assert sum(c["response"]["output_tokens"] for c in calls) == source["output_tokens"], path
                critics = {c["step"]: c for c in calls if c.get("role") == "critic"}
                env = make_env("babyai", full_task, cfg)
                random.seed(seed)
                np.random.seed(seed)
                obs, _ = env.reset(seed=seed)
                assert obs["mission"] == source["mission"], path
                grids, grid_lookup = [], {}
                states = []

                def save_grid(encoded):
                    cells = encoded.transpose(1, 0, 2).reshape(-1, 3).tolist()
                    grid = {"w": int(encoded.shape[0]), "h": int(encoded.shape[1]), "cells": cells}
                    key = json.dumps(grid, separators=(",", ":"))
                    if key not in grid_lookup:
                        grid_lookup[key] = len(grids)
                        grids.append(grid)
                    return grid_lookup[key]

                def save_state():
                    world = env.unwrapped
                    visible, mask = world.gen_obs_grid()
                    carrying = world.carrying
                    states.append({"grid": save_grid(world.grid.encode()),
                                   "view": save_grid(visible.encode(mask)),
                                   "x": int(world.agent_pos[0]), "y": int(world.agent_pos[1]),
                                   "dir": int(world.agent_dir),
                                   "carrying": f"{carrying.color} {carrying.type}" if carrying else "Nothing"})

                save_state()
                initial = obs["text"]["long_term_context"]
                steps = []
                frequency = Counter()
                for i, row in enumerate(rows):
                    assert int(row["Step"]) == i
                    before = initial if i == 0 else rows[i-1]["Observation"]
                    prompt_text = actors[i]["messages"][-1]["content"]
                    assert prompt_text.startswith("Current Observation:\n" + before), (path, i, "prompt observation")
                    action = env.check_action_validity(row["Action"])
                    frequency[action] += 1
                    obs, reward, terminated, truncated, _ = env.step(action)
                    assert strip_feedback(row["Observation"]) == obs["text"]["long_term_context"].strip(), (path, i, "observation")
                    assert math.isclose(float(row["Reward"]), reward, abs_tol=1e-9), (path, i, "reward")
                    assert (row["Done"] == "True") == bool(terminated or truncated), (path, i, "termination")
                    assert not (terminated or truncated) or i == len(rows)-1
                    save_state()
                    decision = json.loads(row["Reasoning"]) if row.get("Reasoning", "").startswith("{") else None
                    steps.append({"candidate": row["Action"], "action": action,
                                  "fallback": action != row["Action"],
                                  "before": before, "after": row["Observation"],
                                  "reward": reward, "done": bool(terminated or truncated),
                                  "decision": decision, "actor": actors[i], "critic": critics.get(i)})
                assert dict(frequency) == source["action_frequency"], path
                assert env.get_stats()["progression"] == source["progression"], path
                env.close()
                episode = {"id": f"e{exercise}-{'qwen' if run_meta['provider'] == 'ollama' else 'gemini'}-{task}-{seed}",
                           "run": ri, "exercise": exercise, "model": manifest["model"],
                           "task": task, "seed": seed, "mission": source["mission"],
                           "success": source["progression"] == 1, "attempt": attempt,
                           "source": str(path.relative_to(ROOT)),
                           "input_tokens": source["input_tokens"], "output_tokens": source["output_tokens"],
                           "invalid": len(source["failed_candidates"]),
                           "reflection_stats": source.get("reflection_stats"),
                           "grids": grids, "states": states, "steps": steps, "verified": True}
                data["episodes"].append(episode)
                audit["episodes"].append({"id": episode["id"], "steps": len(steps), "verified": True, "completed_attempt": attempt})
                for file in (path, path.with_suffix(".csv")):
                    audit["source_hashes"][str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
        print(f"Verified exercise {exercise}, {manifest['model']}: 25 episodes", flush=True)
    assert len(data["episodes"]) == 150
    expected = [7, 21, 2, 15, 3, 22]
    assert [sum(e["success"] for e in data["episodes"] if e["run"] == ri) for ri in range(6)] == expected
    for run in runs:
        p = run / "prompts.jsonl"
        audit["source_hashes"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    audit["total_steps"] = sum(len(e["steps"]) for e in data["episodes"])
    audit["total_episodes"] = len(data["episodes"])
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    secret = re.compile(r"(?:AIza[0-9A-Za-z_-]{30,}|gsk_[0-9A-Za-z]{30,}|sk-(?:proj-)?[0-9A-Za-z_-]{30,})")
    assert not secret.search(payload), "Possible credential in selected logs"
    (HERE / "verified_data.json").write_text(payload)
    (HERE / "validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    render(payload)


def render(payload=None):
    payload = payload or (HERE / "verified_data.json").read_text()
    template = (HERE / "viewer.html").read_text()
    assert template.count("__DATA_PAYLOAD__") == 1
    packed = base64.b64encode(gzip.compress(payload.encode(), mtime=0)).decode()
    html = template.replace("__DATA_PAYLOAD__", packed)
    for target in (HERE / "dist/index.html", ROOT / "index.html"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html)
        render_results(target.parent, target.name)
    print(f"Saved index.html ({len(html.encode())/1024/1024:.1f} MB)", flush=True)


def render_results(directory, viewer_name):
    source = ROOT / "Ass2-3/comparison.csv"
    with source.open(newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        rows = list(reader)
    escape = html_module.escape
    header = "".join(f"<th scope='col'>{escape(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{escape(row[c])}</td>" for c in columns) + "</tr>" for row in rows)
    template = (HERE / "results_template.html").read_text()
    page = template.replace("__VIEWER__", viewer_name).replace("__HEADER__", header).replace("__ROWS__", body).replace("__COUNT__", str(len(rows)))
    (directory / "results.html").write_text(page)
    (directory / "comparison.csv").write_bytes(source.read_bytes())


if __name__ == "__main__":
    if "--render-only" in sys.argv:
        render()
    else:
        build()
