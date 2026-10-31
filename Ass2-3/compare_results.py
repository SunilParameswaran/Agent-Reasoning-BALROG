"""Summarize completed BabyAI episodes; never score interrupted episodes as failures."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

TASKS = ("goto", "pickup", "open", "putnext", "pick_up_seq_go_to")


def read_run(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    tasks = manifest.get("tasks") or (TASKS if manifest["profile"] == "baseline" else ("goto",))
    records = {}
    for seed in manifest["seeds"]:
        for task in tasks:
            full_task = f"BabyAI-MixedTrainLocal-v0/{task}"
            path = folder / f"seed_{seed}/babyai/{full_task}/{full_task}_run_00.json"
            if not path.exists():
                continue
            episode = json.loads(path.read_text())
            if episode["seed"] != seed or episode["task"].split("/")[-1] != task:
                raise ValueError(f"Episode identity mismatch: {path}")
            if episode["progression"] not in (0, 1):
                raise ValueError(f"Expected binary BabyAI success: {path}")
            issues = Counter()
            trajectory = path.with_suffix(".csv")
            if trajectory.exists():
                with trajectory.open() as source:
                    for row in csv.DictReader(source):
                        if row.get("Reasoning", "").startswith('{"memory_before":'):
                            issues.update(json.loads(row["Reasoning"])["issues"])
            episode["format_issues"] = dict(issues)
            records[(seed, task)] = episode
    return manifest, tasks, records


def summarize(folder):
    manifest, tasks, records = read_run(folder)
    rows = []
    for task in ("ALL", *tasks):
        episodes = [v for (_, t), v in records.items() if task == "ALL" or t == task]
        n = len(episodes)
        expected = len(manifest["seeds"]) * (len(tasks) if task == "ALL" else 1)
        successes = sum(e["progression"] for e in episodes)
        p = successes / n if n else None
        rows.append({
            "run": Path(folder).name,
            "model": manifest["model"],
            "agent": manifest.get("agent", "naive"),
            "profile": manifest["profile"],
            "benchmark": "babyai",
            "task": task,
            "successes": int(successes),
            "completed": n,
            "expected": expected,
            "status": "complete" if n == expected else "partial",
            "success_percent": round(100 * p, 3) if n else "",
            "standard_error_percent": round(100 * math.sqrt(p * (1-p) / n), 3) if n else "",
            "actions": sum(e["num_steps"] for e in episodes),
            "invalid_candidates": sum(len(e["failed_candidates"]) for e in episodes),
            "input_tokens": sum(e["input_tokens"] for e in episodes),
            "output_tokens": sum(e["output_tokens"] for e in episodes),
            "critic_calls": sum(e.get("reflection_stats", {}).get("reviews", 0) for e in episodes),
            "accepted_reviews": sum(e.get("reflection_stats", {}).get("accepted_reviews", 0) for e in episodes),
            "critic_input_tokens": sum(e.get("reflection_stats", {}).get("usage", {}).get("critic", {}).get("input_tokens", 0) for e in episodes),
            "critic_output_tokens": sum(e.get("reflection_stats", {}).get("usage", {}).get("critic", {}).get("output_tokens", 0) for e in episodes),
            "format_issues": json.dumps(dict(sum((Counter(e["format_issues"]) for e in episodes), Counter())), sort_keys=True),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "comparison.csv")
    args = parser.parse_args()
    if len({p.resolve() for p in args.runs}) != len(args.runs):
        parser.error("A run folder was supplied twice.")
    rows = [row for run in args.runs for row in summarize(run)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["task"] == "ALL":
            print(f"{row['model']} / {row['agent']}: {row['success_percent']}% "
                  f"({row['successes']}/{row['completed']}; expected {row['expected']}; "
                  f"{row['profile']}, {row['status']})")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
