"""Refresh the comparison and package completed or explicitly partial evidence."""

import argparse
import csv
import json
import re
import time
import zipfile
from pathlib import Path

from compare_results import summarize

HOME = Path(__file__).resolve().parent
SECRET = re.compile(rb"(?:AIza[0-9A-Za-z_-]{30,}|gsk_[0-9A-Za-z]{30,}|sk-(?:proj-)?[0-9A-Za-z_-]{30,})")
ALLOWED = {".py", ".md", ".json", ".jsonl", ".csv", ".yaml", ".txt", ".log"}


def selected_runs():
    entries = json.loads((HOME / "experiments.json").read_text())["runs"]
    runs = []
    for entry in entries:
        original = HOME / entry
        fallback = HOME / "reference" / entry.removeprefix("../")
        runs.append(original if original.exists() else fallback)
    return runs


def write_tables(runs):
    rows = [r for run in runs for r in summarize(run)]
    with (HOME / "comparison.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    overall = [r for r in rows if r["task"] == "ALL"]
    lines = ["# Exercise 3 results", "",
             "Complete comparison." if all(r["status"] == "complete" for r in overall) else "PARTIAL: do not rank incomplete results against complete baselines.", "",
             "| Model | Agent | Success | Completed/expected | Critic calls | Status |",
             "|---|---|---:|---:|---:|---|"]
    for r in overall:
        lines.append(f"| {r['model']} | {r['agent']} | {r['success_percent']}% | {r['completed']}/{r['expected']} | {r['critic_calls']} | {r['status']} |")
    lines += ["", "Task-level results, token counts and accepted critiques are in comparison.csv. "
              "These totals cover completed episodes only. No live pilot was run for Exercise 3.", "",
              "The naive and planning-memory baselines are historical student experiments. "
              "The reflection actor uses the naive prompt/parser plus saved advice. Critic calls add computation. "
              "The comparison to Exercise 2 also changes the response format and memory mechanism.", "",
              "Five seeds per task are not repeated samples of an identical instance. "
              "Earlier results informed the design, so this is exploratory rather than held-out evaluation. "
              "No repeated-sampling stability study or bonus philosophical answer is included.", ""]
    (HOME / "REPORT_RESULTS.md").write_text("\n".join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    runs = selected_runs()
    while True:
        try:
            running = [p.name for p in runs if json.loads((p/"manifest.json").read_text())["status"] == "running"]
        except json.JSONDecodeError:
            if not args.wait:
                raise
            time.sleep(1)
            continue
        if not running:
            break
        if not args.wait:
            parser.error("An evaluation is running; wait or use --wait.")
        print("Waiting for " + ", ".join(running), flush=True)
        time.sleep(30)
    rows = [r for run in runs for r in summarize(run)]
    if any(r["status"] != "complete" for r in rows) and not args.allow_partial:
        parser.error("Missing episodes: resume first or use --allow-partial.")
    write_tables(runs)
    files = {}
    for path in HOME.rglob("*"):
        rel = path.relative_to(HOME)
        if path.is_file() and path.suffix in ALLOWED and not any(p.startswith(".") or p == "__pycache__" for p in rel.parts):
            files["Ass2-3/" + rel.as_posix()] = path
    for entry, run in zip(json.loads((HOME/"experiments.json").read_text())["runs"], runs):
        if not entry.startswith("../"):
            continue
        for path in run.rglob("*"):
            if path.is_file() and path.suffix in ALLOWED:
                files["Ass2-3/reference/" + entry.removeprefix("../") + "/" + path.relative_to(run).as_posix()] = path
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
        path = HOME.parent / "BALROG" / name
        if path.exists():
            files["Ass2-3/reference/BALROG_" + name] = path
    for name, path in files.items():
        if SECRET.search(path.read_bytes()):
            raise ValueError(f"Possible API key in {name}; remove it before packaging.")
    destination = HOME.parent / "Ass2-3.zip"
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
    temporary.replace(destination)
    print(f"Saved {destination} ({len(files)} files)", flush=True)


if __name__ == "__main__":
    main()
