"""Package Exercise 2 evidence after evaluation stops, excluding credentials."""

import argparse
import csv
import json
import re
import shutil
import time
import zipfile
from pathlib import Path

from compare_results import summarize

HOME = Path(__file__).resolve().parent
BASELINES = (
    "20260930_170915_722488_baseline_qwen2.5_1.5b",
    "20261005_133250_414751_baseline_gemini_gemini-3.5-flash-lite",
)
CUSTOM = (
    "20261007_114314_267623_baseline_custom_ollama_qwen2.5_1.5b",
    "20261007_114345_414551_baseline_custom_gemini_gemini-3.5-flash-lite",
)
ALLOWED = {".py", ".md", ".json", ".jsonl", ".csv", ".yaml", ".txt", ".log"}
SECRET = re.compile(rb"(?:AIza[0-9A-Za-z_-]{30,}|gsk_[0-9A-Za-z]{30,}|sk-(?:proj-)?[0-9A-Za-z_-]{30,})")


def run_folders():
    baselines = []
    for name in BASELINES:
        original = HOME.parent / "Ass2-1/results" / name
        baselines.append(original if original.exists() else HOME / "reference/baselines" / name)
    return baselines + [HOME / "results" / name for name in CUSTOM]


def write_report(rows):
    overall = [r for r in rows if r["task"] == "ALL"]
    complete = all(r["status"] == "complete" for r in overall)
    text = ["# Exercise 2 results", "",
            "Completed comparison." if complete else "PARTIAL comparison: finish the missing episodes before making overall agent rankings.", "",
            "| Model | Agent | Success | Episodes completed/expected | Status |",
            "|---|---|---:|---:|---|"]
    for r in overall:
        text.append(f"| {r['model']} | {r['agent']} | {r['success_percent']}% | {r['completed']}/{r['expected']} | {r['status']} |")
    text += ["", "The full task-level matrix is in `comparison.csv`. Success is mission completion within the action limit.",
             "The baseline evidence comes from the student's Exercise 1 experiments, not the teaching assistant's example.", ""]
    for model in dict.fromkeys(r["model"] for r in overall):
        baseline = next(r for r in overall if r["model"] == model and r["agent"] == "naive")
        custom = next(r for r in overall if r["model"] == model and r["agent"] == "custom")
        if baseline["status"] == custom["status"] == "complete":
            delta = custom["success_percent"] - baseline["success_percent"]
            text.append(f"- {model}: custom minus naive = {delta:+.1f} percentage points; "
                        f"{custom['invalid_candidates']} invalid action candidates in {custom['actions']} custom-agent actions "
                        f"versus {baseline['invalid_candidates']} in {baseline['actions']} naive-agent actions.")
    text += ["", "There are five seeds per task and one model trajectory per task/seed/agent combination. "
             "No repeated-sampling brittleness experiment has been collected. Pilots use separate seeds and are excluded here.",
             "", "The custom treatment combines a planning prompt, persistent plan/subgoal memory and a new parser. "
             "It does not isolate the effect of memory. The earlier baseline and current cloud run may also differ in provider-side model version. "
             "Do not infer statistical significance or general reasoning ability from these small samples.", ""]
    (HOME / "REPORT_RESULTS.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--wait", action="store_true", help="Wait for the two main evaluations to finish or fail before packaging")
    args = parser.parse_args()
    runs = run_folders()
    while True:
        running = [r.name for r in runs if json.loads((r / "manifest.json").read_text()).get("status") == "running"]
        if not running:
            break
        if not args.wait:
            parser.error("Evaluation is still running; wait or use --wait.")
        print("Waiting for " + ", ".join(running), flush=True)
        time.sleep(30)
    rows = [row for run in runs for row in summarize(run)]
    if any(r["status"] != "complete" for r in rows) and not args.allow_partial:
        parser.error("Some episodes are missing. Resume first, or explicitly use --allow-partial.")
    with (HOME / "comparison.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(rows)
    active_agent = HOME.parent / "BALROG/balrog/agents/custom.py"
    (HOME / "agent").mkdir(exist_ok=True)
    if active_agent.exists():
        shutil.copy2(active_agent, HOME / "agent/custom.py")
    if not (HOME / "agent/custom.py").exists():
        parser.error("Agent source is missing.")
    files = {}
    for path in HOME.rglob("*"):
        rel = path.relative_to(HOME)
        if path.is_file() and path.suffix in ALLOWED and not any(p.startswith(".") or p == "__pycache__" for p in rel.parts):
            files["Ass2-2/" + rel.as_posix()] = path
    for run in runs[:2]:
        for path in run.rglob("*"):
            if path.is_file() and path.suffix in ALLOWED:
                files[f"Ass2-2/reference/baselines/{run.name}/{path.relative_to(run).as_posix()}"] = path
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
        license_path = HOME.parent / "BALROG" / name
        if license_path.exists():
            files["Ass2-2/reference/BALROG_" + name] = license_path
    # Validate before producing an archive; only allowlisted text file types enter it.
    for name, path in files.items():
        if SECRET.search(path.read_bytes()):
            raise ValueError(f"Possible API key in {name}; remove it before packaging.")
    destination = HOME.parent / "Ass2-2.zip"
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
    temporary.replace(destination)
    print(f"Saved {destination} ({len(files)} files).", flush=True)


if __name__ == "__main__":
    main()
