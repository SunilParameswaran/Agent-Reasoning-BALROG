"""Read evaluation progress from disk. Makes no API requests."""

import json
import time
from pathlib import Path


def main():
    home = Path(__file__).resolve().parent
    for path in sorted((home / "results").glob("*/manifest.json")):
        try:
            manifest = json.loads(path.read_text())
            episodes = [json.loads(p.read_text()) for p in path.parent.glob("seed_*/babyai/**/*_run_*.json")]
            expected = len(manifest["seeds"]) * len(manifest.get("tasks") or (range(5) if manifest["profile"] == "baseline" else range(1)))
            print(f"{manifest['model']} / {manifest['agent']}: {manifest['status']}; "
                  f"{len(episodes)}/{expected} completed, {sum(e['progression'] for e in episodes):g} successes")
            prompts = path.parent / "prompts.jsonl"
            if prompts.exists() and prompts.stat().st_size:
                lines = prompts.read_text().splitlines()
                try:
                    last = json.loads(lines[-1])
                except json.JSONDecodeError:
                    last = json.loads(lines[-2]) if len(lines) > 1 else None
                if last:
                    print(f"  Last response: {last.get('role', 'actor')}, seed {last['seed']}, "
                          f"{last['task'].split('/')[-1]}, action index {last['step']} "
                          f"({round(time.time()-prompts.stat().st_mtime)} seconds since log update)")
            if manifest.get("attempts") and manifest["attempts"][-1].get("reason"):
                print("  Stop reason:", manifest["attempts"][-1]["reason"])
            print(" ", path.parent)
        except (json.JSONDecodeError, OSError):
            print(f"{path.parent.name}: files are being updated; check again shortly.")
    print("A 'running' manifest records the last saved state; a stale log alone does not prove the process is alive.")


if __name__ == "__main__":
    main()
