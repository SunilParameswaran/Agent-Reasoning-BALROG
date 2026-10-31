"""Verify the published comparison from completed episodes, without model calls."""

import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    spec = importlib.util.spec_from_file_location("comparison", ROOT / "Ass2-3/compare_results.py")
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    entries = json.loads((ROOT / "Ass2-3/experiments.json").read_text())["runs"]
    rows = [row for entry in entries for row in comparison.summarize(ROOT / "Ass2-3" / entry)]
    with (ROOT / "Ass2-3/comparison.csv").open(newline="") as source:
        published = list(csv.DictReader(source))
    normalized = [{key: str(value) for key, value in row.items()} for row in rows]
    if normalized != published:
        raise SystemExit("The recorded episodes differ from the published comparison; inspect the source runs.")
    output = ROOT / "generated/comparison.csv"
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["task"] == "ALL":
            print(f"{row['model']} / {row['agent']}: {row['successes']}/{row['completed']} ({row['success_percent']}%)")
    print(f"Verified {len(rows)} comparison rows. Saved {output}")


if __name__ == "__main__":
    main()
