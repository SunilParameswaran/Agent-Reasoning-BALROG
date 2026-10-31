"""Build complete or explicitly partial tables using completed episode JSONs only."""

import csv
import json
import math
from pathlib import Path


def write_result_tables(run, settings):
    from omegaconf import OmegaConf  # type: ignore

    run = Path(run)
    snapshots = sorted(run.glob("seed_*/config.yaml"))
    if not snapshots:
        return {"completed_episodes": 0, "expected_episodes": None}
    template = OmegaConf.load(snapshots[0])
    rows = []
    expected = {env: 0 for env in dict.fromkeys(settings["envs"])}
    for seed in settings["seeds"]:
        seed_dir = run / f"seed_{seed}"
        snapshot = seed_dir / "config.yaml"
        cfg = OmegaConf.load(snapshot) if snapshot.exists() else template
        for env in expected:
            for task in cfg.tasks[f"{env}_tasks"]:
                for index in range(cfg.eval.num_episodes[env]):
                    expected[env] += 1
                    path = seed_dir / env / task / f"{task}_run_{index:02d}.json"
                    if not path.exists():
                        continue  # An unfinished episode is missing data, not a failure score.
                    episode = json.loads(path.read_text())
                    if episode["task"] != task or episode["seed"] != seed:
                        raise ValueError(
                            f"Episode identity does not match its folder: {path}"
                        )
                    rows.append(
                        {
                            "agent": settings["agent"],
                            "model": settings["model"],
                            "benchmark": env,
                            "seed": seed,
                            "task": task,
                            "episode": index,
                            "progression": episode["progression"],
                            "steps": episode["num_steps"],
                            "failed_candidates": episode["failed_candidates"],
                        }
                    )
    # Replace tables only after all saved episode records have been validated.
    with (run / "episodes.csv").open("w", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "agent",
                "model",
                "benchmark",
                "seed",
                "task",
                "episode",
                "progression",
                "steps",
                "failed_candidates",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (run / "matrix.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(
            [
                "agent",
                "model",
                "benchmark",
                "profile",
                "progression_percent",
                "standard_error_percent",
                "episodes",
                "expected_episodes",
                "status",
            ]
        )
        for env, count in expected.items():
            values = [r["progression"] for r in rows if r["benchmark"] == env]
            n = len(values)
            mean = sum(values) / n if n else None
            se = math.sqrt(sum((v - mean) ** 2 for v in values)) / n if n else None
            writer.writerow(
                [
                    settings["agent"],
                    settings["model"],
                    env,
                    settings["profile"],
                    100 * mean if n else "",
                    100 * se if n else "",
                    n,
                    count,
                    "complete" if n == count else "partial",
                ]
            )
    return {
        "completed_episodes": len(rows),
        "expected_episodes": sum(expected.values()),
    }
