#!/usr/bin/env python3
"""Compare naive and planning-memory agents on BabyAI using Ollama or Gemini."""  # noqa: EXE001

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen


def main(output_home=None):
    home = Path(output_home).resolve() if output_home else Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["naive", "custom"], default="custom")
    parser.add_argument("--tasks", nargs="+", choices=["goto", "pickup", "open", "putnext", "pick_up_seq_go_to"], help="Optional predefined BabyAI task subset")
    parser.add_argument("--repo", type=Path, default=home.parent / "BALROG")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume a run folder using its saved experiment settings",
    )
    parser.add_argument("--provider", choices=["ollama", "gemini"], default="ollama")
    parser.add_argument(
        "--model", help="Default: qwen2.5:1.5b locally, gemini-3.5-flash-lite on Gemini"
    )
    parser.add_argument(
        "--base-url", help="Local server URL; Gemini uses its fixed official endpoint"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List server models and exit without evaluation",
    )
    parser.add_argument("--profile", choices=["smoke", "baseline"], default="smoke")
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["babyai"],
        choices=["babyai"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--max-steps", type=int, help="Override the environment step limit"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Default: 128 locally, 1024 on Gemini (including thinking)",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--request-delay",
        type=float,
        help="Seconds between model calls: default 6 on Gemini, 0 locally",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        help="Gemini only: defaults to minimal for Gemini 3, none for 2.5 Flash",
    )
    args = parser.parse_args()
    previous_metadata = None
    if args.resume:
        explicit = {
            item.split("=")[0] for item in sys.argv[1:] if item.startswith("--")
        }
        unsupported = explicit - {"--resume", "--request-delay", "--repo"}
        if unsupported:
            parser.error(
                "Resume restores the original experiment settings. Remove: "
                + ", ".join(sorted(unsupported))
            )
        args.resume = args.resume.resolve()
        try:
            previous_metadata = json.loads((args.resume / "manifest.json").read_text())
            for field in [
                "agent",
                "tasks",
                "model",
                "profile",
                "envs",
                "seeds",
                "max_steps",
                "max_tokens",
                "temperature",
                "base_url",
            ]:
                setattr(args, field, previous_metadata[field])
            args.provider = previous_metadata.get("provider", "ollama")
            args.reasoning_effort = previous_metadata.get("reasoning_effort")
            if "--request-delay" not in explicit:
                args.request_delay = previous_metadata.get("request_delay")
            if "--repo" not in explicit:
                args.repo = Path(previous_metadata["repo"])
            if not list(args.resume.glob("seed_*/config.yaml")):
                parser.error("Cannot resume: no saved seed configuration exists.")
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"Cannot read saved experiment: {exc}")
        if previous_metadata.get("status") == "complete":
            print(f"This run is already complete: {args.resume}")
            return
    if args.seeds is None:
        args.seeds = [100] if args.profile == "smoke" else [42, 43, 44, 45, 46]
    cloud_sdk = None
    if args.provider == "gemini":
        from gemini_client import DEFAULT_MODEL, GEMINI_BASE_URL, connect_gemini

        args.model = (args.model or DEFAULT_MODEL).removeprefix("models/")
        if args.base_url and args.base_url.rstrip("/") != GEMINI_BASE_URL:
            parser.error(
                "Gemini credentials may only be sent to the official Google endpoint."
            )
        args.base_url = GEMINI_BASE_URL
        if args.reasoning_effort is None:
            if args.model.startswith("gemini-2.5-flash"):
                args.reasoning_effort = "none"
            elif "pro" in args.model:
                args.reasoning_effort = "low"
            else:
                args.reasoning_effort = "minimal"
        if args.reasoning_effort == "none" and not args.model.startswith(
            "gemini-2.5-flash"
        ):
            parser.error(
                "Thinking cannot be disabled for Gemini 3 or 2.5 Pro; use minimal/low instead."
            )
    else:
        args.model = args.model or "qwen2.5:1.5b"
        args.base_url = args.base_url or "http://localhost:11434/v1"
        if args.reasoning_effort is not None:
            parser.error("--reasoning-effort is only supported for --provider gemini")
    if args.max_tokens is None:
        args.max_tokens = 1024 if args.provider == "gemini" else 128
    if args.request_delay is None:
        args.request_delay = 6.0 if args.provider == "gemini" else 0.0
    if not math.isfinite(args.request_delay) or args.request_delay < 0:
        parser.error("--request-delay must be a finite non-negative number")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct seeds; repeat runs in separate result directories.")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    repo = args.repo.resolve()
    if not args.list_models and not (repo / "eval.py").is_file():
        parser.error(f"BALROG checkout not found at {repo}")
    if previous_metadata is not None:
        current_commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if current_commit != previous_metadata.get("commit"):
            parser.error(
                "Resume requires the original BALROG Git revision to preserve the experiment."
            )
    source_files = {"runner.py": Path(__file__).resolve(),
                    "gemini_client.py": Path(__file__).resolve().parent / "gemini_client.py",
                    "result_tables.py": Path(__file__).resolve().parent / "result_tables.py"}
    for path in sorted((repo / "balrog").rglob("*")):
        if path.is_file() and path.suffix in {".py", ".yaml"}:
            source_files["BALROG/" + str(path.relative_to(repo))] = path
    source_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for name, path in source_files.items()}
    if previous_metadata is not None and source_hashes != previous_metadata.get("source_hashes"):
        parser.error("Source files changed since this run began. Restore its code_snapshot before resuming, or start a new run.")
    # Fail before starting an experiment if the server, credentials or model are unavailable.
    try:
        if args.provider == "gemini":
            cloud_sdk = connect_gemini()
            server_models = {
                "data": [
                    model.model_dump(mode="json") for model in cloud_sdk.models.list()
                ]
            }
        else:
            with urlopen(args.base_url.rstrip("/") + "/models", timeout=10) as response:
                server_models = json.load(response)
    except Exception as exc:  # noqa: BLE001
        if cloud_sdk is not None:
            cloud_sdk.close()
        parser.error(f"Cannot reach model server: {exc}")
    available = [
        item["id"].removeprefix("models/") if args.provider == "gemini" else item["id"]
        for item in server_models.get("data", [])
    ]
    if args.list_models:
        print("\n".join(sorted(available)))
        if cloud_sdk is not None:
            cloud_sdk.close()
        return
    if args.model not in available:
        if cloud_sdk is not None:
            cloud_sdk.close()
        parser.error(f"Model {args.model!r} is unavailable. Server models: {available}")

    sys.path.insert(0, str(repo))
    from hydra import compose, initialize_config_dir  # type: ignore  # noqa: I001
    from omegaconf import OmegaConf, open_dict  # type: ignore
    from balrog.agents import AgentFactory  # type: ignore
    from balrog.evaluator import EvaluatorManager  # type: ignore
    from balrog.utils import collect_and_summarize_results, print_summary_table  # type: ignore

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # noqa: DTZ005
    run = args.resume or (
        home
        / "results"
        / f"{stamp}_{args.profile}_{args.agent}_{args.provider}_{args.model.replace('/', '_').replace(':', '_')}"
    )
    run.mkdir(parents=True, exist_ok=bool(args.resume))
    logging.basicConfig(filename=run / "eval.log", level=logging.INFO, force=True)
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    metadata = previous_metadata or {
        **{key: value for key, value in vars(args).items() if key != "resume"},
        "repo": str(repo),
        "commit": commit,
        "source_hashes": source_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "server_models": server_models,
        "status": "running",
        "source": "own cloud experiment"
        if args.provider == "gemini"
        else "own local experiment",
    }
    if not args.resume:
        for name, path in source_files.items():
            snapshot = run / "code_snapshot" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, snapshot)
    attempt = {
        "id": stamp,
        "resumed": bool(args.resume),
        "request_delay": args.request_delay,
        "python": sys.version,
        "status": "running",
    }
    metadata.setdefault("attempts", []).append(attempt)
    metadata["status"] = "running"
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    package_file = run / (
        f"packages_resume_{stamp}.txt" if args.resume else "packages.txt"
    )
    with package_file.open("w") as target:
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], stdout=target, check=True
        )
    os.chdir(repo)  # TextWorld and demonstrations use paths relative to the checkout.

    context = {}
    next_request_at = 0.0
    # One worker keeps each prompt associated with the correct episode and step.
    with (run / "prompts.jsonl").open("a" if args.resume else "w") as prompts, (run / "decisions.jsonl").open("a" if args.resume else "w") as decisions:

        class RecordingAgentFactory(AgentFactory):
            def create_agent(self):
                agent = super().create_agent()
                if args.provider == "gemini":
                    from gemini_client import GeminiClient

                    agent.client = GeminiClient(cloud_sdk, self.config.client)
                original_generate = agent.client.generate

                def recorded_generate(messages):
                    nonlocal next_request_at
                    wait = next_request_at - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                    prompt = [{"role": m.role, "content": m.content} for m in messages]
                    response = original_generate(messages)
                    next_request_at = time.monotonic() + args.request_delay
                    prompts.write(
                        json.dumps(
                            {
                                **context,
                                "attempt": stamp,
                                "messages": prompt,
                                "response": response._asdict(),
                            }
                        )
                        + "\n"
                    )
                    prompts.flush()
                    context["step"] += 1
                    return response

                agent.client.generate = recorded_generate
                original_act = agent.act

                def recorded_act(obs, prev_action=None):
                    step = context["step"]
                    response = original_act(obs, prev_action=prev_action)
                    if args.agent == "custom":
                        decisions.write(json.dumps({**context, "step": step, "attempt": stamp,
                                                    **agent.last_decision}) + "\n")
                        decisions.flush()
                    return response

                agent.act = recorded_act
                return agent

        try:
            for seed in args.seeds:
                with initialize_config_dir(
                    config_dir=str(repo / "balrog/config"), version_base="1.1"
                ):
                    cfg = compose(config_name="config")
                cfg.agent.type = args.agent
                cfg.agent.max_image_history = 0
                cfg.agent.max_text_history = 16
                cfg.client.client_name = (
                    "gemini-openai" if args.provider == "gemini" else "vllm"
                )
                # AgentFactory accepts the openai substring; RecordingAgentFactory
                # replaces the lazy placeholder with our authenticated Gemini client.
                cfg.client.model_id = args.model
                cfg.client.base_url = args.base_url
                cfg.client.generate_kwargs.max_tokens = args.max_tokens
                cfg.client.generate_kwargs.temperature = args.temperature
                cfg.client.max_retries = 2
                if args.provider == "gemini":
                    cfg.client.max_retries = 5
                    with open_dict(cfg.client.generate_kwargs):
                        cfg.client.generate_kwargs.reasoning_effort = (
                            args.reasoning_effort
                        )
                cfg.eval.num_workers = 1
                cfg.eval.save_trajectories = True
                cfg.eval.save_images = False
                cfg.eval.max_steps_per_episode = (
                    args.max_steps
                    if args.max_steps is not None
                    else (20 if args.profile == "smoke" else None)
                )
                cfg.envs.names = "-".join(dict.fromkeys(args.envs))
                cfg.envs.env_kwargs.seed = seed
                seed_dir = run / f"seed_{seed}"
                seed_dir.mkdir(exist_ok=bool(args.resume))
                cfg.eval.output_dir = str(seed_dir)
                for env in dict.fromkeys(args.envs):
                    cfg.eval.num_episodes[env] = 1
                    if args.tasks:
                        cfg.tasks[f"{env}_tasks"] = [f"BabyAI-MixedTrainLocal-v0/{task}" for task in dict.fromkeys(args.tasks)]
                    elif args.profile == "smoke":
                        cfg.tasks[f"{env}_tasks"] = cfg.tasks[f"{env}_tasks"][:1]
                snapshot = seed_dir / "config.yaml"
                if args.resume:
                    # Preserve the original task lists and all environment settings,
                    # including for seeds that had not started when the run stopped.
                    saved = (
                        snapshot
                        if snapshot.exists()
                        else sorted(run.glob("seed_*/config.yaml"))[0]  # noqa: FURB192
                    )  # noqa: FURB192, RUF100
                    cfg = OmegaConf.load(saved)
                    cfg.envs.env_kwargs.seed = seed
                    cfg.eval.output_dir = str(seed_dir)
                if not snapshot.exists():
                    OmegaConf.save(cfg, snapshot)
                manager = EvaluatorManager(
                    cfg, original_cwd=str(repo), output_dir=str(seed_dir)
                )
                # Wrap each evaluator instance only to attach logging context; keep its behavior.
                for env, evaluator in manager.env_evaluators.items():
                    original_episode = evaluator.run_episode

                    def recorded_episode(
                        task, agent, *pos, _original=original_episode, _env=env, **kw
                    ):
                        context.update(
                            seed=seed,  # noqa: B023
                            environment=_env,
                            task=task,  # noqa: B023, RUF100
                            episode=kw.get("episode_idx", 0),
                            step=0,
                        )
                        if args.resume:
                            # The environment cannot restart mid-episode. Preserve the
                            # old partial trajectory before rerunning the same task/seed.
                            index = kw.get("episode_idx", 0)
                            csv_path = (
                                seed_dir / _env / task / f"{task}_run_{index:02d}.csv"  # noqa: B023
                            )  # noqa: B023, RUF100
                            if (
                                csv_path.exists()
                                and not csv_path.with_suffix(".json").exists()
                            ):
                                backup = (
                                    run
                                    / "interrupted_attempts"
                                    / stamp
                                    / csv_path.relative_to(run)
                                )
                                backup.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(csv_path, backup)
                        return _original(task, agent, *pos, **kw)

                    evaluator.run_episode = recorded_episode
                print(
                    f"Running {args.profile}/{args.agent}: {args.model}, seed {seed}; {len(manager.tasks)} pending episodes. Output: {run}",
                    flush=True,
                )
                manager.run(RecordingAgentFactory(cfg))
                summary = collect_and_summarize_results(str(seed_dir))
                print_summary_table(summary)
                (seed_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
            metadata["status"] = "complete"
        except BaseException:
            metadata["status"] = "failed_or_interrupted"
            raise
        finally:
            from result_tables import write_result_tables

            try:
                metadata.update(write_result_tables(run, metadata))
                if metadata["completed_episodes"] != metadata["expected_episodes"]:
                    metadata["status"] = "failed_or_interrupted"
            finally:
                attempt["status"] = metadata["status"]
                if cloud_sdk is not None:
                    cloud_sdk.close()
                manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved prompts, trajectories, configuration and matrix to {run}")


if __name__ == "__main__":
    main()
