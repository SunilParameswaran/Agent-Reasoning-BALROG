# Setup and reproducibility

Run commands from the root of this repository. Viewing `index.html`, `results.html` or the saved CSV requires no Python environment, API key or running model server.

## 1. Clone the project and create an environment

Prerequisites: Git and Conda. The recorded experiments used Python 3.10.21.

```bash
git clone https://github.com/SunilParameswaran/Agentic_LLM_Assignment_2.git
cd Agentic_LLM_Assignment_2
conda create -n balrog python=3.10 -y
conda activate balrog
python --version
```

## 2. Install the recorded BALROG revision

```bash
git clone https://github.com/balrog-ai/BALROG.git
git -C BALROG checkout b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5
python -m pip install -e ./BALROG

# Exercise 2 imports the planning agent from inside BALROG.
cp Ass2-2/agent/custom.py BALROG/balrog/agents/custom.py
```

The upstream installation includes dependencies for more environments than this project evaluates. Follow [`BALROG/docs/installation.md`](https://github.com/balrog-ai/BALROG/blob/b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5/docs/installation.md) in your new checkout for platform-specific requirements. Its `balrog-post-install` command downloads additional MiniHack/TextWorld resources; those downloads are not needed for the BabyAI runs shown here. `Ass2-2/reference/custom_original.py` preserves the original custom-agent implementation. Exercise 3 registers its reflection agent locally and needs no additional BALROG file replacement.

Recorded package inventories are in each run's `packages.txt`. The BALROG revision is pinned, but `pip install -e ./BALROG` does not lock every transitive dependency. A fresh installation can therefore differ from the original environment. Use those inventories and `code_snapshot/` directories when investigating reproducibility differences.

## 3. Choose a model provider

### Local: Qwen through Ollama

Install [Ollama](https://ollama.com/) and start its service. If the desktop app is not already serving, run `ollama serve` in a separate terminal. Then:

```bash
ollama pull qwen2.5:1.5b
python Ass2-1/run_local.py --provider ollama --list-models
```

The runner defaults to `qwen2.5:1.5b` and the local OpenAI-compatible endpoint at `http://localhost:11434/v1`. `--list-models` checks availability without running an episode.

### Cloud: Gemini

The runners read `GEMINI_API_KEY` from the current shell. This portable prompt hides the key while you type:

```bash
export GEMINI_API_KEY="$(python -c 'import getpass; print(getpass.getpass("Gemini API key: "))')"
python Ass2-1/run_local.py --provider gemini --list-models
```

They do not automatically read `BALROG/SECRETS`. The historical model ID is `gemini-3.5-flash-lite`; it is also the runner default. Confirm it is available to your account before using it. If you choose another listed model via `--model`, its results form a new comparison, not a rerun with the historical model.

## 4. Run the full evaluation

Each command evaluates 25 episodes: five tasks × seeds 42–46. Run commands sequentially, particularly on a rate-limited cloud account.

```bash
# Local Qwen
python Ass2-1/run_local.py --provider ollama --profile baseline --seeds 42 43 44 45 46
python Ass2-2/run_exercise2.py --provider ollama --agent custom --profile baseline --seeds 42 43 44 45 46
python Ass2-3/run_exercise3.py --provider ollama --agent reflection --profile baseline --seeds 42 43 44 45 46

# Cloud Gemini: uses the model configured in the runner unless --model is supplied.
python Ass2-1/run_local.py --provider gemini --profile baseline --seeds 42 43 44 45 46
python Ass2-2/run_exercise2.py --provider gemini --agent custom --profile baseline --seeds 42 43 44 45 46
python Ass2-3/run_exercise3.py --provider gemini --agent reflection --profile baseline --seeds 42 43 44 45 46
```

`--profile baseline` selects the full experiment settings even for custom and reflection agents. Without it, the default is a short diagnostic run. See each runner's `--help` for options. These commands make model calls; the comparison script below does not.

Recorded settings: temperature 1.0; 16 text observations of history; no images; one worker; 64 actions for `goto`/`pickup` and 128 for the other tasks. Local output limit: 128 tokens. Gemini output limit: 1024 tokens, including thinking, with minimal reasoning effort. Critic calls add computation without consuming environment-action steps.

## 5. Resume and inspect results

New evaluations create timestamped directories under the relevant `results/` folder. Copy the run directory printed by the runner and substitute it below:

```bash
python Ass2-1/run_local.py --resume Ass2-1/results/YOUR_RUN_FOLDER
python Ass2-2/run_exercise2.py --resume Ass2-2/results/YOUR_RUN_FOLDER
python Ass2-3/run_exercise3.py --resume Ass2-3/results/YOUR_RUN_FOLDER
```

Resume skips completed episodes and restarts an interrupted episode from its beginning. It restores saved settings; do not combine it with new agent/model/profile arguments. Exercises 2 and 3 also check source hashes. A persistent Gemini quota error stops the run while preserving completed episodes. The default cloud delay is six seconds after a response; increasing `--request-delay` may help per-minute limits, but does not replenish daily quota.

```bash
# Recompute and verify the historical comparison; standard library only.
python scripts/summarize_results.py

# Status of the runs selected in Ass2-3/experiments.json; no model calls.
python Ass2-3/status.py

# Compare your own runs: supply their actual directories.
python Ass2-3/compare_results.py PATH_TO_RUN_1 PATH_TO_RUN_2 --output generated/my_comparison.csv
```

`scripts/summarize_results.py` uses the historical run selection, not the latest folders. It checks the generated table against `Ass2-3/comparison.csv` and writes a separate output. Set up a new run selection deliberately when comparing new experiments. Environment seeds fix task instances, not stochastic model outputs.

## 6. Run validation without model calls

Run each exercise's suite separately; some test modules share names.

```bash
python -m pytest Ass2-1/test_cloud_runner.py -q
python -m pytest Ass2-2/test_agent.py Ass2-2/test_runner.py Ass2-2/test_comparison.py Ass2-2/test_package.py -q
python -m pytest Ass2-3/test_reflection.py Ass2-3/test_runner.py Ass2-3/test_artifacts.py -q
```

These use fake clients to check parsing, memory reset, reflection triggers, resume behavior and artifact handling. They do not measure live model quality.

Repository preparation checks on 31 October 2026: **54 tests passed** (8 baseline, 23 planning, 23 reflection) in the existing `balrog` environment; all 36 published comparison rows matched the saved episode records; the viewer builder verified all 150 episodes and 10,279 actions. A fresh dependency installation on another machine was not tested.

## 7. Rebuild the static viewer

With BALROG and its environment dependencies installed:

```bash
python run_viewer/build_viewer.py
```

The builder replays recorded actions and verifies observations, rewards and outcomes against the saved logs. It does not call an LLM. It writes `index.html`, `results.html`, `comparison.csv` and copies under `run_viewer/dist/`. The first build creates `run_viewer/verified_data.json`; subsequent presentation-only edits can use:

```bash
python run_viewer/build_viewer.py --render-only
```

Open `index.html` in a modern browser. Keep `results.html` and `comparison.csv` beside it so the results links work. Updating local files does not automatically update the separately hosted Sites demo.
