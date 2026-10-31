# Exercise 3: bounded self-reflection

This exercise tests whether occasional review of recent behavior improves BabyAI
success. Both main evaluations are complete; no live pilot was run for this exercise.

## Agent behavior

The same model has two roles: the **actor** outputs one action, while a separate
**critic call** reviews the mission, previous advice and the last eight observed
transitions. Accepted advice replaces previous advice and is passed to the actor.

Reflection happens before the next action when:

- Four consecutive actions left the observations unchanged; or
- A transition cycle of length 2–4 repeated twice; or
- Another multiple of 16 actions has been executed (scheduled review).

There must be at least **8 actions between reviews**, including before the first
review, with at most **4 reviews per episode**. A scheduled review can therefore
be skipped during the cooldown. The critic's advice is limited to 60 words;
empty, overlong, truncated or bare-action responses are rejected. All advice and
counters reset between episodes. Accepted advice is not guaranteed to be correct.

This is an own-design reflection approach, not model training or a reproduction
of ReAct/Reflexion. It uses public text observations, not hidden environment state.

## Setup

Keep `BALROG/` and this exercise folder beside each other. Commands below run
from their parent directory (Assignment 2). The experiments used Python 3.10.21.
For a fresh setup, install Conda and Git, then follow the checkout's installation
instructions after selecting the recorded revision:

```bash
conda create -n balrog python=3.10 -y
conda activate balrog
git clone https://github.com/balrog-ai/BALROG.git
git -C BALROG checkout b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5
python -m pip install -e ./BALROG
```

See `BALROG/docs/installation.md` for additional installation steps; each run's
`packages.txt` records the versions used. If already installed, just activate
`balrog`. For local evaluation, install/start Ollama and download the model with
`ollama pull qwen2.5:1.5b`. The server should be available at `localhost:11434`.
For Gemini, set the key in the same terminal (macOS zsh):

```zsh
read -s "GEMINI_API_KEY?Paste your Gemini key: "
export GEMINI_API_KEY
```

The runners read this environment variable; they do not automatically read
`BALROG/SECRETS`. Credentials are not required to inspect saved results.

No BALROG agent file needs replacing. This exercise's runner registers the new
agent locally; upstream `eval.py` does not recognize `agent.type=reflection`.

## Run, inspect or resume

```bash
python Ass2-3/run_exercise3.py --agent reflection --profile baseline
python Ass2-3/run_exercise3.py --provider gemini --agent reflection --profile baseline
python Ass2-3/status.py
python Ass2-3/run_exercise3.py --resume Ass2-3/results/YOUR_RUN_FOLDER
```

Replace `YOUR_RUN_FOLDER` with the existing experiment folder. The status command
reads saved files without API calls. `--agent naive` and `--agent no_reflection`
are available as controls; the latter disables critic calls and advice. The main
comparison reuses existing baselines rather than starting new cloud control runs.

Completed episodes are skipped on resume; an interrupted episode restarts from
its beginning. Missing episodes are excluded from scores and labelled partial.
A new run without `--resume` creates a separate experiment. Resume restores saved
settings; Exercises 2/3 also verify source hashes. Environment seeds do not make
model outputs deterministic.

## Experiment settings

- **Benchmark:** BabyAI, with `goto`, `pickup`, `open`, `putnext` and `pick_up_seq_go_to`.
- **Seeds:** 42–46; one episode per task/seed gives **25 episodes per model**.
- **Action limits:** 64 for goto/pickup; 128 for the other tasks.
- **Shared settings:** temperature 1.0, 16 text observations, no images, one worker.
- **Models:** local `qwen2.5:1.5b` (128 output tokens); cloud
  `gemini-3.5-flash-lite` (1024 tokens, including thinking; minimal reasoning effort).
- **Cloud pacing:** six seconds after each response; persistent quota errors stop the run.

Success is complete mission achievement within the action limit: progression 1
for success, 0 for failure. Success percentage is the fraction of successful
episodes multiplied by 100. These are five task types within one benchmark family.

The actor uses the naive agent's single-action prompt/parser plus saved advice.
Critic calls use the same model settings and add tokens/time, but do not consume
game-action steps. Interrupted attempts can incur additional usage.

## Final results

| Model | Naive (Exercise 1) | Planning (Exercise 2) | Reflection (Exercise 3) |
|---|---:|---:|---:|
| Qwen2.5 1.5B | 28% (7/25) | 8% (2/25) | 12% (3/25) |
| Gemini 3.5 Flash-Lite | 84% (21/25) | 60% (15/25) | 88% (22/25) |

Gemini gained one success over its naive baseline; Qwen remained below its naive
baseline. Reflection added computation, so this is not an equal-compute comparison.
Earlier results informed the design, and five seeds per task do not establish
statistical significance or repeated-sampling stability. See `comparison.csv`
for task-level results and critic usage, and `REPORT_RESULTS.md` for the summary.

## Code and output files

| File | Purpose |
|---|---|
| `agent/reflection.py` | Actor/critic prompts, reflection triggers, advice validation and memory reset. |
| `run_exercise3.py`, `gemini_client.py` | BALROG integration, logging, resume and cloud transport. |
| `result_tables.py`, `compare_results.py` | Per-run and cross-experiment tables. |
| `experiments.json` | Selects the six historical/current runs used for comparison and packaging. |
| `results/<run>/prompts.jsonl` | Exact inputs/responses labelled actor or critic, action index and attempt. |
| `results/<run>/reflections.jsonl` | Review trigger, raw advice, acceptance, old/new advice and tokens. |
| `results/<run>/decisions.jsonl` | Actor decision, advice used, candidate action and fallback-aware executed action. |
| `results/<run>/seed_*/…/*_run_*.csv` | Candidate actions and post-action observations/rewards. |
| `results/<run>/seed_*/…/*_run_*.json` | Outcome and total tokens, plus actor/critic usage in `reflection_stats`. |
| `episodes.csv`, `matrix.csv` within each run | Completed-episode and overall performance tables. |
| `manifest.json`, `seed_*/config.yaml`, `packages.txt`, `code_snapshot/` within each run | Experiment settings, status, versions and source snapshots. |

Actor and critic calls for the same decision share an action index. Raw logs
include interrupted attempts; comparison totals cover completed episodes only.
Top-level episode token totals include both roles. To assess a critique, inspect
the actions and observations after it; acceptance alone does not show benefit.

## Validate and package

```bash
python -m pytest Ass2-3/test_reflection.py Ass2-3/test_runner.py Ass2-3/test_artifacts.py -q
python Ass2-3/package_submission.py
```

