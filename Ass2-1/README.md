# Exercise 1: baseline evaluation

This exercise evaluates BALROG's naive agent with a local and a cloud LLM.
The agent receives the mission, current observation and recent history, then
returns one action per model call. Both main evaluations are complete.

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

## Run or resume

```bash
python Ass2-1/run_local.py --profile baseline --seeds 42 43 44 45 46
python Ass2-1/run_local.py --provider gemini --profile baseline --seeds 42 43 44 45 46
python Ass2-1/run_local.py --resume Ass2-1/results/YOUR_RUN_FOLDER
```

Replace `YOUR_RUN_FOLDER` with the existing experiment folder. Without
`--profile baseline`, the default is a short diagnostic run, not the full evaluation.

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

## Final results

| Model | Successful episodes | Success rate |
|---|---:|---:|
| Qwen2.5 1.5B | 7/25 | 28% |
| Gemini 3.5 Flash-Lite | 21/25 | 84% |

This is a focused BabyAI evaluation, not a reproduction of the complete BALROG paper.

## Code and output files

| File | Purpose |
|---|---|
| `run_local.py` | Runs the naive agent through BALROG and records model calls. |
| `gemini_client.py` | Connects the runner to Gemini. |
| `result_tables.py` | Helper called automatically to create episode and overall tables. |
| `results/<run>/matrix.csv` | Overall success, standard error and completed/expected counts. |
| `results/<run>/episodes.csv` | One row per completed episode, including task, seed and steps. |
| `results/<run>/prompts.jsonl` | One JSON record per returned model response, with its exact input. |
| `results/<run>/seed_*/…/*_run_*.csv` | Candidate actions and the observations/rewards after them. |
| `results/<run>/seed_*/…/*_run_*.json` | Final episode outcome, action counts and token usage. |
| `manifest.json`, `seed_*/config.yaml`, `packages.txt` within each run | Settings, completion status and reproducibility information. |
| `eval.log` within each run | Execution and diagnostic messages. |

Start with `matrix.csv`, select an episode from `episodes.csv`, then inspect its
trajectory and prompts. CSV observations occur **after** the action in that row;
`prompts.jsonl` shows what the model saw **before** acting. Invalid candidates
trigger BALROG's `go forward` fallback. A `complete` run means all episodes were
evaluated, not that all missions succeeded. Raw prompt logs include interrupted
attempts; scores use completed episodes only.


