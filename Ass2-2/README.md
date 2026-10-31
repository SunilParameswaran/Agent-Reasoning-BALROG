# Exercise 2: planning and subgoal memory

This exercise tests whether an explicit plan and current subgoal improve BabyAI
success. Both main evaluations are complete.

## Agent behavior

Before **every action**, one model call produces the following fields:

```text
PLAN: Find the key, pick it up, then open the door.
SUBGOAL: Approach the key.
ACTION: turn left
```

The plan describes the broader strategy; the subgoal describes the immediate
objective. The next prompt includes both saved fields. `KEEP` preserves an
existing field, and memory resets at each new episode. Plans are limited to 24
words and subgoals to 10. Invalid memory updates preserve the previous memory;
an invalid action triggers BALROG's `go forward` fallback. No repair call is made.
This is inference-time memory, not model training. The action parser is BabyAI-specific.

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

**Install the submitted agent into a fresh BALROG checkout before running:**

```bash
cp Ass2-2/agent/custom.py BALROG/balrog/agents/custom.py
```

The runner uses the file inside BALROG. `reference/custom_original.py` preserves
the original supplied implementation.

## Run or resume

```bash
python Ass2-2/run_exercise2.py --agent custom --profile baseline
python Ass2-2/run_exercise2.py --provider gemini --agent custom --profile baseline
python Ass2-2/run_exercise2.py --resume Ass2-2/results/YOUR_RUN_FOLDER
```

Replace `YOUR_RUN_FOLDER` with the existing experiment folder. `--agent` selects
the agent; `--profile baseline` selects the full evaluation settings. Without
that profile, the runner performs a short diagnostic. `--agent naive` is also supported.

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

| Model | Naive baseline (Exercise 1) | Planning memory (Exercise 2) |
|---|---:|---:|
| Qwen2.5 1.5B | 28% (7/25) | 8% (2/25) |
| Gemini 3.5 Flash-Lite | 84% (21/25) | 60% (15/25) |

Planning memory reduced success for both models. The treatment changes the
prompt, memory and parser together, so the effect cannot be attributed to memory
alone. Short diagnostic runs are excluded. `comparison.csv` includes task-level
results; `REPORT_RESULTS.md` provides the overall summary. The baselines are the
student's completed Exercise 1 runs.

## Code and output files

| File | Purpose |
|---|---|
| `agent/custom.py` | Submission copy of the planning agent and its exact instructions. |
| `run_exercise2.py`, `gemini_client.py` | Evaluation runner and cloud transport. |
| `result_tables.py`, `compare_results.py` | Per-run tables and comparisons across supplied run folders. |
| `results/<run>/prompts.jsonl` | Exact model inputs and raw responses, including interrupted attempts. |
| `results/<run>/decisions.jsonl` | Memory before/after, parsed action, format issues and fallback. |
| `results/<run>/seed_*/…/*_run_*.csv` | Candidate actions, post-action observations and structured decision records. |
| `results/<run>/seed_*/…/*_run_*.json` | Completed episode outcome, action counts and token usage. |
| `episodes.csv`, `matrix.csv` within each run | Episode scores and overall success; missing episodes are labelled partial. |
| `manifest.json`, `seed_*/config.yaml`, `packages.txt`, `code_snapshot/` within each run | Settings, versions and source used for that experiment. |

The trajectory CSV records the **candidate**, not necessarily the executed
fallback action. Use the decision log and observation feedback to identify
fallbacks. Its `Reasoning` field stores the harness's decision record, not hidden
model reasoning. Five seeds per task do not establish repeated-sampling stability.

## Validate and package

```bash
python -m pytest Ass2-2/test_agent.py Ass2-2/test_runner.py Ass2-2/test_comparison.py Ass2-2/test_package.py -q
python Ass2-2/package_submission.py
```

