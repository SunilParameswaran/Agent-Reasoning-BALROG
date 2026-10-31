# Exercise 2 results

Completed comparison.

| Model | Agent | Success | Episodes completed/expected | Status |
|---|---|---:|---:|---|
| qwen2.5:1.5b | naive | 28.0% | 25/25 | complete |
| gemini-3.5-flash-lite | naive | 84.0% | 25/25 | complete |
| qwen2.5:1.5b | custom | 8.0% | 25/25 | complete |
| gemini-3.5-flash-lite | custom | 60.0% | 25/25 | complete |

The full task-level matrix is in `comparison.csv`. Success is mission completion within the action limit.
The baseline evidence comes from the student's Exercise 1 experiments, not the teaching assistant's example.

- qwen2.5:1.5b: custom minus naive = -20.0 percentage points; 549 invalid action candidates in 2483 custom-agent actions versus 9 in 2203 naive-agent actions.
- gemini-3.5-flash-lite: custom minus naive = -24.0 percentage points; 64 invalid action candidates in 1409 custom-agent actions versus 1 in 921 naive-agent actions.

There are five seeds per task and one model trajectory per task/seed/agent combination. No repeated-sampling brittleness experiment has been collected. Pilots use separate seeds and are excluded here.

The custom treatment combines a planning prompt, persistent plan/subgoal memory and a new parser. It does not isolate the effect of memory. The earlier baseline and current cloud run may also differ in provider-side model version. Do not infer statistical significance or general reasoning ability from these small samples.
