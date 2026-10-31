# Exercise 3 results

Complete comparison.

| Model | Agent | Success | Completed/expected | Critic calls | Status |
|---|---|---:|---:|---:|---|
| qwen2.5:1.5b | naive | 28.0% | 25/25 | 0 | complete |
| gemini-3.5-flash-lite | naive | 84.0% | 25/25 | 0 | complete |
| qwen2.5:1.5b | custom | 8.0% | 25/25 | 0 | complete |
| gemini-3.5-flash-lite | custom | 60.0% | 25/25 | 0 | complete |
| qwen2.5:1.5b | reflection | 12.0% | 25/25 | 91 | complete |
| gemini-3.5-flash-lite | reflection | 88.0% | 25/25 | 31 | complete |

Task-level results, token counts and accepted critiques are in comparison.csv. These totals cover completed episodes only. No live pilot was run for Exercise 3.

The naive and planning-memory baselines are historical student experiments. The reflection actor uses the naive prompt/parser plus saved advice. Critic calls add computation. The comparison to Exercise 2 also changes the response format and memory mechanism.

Five seeds per task are not repeated samples of an identical instance. Earlier results informed the design, so this is exploratory rather than held-out evaluation. No repeated-sampling stability study or bonus philosophical answer is included.
