# Teach-back grader evaluation

Test set: 150 answers, 341 facts, 7 cases, 12 languages. Gold verdicts: {'correct': 138, 'incorrect': 96, 'partial': 65, 'missing': 42}. Generated 2026-10-17 14:25:50 IST.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| homeward_run2 | 94.1% (91.4%–96.7%) | 0.940 | 1.4% (0.0%–3.2%) | 1.9% (2/108) | 2.9% | 100.0% |
| homeward_run4 | 94.7% (91.9%–97.5%) | 0.946 | 1.4% (0.0%–3.1%) | 0.9% (1/108) | 2.9% | 100.0% |
| e4b_base | 83.0% (80.6%–85.8%) | 0.832 | 4.3% (1.5%–7.2%) | 4.6% (5/108) | 10.9% | 100.0% |
| gemma4_31b | 97.1% (94.9%–99.1%) | 0.971 | 0.7% (0.0%–2.1%) | 0.9% (1/108) | 2.9% | 100.0% |

## Accuracy by language

| System | ar | bn | en | es | fr | hi | it | pt | ta | tl | vi | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| homeward_run2 | 87% | 84% | 89% | 100% | 97% | 100% | 96% | 94% | 93% | 92% | 96% | 100% |
| homeward_run4 | 94% | 92% | 96% | 96% | 97% | 93% | 93% | 94% | 97% | 92% | 96% | 97% |
| e4b_base | 77% | 80% | 71% | 89% | 90% | 80% | 89% | 94% | 76% | 85% | 81% | 83% |
| gemma4_31b | 100% | 96% | 100% | 100% | 93% | 97% | 96% | 100% | 93% | 100% | 92% | 97% |

## Speed and tokens

| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |
|---|---|---|---|---|---|---|
| homeward_run2 | AMD Radeon Pro W7900D | 5.15 s | 7.84 s | 19.1 (bs1), 169 (bs16) | 369 | 116 |
| homeward_run4 | AMD Radeon Pro W7900D | 4.89 s | 7.62 s | 19.8 (bs1), 167 (bs16) | 369 | 114 |
| e4b_base | AMD Radeon Pro W7900D | 7.44 s | 10.80 s | 20.1 (bs1), 177 (bs16) | 369 | 164 |
| gemma4_31b | AMD Token Factory | 185.65 s | 614.09 s | n/a | 373 | 133 |

### Confusion matrix: homeward_run2 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 134 | 4 | 0 | 0 | 0 |
| partial | 2 | 62 | 1 | 0 | 0 |
| incorrect | 2 | 10 | 84 | 0 | 0 |
| missing | 0 | 1 | 0 | 41 | 0 |

### Confusion matrix: homeward_run4 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 134 | 4 | 0 | 0 | 0 |
| partial | 1 | 63 | 0 | 1 | 0 |
| incorrect | 2 | 10 | 84 | 0 | 0 |
| missing | 0 | 0 | 0 | 42 | 0 |

### Confusion matrix: e4b_base (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 123 | 15 | 0 | 0 | 0 |
| partial | 5 | 57 | 0 | 3 | 0 |
| incorrect | 6 | 28 | 62 | 0 | 0 |
| missing | 0 | 1 | 0 | 41 | 0 |

### Confusion matrix: gemma4_31b (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 134 | 4 | 0 | 0 | 0 |
| partial | 0 | 63 | 2 | 0 | 0 |
| incorrect | 1 | 3 | 92 | 0 | 0 |
| missing | 0 | 0 | 0 | 42 | 0 |
