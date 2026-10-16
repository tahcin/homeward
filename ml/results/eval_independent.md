# Teach-back grader evaluation

Test set: 150 answers, 341 facts, 7 cases, 12 languages. Gold verdicts: {'correct': 138, 'incorrect': 96, 'partial': 65, 'missing': 42}. Generated 2026-10-16 13:55:53 IST.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| e4b_base | 83.0% (80.6%–85.8%) | 0.832 | 4.3% (1.5%–7.2%) | 4.6% (5/108) | 10.9% | 100.0% |
| run1 | 88.3% (85.4%–91.1%) | 0.878 | 5.8% (2.9%–8.6%) | 6.5% (7/108) | 6.5% | 100.0% |
| homeward_run2 | 94.1% (91.4%–96.7%) | 0.940 | 1.4% (0.0%–3.2%) | 1.9% (2/108) | 2.9% | 100.0% |
| gemma4_31b | 97.1% (94.9%–99.1%) | 0.971 | 0.7% (0.0%–2.1%) | 0.9% (1/108) | 2.9% | 100.0% |
| deepseek_v4_flash | 79.2% (72.3%–86.2%) | 0.817 | 2.9% (0.0%–6.8%) | 2.8% (3/108) | 14.5% | 90.7% |

## Accuracy by language

| System | ar | bn | en | es | fr | hi | it | pt | ta | tl | vi | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b_base | 77% | 80% | 71% | 89% | 90% | 80% | 89% | 94% | 76% | 85% | 81% | 83% |
| run1 | 84% | 84% | 82% | 96% | 90% | 90% | 89% | 87% | 76% | 96% | 92% | 93% |
| homeward_run2 | 87% | 84% | 89% | 100% | 97% | 100% | 96% | 94% | 93% | 92% | 96% | 100% |
| gemma4_31b | 100% | 96% | 100% | 100% | 93% | 97% | 96% | 100% | 93% | 100% | 92% | 97% |
| deepseek_v4_flash | 68% | 68% | 61% | 81% | 80% | 87% | 96% | 87% | 62% | 88% | 92% | 80% |

## Speed and tokens

| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |
|---|---|---|---|---|---|---|
| e4b_base | AMD Radeon Pro W7900D | 7.44 s | 10.80 s | 20.1 (bs1), 177 (bs16) | 369 | 164 |
| run1 | AMD Radeon Pro W7900D | 4.89 s | 8.27 s | 18.7 (bs1), 169 (bs16) | 369 | 108 |
| homeward_run2 | AMD Radeon Pro W7900D | 5.15 s | 7.84 s | 19.1 (bs1), 169 (bs16) | 369 | 116 |
| gemma4_31b | AMD Token Factory | 185.65 s | 614.09 s | n/a | 373 | 133 |
| deepseek_v4_flash | AMD Token Factory | 285.47 s | 642.42 s | n/a | 359 | 128 |

### Confusion matrix: e4b_base (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 123 | 15 | 0 | 0 | 0 |
| partial | 5 | 57 | 0 | 3 | 0 |
| incorrect | 6 | 28 | 62 | 0 | 0 |
| missing | 0 | 1 | 0 | 41 | 0 |

### Confusion matrix: run1 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 129 | 7 | 2 | 0 | 0 |
| partial | 8 | 50 | 2 | 5 | 0 |
| incorrect | 8 | 8 | 80 | 0 | 0 |
| missing | 0 | 0 | 0 | 42 | 0 |

### Confusion matrix: homeward_run2 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 134 | 4 | 0 | 0 | 0 |
| partial | 2 | 62 | 1 | 0 | 0 |
| incorrect | 2 | 10 | 84 | 0 | 0 |
| missing | 0 | 1 | 0 | 41 | 0 |

### Confusion matrix: gemma4_31b (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 134 | 4 | 0 | 0 | 0 |
| partial | 0 | 63 | 2 | 0 | 0 |
| incorrect | 1 | 3 | 92 | 0 | 0 |
| missing | 0 | 0 | 0 | 42 | 0 |

### Confusion matrix: deepseek_v4_flash (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 118 | 10 | 1 | 1 | 8 |
| partial | 3 | 50 | 3 | 1 | 8 |
| incorrect | 2 | 25 | 64 | 0 | 5 |
| missing | 2 | 1 | 0 | 38 | 1 |
