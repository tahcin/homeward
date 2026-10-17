# Teach-back grader evaluation

Test set: 348 answers, 744 facts, 18 cases, 12 languages. Gold verdicts: {'correct': 382, 'incorrect': 176, 'missing': 63, 'partial': 123}. Generated 2026-10-17 14:26:10 IST.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| e4b_base | 82.1% (79.1%–84.9%) | 0.763 | 1.7% (0.4%–3.1%) | 1.2% (2/173) | 5.0% | 100.0% |
| run1 | 87.5% (85.2%–89.7%) | 0.829 | 1.7% (0.4%–3.1%) | 1.2% (2/173) | 3.7% | 100.0% |
| homeward_run2 | 87.0% (84.7%–89.4%) | 0.814 | 2.1% (0.4%–4.1%) | 1.7% (3/173) | 2.9% | 100.0% |
| homeward_run4 | 86.0% (83.3%–88.7%) | 0.814 | 2.1% (0.4%–4.1%) | 1.7% (3/173) | 6.5% | 100.0% |
| gemma4_31b | 85.3% (82.0%–88.2%) | 0.801 | 2.5% (0.8%–4.5%) | 1.2% (2/173) | 6.8% | 100.0% |
| deepseek_v4_flash | 71.6% (66.1%–76.7%) | 0.709 | 3.8% (1.3%–6.7%) | 3.5% (6/173) | 23.6% | 84.5% |

## Accuracy by language

| System | ar | bn | en | es | fr | hi | it | pt | ta | tl | vi | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b_base | 80% | 81% | 86% | 88% | 89% | 90% | 71% | 80% | 90% | 80% | 78% | 83% |
| run1 | 88% | 84% | 95% | 91% | 84% | 98% | 79% | 86% | 90% | 86% | 84% | 90% |
| homeward_run2 | 90% | 88% | 93% | 88% | 86% | 95% | 81% | 82% | 92% | 85% | 83% | 90% |
| homeward_run4 | 78% | 88% | 91% | 91% | 80% | 92% | 77% | 82% | 96% | 84% | 85% | 90% |
| gemma4_31b | 82% | 91% | 91% | 91% | 77% | 88% | 67% | 91% | 92% | 85% | 86% | 86% |
| deepseek_v4_flash | 48% | 75% | 86% | 94% | 77% | 65% | 56% | 66% | 75% | 69% | 66% | 82% |

## Speed and tokens

| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |
|---|---|---|---|---|---|---|
| e4b_base | AMD Radeon Pro W7900D | 8.57 s | 11.85 s | 18.8 (bs1), 174 (bs16) | 398 | 157 |
| run1 | AMD Radeon Pro W7900D | 5.37 s | 7.17 s | 19.1 (bs1), 169 (bs16) | 398 | 103 |
| homeward_run2 | AMD Radeon Pro W7900D | 5.67 s | 8.27 s | 18.5 (bs1), 162 (bs16) | 398 | 108 |
| homeward_run4 | AMD Radeon Pro W7900D | 4.96 s | 7.59 s | 19.8 (bs1), 165 (bs16) | 398 | 106 |
| gemma4_31b | AMD Token Factory | 736.11 s | 1393.01 s | n/a | 402 | 128 |
| deepseek_v4_flash | AMD Token Factory | 555.41 s | 1211.82 s | n/a | 366 | 112 |

### Confusion matrix: e4b_base (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 363 | 16 | 3 | 0 | 0 |
| partial | 27 | 90 | 6 | 0 | 0 |
| incorrect | 2 | 51 | 117 | 6 | 0 |
| missing | 2 | 15 | 5 | 41 | 0 |

### Confusion matrix: run1 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 368 | 8 | 6 | 0 | 0 |
| partial | 28 | 80 | 15 | 0 | 0 |
| incorrect | 2 | 13 | 158 | 3 | 0 |
| missing | 2 | 4 | 12 | 45 | 0 |

### Confusion matrix: homeward_run2 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 371 | 6 | 5 | 0 | 0 |
| partial | 31 | 81 | 11 | 0 | 0 |
| incorrect | 3 | 17 | 156 | 0 | 0 |
| missing | 2 | 8 | 14 | 39 | 0 |

### Confusion matrix: homeward_run4 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 357 | 20 | 5 | 0 | 0 |
| partial | 25 | 88 | 10 | 0 | 0 |
| incorrect | 3 | 17 | 153 | 3 | 0 |
| missing | 2 | 8 | 11 | 42 | 0 |

### Confusion matrix: gemma4_31b (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 356 | 19 | 7 | 0 | 0 |
| partial | 28 | 82 | 12 | 1 | 0 |
| incorrect | 3 | 11 | 155 | 7 | 0 |
| missing | 3 | 4 | 14 | 42 | 0 |

### Confusion matrix: deepseek_v4_flash (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 292 | 24 | 14 | 7 | 45 |
| partial | 28 | 71 | 7 | 2 | 15 |
| incorrect | 7 | 14 | 133 | 5 | 17 |
| missing | 2 | 8 | 13 | 37 | 3 |
