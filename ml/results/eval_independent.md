# Teach-back grader evaluation

Test set: 150 answers, 341 facts, 7 cases, 12 languages. Gold verdicts: {'correct': 138, 'incorrect': 96, 'partial': 65, 'missing': 42}. Generated 2026-10-17 14:25:46 IST.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| e4b_base | 83.0% (80.6%–85.8%) | 0.832 | 4.3% (1.5%–7.2%) | 4.6% (5/108) | 10.9% | 100.0% |
| run1 | 88.3% (85.4%–91.1%) | 0.878 | 5.8% (2.9%–8.6%) | 6.5% (7/108) | 6.5% | 100.0% |
| homeward_run2 | 94.1% (91.4%–96.7%) | 0.940 | 1.4% (0.0%–3.2%) | 1.9% (2/108) | 2.9% | 100.0% |
| homeward_run4 | 94.7% (91.9%–97.5%) | 0.946 | 1.4% (0.0%–3.1%) | 0.9% (1/108) | 2.9% | 100.0% |
| gemma4_31b | 97.1% (94.9%–99.1%) | 0.971 | 0.7% (0.0%–2.1%) | 0.9% (1/108) | 2.9% | 100.0% |
| deepseek_v4_flash | 80.4% (73.4%–86.3%) | 0.825 | 2.9% (0.0%–6.8%) | 2.8% (3/108) | 14.5% | 90.7% |
| qwen3_8_flash_next | 91.2% (89.7%–92.8%) | 0.913 | 0.7% (0.0%–2.1%) | 0.9% (1/108) | 8.7% | 99.3% |
| minicpm_v_4_7_35b_a3b | 85.9% (79.4%–90.7%) | 0.852 | 2.2% (0.0%–6.1%) | 2.8% (3/108) | 13.8% | 100.0% |
| clef | 88.9% (86.1%–92.7%) | 0.897 | 0.7% (0.0%–2.0%) | 0.9% (1/108) | 13.0% | 100.0% |

## Accuracy by language

| System | ar | bn | en | es | fr | hi | it | pt | ta | tl | vi | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b_base | 77% | 80% | 71% | 89% | 90% | 80% | 89% | 94% | 76% | 85% | 81% | 83% |
| run1 | 84% | 84% | 82% | 96% | 90% | 90% | 89% | 87% | 76% | 96% | 92% | 93% |
| homeward_run2 | 87% | 84% | 89% | 100% | 97% | 100% | 96% | 94% | 93% | 92% | 96% | 100% |
| homeward_run4 | 94% | 92% | 96% | 96% | 97% | 93% | 93% | 94% | 97% | 92% | 96% | 97% |
| gemma4_31b | 100% | 96% | 100% | 100% | 93% | 97% | 96% | 100% | 93% | 100% | 92% | 97% |
| deepseek_v4_flash | 68% | 68% | 75% | 81% | 80% | 87% | 96% | 87% | 62% | 88% | 92% | 80% |
| qwen3_8_flash_next | 87% | 80% | 96% | 96% | 93% | 90% | 96% | 84% | 83% | 100% | 96% | 93% |
| minicpm_v_4_7_35b_a3b | 87% | 80% | 86% | 93% | 83% | 90% | 93% | 81% | 72% | 96% | 85% | 87% |
| clef | 94% | 88% | 96% | 96% | 87% | 93% | 89% | 90% | 86% | 85% | 81% | 80% |

## Speed and tokens

| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |
|---|---|---|---|---|---|---|
| e4b_base | AMD Radeon Pro W7900D | 7.44 s | 10.80 s | 20.1 (bs1), 177 (bs16) | 369 | 164 |
| run1 | AMD Radeon Pro W7900D | 4.89 s | 8.27 s | 18.7 (bs1), 169 (bs16) | 369 | 108 |
| homeward_run2 | AMD Radeon Pro W7900D | 5.15 s | 7.84 s | 19.1 (bs1), 169 (bs16) | 369 | 116 |
| homeward_run4 | AMD Radeon Pro W7900D | 4.89 s | 7.62 s | 19.8 (bs1), 167 (bs16) | 369 | 114 |
| gemma4_31b | AMD Token Factory | 185.65 s | 614.09 s | n/a | 373 | 133 |
| deepseek_v4_flash | AMD Token Factory | 285.47 s | 642.42 s | n/a | 359 | 128 |
| qwen3_8_flash_next | AMD Token Factory | 1325.47 s | 2387.86 s | n/a | 386 | 171 |
| minicpm_v_4_7_35b_a3b | AMD Token Factory | 1283.95 s | 3116.19 s | n/a | 386 | 171 |
| clef | AMD Token Factory | 2.20 s | 3.67 s | n/a | 679 | 0 |

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

### Confusion matrix: homeward_run4 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 134 | 4 | 0 | 0 | 0 |
| partial | 1 | 63 | 0 | 1 | 0 |
| incorrect | 2 | 10 | 84 | 0 | 0 |
| missing | 0 | 0 | 0 | 42 | 0 |

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
| correct | 118 | 11 | 1 | 1 | 7 |
| partial | 3 | 52 | 3 | 1 | 6 |
| incorrect | 2 | 25 | 65 | 0 | 4 |
| missing | 2 | 1 | 0 | 39 | 0 |

### Confusion matrix: qwen3_8_flash_next (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 126 | 9 | 3 | 0 | 0 |
| partial | 0 | 55 | 9 | 1 | 0 |
| incorrect | 1 | 5 | 89 | 0 | 1 |
| missing | 0 | 0 | 0 | 41 | 1 |

### Confusion matrix: minicpm_v_4_7_35b_a3b (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 119 | 14 | 2 | 3 | 0 |
| partial | 5 | 50 | 5 | 5 | 0 |
| incorrect | 3 | 10 | 83 | 0 | 0 |
| missing | 0 | 0 | 1 | 41 | 0 |

### Confusion matrix: clef (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 120 | 18 | 0 | 0 | 0 |
| partial | 2 | 63 | 0 | 0 | 0 |
| incorrect | 1 | 17 | 78 | 0 | 0 |
| missing | 0 | 0 | 0 | 42 | 0 |
