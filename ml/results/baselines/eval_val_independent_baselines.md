# Teach-back grader evaluation

Test set: 120 answers, 284 facts, 5 cases, 12 languages. Gold verdicts: {'incorrect': 78, 'correct': 113, 'partial': 58, 'missing': 35}. Generated 2026-10-17 02:49:53 IST.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| homeward_run2 | 91.5% (83.3%–95.7%) | 0.916 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 1.8% | 98.3% |
| gemma4_31b | 95.1% (92.5%–98.1%) | 0.948 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 1.8% | 100.0% |
| e4b_base | 75.4% (66.4%–80.7%) | 0.766 | 1.8% (0.0%–3.0%) | 0.0% (0/74) | 13.3% | 100.0% |
| mimo_v2_6_flash | 92.3% (89.3%–95.2%) | 0.923 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 10.6% | 98.3% |
| qwen3_8_flash_next | 70.1% (64.6%–77.1%) | 0.792 | 0.9% (0.0%–3.2%) | 1.4% (1/74) | 46.9% | 82.5% |
| qwen3_8_27b | 80.6% (76.5%–84.2%) | 0.852 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 31.9% | 91.7% |
| minicpm_v_4_7_35b_a3b | 85.6% (79.3%–89.1%) | 0.852 | 2.7% (0.0%–5.4%) | 4.1% (3/74) | 8.0% | 100.0% |

## Accuracy by language

| System | ar | bn | en | es | fr | hi | it | pt | ta | tl | vi | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| homeward_run2 | 90% | 92% | 83% | 92% | 95% | 88% | 89% | 92% | 91% | 92% | 100% | 96% |
| gemma4_31b | 93% | 96% | 96% | 96% | 100% | 94% | 93% | 96% | 91% | 100% | 100% | 87% |
| e4b_base | 80% | 62% | 75% | 81% | 70% | 75% | 82% | 75% | 68% | 75% | 83% | 74% |
| mimo_v2_6_flash | 87% | 88% | 100% | 96% | 100% | 81% | 86% | 92% | 82% | 100% | 100% | 96% |
| qwen3_8_flash_next | 80% | 67% | 58% | 58% | 65% | 75% | 75% | 83% | 18% | 75% | 96% | 87% |
| qwen3_8_27b | 60% | 88% | 92% | 92% | 100% | 62% | 86% | 83% | 55% | 75% | 83% | 91% |
| minicpm_v_4_7_35b_a3b | 93% | 71% | 83% | 92% | 85% | 75% | 86% | 88% | 73% | 92% | 96% | 87% |

## Speed and tokens

| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |
|---|---|---|---|---|---|---|
| homeward_run2 | AMD Radeon Pro W7900D | n/a | n/a | 173 (bs16) | 371 | 118 |
| gemma4_31b | AMD Token Factory | 633.57 s | 1182.73 s | n/a | 375 | 138 |
| e4b_base | AMD Radeon Pro W7900D | n/a | n/a | 179 (bs16) | 371 | 166 |
| mimo_v2_6_flash | AMD Token Factory | 1333.00 s | 2472.39 s | n/a | 380 | 261 |
| qwen3_8_flash_next | AMD Token Factory | 1479.58 s | 2999.50 s | n/a | 420 | 746 |
| qwen3_8_27b | AMD Token Factory | 1794.49 s | 4164.34 s | n/a | 423 | 961 |
| minicpm_v_4_7_35b_a3b | AMD Token Factory | 1284.47 s | 2352.74 s | n/a | 388 | 169 |

### Confusion matrix: homeward_run2 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 111 | 1 | 1 | 0 | 0 |
| partial | 10 | 48 | 0 | 0 | 0 |
| incorrect | 0 | 12 | 66 | 0 | 0 |
| missing | 0 | 0 | 0 | 35 | 0 |

### Confusion matrix: gemma4_31b (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 111 | 2 | 0 | 0 | 0 |
| partial | 7 | 50 | 1 | 0 | 0 |
| incorrect | 0 | 2 | 75 | 1 | 0 |
| missing | 0 | 1 | 0 | 34 | 0 |

### Confusion matrix: e4b_base (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 98 | 11 | 4 | 0 | 0 |
| partial | 17 | 41 | 0 | 0 | 0 |
| incorrect | 2 | 35 | 41 | 0 | 0 |
| missing | 0 | 1 | 0 | 34 | 0 |

### Confusion matrix: mimo_v2_6_flash (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 101 | 10 | 1 | 0 | 1 |
| partial | 2 | 53 | 1 | 1 | 1 |
| incorrect | 0 | 1 | 76 | 1 | 0 |
| missing | 0 | 2 | 0 | 32 | 1 |

### Confusion matrix: qwen3_8_flash_next (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 60 | 24 | 0 | 0 | 29 |
| partial | 0 | 43 | 2 | 0 | 13 |
| incorrect | 1 | 0 | 69 | 0 | 8 |
| missing | 0 | 0 | 1 | 27 | 7 |

### Confusion matrix: qwen3_8_27b (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 77 | 21 | 2 | 0 | 13 |
| partial | 1 | 49 | 2 | 0 | 6 |
| incorrect | 0 | 1 | 74 | 0 | 3 |
| missing | 0 | 1 | 1 | 29 | 4 |

### Confusion matrix: minicpm_v_4_7_35b_a3b (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 104 | 7 | 1 | 1 | 0 |
| partial | 12 | 42 | 1 | 3 | 0 |
| incorrect | 2 | 13 | 63 | 0 | 0 |
| missing | 1 | 0 | 0 | 34 | 0 |
