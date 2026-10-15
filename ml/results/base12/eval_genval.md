# Teach-back grader evaluation

Test set: 68 answers, 130 facts, 5 cases, 5 languages. Gold verdicts: {'incorrect': 23, 'correct': 84, 'partial': 15, 'missing': 8}. Generated 2026-10-15 17:43:03 IST.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| e4b_base | 93.1% (91.1%–95.2%) | 0.872 | 0.0% (0.0%–0.0%) | 0.0% (0/26) | 1.2% | 100.0% |
| e4b_lora | 96.9% (92.8%–100.0%) | 0.942 | 0.0% (0.0%–0.0%) | 0.0% (0/26) | 0.0% | 98.5% |
| gemma4_12b_base | 86.9% (72.9%–96.1%) | 0.912 | 0.0% (0.0%–0.0%) | 0.0% (0/26) | 13.1% | 97.1% |

## Accuracy by language

| System | ar | en | it | pt | ta |
|---|---|---|---|---|---|
| e4b_base | 92% | 89% | 97% | 92% | 94% |
| e4b_lora | 100% | 89% | 100% | 92% | 100% |
| gemma4_12b_base | 96% | 95% | 97% | 92% | 62% |

## Speed and tokens

| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |
|---|---|---|---|---|---|---|
| e4b_base | AMD Radeon Pro W7900D | n/a | n/a | 124 (bs16) | 390 | 137 |
| e4b_lora | AMD Radeon Pro W7900D | n/a | n/a | 114 (bs16) | 390 | 88 |
| gemma4_12b_base | AMD Radeon Pro W7900D | 9.09 s | 17.41 s | 14.5 (bs1), 81 (bs16) | 394 | 93 |

### Confusion matrix: e4b_base (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 83 | 1 | 0 | 0 | 0 |
| partial | 2 | 12 | 0 | 1 | 0 |
| incorrect | 0 | 4 | 19 | 0 | 0 |
| missing | 0 | 1 | 0 | 7 | 0 |

### Confusion matrix: e4b_lora (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 84 | 0 | 0 | 0 | 0 |
| partial | 3 | 11 | 0 | 1 | 0 |
| incorrect | 0 | 0 | 23 | 0 | 0 |
| missing | 0 | 0 | 0 | 8 | 0 |

### Confusion matrix: gemma4_12b_base (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 73 | 1 | 0 | 0 | 10 |
| partial | 2 | 12 | 0 | 0 | 1 |
| incorrect | 0 | 2 | 20 | 0 | 1 |
| missing | 0 | 0 | 0 | 8 | 0 |
