# Teach-back grader evaluation

Test set: 120 answers, 284 facts, 5 cases, 12 languages. Gold verdicts: {'incorrect': 78, 'correct': 113, 'partial': 58, 'missing': 35}. Generated 2026-10-15 17:43:02 IST.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| e4b_base | 75.4% (66.4%–80.7%) | 0.766 | 1.8% (0.0%–3.0%) | 0.0% (0/74) | 13.3% | 100.0% |
| e4b_lora | 85.6% (81.7%–89.0%) | 0.857 | 1.8% (0.0%–3.0%) | 0.0% (0/74) | 7.1% | 100.0% |
| gemma4_12b_base | 81.0% (76.1%–85.4%) | 0.845 | 2.7% (1.0%–4.3%) | 2.7% (2/74) | 15.9% | 95.0% |

## Accuracy by language

| System | ar | bn | en | es | fr | hi | it | pt | ta | tl | vi | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b_base | 80% | 62% | 75% | 81% | 70% | 75% | 82% | 75% | 68% | 75% | 83% | 74% |
| e4b_lora | 93% | 79% | 83% | 88% | 85% | 75% | 86% | 88% | 73% | 83% | 96% | 91% |
| gemma4_12b_base | 83% | 83% | 83% | 85% | 95% | 81% | 61% | 88% | 82% | 88% | 70% | 78% |

## Speed and tokens

| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |
|---|---|---|---|---|---|---|
| e4b_base | AMD Radeon Pro W7900D | n/a | n/a | 179 (bs16) | 371 | 166 |
| e4b_lora | AMD Radeon Pro W7900D | n/a | n/a | 169 (bs16) | 371 | 110 |
| gemma4_12b_base | AMD Radeon Pro W7900D | 9.09 s | 17.41 s | 14.5 (bs1), 81 (bs16) | 375 | 116 |

### Confusion matrix: e4b_base (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 98 | 11 | 4 | 0 | 0 |
| partial | 17 | 41 | 0 | 0 | 0 |
| incorrect | 2 | 35 | 41 | 0 | 0 |
| missing | 0 | 1 | 0 | 34 | 0 |

### Confusion matrix: e4b_lora (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 105 | 5 | 3 | 0 | 0 |
| partial | 19 | 37 | 1 | 1 | 0 |
| incorrect | 2 | 10 | 66 | 0 | 0 |
| missing | 0 | 0 | 0 | 35 | 0 |

### Confusion matrix: gemma4_12b_base (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 95 | 7 | 2 | 0 | 9 |
| partial | 10 | 43 | 1 | 0 | 4 |
| incorrect | 3 | 10 | 61 | 0 | 4 |
| missing | 0 | 1 | 0 | 31 | 3 |
