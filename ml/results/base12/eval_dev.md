# Teach-back grader evaluation

Test set: 60 answers, 119 facts, 4 cases, 12 languages. Gold verdicts: {'incorrect': 33, 'correct': 50, 'missing': 12, 'partial': 24}. Generated 2026-10-15 17:43:03 IST.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| e4b_base | 77.3% (71.0%–86.0%) | 0.781 | 0.0% (0.0%–0.0%) | 0.0% (0/35) | 12.0% | 100.0% |
| e4b_lora | 84.9% (78.3%–94.0%) | 0.835 | 0.0% (0.0%–0.0%) | 0.0% (0/35) | 8.0% | 100.0% |
| gemma4_12b_base | 80.7% (73.8%–87.9%) | 0.844 | 0.0% (0.0%–0.0%) | 0.0% (0/35) | 16.0% | 93.3% |

## Accuracy by language

| System | ar | bn | en | es | fr | hi | it | pt | ta | tl | vi | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b_base | 75% | 50% | 89% | 100% | 90% | 71% | 67% | 77% | 70% | 85% | 70% | 77% |
| e4b_lora | 75% | 75% | 78% | 89% | 90% | 100% | 89% | 100% | 60% | 85% | 80% | 92% |
| gemma4_12b_base | 88% | 75% | 78% | 100% | 80% | 29% | 78% | 77% | 50% | 100% | 100% | 92% |

## Speed and tokens

| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |
|---|---|---|---|---|---|---|
| e4b_base | AMD Radeon Pro W7900D | n/a | n/a | 188 (bs16) | 345 | 143 |
| e4b_lora | AMD Radeon Pro W7900D | n/a | n/a | 177 (bs16) | 345 | 92 |
| gemma4_12b_base | AMD Radeon Pro W7900D | 9.09 s | 17.41 s | 14.5 (bs1), 81 (bs16) | 349 | 99 |

### Confusion matrix: e4b_base (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 44 | 4 | 2 | 0 | 0 |
| partial | 4 | 20 | 0 | 0 | 0 |
| incorrect | 0 | 16 | 17 | 0 | 0 |
| missing | 0 | 1 | 0 | 11 | 0 |

### Confusion matrix: e4b_lora (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 46 | 1 | 3 | 0 | 0 |
| partial | 9 | 13 | 1 | 1 | 0 |
| incorrect | 0 | 2 | 31 | 0 | 0 |
| missing | 0 | 0 | 1 | 11 | 0 |

### Confusion matrix: gemma4_12b_base (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 42 | 3 | 2 | 0 | 3 |
| partial | 4 | 17 | 0 | 0 | 3 |
| incorrect | 0 | 5 | 26 | 0 | 2 |
| missing | 0 | 0 | 0 | 11 | 1 |
