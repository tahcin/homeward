# Teach-back grader evaluation

Test set: 120 answers, 284 facts, 5 cases, 12 languages. Gold verdicts: {'incorrect': 78, 'correct': 113, 'partial': 58, 'missing': 35}. Generated 2026-10-16 12:18:46 IST.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| e4b_base | 75.4% (66.4%–80.7%) | 0.766 | 1.8% (0.0%–3.0%) | 0.0% (0/74) | 13.3% | 100.0% |
| run1 | 85.6% (81.7%–89.0%) | 0.857 | 1.8% (0.0%–3.0%) | 0.0% (0/74) | 7.1% | 100.0% |
| runA_final | 89.1% (86.2%–93.0%) | 0.887 | 1.8% (0.0%–5.4%) | 2.7% (2/74) | 0.0% | 100.0% |

## Accuracy by language

| System | ar | bn | en | es | fr | hi | it | pt | ta | tl | vi | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b_base | 80% | 62% | 75% | 81% | 70% | 75% | 82% | 75% | 68% | 75% | 83% | 74% |
| run1 | 93% | 79% | 83% | 88% | 85% | 75% | 86% | 88% | 73% | 83% | 96% | 91% |
| runA_final | 90% | 83% | 96% | 85% | 95% | 81% | 89% | 88% | 77% | 92% | 100% | 91% |

## Speed and tokens

| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |
|---|---|---|---|---|---|---|
| e4b_base | AMD Radeon Pro W7900D | n/a | n/a | 179 (bs16) | 371 | 166 |
| run1 | AMD Radeon Pro W7900D | n/a | n/a | 169 (bs16) | 371 | 110 |
| runA_final | AMD Radeon Pro W7900D | n/a | n/a | 170 (bs16) | 371 | 113 |

### Confusion matrix: e4b_base (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 98 | 11 | 4 | 0 | 0 |
| partial | 17 | 41 | 0 | 0 | 0 |
| incorrect | 2 | 35 | 41 | 0 | 0 |
| missing | 0 | 1 | 0 | 34 | 0 |

### Confusion matrix: run1 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 105 | 5 | 3 | 0 | 0 |
| partial | 19 | 37 | 1 | 1 | 0 |
| incorrect | 2 | 10 | 66 | 0 | 0 |
| missing | 0 | 0 | 0 | 35 | 0 |

### Confusion matrix: runA_final (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 113 | 0 | 0 | 0 | 0 |
| partial | 20 | 37 | 1 | 0 | 0 |
| incorrect | 2 | 7 | 69 | 0 | 0 |
| missing | 0 | 0 | 1 | 34 | 0 |
