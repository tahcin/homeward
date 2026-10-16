# Teach-back grader evaluation

Test set: 120 answers, 284 facts, 5 cases, 12 languages. Gold verdicts: {'incorrect': 78, 'correct': 113, 'partial': 58, 'missing': 35}. Generated 2026-10-16 12:20:02 IST.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| runA_final | 89.1% (86.2%–93.0%) | 0.887 | 1.8% (0.0%–5.4%) | 2.7% (2/74) | 0.0% | 100.0% |
| runA_avg | 88.7% (85.6%–92.1%) | 0.883 | 1.8% (0.0%–5.4%) | 2.7% (2/74) | 0.9% | 100.0% |

## Accuracy by language

| System | ar | bn | en | es | fr | hi | it | pt | ta | tl | vi | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| runA_final | 90% | 83% | 96% | 85% | 95% | 81% | 89% | 88% | 77% | 92% | 100% | 91% |
| runA_avg | 87% | 79% | 96% | 85% | 95% | 81% | 89% | 88% | 82% | 92% | 100% | 91% |

## Speed and tokens

| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |
|---|---|---|---|---|---|---|
| runA_final | AMD Radeon Pro W7900D | n/a | n/a | 170 (bs16) | 371 | 113 |
| runA_avg | AMD Token Factory | n/a | n/a | n/a | 371 | 114 |

### Confusion matrix: runA_final (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 113 | 0 | 0 | 0 | 0 |
| partial | 20 | 37 | 1 | 0 | 0 |
| incorrect | 2 | 7 | 69 | 0 | 0 |
| missing | 0 | 0 | 1 | 34 | 0 |

### Confusion matrix: runA_avg (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 112 | 1 | 0 | 0 | 0 |
| partial | 19 | 38 | 1 | 0 | 0 |
| incorrect | 2 | 7 | 68 | 1 | 0 |
| missing | 0 | 0 | 1 | 34 | 0 |
