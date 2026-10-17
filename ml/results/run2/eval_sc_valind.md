# Teach-back grader evaluation

Test set: 120 answers, 284 facts, 5 cases, 12 languages. Gold verdicts: {'incorrect': 78, 'correct': 113, 'partial': 58, 'missing': 35}. Generated 2026-10-16 13:59:11 IST.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| greedy | 91.5% (83.3%–95.7%) | 0.916 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 1.8% | 98.3% |
| sample1 | 91.5% (86.4%–95.1%) | 0.917 | 0.9% (0.0%–2.7%) | 0.0% (0/74) | 1.8% | 98.3% |
| sample2 | 90.8% (84.0%–94.5%) | 0.909 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 3.5% | 98.3% |
| sample3 | 91.5% (83.3%–96.4%) | 0.916 | 0.9% (0.0%–2.1%) | 1.4% (1/74) | 1.8% | 97.5% |
| vote3_samples | 91.5% (84.1%–96.0%) | 0.916 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 2.7% | 100.0% |
| vote_greedy_s1_s2 | 91.5% (84.1%–95.3%) | 0.917 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 2.7% | 100.0% |

## Accuracy by language

| System | ar | bn | en | es | fr | hi | it | pt | ta | tl | vi | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 90% | 92% | 83% | 92% | 95% | 88% | 89% | 92% | 91% | 92% | 100% | 96% |
| sample1 | 90% | 92% | 88% | 88% | 95% | 88% | 89% | 96% | 91% | 92% | 100% | 91% |
| sample2 | 90% | 92% | 79% | 92% | 100% | 88% | 89% | 92% | 82% | 92% | 100% | 96% |
| sample3 | 93% | 92% | 92% | 88% | 100% | 88% | 89% | 96% | 82% | 92% | 96% | 91% |
| vote3_samples | 93% | 92% | 88% | 88% | 100% | 88% | 89% | 96% | 82% | 92% | 100% | 91% |
| vote_greedy_s1_s2 | 90% | 92% | 83% | 92% | 100% | 88% | 89% | 92% | 86% | 92% | 100% | 96% |

## Speed and tokens

| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |
|---|---|---|---|---|---|---|
| greedy | AMD Radeon Pro W7900D | n/a | n/a | 173 (bs16) | 371 | 118 |
| sample1 | AMD Radeon Pro W7900D | n/a | n/a | 168 (bs16) | 371 | 117 |
| sample2 | AMD Token Factory | n/a | n/a | n/a | 371 | 119 |
| sample3 | AMD Token Factory | n/a | n/a | n/a | 371 | 118 |
| vote3_samples | AMD Token Factory | n/a | n/a | n/a | 0 | 0 |
| vote_greedy_s1_s2 | AMD Token Factory | n/a | n/a | n/a | 0 | 0 |

### Confusion matrix: greedy (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 111 | 1 | 1 | 0 | 0 |
| partial | 10 | 48 | 0 | 0 | 0 |
| incorrect | 0 | 12 | 66 | 0 | 0 |
| missing | 0 | 0 | 0 | 35 | 0 |

### Confusion matrix: sample1 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 111 | 2 | 0 | 0 | 0 |
| partial | 13 | 45 | 0 | 0 | 0 |
| incorrect | 1 | 8 | 69 | 0 | 0 |
| missing | 0 | 0 | 0 | 35 | 0 |

### Confusion matrix: sample2 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 109 | 2 | 2 | 0 | 0 |
| partial | 9 | 49 | 0 | 0 | 0 |
| incorrect | 0 | 12 | 66 | 0 | 0 |
| missing | 0 | 0 | 1 | 34 | 0 |

### Confusion matrix: sample3 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 111 | 1 | 1 | 0 | 0 |
| partial | 11 | 47 | 0 | 0 | 0 |
| incorrect | 1 | 9 | 68 | 0 | 0 |
| missing | 0 | 0 | 1 | 34 | 0 |

### Confusion matrix: vote3_samples (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 110 | 2 | 1 | 0 | 0 |
| partial | 11 | 47 | 0 | 0 | 0 |
| incorrect | 0 | 9 | 69 | 0 | 0 |
| missing | 0 | 0 | 1 | 34 | 0 |

### Confusion matrix: vote_greedy_s1_s2 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 110 | 2 | 1 | 0 | 0 |
| partial | 10 | 48 | 0 | 0 | 0 |
| incorrect | 0 | 11 | 67 | 0 | 0 |
| missing | 0 | 0 | 0 | 35 | 0 |
