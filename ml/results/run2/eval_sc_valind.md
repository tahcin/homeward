# Teach-back grader evaluation

Test set: 120 answers, 284 facts, 5 cases, 12 languages. Gold verdicts: {'incorrect': 78, 'correct': 113, 'partial': 58, 'missing': 35}. Generated 2026-10-16 13:54:20 IST.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| greedy | 87.3% (76.6%–94.6%) | 0.888 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 6.2% | 98.3% |
| sample1 | 87.7% (79.7%–93.6%) | 0.892 | 0.9% (0.0%–2.7%) | 0.0% (0/74) | 6.2% | 98.3% |
| vote3_samples | 89.1% (83.1%–94.0%) | 0.899 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 6.2% | 99.2% |
| vote_greedy_s1_s2 | 87.3% (77.5%–94.0%) | 0.889 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 7.1% | 98.3% |

## Accuracy by language

| System | ar | bn | en | es | fr | hi | it | pt | ta | tl | vi | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 70% | 92% | 83% | 69% | 95% | 88% | 89% | 92% | 91% | 92% | 100% | 96% |
| sample1 | 70% | 92% | 88% | 69% | 95% | 88% | 89% | 96% | 91% | 92% | 100% | 91% |
| vote3_samples | 70% | 92% | 88% | 88% | 100% | 88% | 89% | 96% | 82% | 92% | 100% | 91% |
| vote_greedy_s1_s2 | 70% | 92% | 83% | 69% | 100% | 88% | 89% | 92% | 86% | 92% | 100% | 96% |

## Speed and tokens

| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |
|---|---|---|---|---|---|---|
| greedy | AMD Radeon Pro W7900D | n/a | n/a | 173 (bs16) | 371 | 118 |
| sample1 | AMD Radeon Pro W7900D | n/a | n/a | 168 (bs16) | 371 | 117 |
| vote3_samples | AMD Token Factory | n/a | n/a | n/a | 0 | 0 |
| vote_greedy_s1_s2 | AMD Token Factory | n/a | n/a | n/a | 0 | 0 |

### Confusion matrix: greedy (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 106 | 1 | 1 | 0 | 5 |
| partial | 10 | 45 | 0 | 0 | 3 |
| incorrect | 0 | 12 | 66 | 0 | 0 |
| missing | 0 | 0 | 0 | 31 | 4 |

### Confusion matrix: sample1 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 106 | 2 | 0 | 0 | 5 |
| partial | 12 | 43 | 0 | 0 | 3 |
| incorrect | 1 | 8 | 69 | 0 | 0 |
| missing | 0 | 0 | 0 | 31 | 4 |

### Confusion matrix: vote3_samples (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 106 | 2 | 2 | 0 | 3 |
| partial | 11 | 46 | 0 | 0 | 1 |
| incorrect | 0 | 9 | 69 | 0 | 0 |
| missing | 0 | 0 | 1 | 32 | 2 |

### Confusion matrix: vote_greedy_s1_s2 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 105 | 2 | 1 | 0 | 5 |
| partial | 10 | 45 | 0 | 0 | 3 |
| incorrect | 0 | 11 | 67 | 0 | 0 |
| missing | 0 | 0 | 0 | 31 | 4 |
