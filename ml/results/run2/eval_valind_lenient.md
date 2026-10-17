# Teach-back grader evaluation

Test set: 120 answers, 284 facts, 5 cases, 12 languages. Gold verdicts: {'incorrect': 78, 'correct': 113, 'partial': 58, 'missing': 35}. Generated 2026-10-16 13:58:58 IST.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| final_greedy | 91.5% (83.3%–95.7%) | 0.916 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 1.8% | 98.3% |
| vote3 | 89.1% (83.1%–94.0%) | 0.899 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 6.2% | 99.2% |
| gemma4_31b | 95.1% (92.5%–98.1%) | 0.948 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 1.8% | 100.0% |

## Accuracy by language

| System | ar | bn | en | es | fr | hi | it | pt | ta | tl | vi | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final_greedy | 90% | 92% | 83% | 92% | 95% | 88% | 89% | 92% | 91% | 92% | 100% | 96% |
| vote3 | 70% | 92% | 88% | 88% | 100% | 88% | 89% | 96% | 82% | 92% | 100% | 91% |
| gemma4_31b | 93% | 96% | 96% | 96% | 100% | 94% | 93% | 96% | 91% | 100% | 100% | 87% |

## Speed and tokens

| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |
|---|---|---|---|---|---|---|
| final_greedy | AMD Radeon Pro W7900D | n/a | n/a | 173 (bs16) | 371 | 118 |
| vote3 | AMD Token Factory | n/a | n/a | n/a | 0 | 0 |
| gemma4_31b | AMD Token Factory | 633.57 s | 1182.73 s | n/a | 375 | 138 |

### Confusion matrix: final_greedy (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 111 | 1 | 1 | 0 | 0 |
| partial | 10 | 48 | 0 | 0 | 0 |
| incorrect | 0 | 12 | 66 | 0 | 0 |
| missing | 0 | 0 | 0 | 35 | 0 |

### Confusion matrix: vote3 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 106 | 2 | 2 | 0 | 3 |
| partial | 11 | 46 | 0 | 0 | 1 |
| incorrect | 0 | 9 | 69 | 0 | 0 |
| missing | 0 | 0 | 1 | 32 | 2 |

### Confusion matrix: gemma4_31b (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 111 | 2 | 0 | 0 | 0 |
| partial | 7 | 50 | 1 | 0 | 0 |
| incorrect | 0 | 2 | 75 | 1 | 0 |
| missing | 0 | 1 | 0 | 34 | 0 |
