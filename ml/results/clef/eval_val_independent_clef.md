# Teach-back grader evaluation

Test set: 120 answers, 284 facts, 5 cases, 12 languages. Gold verdicts: {'incorrect': 78, 'correct': 113, 'partial': 58, 'missing': 35}. Generated 2026-10-16 18:58:27 IST.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| clef | 85.6% (75.4%–91.6%) | 0.865 | 1.8% (0.0%–5.4%) | 2.7% (2/74) | 12.4% | 100.0% |
| gemma4_31b | 95.1% (92.5%–98.1%) | 0.948 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 1.8% | 100.0% |
| homeward_run2 | 91.5% (83.3%–95.7%) | 0.916 | 0.0% (0.0%–0.0%) | 0.0% (0/74) | 1.8% | 98.3% |
| e4b_base | 75.4% (66.4%–80.7%) | 0.766 | 1.8% (0.0%–3.0%) | 0.0% (0/74) | 13.3% | 100.0% |

## Accuracy by language

| System | ar | bn | en | es | fr | hi | it | pt | ta | tl | vi | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clef | 87% | 79% | 83% | 88% | 90% | 88% | 93% | 88% | 68% | 83% | 91% | 87% |
| gemma4_31b | 93% | 96% | 96% | 96% | 100% | 94% | 93% | 96% | 91% | 100% | 100% | 87% |
| homeward_run2 | 90% | 92% | 83% | 92% | 95% | 88% | 89% | 92% | 91% | 92% | 100% | 96% |
| e4b_base | 80% | 62% | 75% | 81% | 70% | 75% | 82% | 75% | 68% | 75% | 83% | 74% |

## Speed and tokens

| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |
|---|---|---|---|---|---|---|
| clef | AMD Token Factory | n/a | n/a | n/a | 700 | 0 |
| gemma4_31b | AMD Token Factory | 633.57 s | 1182.73 s | n/a | 375 | 138 |
| homeward_run2 | AMD Radeon Pro W7900D | n/a | n/a | 173 (bs16) | 371 | 118 |
| e4b_base | AMD Radeon Pro W7900D | n/a | n/a | 179 (bs16) | 371 | 166 |

### Confusion matrix: clef (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 99 | 14 | 0 | 0 | 0 |
| partial | 1 | 57 | 0 | 0 | 0 |
| incorrect | 2 | 23 | 53 | 0 | 0 |
| missing | 0 | 1 | 0 | 34 | 0 |

### Confusion matrix: gemma4_31b (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 111 | 2 | 0 | 0 | 0 |
| partial | 7 | 50 | 1 | 0 | 0 |
| incorrect | 0 | 2 | 75 | 1 | 0 |
| missing | 0 | 1 | 0 | 34 | 0 |

### Confusion matrix: homeward_run2 (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 111 | 1 | 1 | 0 | 0 |
| partial | 10 | 48 | 0 | 0 | 0 |
| incorrect | 0 | 12 | 66 | 0 | 0 |
| missing | 0 | 0 | 0 | 35 | 0 |

### Confusion matrix: e4b_base (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 98 | 11 | 4 | 0 | 0 |
| partial | 17 | 41 | 0 | 0 | 0 |
| incorrect | 2 | 35 | 41 | 0 | 0 |
| missing | 0 | 1 | 0 | 34 | 0 |
