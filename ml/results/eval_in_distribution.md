# Teach-back grader evaluation

Test set: 348 answers, 744 facts, 18 cases, 12 languages. Gold verdicts: {'correct': 382, 'incorrect': 176, 'missing': 63, 'partial': 123}. Generated 2026-10-15 16:58:16 IST.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| e4b_base | 82.1% (79.1%–84.9%) | 0.763 | 1.7% (0.4%–3.1%) | 1.2% (2/173) | 5.0% | 100.0% |
| e4b_lora | 87.5% (85.2%–89.7%) | 0.829 | 1.7% (0.4%–3.1%) | 1.2% (2/173) | 3.7% | 100.0% |
| gemma4_31b | 85.3% (82.0%–88.2%) | 0.801 | 2.5% (0.8%–4.5%) | 1.2% (2/173) | 6.8% | 100.0% |
| deepseek_v4_flash | 70.8% (65.6%–75.9%) | 0.706 | 3.8% (1.3%–6.7%) | 3.5% (6/173) | 24.3% | 84.5% |

## Accuracy by language

| System | ar | bn | en | es | fr | hi | it | pt | ta | tl | vi | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b_base | 80% | 81% | 86% | 88% | 89% | 90% | 71% | 80% | 90% | 80% | 78% | 83% |
| e4b_lora | 88% | 84% | 95% | 91% | 84% | 98% | 79% | 86% | 90% | 86% | 84% | 90% |
| gemma4_31b | 82% | 91% | 91% | 91% | 77% | 88% | 67% | 91% | 92% | 85% | 86% | 86% |
| deepseek_v4_flash | 48% | 72% | 86% | 94% | 77% | 65% | 56% | 66% | 75% | 69% | 66% | 78% |

## Speed and tokens

| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |
|---|---|---|---|---|---|---|
| e4b_base | AMD Radeon Pro W7900D | 8.57 s | 11.85 s | 18.8 (bs1), 174 (bs16) | 398 | 157 |
| e4b_lora | AMD Radeon Pro W7900D | 5.37 s | 7.17 s | 19.1 (bs1), 169 (bs16) | 398 | 103 |
| gemma4_31b | AMD Token Factory | 736.11 s | 1393.01 s | n/a | 402 | 128 |
| deepseek_v4_flash | AMD Token Factory | 555.41 s | 1211.82 s | n/a | 366 | 112 |

### Confusion matrix: e4b_base (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 363 | 16 | 3 | 0 | 0 |
| partial | 27 | 90 | 6 | 0 | 0 |
| incorrect | 2 | 51 | 117 | 6 | 0 |
| missing | 2 | 15 | 5 | 41 | 0 |

### Confusion matrix: e4b_lora (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 368 | 8 | 6 | 0 | 0 |
| partial | 28 | 80 | 15 | 0 | 0 |
| incorrect | 2 | 13 | 158 | 3 | 0 |
| missing | 2 | 4 | 12 | 45 | 0 |

### Confusion matrix: gemma4_31b (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 356 | 19 | 7 | 0 | 0 |
| partial | 28 | 82 | 12 | 1 | 0 |
| incorrect | 3 | 11 | 155 | 7 | 0 |
| missing | 3 | 4 | 14 | 42 | 0 |

### Confusion matrix: deepseek_v4_flash (rows gold, columns predicted)

| gold \ pred | correct | partial | incorrect | missing | invalid |
|---|---|---|---|---|---|
| correct | 289 | 23 | 14 | 7 | 49 |
| partial | 28 | 70 | 7 | 2 | 16 |
| incorrect | 7 | 14 | 131 | 5 | 19 |
| missing | 2 | 8 | 13 | 37 | 3 |
