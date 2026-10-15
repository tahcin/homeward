# Homeward grader: evaluation (run 1)

Generated 2026-10-15 16:59 IST. All numbers come from the files in ml/results; 95% intervals are case-level bootstrap (1,000 resamples).

## Independent test set (headline)

150 answers, 341 facts, 7 cases, 12 languages; gold {'correct': 138, 'incorrect': 96, 'partial': 65, 'missing': 42}.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | Critical facts graded correct when wrong/missing | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| e4b_base | 83.0% (80.6–85.8) | 0.832 | 4.3% (1.5–7.2) | 5/108 | 10.9% | 100.0% |
| e4b_lora | 88.3% (85.4–91.1) | 0.878 | 5.8% (2.9–8.6) | 7/108 | 6.5% | 100.0% |
| gemma4_31b | 97.1% (94.9–99.1) | 0.971 | 0.7% (0.0–2.1) | 1/108 | 2.9% | 100.0% |
| deepseek_v4_flash | 79.2% (72.3–86.2) | 0.817 | 2.9% (0.0–6.8) | 3/108 | 14.5% | 90.7% |

- e4b_lora vs base E4B: right where base is wrong on 28 facts, wrong where base is right on 10 (exact McNemar p = 0.0051).
- gemma4_31b vs base E4B: right where base is wrong on 51 facts, wrong where base is right on 3 (exact McNemar p = 2.9e-12).
- deepseek_v4_flash vs base E4B: right where base is wrong on 25 facts, wrong where base is right on 38 (exact McNemar p = 0.13).

## In-distribution test set (generator-labelled)

348 answers, 744 facts, 18 cases, 12 languages; gold {'correct': 382, 'incorrect': 176, 'missing': 63, 'partial': 123}.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | Critical facts graded correct when wrong/missing | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| e4b_base | 82.1% (79.1–84.9) | 0.763 | 1.7% (0.4–3.1) | 2/173 | 5.0% | 100.0% |
| e4b_lora | 87.5% (85.2–89.7) | 0.829 | 1.7% (0.4–3.1) | 2/173 | 3.7% | 100.0% |
| gemma4_31b | 85.3% (82.0–88.2) | 0.801 | 2.5% (0.8–4.5) | 2/173 | 6.8% | 100.0% |
| deepseek_v4_flash | 70.8% (65.6–75.9) | 0.706 | 3.8% (1.3–6.7) | 6/173 | 24.3% | 84.5% |

- e4b_lora vs base E4B: right where base is wrong on 54 facts, wrong where base is right on 14 (exact McNemar p = 1.1e-06).
- gemma4_31b vs base E4B: right where base is wrong on 62 facts, wrong where base is right on 38 (exact McNemar p = 0.021).
- deepseek_v4_flash vs base E4B: right where base is wrong on 42 facts, wrong where base is right on 126 (exact McNemar p = 6e-11).

## Training (AMD Radeon PRO W7900D, ROCm)

LoRA r=16, alpha=32, lr 0.0001, 2.0 epochs, 672 verifier-agreed examples, 84 steps; 34,881,536 trainable of 7,975,982,368 params; 9.6 min wall, 1115 tokens/s, peak 21.3 GiB allocated; val loss 0.544 -> 0.203.

## Safety threshold

both selection sets (generator val split and a 60-answer independent dev set written by Claude) had zero false reassurance at every tau, so the rule chose tau = 0 (no escalation); the threshold could not be calibrated and test numbers are unchanged.

## Capacity on one W7900D (fine-tuned grader)

| Batch | Teach-backs/s | Teach-backs/hour | Gen tokens/s | Peak GiB |
|---|---|---|---|---|
| 16 | 1.64 | 5917 | 169 | 16.4 |
| 32 | 2.16 | 7784 | 222 | 17.9 |
| 64 | 2.51 | 9027 | 257 | 20.6 |

Discharges per hour per card, ASR plus grading: 276 (6 graded answers per discharge; grading batched at 64; ASR unbatched at batch 1 (p50 of the on-prem loop clips); one card; no other load). Grading only: 1505.

## Hosted cost per discharge (Token Factory list prices)

- gemma4_31b: $0.00013 per discharge ($0.000022 per answer)
- deepseek_v4_flash: $0.00052 per discharge ($0.000086 per answer)

## On-prem loop

25 clips: E4B ASR then fine-tuned E4B grading on one card, p50 10.9 s, p95 14.9 s, peak 15.1 GiB.

