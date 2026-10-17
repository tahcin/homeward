# Homeward grader: evaluation (run 4, final)

Generated 2026-10-18 02:01 IST. All numbers come from the files in ml/results; 95% intervals are case-level bootstrap (1,000 resamples).

## Scoring

Facts whose verdict cannot be read count as wrong. Since 2026-10-16 the parser has a lenient fallback for malformed JSON (a dropped colon or quote): each fact's id and verdict are read with a regex. It is applied identically to every system and every set; JSON validity is still reported strictly. It was added after model selection, so it did not influence which model was chosen. On the test sets it changes only DeepSeek-V4-Flash (79.2% to 80.4% independent).

## Independent test set (headline)

150 answers, 341 facts, 7 cases, 12 languages; gold {'correct': 138, 'incorrect': 96, 'partial': 65, 'missing': 42}.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | Critical facts graded correct when wrong/missing | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| e4b_base | 83.0% (80.6–85.8) | 0.832 | 4.3% (1.5–7.2) | 5/108 | 10.9% | 100.0% |
| run1 | 88.3% (85.4–91.1) | 0.878 | 5.8% (2.9–8.6) | 7/108 | 6.5% | 100.0% |
| homeward_run2 | 94.1% (91.4–96.7) | 0.940 | 1.4% (0.0–3.2) | 2/108 | 2.9% | 100.0% |
| homeward_run4 | 94.7% (91.9–97.5) | 0.946 | 1.4% (0.0–3.1) | 1/108 | 2.9% | 100.0% |
| gemma4_31b | 97.1% (94.9–99.1) | 0.971 | 0.7% (0.0–2.1) | 1/108 | 2.9% | 100.0% |
| deepseek_v4_flash | 80.4% (73.4–86.3) | 0.825 | 2.9% (0.0–6.8) | 3/108 | 14.5% | 90.7% |
| qwen3_8_flash_next | 91.2% (89.7–92.8) | 0.913 | 0.7% (0.0–2.1) | 1/108 | 8.7% | 99.3% |
| minicpm_v_4_7_35b_a3b | 85.9% (79.4–90.7) | 0.852 | 2.2% (0.0–6.1) | 3/108 | 13.8% | 100.0% |
| clef | 88.9% (86.1–92.7) | 0.897 | 0.7% (0.0–2.0) | 1/108 | 13.0% | 100.0% |
| mimo_v2_6_flash | 88.9% (86.3–91.8) | 0.891 | 0.7% (0.0–2.0) | 1/108 | 15.9% | 100.0% |
| qwen3_8_27b | 87.7% (85.9–90.1) | 0.895 | 0.7% (0.0–2.0) | 1/108 | 23.2% | 99.3% |
| glm_5_3_flash | 90.3% (85.9–93.7) | 0.921 | 0.7% (0.0–2.0) | 1/108 | 12.3% | 94.0% |

- run1 vs base E4B: right where base is wrong on 28 facts, wrong where base is right on 10 (exact McNemar p = 0.0051).
- homeward_run2 vs base E4B: right where base is wrong on 41 facts, wrong where base is right on 3 (exact McNemar p = 1.6e-09).
- homeward_run4 vs base E4B: right where base is wrong on 43 facts, wrong where base is right on 3 (exact McNemar p = 4.6e-10).
- gemma4_31b vs base E4B: right where base is wrong on 51 facts, wrong where base is right on 3 (exact McNemar p = 2.9e-12).
- deepseek_v4_flash vs base E4B: right where base is wrong on 27 facts, wrong where base is right on 36 (exact McNemar p = 0.31).
- qwen3_8_flash_next vs base E4B: right where base is wrong on 45 facts, wrong where base is right on 17 (exact McNemar p = 0.0005).
- minicpm_v_4_7_35b_a3b vs base E4B: right where base is wrong on 37 facts, wrong where base is right on 27 (exact McNemar p = 0.26).
- clef vs base E4B: right where base is wrong on 36 facts, wrong where base is right on 16 (exact McNemar p = 0.0078).
- mimo_v2_6_flash vs base E4B: right where base is wrong on 43 facts, wrong where base is right on 23 (exact McNemar p = 0.019).
- qwen3_8_27b vs base E4B: right where base is wrong on 49 facts, wrong where base is right on 33 (exact McNemar p = 0.097).
- glm_5_3_flash vs base E4B: right where base is wrong on 47 facts, wrong where base is right on 22 (exact McNemar p = 0.0035).

## In-distribution test set (generator-labelled)

348 answers, 744 facts, 18 cases, 12 languages; gold {'correct': 382, 'incorrect': 176, 'missing': 63, 'partial': 123}.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | Critical facts graded correct when wrong/missing | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| e4b_base | 82.1% (79.1–84.9) | 0.763 | 1.7% (0.4–3.1) | 2/173 | 5.0% | 100.0% |
| run1 | 87.5% (85.2–89.7) | 0.829 | 1.7% (0.4–3.1) | 2/173 | 3.7% | 100.0% |
| homeward_run2 | 87.0% (84.7–89.4) | 0.814 | 2.1% (0.4–4.1) | 3/173 | 2.9% | 100.0% |
| homeward_run4 | 86.0% (83.3–88.7) | 0.814 | 2.1% (0.4–4.1) | 3/173 | 6.5% | 100.0% |
| gemma4_31b | 85.3% (82.0–88.2) | 0.801 | 2.5% (0.8–4.5) | 2/173 | 6.8% | 100.0% |
| deepseek_v4_flash | 71.6% (66.1–76.7) | 0.709 | 3.8% (1.3–6.7) | 6/173 | 23.6% | 84.5% |

- run1 vs base E4B: right where base is wrong on 54 facts, wrong where base is right on 14 (exact McNemar p = 1.1e-06).
- homeward_run2 vs base E4B: right where base is wrong on 57 facts, wrong where base is right on 21 (exact McNemar p = 5.6e-05).
- homeward_run4 vs base E4B: right where base is wrong on 56 facts, wrong where base is right on 27 (exact McNemar p = 0.0019).
- gemma4_31b vs base E4B: right where base is wrong on 62 facts, wrong where base is right on 38 (exact McNemar p = 0.021).
- deepseek_v4_flash vs base E4B: right where base is wrong on 42 facts, wrong where base is right on 120 (exact McNemar p = 6.8e-10).

## Training (AMD Radeon PRO W7900D, ROCm)

LoRA r=32, alpha=32, lr 0.0002, 2.0 epochs, 3738 training rows, 468 steps; 69,763,072 trainable of 8,010,863,904 params; 58.8 min wall, 1045 tokens/s, peak 23.6 GiB allocated; val loss 0.792 -> 0.331.

## Model selection (validation only)

Declared policy, fixed on validation before any test scoring: a missed critical fact is never acceptable; extra re-checks are. Among all candidates (adapters, LoRA scales) and escalation thresholds tau on val_independent, require zero critical false reassurance and total false reassurance no worse than base E4B on the same set; then minimise total false reassurance, then over-flags, then maximise accuracy.

Chosen: run4_final (LoRA scale 1.0, tau 0.5): val accuracy 93.0%, false reassurance 0, critical 0, over-flags 2.
For transparency, the earlier rule (minimise 5 x false reassurance + over-flags) would have picked run4_final at tau 0.5: accuracy 93.0%, false reassurance 0, critical 0, over-flags 2.

## Run-4 recipe

- Data (splits_v3): 3602 unique teach-back answers (3738 rows; 89 answers where run 2 disagreed with agreed labels x2, 76 low-confidence correct answers x1.5; near-miss and weak-language oversampling off to keep the run inside the GPU window). Sources {'v3': 2754, 'v1': 282, 'v2': 566}. All answers generated and labelled by open models on AMD Token Factory. Labels: per-fact vote of the designed label, Gemma-4-31B-IT, Qwen3.8-Flash-Next or DeepSeek-V4-Flash where present, and Clef (Cloudflare's grading model) as one ordinary vote; strict majority, else the 31B verdict if it is the unique plurality, else the answer is dropped; a vote never turns a 31B incorrect/missing into correct; answers with a final 'correct' fact that Clef gives P(correct) < 0.2 are dropped as likely label errors.
- Contamination check: passed (transcript hashes, cases, held-out conditions); dev, val_independent and test cases excluded from training.
- Target order: 'heard' before 'verdict'. LoRA r=32, alpha=32, lr 0.0002, 2.0 epochs, loss weight 3.0 on verdict tokens, unlikelihood 0.5 on P('correct') where gold is incorrect/missing (half for partial).
- Inference: the final (epoch-2) adapter at LoRA scale 1.0, greedy decoding; a fact graded correct with P(correct) < 0.5 goes to the nurse (tau 0.5, chosen on validation).

## Candidates on validation (val_independent)

| Candidate | LoRA scale | tau | Accuracy | False reassurance | Critical | Over-flags |
|---|---|---|---|---|---|---|
| run4_final | 1.0 | 0.5 | 93.0% | 0 | 0 | 2 |
| run2_final | 0.5 | 0.0 | 91.5% | 0 | 0 | 2 |
| run4_final_sc | 1.0 | 0.5 | 93.0% | 0 | 0 | 3 |
| run4_final_s050 | 0.5 | 0.0 | 92.3% | 0 | 0 | 3 |
| run4_avg_s050 | 0.5 | 0.0 | 91.9% | 0 | 0 | 3 |

Run 2's frozen model was a candidate, so run 4 had to beat it on validation. Two planned candidates did not run: the epoch-averaged adapter at scale 1.0 (its model load stalled twice on a ROCm host-to-GPU copy) and scale 0.75 (out of time). The self-consistency candidate (greedy plus 2 samples at T 0.7) tied on accuracy with one more over-flag at 3x the generation.

## Run 4 vs run 2 (independent test, paired)

Run 4 is right where run 2 is wrong on 9 facts and wrong where run 2 is right on 7 (exact McNemar p = 0.8): neither the accuracy gain nor the critical-miss difference (1 vs 2 of 108) is statistically significant on 150 answers.

## Gemma-4-31B on the validation set

val_independent: accuracy 95.1%, false reassurance 0/113, over-flag 1.8%.

## Gemma 4 12B screen (not trained)

Base 12B on val_independent: 81.0%, critical misses 2/74, JSON valid 95.0%; below the bar for training it, and its ASR on the recorded voices was worse (WER 41% vs 20%).

## Capacity on one W7900D (fine-tuned grader)

| Batch | Teach-backs/s | Teach-backs/hour | Gen tokens/s | Peak GiB |
|---|---|---|---|---|
| 16 | 1.55 | 5591 | 165 | 16.4 |
| 64 | 2.42 | 8707 | 257 | 20.6 |

Discharges per hour per card, ASR plus grading: 274 (6 graded answers per discharge; grading batched at 64; ASR unbatched at batch 1 (p50 of the on-prem loop clips); one card; no other load). Grading only: 1451.

## Hosted cost per discharge (Token Factory list prices)

- gemma4_31b: $0.00013 per discharge ($0.000022 per answer)
- deepseek_v4_flash: $0.00052 per discharge ($0.000086 per answer)

## On-prem loop

25 clips (timed with the run-1 adapter; same model size and code path): E4B ASR then fine-tuned E4B grading on one card, p50 10.9 s, p95 14.9 s, peak 15.1 GiB.

