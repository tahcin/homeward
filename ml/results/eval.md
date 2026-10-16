# Homeward grader: evaluation (run 2, final)

Generated 2026-10-16 13:56 IST. All numbers come from the files in ml/results; 95% intervals are case-level bootstrap (1,000 resamples).

## Independent test set (headline)

150 answers, 341 facts, 7 cases, 12 languages; gold {'correct': 138, 'incorrect': 96, 'partial': 65, 'missing': 42}.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | Critical facts graded correct when wrong/missing | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| e4b_base | 83.0% (80.6–85.8) | 0.832 | 4.3% (1.5–7.2) | 5/108 | 10.9% | 100.0% |
| run1 | 88.3% (85.4–91.1) | 0.878 | 5.8% (2.9–8.6) | 7/108 | 6.5% | 100.0% |
| homeward_run2 | 94.1% (91.4–96.7) | 0.940 | 1.4% (0.0–3.2) | 2/108 | 2.9% | 100.0% |
| gemma4_31b | 97.1% (94.9–99.1) | 0.971 | 0.7% (0.0–2.1) | 1/108 | 2.9% | 100.0% |
| deepseek_v4_flash | 79.2% (72.3–86.2) | 0.817 | 2.9% (0.0–6.8) | 3/108 | 14.5% | 90.7% |

- run1 vs base E4B: right where base is wrong on 28 facts, wrong where base is right on 10 (exact McNemar p = 0.0051).
- homeward_run2 vs base E4B: right where base is wrong on 41 facts, wrong where base is right on 3 (exact McNemar p = 1.6e-09).
- gemma4_31b vs base E4B: right where base is wrong on 51 facts, wrong where base is right on 3 (exact McNemar p = 2.9e-12).
- deepseek_v4_flash vs base E4B: right where base is wrong on 25 facts, wrong where base is right on 38 (exact McNemar p = 0.13).

## In-distribution test set (generator-labelled)

348 answers, 744 facts, 18 cases, 12 languages; gold {'correct': 382, 'incorrect': 176, 'missing': 63, 'partial': 123}.

| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | Critical facts graded correct when wrong/missing | Over-flag | JSON valid |
|---|---|---|---|---|---|---|
| e4b_base | 82.1% (79.1–84.9) | 0.763 | 1.7% (0.4–3.1) | 2/173 | 5.0% | 100.0% |
| run1 | 87.5% (85.2–89.7) | 0.829 | 1.7% (0.4–3.1) | 2/173 | 3.7% | 100.0% |
| homeward_run2 | 87.0% (84.7–89.4) | 0.814 | 2.1% (0.4–4.1) | 3/173 | 2.9% | 100.0% |
| gemma4_31b | 85.3% (82.0–88.2) | 0.801 | 2.5% (0.8–4.5) | 2/173 | 6.8% | 100.0% |
| deepseek_v4_flash | 70.8% (65.6–75.9) | 0.706 | 3.8% (1.3–6.7) | 6/173 | 24.3% | 84.5% |

- run1 vs base E4B: right where base is wrong on 54 facts, wrong where base is right on 14 (exact McNemar p = 1.1e-06).
- homeward_run2 vs base E4B: right where base is wrong on 57 facts, wrong where base is right on 21 (exact McNemar p = 5.6e-05).
- gemma4_31b vs base E4B: right where base is wrong on 62 facts, wrong where base is right on 38 (exact McNemar p = 0.021).
- deepseek_v4_flash vs base E4B: right where base is wrong on 42 facts, wrong where base is right on 126 (exact McNemar p = 6e-11).

## Training (AMD Radeon PRO W7900D, ROCm)

LoRA r=32, alpha=32, lr 0.0002, 2.0 epochs, 2743 training rows, 344 steps; 69,763,072 trainable of 8,010,863,904 params; 43.5 min wall, 1036 tokens/s, peak 23.5 GiB allocated; val loss 0.792 -> 0.338.

## Model selection (validation only)

Declared policy, fixed on validation before any test scoring: a missed critical fact is never acceptable; extra re-checks are. Among all candidates (adapters, LoRA scales) and escalation thresholds tau on val_independent, require zero critical false reassurance and total false reassurance no worse than base E4B on the same set; then minimise total false reassurance, then over-flags, then maximise accuracy.

Chosen: runB_avg_s050 (LoRA scale 0.5, tau 0.0): val accuracy 87.3%, false reassurance 0, critical 0, over-flags 7.
For transparency, the earlier rule (minimise 5 x false reassurance + over-flags) would have picked runB_avg_s050 at tau 0.0: accuracy 87.3%, false reassurance 0, critical 0, over-flags 7.

## Run-2 recipe

- Data: 1793 unique teach-back answers (2743 rows after oversampling near-miss pairs x1.5 and weaker languages x1.25), v1 generator answers plus multi-model v2 answers. Labels: kept when Gemma-4-31B-IT agreed with the designed label on every fact (heard/reteach text distilled from 31B), otherwise a per-fact 2-of-3 vote of designed, 31B and Qwen3.8-Flash-Next labels; answers without a majority dropped. Counts: {'excluded_eval_or_heldout_case': 580, 'kept_teacher_agree_v1': 609, 'dropped_no_majority_v1': 21, 'kept_majority_vote_v1': 226, 'facts_relabelled_by_vote': 460, 'kept_teacher_agree_v2': 847, 'dropped_no_majority_v2': 45, 'kept_majority_vote_v2': 348, 'dropped_case_cap': 237}.
- Contamination check: passed (transcript hashes, cases, held-out conditions); dev, val_independent and test cases excluded from training.
- Target order: 'heard' before 'verdict' (run B). LoRA r=32, alpha=32, lr 0.0002, 2.0 epochs, loss weight 3.0 on verdict tokens, unlikelihood 0.5 on P('correct') where gold is incorrect/missing (half for partial).
- Inference: average of the epoch-1 and epoch-2 adapters, applied at LoRA scale 0.5 (chosen on validation), greedy decoding, no escalation (tau 0).

## Self-consistency (validation only)

- greedy: accuracy 87.3%, false reassurance 0/113, over-flag 6.2%
- sample1: accuracy 87.7%, false reassurance 1/113, over-flag 6.2%
- vote3_samples: accuracy 89.1%, false reassurance 0/113, over-flag 6.2%
- vote_greedy_s1_s2: accuracy 87.3%, false reassurance 0/113, over-flag 7.1%
- Majority of 3 samples (T 0.7) costs 3x the generation; candidate for run 4.

## Gemma-4-31B on the validation set

val_independent: accuracy 95.1%, false reassurance 0/113, over-flag 1.8%.

## Gemma 4 12B screen (not trained)

Base 12B on val_independent: 81.0%, critical misses 2/74, JSON valid 95.0%; below the bar for training it, and its ASR on the recorded voices was worse (WER 41% vs 20%).

## Capacity on one W7900D (fine-tuned grader)

| Batch | Teach-backs/s | Teach-backs/hour | Gen tokens/s | Peak GiB |
|---|---|---|---|---|
| 16 | 1.50 | 5398 | 162 | 16.4 |
| 64 | 2.42 | 8697 | 262 | 20.6 |

Discharges per hour per card, ASR plus grading: 274 (6 graded answers per discharge; grading batched at 64; ASR unbatched at batch 1 (p50 of the on-prem loop clips); one card; no other load). Grading only: 1449.

## Hosted cost per discharge (Token Factory list prices)

- gemma4_31b: $0.00013 per discharge ($0.000022 per answer)
- deepseek_v4_flash: $0.00052 per discharge ($0.000086 per answer)

## On-prem loop

25 clips (timed with the run-1 adapter; same model size and code path): E4B ASR then fine-tuned E4B grading on one card, p50 10.9 s, p95 14.9 s, peak 15.1 GiB.

