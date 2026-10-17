# Homeward

**Make sure patients understand before they go home.**

Homeward rewrites hospital discharge instructions in the patient's language, with every dose and date verifiably carried over or flagged to the nurse. It then listens to the patient (or their caregiver) explain the instructions back, grades each fact, and tells the nurse exactly which instruction was misunderstood. The teach-back grader is a small model, Gemma 4 E4B, fine-tuned on an AMD Radeon PRO W7900D so it can run on one workstation GPU inside the hospital.

- Live demo (synthetic patients, no login): https://homeward.tahcin.tech
- Fine-tuned grader (LoRA adapter for Gemma 4 E4B): https://huggingface.co/oktash/homeward-teachback-gemma4-e4b (publishing Oct 18)
- Built for the AMD Developer Hackathon: ACT III, Health and Wellbeing track, by team Solus.

> Hosted demo: synthetic data only, not medical advice. Homeward never diagnoses or recommends treatment; it only restates the clinician's own instructions and checks they were understood.

## Results

Teach-back grading on an independent test set that no training-data generator ever saw: 150 answers, 341 facts, 7 discharge cases, 12 languages, messy and code-mixed phrasing, written by Claude (a different model family from the generators) and never used for training. Each system grades every fact as correct, partial, incorrect or missing.

- The fine-tuned Gemma 4 E4B (8B, 4.5B effective) scores **94.1%** per-fact accuracy, up from 83.0% for the same model before fine-tuning, and passes 2 critical misses as understood instead of 5.
- It is more accurate than every other model we tested except its own teacher, including Qwen3.8 Flash Next (180B total, 6B active) at 91.2% and DeepSeek V4 Flash (284B total, 13B active) at 80.4%. Two comparators, Qwen3.8 Flash Next and Clef, pass fewer critical misses (1 against our 2), and our interval overlaps Qwen3.8 Flash Next's.
- Gemma 4 31B is the teacher whose labels the training data was filtered with. The fine-tune keeps 97% of its accuracy (94.1% vs 97.1%) at about a quarter of the size (8B vs 30.7B); the 31B passes 1 critical miss against our 2.

| System | Size (total / active) | Per-fact accuracy (95% CI) | Macro-F1 | Critical misses passed as understood | Over-flag | Valid JSON |
|---|---|---|---|---|---|---|
| **Homeward: Gemma 4 E4B + LoRA (W7900D)** | **8B / 4.5B effective** | **94.1% (91.4–96.7)** | **0.940** | **2 of 108** | **2.9%** | **100%** |
| Gemma 4 E4B base, same prompt (W7900D) | 8B / 4.5B effective | 83.0% (80.6–85.8) | 0.832 | 5 of 108 | 10.9% | 100% |
| Gemma 4 31B, teacher (Token Factory) | 30.7B dense | 97.1% (94.9–99.1) | 0.971 | 1 of 108 | 2.9% | 100% |
| DeepSeek V4 Flash | 284B / 13B | 80.4% (73.4–86.3) | 0.825 | 3 of 108 | 14.5% | 90.7% |
| Qwen3.8 Flash Next | 180B / 6B | 91.2% (89.7–92.8) | 0.913 | 1 of 108 | 8.7% | 99.3% |
| MiniCPM-V 4.7 35B-A3B | 35B / about 3B | 85.9% (79.4–90.7) | 0.852 | 3 of 108 | 13.8% | 100% |
| Clef (Cloudflare decision model) | 27B dense | 88.9% (86.1–92.7) | 0.897 | 1 of 108 | 13.0% | 100% |

How to read it:

- Ours is fine-tuned; every comparator ran zero-shot on our exact grader prompt (the one in `prompts/grader.json`), on AMD GPUs via AMD Token Factory. Clef is a decision model with its own endpoint, so it was asked one multiple-choice question per fact with the same grading criteria instead.
- Critical misses: of the 108 critical facts (medicines, warning signs) that the patient got wrong or left out, how many the system graded "understood". This is the unsafe error. Over-flag: facts the patient got right that the system sent back for re-teaching (extra nurse time, not unsafe).
- 95% intervals are case-level bootstrap. Against its base, the fine-tune fixes 41 facts and breaks 3 (exact McNemar p = 1.6e-9).
- Scoring: a fact whose verdict cannot be read counts as wrong. A lenient fallback for malformed JSON is applied identically to every system; it changes only DeepSeek V4 Flash (79.2% under the strict parser, 80.4% with the fallback). Valid JSON is reported strictly.
- Sizes are the makers' published figures (the MiniCPM active count comes from its "A3B" name only).
- Pending: GLM-5.3 Flash (320B / 18B), MiMo-V2.6 Flash (309B / 15B) and Qwen3.8 27B (27B dense) did not finish within the event's daily API budget; they will be added to this table tonight (Oct 17), whatever they score. Partial predictions are in `ml/results/`.
- On the in-distribution test set (348 answers, labels designed by the generator) the fine-tune scores 87.0% and the 31B 85.3%; the intervals overlap and the labels come from the generator, so we do not claim it beats its teacher.

The model was chosen on a separate validation set (also written by Claude) with a rule declared before the test set was scored: zero critical misses, then fewest total misses, then fewest re-checks, then accuracy. The chosen model was then scored on the test set once. A further training run (run 4, larger jury-labelled training set) is running on Oct 17; it replaces run 2 only if it beats run 2 on validation under the same rule, and this table will say which run it shows. Full details and every intermediate number: [ml/results/eval.md](ml/results/eval.md).

**Fact lock** on 24 synthetic notes in 8 languages with planted errors: 16 of 17 caught (every wrong dose, date, frequency and dropped negation); 1 missed (a dropped warning symptom). Details: [ml/results/lock_bench.md](ml/results/lock_bench.md).

**Speech recognition** on 15 real recordings (Hindi, Bengali, English; the author's own voice reading scripted answers), word error rate:

| System (both on the W7900D) | All | Hindi | Bengali | English |
|---|---|---|---|---|
| Gemma 4 E4B | 20.2% | 22.7% | 33.3% | 6.2% |
| Whisper large-v3-turbo | 38.2% | 23.9% | 91.3% | 8.6% |

Small sample (238 words); synthetic-voice results for 11 languages are in [ml/results/asr_wer_only.md](ml/results/asr_wer_only.md).

**On one W7900D**, speech recognition plus grading with the fine-tuned E4B handles about 274 discharges per hour (6 graded answers per discharge, grading batched at 64, no other load); peak memory in that batched run is 20.6 GiB of 48.

## Why

- 78% of emergency patients in one study had a gap in understanding at least one area of their discharge instructions (Engel et al., 2009, n=140).
- When researchers followed 33 patients through discharge, only 1 was asked to explain the plan back (Trivedi et al., JAMA Internal Medicine, 2023).
- Teach-back works and is fast when it happens: in a Dutch emergency department trial (n=483) the share leaving with a comprehension gap fell from 49% to 11.9%, and teach-back took 1:39 minutes against 3:11 for the usual discharge talk.
- Machine translation alone is not safe for this: in one study 8% of machine-translated Chinese discharge sentences could have caused significant harm (Khoong et al., JAMA Internal Medicine, 2019). That is why Homeward checks every fact in code.

Sources with links: [docs/research.md](docs/research.md).

## How it works

1. **Facts.** Gemma 4 31B pulls out the facts a patient must understand (medicines, changes, warning signs, follow-ups, limits) with their anchors (medicine names, doses, dates). The nurse edits and confirms them. Nothing goes further unconfirmed.
2. **Patient version.** Gemma writes the instructions in the patient's language (12 languages) and reading level.
3. **Fact lock.** An independent model (DeepSeek V4 Flash) quotes the sentences that carry each fact and back-translates them literally. Code checks that every anchor is in those sentences (with digit, unit, date and time normalisation per language), then the meaning is compared for negation, frequency, dose, timing and action. A failing fact is rewritten (up to two targeted repairs) or flagged for the nurse. The model proposes, code decides.
4. **Teach-back.** The patient answers open questions by voice, one per screen, in their own language. They see what was heard before it is graded. The grader marks each fact correct, partial, incorrect or missing, quoting what the patient said and what to re-teach.
5. **Re-teach.** Missed facts are explained again in simpler words and asked once more.
6. **Readiness board.** "Understanding confirmed" only when every critical fact is understood, a rule in code the model cannot waive. The board shows the patient's own words with an English rendering, and an audit trail of every model call.

## What runs on AMD

| Step | Hosted demo | Hospital deployment |
|---|---|---|
| Extract facts, write patient version | Gemma 4 31B on AMD GPUs via AMD Token Factory | Gemma 4 31B on hospital AMD hardware |
| Independent check (quote, back-translate, compare) | DeepSeek V4 Flash on AMD GPUs via AMD Token Factory | same, on site |
| OCR of uploaded notes | MinerU 2.5 Pro on AMD GPUs via AMD Token Factory | same |
| Listen to the patient (speech to text) | browser speech recognition (Token Factory had no working audio model during the event) | **Gemma 4 E4B on one AMD Radeon PRO W7900D** |
| Grade the teach-back | Gemma 4 31B on AMD GPUs via AMD Token Factory; the results board can also replay verdicts recorded from the fine-tuned E4B on the W7900D | **Gemma 4 E4B fine-tuned with LoRA, on the same W7900D** |
| Fine-tuning and evaluation | | AMD Radeon PRO W7900D (48 GB), ROCm 10, PyTorch 2.13 |
| Comparator models in the results table | AMD GPUs via AMD Token Factory | |

Why a small model on site: hospitals should not send patient data to a cloud API (AMD Token Factory's own terms forbid health data). One workstation GPU in the hospital that both listens and grades is the deployable version.

Training (run 2): LoRA rank 32 on Gemma 4 E4B, 2,743 training rows from 1,793 unique answers, 2 epochs, 43.5 minutes on one W7900D, peak 23.5 GiB; the epoch-1 and epoch-2 adapters are averaged and applied at LoRA scale 0.5 (chosen on validation).

## Repository

- `web/`: the app (React + Cloudflare Worker). Setup, environment variables and architecture: [web/README.md](web/README.md).
- `ml/`: synthetic data generation, labelling, LoRA fine-tuning, evaluation, speech evaluation. Details: [ml/README.md](ml/README.md).
- `ml/notebook/`: the exact scripts run on the AMD notebook (`setup.sh`, `run_day1.sh` for run 2, `run_day2.sh` for run 4).
- `prompts/grader.json`: the one grader prompt shared by training, evaluation and the app.
- `data/`: synthetic discharge cases, teach-back answers and labels, the training splits, the independent test, validation and dev sets, and speech references. All fictional.

## Reproduce

Full steps, the notebook layout and every flag: [ml/README.md](ml/README.md). In short:

```bash
# 1. Score the committed predictions (CPU, no key): reproduces the results table exactly
python3 ml/eval.py score --rows data/test_independent.jsonl \
  --system e4b_base=ml/results/preds_base_test_independent.jsonl \
  --system homeward_run2=ml/results/run2/final_preds_test_independent_tau.jsonl \
  --system gemma4_31b=ml/results/preds_gemma31b_test_independent.jsonl \
  --out /tmp/eval_independent.json --md /tmp/eval_independent.md
python3 ml/report.py --dry-run          # rebuild the eval.md summary, write nothing

# 2. Comparators, zero-shot on Token Factory (needs TOKENFACTORY_API_KEY)
uv run --with httpx python ml/eval.py predict-api --model DeepSeek-V4-Flash \
  --rows data/test_independent.jsonl --out ml/results/preds_deepseek_test_independent.jsonl

# 3. Train and predict run 2 on the AMD notebook (W7900D, ROCm 10), after copying ml/, prompts/ and data/ as in ml/README.md
bash /workspace/setup.sh
cd /workspace/code; W=/workspace; D=$W/data; M=$W/models/gemma-4-E4B-it
python train_lora.py --model $M --train $D/v2/sft_train_heardfirst.jsonl --val $D/v2/sft_val_heardfirst.jsonl \
  --out $W/runs/runB --epochs 2 --bsz 4 --accum 4 --rank 32 --alpha 32 --lr 2e-4 \
  --verdict-weight 3 --ul-lambda 0.5 --max-len 1536
python avg_adapters.py $W/runs/runB/adapter_avg $W/runs/runB/adapter_epoch1 $W/runs/runB/adapter
python eval.py predict-local --model $M --adapter $W/runs/runB/adapter_avg --lora-scale 0.5 \
  --rows $D/test_independent.jsonl --out preds_test_independent.jsonl
```

The full run-2 search (runs A, B, C, scale sweep, selection) is `ml/notebook/run_day1.sh`; run 4 is `ml/notebook/run_day2.sh`. The training data is committed (`data/splits_v2/` for run 2, `data/splits_v3/` for run 4) and can be rebuilt with `ml/build_sft_v2.py`. Once the adapter is on Hugging Face, step 3 can start from the download instead of retraining.

## Honest limits

- All patients, notes and teach-back answers are synthetic. The grader has not been tested with real patients, and the speech sample is small (15 clips, one speaker).
- The independent test set has 150 answers from 7 cases; several intervals overlap, so small differences between systems (ours vs Qwen3.8 Flash Next, for example) are not significant.
- The test, validation and dev sets were written by one model (Claude) and labelled by its design, with one logged fix (`data/test_independent.CHANGELOG.txt`). Real patient speech will be messier.
- Some comparators (DeepSeek V4 Flash, Qwen3.8 Flash Next, Gemma 4 31B, MiMo-V2.6 Flash) also wrote or labelled part of our training data. They did not see the test set.
- Comparators ran zero-shot on a prompt we wrote for our own model; a prompt tuned for each of them might score higher.
- The fine-tune learns from the 31B's labels, so it inherits the teacher's judgement and still trails it.
- Retraining on a GPU is not bit-for-bit deterministic; a new run can differ by a few facts.
- Browser speech recognition in the hosted demo sends audio to the browser vendor's service; the hospital build transcribes on site. Read-aloud depends on the device's voices.
- This is a hackathon prototype, not a medical device.

## AI assistance

- Built with Claude Code (code, scripts, documentation).
- The independent test set and the validation and dev sets were written by Claude, a different model family from the data generators, and were never used in training.
- Training data (cases and teach-back answers) was generated and labelled only by open models on AMD Token Factory: DeepSeek V4 Flash wrote the cases, and DeepSeek V4 Flash, Gemma 4 31B, Qwen3.8 Flash Next and MiMo-V2.6 Flash wrote answers; Gemma 4 31B (teacher), Qwen3.8 Flash Next and DeepSeek V4 Flash labelled them. For run 4, Cloudflare's Clef (on Token Factory) adds one label vote.
- No Claude-written data is in any training set; a hash check enforces it.

## Licence

MIT (see [LICENSE](LICENSE)). The fine-tuned adapter is also subject to the Gemma 4 base model's terms. Fonts and icons: see [web/README.md](web/README.md#licences).
