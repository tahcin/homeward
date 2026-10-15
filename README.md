# Homeward

**Make sure patients understand before they go home.**

Homeward rewrites hospital discharge instructions in the patient's language, with every dose and date verifiably carried over or flagged to the nurse. It then listens to the patient (or their caregiver) explain the instructions back, and tells the nurse exactly which instruction was misunderstood.

- Live demo (synthetic patients, no login): https://homeward.tahcin.tech
- Built for the AMD Developer Hackathon: ACT III, Health and Wellbeing track, by team Solus.

> Hosted demo: synthetic data only, not medical advice. Homeward never diagnoses or recommends treatment; it only restates the clinician's own instructions and checks they were understood.

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
4. **Teach-back.** The patient answers open questions by voice, one per screen, in their own language. They see what was heard before it is graded. Gemma grades each fact as correct, partial, incorrect or missing, quoting what the patient said and what to re-teach.
5. **Re-teach.** Missed facts are explained again in simpler words and asked once more.
6. **Readiness board.** "Understanding confirmed" only when every critical fact is understood, a rule in code the model cannot waive. The board shows the patient's own words with an English rendering, and an audit trail of every model call.

## What runs on AMD

| Step | Hosted demo | Hospital deployment |
|---|---|---|
| Extract facts, write patient version, grade | Gemma 4 31B on AMD GPUs (AMD Token Factory) | Gemma 4 31B on hospital AMD hardware, or the fine-tuned E4B for grading |
| Independent check (quote, back-translate, compare) | DeepSeek V4 Flash on AMD GPUs (Token Factory) | same, on site |
| OCR of uploaded notes | MinerU 2.5 Pro on AMD (Token Factory) | same |
| Listen to the patient (speech to text) | browser speech recognition (Token Factory had no working audio model during the event) | **Gemma 4 E4B on one AMD Radeon PRO W7900D** |
| Grade the teach-back | Gemma 4 31B | **Gemma 4 E4B fine-tuned with LoRA on the AMD Radeon PRO W7900D** |
| Fine-tuning and evaluation | | AMD Radeon PRO W7900D (48 GB), ROCm 10, PyTorch 2.13 |

Why a small model on site: hospitals should not send patient data to a cloud API (AMD Token Factory's own terms forbid health data). One workstation GPU in the hospital that both listens and grades is the deployable version.

## Results

All numbers are measured; details, confidence intervals and commands are in [ml/results/eval.md](ml/results/eval.md).

**Teach-back grading** on an independent test set the training-data generator never saw ([N] answers, [N] facts, 12 languages, written by a different model family with messy, code-mixed phrasing):

| System | Per-fact accuracy | Macro-F1 | Critical misses graded "understood" (false reassurance) |
|---|---|---|---|
| Gemma 4 E4B base (W7900D) | [ ] | [ ] | [ ] |
| **Gemma 4 E4B fine-tuned (W7900D)** | [ ] | [ ] | [ ] |
| Gemma 4 31B (Token Factory) | [ ] | [ ] | [ ] |
| DeepSeek V4 Flash (Token Factory) | [ ] | [ ] | [ ] |

**Speech recognition** on 15 real recordings (Hindi, Bengali, English; the author's own voice reading scripted answers), word error rate:

| System (both on the W7900D) | All | Hindi | Bengali | English |
|---|---|---|---|---|
| Gemma 4 E4B | 20.2% | 22.7% | 33.3% | 6.2% |
| Whisper large-v3-turbo | 38.2% | 23.9% | 91.3% | 8.6% |

Small sample (238 words); synthetic-voice results for 11 languages are in [ml/results/asr_wer_only.md](ml/results/asr_wer_only.md).

## Repository

- `web/`: the app (React + Cloudflare Worker). Setup, environment variables and architecture: [web/README.md](web/README.md).
- `ml/`: synthetic data generation, label verification, LoRA fine-tuning, evaluation, speech evaluation.
- `prompts/grader.json`: the one grader prompt shared by training, evaluation and the app.
- `data/`: synthetic discharge cases, teach-back answers, the independent test set, and speech references. All fictional.

## Reproduce

```bash
# 1. synthetic data on AMD Token Factory (needs TOKENFACTORY_API_KEY in the environment)
cd ml
uv run --with httpx python gen_cases.py --n 72 --out ../data/cases.jsonl
uv run --with httpx python gen_answers.py --cases ../data/cases.jsonl --out ../data/answers.jsonl
uv run --with httpx python verify.py            # blind second-model labels
uv run --with httpx python build_sft.py         # case-level split, held-out conditions

# 2. on the AMD GPU (ROCm 10, PyTorch 2.13): see ml/notebook/setup.sh and ml/notebook/run_day1.sh
python train_lora.py
python eval.py predict-local --adapter <run>
python eval.py score
```

## Honest limits

- The data is synthetic. The grader has not been tested with real patients, and the speech sample is small.
- Browser speech recognition in the hosted demo sends audio to the browser vendor's service; the hospital build transcribes on site.
- Read-aloud depends on the device's voices.
- This is a hackathon prototype, not a medical device.

## AI assistance

Built with AI coding assistance (Claude). Synthetic data was generated with models on AMD Token Factory. The independent test set was written by Claude, a different model family from the data generator, to reduce circularity.

## Licence

MIT. Fonts and icons: see [web/README.md](web/README.md#licences).
