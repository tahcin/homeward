---
base_model: google/gemma-4-E4B-it
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
language:
- ar
- bn
- en
- es
- fr
- hi
- it
- pt
- ta
- tl
- vi
- zh
tags:
- lora
- peft
- gemma4
- healthcare
- medical
- patient-education
- teach-back
- multilingual
- llm-as-judge
- rocm
- amd
---

<!-- Generated in part by ml/hf/build_card.py. Every number between v:/b: markers is filled from ml/results; do not edit those by hand. Rebuild: python3 ml/hf/build_card.py --system <eval.json key>. -->

# Homeward teach-back grader: Gemma 4 E4B LoRA

A LoRA adapter for [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) that grades a patient's teach-back before hospital discharge. Given the discharge facts a patient must understand and what the patient (or a caregiver) said back in their own language, it grades every fact as **correct**, **partial**, **incorrect** or **missing**, with a short English rendering of what it heard and one sentence to re-teach.

It is the on-site grader of [Homeward](https://homeward.tahcin.tech), built for the AMD Developer Hackathon: ACT III (Health and Wellbeing track) by team Solus. Trained and evaluated on one AMD Radeon PRO W7900D (48 GB) with ROCm.

Published model: Homeward adapter, <!--v:run-->run 4<!--/v--> (eval key <!--v:system-->`homeward_run4`<!--/v-->), applied at LoRA scale <!--v:scale-->1<!--/v-->.<!--v:scale_matters--><!--/v-->

## Results at a glance

<!--b:headline-->
- Per-fact accuracy 94.7% (95% CI 91.9–97.5) on an independent test set of 150 answers and 341 facts in 12 languages. Critical facts the patient got wrong or left out but that were graded "understood": 1 of 108.
- Against the same model without the adapter: 83.0% accuracy and 5 of 108 critical misses. The adapter fixes 43 fact grades and breaks 3 (exact McNemar p = 4.6e-10).
- It keeps 97.6% of the accuracy of its teacher, Gemma 4 31B (97.1%, 1 of 108 critical misses), at about a quarter of the size (8B against 30.7B parameters). It ties it on critical misses but is below it on accuracy.
- Higher point accuracy than MiMo-V2.6 Flash (309B total, 15B active, 88.9%), DeepSeek V4 Flash (284B total, 13B active, 80.4%), Qwen3.8 Flash Next (180B total, 6B active, 91.2%), MiniCPM-V 4.7 35B-A3B (35B total, about 3B active, 85.9%), Qwen3.8 27B (27B dense, 87.7%) and Cloudflare Clef (27B dense, 88.9%). The largest of these is MiMo-V2.6 Flash (309B total, 15B active). The 95% intervals overlap for Qwen3.8 Flash Next and Cloudflare Clef, so those gaps are not conclusive.
<!--/b:headline-->

The test set is synthetic. The grader has not been tested with real patients.

## What it does

Teach-back means asking the patient to explain their instructions in their own words. Homeward asks open questions by voice, one per screen, in the patient's language, and this adapter grades each answer against the facts the nurse confirmed:

| Verdict | Meaning |
|---|---|
| correct | conveys the fact accurately enough to act on safely (paraphrase and everyday medicine names are fine if unambiguous) |
| partial | conveys part of it but leaves out or is vague about a key element; says nothing wrong |
| incorrect | says something that conflicts with the fact (wrong dose, timing, medicine, action, threshold, date or person) |
| missing | does not mention it |

When unsure between two verdicts, the prompt asks for the less reassuring one. Output is JSON only:

```json
{"facts": [{"id": "f1", "verdict": "correct|partial|incorrect|missing", "heard": "short English rendering, empty if missing", "reteach": "one short English sentence, empty if correct"}]}
```

The exact system and user prompt is in `grader_prompt.json` (the same file the app, training and evaluation use). In the app, a code rule (not the model) marks "understanding confirmed" only when every critical fact is graded correct; anything else goes back to the nurse.

Languages: Arabic, Bengali, Chinese, English, French, Hindi, Italian, Portuguese, Spanish, Tagalog, Tamil, Vietnamese.

## Intended use

- Decision support for a nurse or discharge educator: it points at which instruction a patient seems to have misunderstood, so the nurse can re-teach it.
- On-site deployment, where patient speech should not leave the hospital: the same Gemma 4 E4B model transcribes the speech (adapter off) and grades the transcript (adapter on) on one workstation GPU.
- Research on small multilingual graders for patient communication.

## Out of scope

- Not a medical device. It has no regulatory clearance and must not be used as one.
- It does not diagnose, recommend treatment or give medical advice. It only compares what a patient said with instructions a clinician already wrote and confirmed.
- It never decides that a patient may go home. The nurse decides; the model's verdicts are a prompt to check, not a sign-off.
- Not tested with real patients, real clinical notes or real hospital workflows. All training and test data is synthetic and fictional.
- Not for languages other than the 12 above, or for tasks other than grading teach-back against a given fact list.

## Training data

<!--b:data_intro-->
All training data is synthetic and fictional. Cases and patient answers were written only by open models served on AMD Token Factory; the labels come from those open models plus Cloudflare Clef, also served on Token Factory, as one extra vote. Clef is post-trained from Qwen3.8-27B; we could not confirm that its own weights are open. No Claude output and no real patient data was used for training.
<!--/b:data_intro-->

- Cases and answers: fictional discharge cases (condition, medicines, changes, warning signs, follow-ups) and spoken-style teach-back answers written to a designed per-fact plan (which facts the speaker gets right, partly right, wrong or leaves out), in 12 languages, with varied speakers (patient or caregiver) and styles (hesitant, code-mixed, no punctuation). The later answer batches record their writer: DeepSeek-V4-Flash, Qwen3.8-Flash-Next, MiMo-V2.6-Flash and Gemma-4-31B-IT, in rotation. The first batch of cases and answers does not record its writer; the generator scripts default to DeepSeek-V4-Flash.
- Teacher: Gemma-4-31B-IT re-graded the answers. Its verdicts, together with the designed labels and other models (below), decide the verdict targets. The `heard` and `reteach` targets are the 31B's wording where it is the label source, otherwise the wording of the label that won.
- Held out: whole cases (never single answers) are split, three conditions never appear in training, and every evaluation case is excluded from training.

<!--b:data-->
- 3,602 unique teach-back answers (3,738 training rows after oversampling) from fictional discharge cases in 12 languages. Unique answers by source batch: v3 2,754, v1 282, v2 566. Verdicts: correct 3,597, partial 1,509, incorrect 1,951, missing 468 facts.
- Labels: each answer was written to a designed per-fact plan. Every fact then went to a vote of the designed label, Gemma-4-31B-IT (the teacher), Qwen3.8-Flash-Next and DeepSeek-V4-Flash where they had labelled it, and Cloudflare Clef (a decision model post-trained from Qwen3.8-27B, served on Token Factory). A strict majority wins; otherwise the 31B's verdict if it is the unique plurality; otherwise the answer is dropped. A vote never turns a 31B "incorrect" or "missing" into "correct". Answers with a "correct" fact that Clef gives P(correct) < 0.2 are dropped as likely label errors. Kept by the vote: 5,092; dropped without a majority: 588; dropped as label errors: 52; dropped by a per-case cap: 1,487.
- Oversampling: 89 hard answers x2 (training-side answers where the previous adapter disagreed with agreed labels) and 76 low-confidence correct answers x1.5.
- Contamination check: passed (transcript hashes, cases, held-out conditions).
<!--/b:data-->

## Training procedure

LoRA on the attention and MLP projections of the language model only (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`); the vision and audio towers are frozen and untouched. Loss on the assistant reply tokens only.

| Setting | Value |
|---|---|
| LoRA rank / alpha / dropout | <!--v:rank-->32<!--/v--> / <!--v:alpha-->32<!--/v--> / <!--v:dropout-->0.05<!--/v--> |
| LoRA modules | <!--v:lora_mods-->258<!--/v--> |
| Trainable parameters | <!--v:trainable-->69,763,072<!--/v--> of <!--v:total_params-->8,010,863,904<!--/v--> |
| Learning rate, schedule | <!--v:lr-->0.0002<!--/v-->, linear warm-up (<!--v:warmup-->0.05<!--/v--> of steps) then cosine, AdamW, no weight decay |
| Epochs, optimiser steps | <!--v:epochs-->2<!--/v-->, <!--v:steps-->468<!--/v--> |
| Batch | <!--v:bsz-->4<!--/v--> x <!--v:accum-->4<!--/v--> gradient accumulation = <!--v:eff_bsz-->16<!--/v--> sequences per step, max length <!--v:max_len-->1536<!--/v--> tokens |
| Training rows | <!--v:rows-->3,738<!--/v--> |
| Target order | `heard` before `verdict` in each fact's JSON |
| Verdict loss weight | <!--v:vw-->3<!--/v--> on the verdict-value tokens |
| Unlikelihood term | <!--v:ul-->0.5<!--/v--> on P("correct") wherever the gold verdict is incorrect or missing (half for partial), to push against false reassurance |
| Gradient checkpointing | <!--v:ckpt-->on<!--/v--> |
| Precision | bf16 |
| Seed | <!--v:seed-->1234<!--/v--> |
| Epoch averaging | <!--v:avg-->none: the published weights are the adapter at the end of training<!--/v--> |
| LoRA scale at inference | <!--v:scale-->1<!--/v--> (each LoRA module's scaling multiplied by this before merging; chosen on validation) |
| Decoding | <!--v:decoding-->greedy decoding, one pass<!--/v-->; escalation threshold <!--v:tau-->0.5<!--/v--> (0 means no extra escalation) |

Hardware and software: one AMD Radeon PRO W7900D (48 GB, gfx1100) on the AMD hackathon notebook service, ROCm, PyTorch <!--v:torch-->2.13.0+rocm10.0.0<!--/v-->, Transformers <!--v:transformers-->5.19.0<!--/v-->, PEFT <!--v:peft-->0.21.2<!--/v-->. Training took <!--v:wall-->58.8<!--/v--> minutes at <!--v:tokps-->1,045<!--/v--> tokens/s, peak <!--v:peak-->23.6<!--/v--> GiB allocated; validation loss <!--v:val0-->0.792<!--/v--> before training, <!--v:val1-->0.331<!--/v--> after.

### Model selection

<!--b:selection-->
- Declared policy, fixed before any test scoring: safety-first: zero critical false reassurance and total FR <= base on val_independent, then fewest total FR, then fewest over-flags, then accuracy.
- 5 candidate settings (adapters, LoRA scales, escalation thresholds) were compared on a validation set written by Claude (`val_independent`, cases separate from the test set). Chosen: `run4_final` at LoRA scale 1, escalation threshold 0.5.
- The frozen choice was then scored once on the independent test set.
<!--/b:selection-->

## Evaluation

### Independent test set

<!--v:answers-->150<!--/v--> teach-back answers, <!--v:facts-->341<!--/v--> facts, <!--v:cases-->7<!--/v--> cases, <!--v:nlang-->12<!--/v--> languages. It was written by Claude (Anthropic), a different model family from every model that wrote or labelled the training data, to reduce circularity, and it was never used for training or for choosing the model. Its cases come from conditions held out of training. One relabel was made and logged before any model graded that part (`data/test_independent.CHANGELOG.txt` in the GitHub repo). Each frozen model was scored on it once. The labels are the designed labels, not human clinical judgements.

Metrics: accuracy is the share of facts whose verdict equals the designed verdict. A "critical miss passed" is a fact marked critical that the speaker got wrong or left out but that the system graded correct: the unsafe error. False reassurance is the same over all facts. Over-flag is the share of facts the speaker got right that were not graded correct (extra nurse work, not unsafe). 95% intervals are case-level bootstrap with 1,000 resamples. Unreadable verdicts count as wrong for every system.

<!--b:results_table-->
| System | Size | Ran on | Accuracy (95% CI) | Macro-F1 | Critical misses passed | False reassurance, all facts | Over-flag | Valid JSON |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 31B (teacher) | 30.7B dense | AMD Token Factory | 97.1% (94.9–99.1) | 0.971 | 1 of 108 | 0.7% | 2.9% | 100.0% |
| **Homeward adapter (run 4)** | 8B total, 4.5B effective, plus 69.8M LoRA parameters | AMD Radeon PRO W7900D | **94.7% (91.9–97.5)** | 0.946 | **1 of 108** | 1.4% | 2.9% | 100.0% |
| Qwen3.8 Flash Next | 180B total, 6B active | AMD Token Factory | 91.2% (89.7–92.8) | 0.913 | 1 of 108 | 0.7% | 8.7% | 99.3% |
| MiMo-V2.6 Flash | 309B total, 15B active | AMD Token Factory | 88.9% (86.3–91.8) | 0.891 | 1 of 108 | 0.7% | 15.9% | 100.0% |
| Cloudflare Clef | 27B dense | AMD Token Factory | 88.9% (86.1–92.7) | 0.897 | 1 of 108 | 0.7% | 13.0% | 100.0% |
| Qwen3.8 27B | 27B dense | AMD Token Factory | 87.7% (85.9–90.1) | 0.895 | 1 of 108 | 0.7% | 23.2% | 99.3% |
| MiniCPM-V 4.7 35B-A3B | 35B total, about 3B active | AMD Token Factory | 85.9% (79.4–90.7) | 0.852 | 3 of 108 | 2.2% | 13.8% | 100.0% |
| Gemma 4 E4B, base (no adapter) | 8B total, 4.5B effective | AMD Radeon PRO W7900D | 83.0% (80.6–85.8) | 0.832 | 5 of 108 | 4.3% | 10.9% | 100.0% |
| DeepSeek V4 Flash | 284B total, 13B active | AMD Token Factory | 80.4% (73.4–86.3) | 0.825 | 3 of 108 | 2.9% | 14.5% | 90.7% |
<!--/b:results_table-->

Sizes are total parameters from the makers' model cards. Gemma 4 31B is the teacher whose labels trained this adapter, so it is shown as the reference this adapter learned from rather than as a competitor. Comparators ran zero-shot on AMD Token Factory with the same grader prompt and the same scoring; only this adapter was fine-tuned for the task. Cloudflare Clef is a decision model with its own endpoint, so it was asked one multiple-choice question per fact with the same grading criteria instead of the free-text prompt. The difference between this adapter and the run 2 adapter (branch `run2`, below) is not significant on this set (exact McNemar p = 0.8). Gemma 4 E4B base and this adapter ran on the W7900D.

Per language (this adapter):

<!--b:per_language-->
| Language | Facts | Accuracy |
|---|---|---|
| Arabic | 31 | 93.5% |
| Bengali | 25 | 92.0% |
| Chinese | 30 | 96.7% |
| English | 28 | 96.4% |
| French | 30 | 96.7% |
| Hindi | 30 | 93.3% |
| Italian | 28 | 92.9% |
| Portuguese | 31 | 93.5% |
| Spanish | 27 | 96.3% |
| Tagalog | 26 | 92.3% |
| Tamil | 29 | 96.6% |
| Vietnamese | 26 | 96.2% |
<!--/b:per_language-->

Confusion matrix (this adapter, facts):

<!--b:confusion-->
| Gold \ predicted | correct | partial | incorrect | missing | unreadable |
|---|---|---|---|---|---|
| correct | 134 | 4 | 0 | 0 | 0 |
| partial | 1 | 63 | 0 | 1 | 0 |
| incorrect | 2 | 10 | 84 | 0 | 0 |
| missing | 0 | 0 | 0 | 42 | 0 |
<!--/b:confusion-->

<!--b:in_dist-->
On a second, generator-labelled test split (348 answers from held-out cases, labels written by the same models that wrote the training data, so noisier) the adapter scores 86.0% (83.3–88.7) and the 31B teacher 85.3%; the run 2 adapter scores 87.0%. The independent set above is the headline because its labels do not come from the training-data pipeline.
<!--/b:in_dist-->

### Speed on one W7900D

Bf16, LoRA merged: <!--v:aps-->1.47<!--/v--> answers per second at batch <!--v:bsz_inf-->16<!--/v-->, <!--v:bs1-->4.9 s<!--/v--> per answer at batch 1 (median), peak <!--v:vram_inf-->16.3<!--/v--> GiB allocated during the test run.

## Speech

The adapter grades text. In the on-site setup the same Gemma 4 E4B, with the adapter switched off, transcribes the patient's speech first. On <!--v:asr_clips-->15<!--/v--> real recordings (<!--v:asr_words-->238<!--/v--> words in Hindi, Bengali and English; the author's own voice reading scripted answers), word error rate was <!--v:wer_e4b-->20.2%<!--/v--> for Gemma 4 E4B against <!--v:wer_whisper-->38.2%<!--/v--> for Whisper large-v3-turbo on the same GPU (Bengali: <!--v:wer_bn_e4b-->33.3%<!--/v--> against <!--v:wer_bn_whisper-->91.3%<!--/v-->). This is a small sample from one speaker.

## Limitations

- Synthetic data only. Cases, answers and training labels come from language models, and the test labels come from the test writer's design. Real patients speak differently, and real transcripts carry recognition errors.
- Not validated with patients, nurses or clinicians, and not reviewed by native speakers in every language.
- It still passes some critical misses (see the table). In Homeward the nurse reviews every answer and a code rule blocks sign-off unless every critical fact is graded correct, but the model alone is not a safety net.
<!--b:error_direction-->
- Of its 18 wrong fact grades, 13 are more reassuring than the gold verdict and 5 less reassuring. 3 of the more reassuring ones reach "correct"; the rest stop at "partial" (for example "incorrect" graded "partial"), which in Homeward still triggers a re-teach but understates the error to the nurse.
<!--/b:error_direction-->
- The `heard` and `reteach` fields are model-written English and can be wrong; the verdict is what was measured.
- On the generator-labelled split, run 4 (86.0%) is slightly below run 2 (87.0%). Run 4 was picked by the declared validation rule before any test scoring, and neither difference between the two runs is conclusive.
- Small test set (<!--v:cases-->7<!--/v--> cases), so intervals are wide; differences that sit inside overlapping intervals are not conclusive.
<!--b:scale_limit-->

<!--/b:scale_limit-->
- Inherits the limitations and biases of Gemma 4 E4B.

## How to load

<!--b:load_intro-->
The adapter is used at its trained scale (1.0); `homeward_inference.json` records the inference settings.
<!--/b:load_intro-->

<!--b:load_snippet-->
```python
import json
import torch
import transformers
from huggingface_hub import hf_hub_download
from peft import PeftModel

BASE = "google/gemma-4-E4B-it"
ADAPTER = "oktash/homeward-teachback-gemma4-e4b"
LORA_SCALE = 1  # the evaluated setting; also in homeward_inference.json

tok = transformers.AutoProcessor.from_pretrained(BASE).tokenizer
cls = getattr(transformers, "AutoModelForMultimodalLM", None) or transformers.AutoModelForImageTextToText
# On AMD gfx1100 (Radeon PRO W7900 / W7900D, RX 7900) SDPA in the text model gave NaN logits in bf16,
# so the text model uses eager attention; the audio tower needs SDPA. Elsewhere the default is fine.
attn = {"text_config": "eager", "audio_config": "sdpa", "vision_config": "eager"}
model = cls.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto", attn_implementation=attn)

model = PeftModel.from_pretrained(model, ADAPTER)
for module in model.modules():  # apply the LoRA scale before merging
    if hasattr(module, "scaling") and isinstance(module.scaling, dict) and "default" in module.scaling:
        module.scaling["default"] *= LORA_SCALE
model = model.merge_and_unload().eval()

prompt = json.load(open(hf_hub_download(ADAPTER, "grader_prompt.json")))
facts = [  # one line per fact: id, category, critical flag, statement (fictional example)
    {"id": "f1", "category": "medication", "critical": True, "statement": "Take furosemide 40 mg once every morning."},
    {"id": "f2", "category": "warning sign", "critical": True, "statement": "Call the clinic if your weight goes up by 2 kg in 3 days."},
]
fact_lines = "\n".join(f"- {f['id']} [{f['category']}{', critical' if f['critical'] else ''}]: {f['statement']}" for f in facts)
user = prompt["user"].format(facts=fact_lines, question="How will you take your water tablet, and when do you call us?",
                             language_name="Spanish",
                             transcript="la pastilla del agua la tomo por la mañana, una, y si me hincho llamo")
messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(text, add_special_tokens=False, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=768, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# -> {"facts": [{"id": "f1", "verdict": "...", "heard": "...", "reteach": "..."}, ...]}
```
<!--/b:load_snippet-->

The reported numbers also apply an escalation rule on top of greedy decoding: for each fact graded "correct", the probability of each verdict is computed (softmax over the four verdicts' sequence log-probabilities), and if P(correct) is below the threshold (0.5) the verdict becomes the most likely other verdict, so the nurse re-checks it. It changed 2 of 341 facts on the independent test. The snippet above does not apply it; see `verdict-probs` and `escalate` in `ml/eval.py` in the GitHub repo.

Requirements used for the reported numbers: <!--v:reqs-->`transformers==5.19.0`, `peft==0.21.2`<!--/v-->, PyTorch <!--v:torch-->2.13.0+rocm10.0.0<!--/v-->. Chat template with thinking off, <!--v:decoding-->greedy decoding, one pass<!--/v-->, up to 768 new tokens.

## Files

- `adapter_model.safetensors`: the LoRA weights exactly as evaluated.
- `adapter_config.json`: the training config; the only change is `base_model_name_or_path`, set to `google/gemma-4-E4B-it` instead of the notebook's local path.
- `homeward_inference.json`: LoRA scale, decoding, attention settings and the source of each number on this card.
- `grader_prompt.json`: the grader prompt.

Branch `run2` holds the earlier run 2 adapter (an average of its epoch checkpoints), which was evaluated at LoRA scale 0.5, not 1.0: 94.1% accuracy and 2 of 108 critical misses on the independent test set. `main` (run 4) is the published model and is used at LoRA scale 1.0.

## Licence and Gemma terms

<!--v:license_text-->This adapter is released under the Apache 2.0 licence, the same licence as the base model [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it). Check the base model card for its terms.<!--/v-->

The Homeward app and training code are MIT-licensed in the GitHub repository.

## Links

- Code, data generators, evaluation and full results: [github.com/tahcin/homeward](https://github.com/tahcin/homeward) (see `ml/results/eval.md`)
- Live demo with synthetic patients: [homeward.tahcin.tech](https://homeward.tahcin.tech)
- Base model: [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)

## Citation

```bibtex
@misc{homeward2026,
  title  = {Homeward teach-back grader: a Gemma 4 E4B LoRA for multilingual discharge teach-back},
  author = {Team Solus},
  year   = {2026},
  note   = {AMD Developer Hackathon: ACT III},
  url    = {https://huggingface.co/oktash/homeward-teachback-gemma4-e4b}
}
```

## AI assistance

Built with AI coding assistance (Claude). Training data was generated and labelled by models on AMD Token Factory (see Training data); the independent validation and test sets were written by Claude and never used for training.
