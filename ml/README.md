# Homeward ML: data, fine-tuning and evaluation

Everything behind the grader numbers in the main [README](../README.md). Results live in [results/](results/); the summary is [results/eval.md](results/eval.md) and the machine-readable version is [results/eval.json](results/eval.json). Both are written by `report.py` from the per-set score files, so no number in them is typed by hand.

## What is where

| Path | What |
|---|---|
| `../prompts/grader.json` | the one grader prompt used by training, every evaluated model and the app |
| `tf.py` | AMD Token Factory client (OpenAI-compatible). Reads `TOKENFACTORY_API_KEY`; optional `TOKENFACTORY_BASE_URL`, `TF_RPM`, `TF_CONCURRENCY` |
| `gen_cases.py`, `gen_answers.py` | synthetic discharge cases and teach-back answers with designed per-fact labels, written by open models on Token Factory |
| `verify.py` | blind second labels from another model (Qwen3.8-Flash-Next by default; also used with `--model Gemma-4-31B-IT` for the teacher labels and DeepSeek-V4-Flash as an extra juror) |
| `build_sft.py` | run-1 split: case-level train/val/test, held-out conditions (`data/splits/`) |
| `build_sft_v2.py` | run-2 and run-4 training sets: teacher and jury labelling, dedup, case cap, oversampling, contamination check (`data/splits_v2/`, `data/splits_v3/`) |
| `train_lora.py` | LoRA fine-tune of Gemma 4 E4B (verdict-token loss weight, unlikelihood on wrong "correct" calls), saves an adapter per epoch |
| `avg_adapters.py` | averages LoRA adapters (epoch averaging) |
| `eval.py` | `predict-local` (E4B on the GPU), `predict-api` (any Token Factory model, zero-shot, same prompt), `verdict-probs`, `escalate`, `vote`, `score` |
| `select_model.py` | declared safety-first model selection on `val_independent` only |
| `report.py` | builds `results/eval.{json,md}` and fills `web/public/eval.json` |
| `asr_eval.py`, `onprem_demo.py`, `onprem_replay.py`, `serve.py` | speech evaluation, the on-site demo, replay files for the web app, a small local server |
| `notebook/` | the exact scripts run on the AMD notebook: `setup.sh`, `run_day1.sh` (runs A, B, C and the run-2 selection), `run_day2.sh` (run 4), `run_day2b.sh` (run 4 continuation with per-load timeouts and retries), `fhwrap.py` (faulthandler wrapper that dumps a stalled load's stack), `sitter.sh` (watchdog that kills a load stuck for 2 minutes), `demo.sh` |
| `hf/` | Hugging Face model card (`build_card.py` fills it from `results/`), upload script and publishing notes |

Data (all fictional, in `../data/`):

| File | What |
|---|---|
| `cases.jsonl` (72), `cases_v2.jsonl` | synthetic discharge cases |
| `answers.jsonl`, `answers_v2.jsonl`, `answers_v3.jsonl`, `answers_targeted_t1.jsonl`, `answers_targeted_t2.jsonl` | generated teach-back answers with designed labels |
| `verify*.jsonl`, `teacher_31b*.jsonl`, `juror_ds_v3.jsonl`, `juror_clef.jsonl` | model labels used for filtering and voting (Qwen3.8-Flash-Next, Gemma-4-31B-IT, DeepSeek-V4-Flash, Clef) |
| `test_independent.jsonl` (150 answers), `val_independent.jsonl` (120), `dev_independent.jsonl` (60) | written by Claude, a different model family from the generators; never used for training. `test_independent.CHANGELOG.txt` logs the one label fix, made before any model graded those rows |
| `splits/` | run-1 split; `splits/test.jsonl` is the in-distribution test set (348 answers, generator-labelled) |
| `splits_v2/` | run-2 training set |
| `splits_v3/` | run-4 training set (jury labels incl. Clef; the final model) |

## Environment

- Data generation, labelling, API baselines and scoring: any machine with Python 3.11+ and `uv` (`uv run --with httpx ...`), plus `TOKENFACTORY_API_KEY` for anything that calls Token Factory. Scoring and `report.py` need no key and no GPU.
- Training and local inference: the AMD Developer Cloud notebook, one AMD Radeon PRO W7900D (48 GB), ROCm 10, PyTorch 2.13 (`2.13.0+rocm10.0.0`). `notebook/setup.sh` installs `transformers==5.19.0 peft==0.21.2 accelerate==1.15.0` (run it at the start of every session). Use eager attention for text on gfx1100 (the code does this).

The notebook scripts expect this layout under the persistent `/workspace`:

```bash
mkdir -p /workspace/{code,data/v2,data/v3,logs,runs,models/gemma-4-E4B-it}
cp ml/*.py prompts/grader.json /workspace/code/
cp ml/notebook/*.sh /workspace/
cp data/splits_v2/*.jsonl /workspace/data/v2/
cp data/splits_v3/*.jsonl /workspace/data/v3/
cp data/val_independent.jsonl data/test_independent.jsonl /workspace/data/
cp data/splits/test.jsonl /workspace/data/test.jsonl
```

`run_day1.sh` downloads Gemma 4 E4B (`google/gemma-4-E4B-it`) into `/workspace/models/gemma-4-E4B-it` if it is missing (ModelScope mirror first, Hugging Face fallback).

## Reproduce run 2 (earlier model, branch `run2` on Hugging Face)

1. Training set. `data/splits_v2/` is committed. To rebuild it from the committed answers and labels:

   ```bash
   python3 ml/build_sft_v2.py --out data/splits_v2
   ```

   Labels: an answer is kept when Gemma-4-31B-IT agrees with the designed label on every fact; otherwise a per-fact 2-of-3 vote of designed, 31B and Qwen3.8-Flash-Next labels; answers with no majority are dropped. The contamination check (transcript hashes, cases, held-out conditions) must print "passed".

2. Train and average (on the notebook). This is run B from `run_day1.sh`, the one the selection picked:

   ```bash
   cd /workspace/code; W=/workspace; D=$W/data; M=$W/models/gemma-4-E4B-it
   python train_lora.py --model $M --train $D/v2/sft_train_heardfirst.jsonl --val $D/v2/sft_val_heardfirst.jsonl \
     --out $W/runs/runB --epochs 2 --bsz 4 --accum 4 --rank 32 --alpha 32 --lr 2e-4 \
     --verdict-weight 3 --ul-lambda 0.5 --max-len 1536
   python avg_adapters.py $W/runs/runB/adapter_avg $W/runs/runB/adapter_epoch1 $W/runs/runB/adapter
   ```

   The full search that led to this choice (runs A, B and C, LoRA-scale sweep, selection) is `bash /workspace/run_day1.sh`. It also lists the run-1 adapter as a candidate; remove that line if you have not trained run 1.

3. Selection (validation only). `select_model.py --rows $D/val_independent.jsonl --cand name:adapter:scale:preds:probs ...` applies the declared rule: zero critical false reassurance and total false reassurance no worse than base E4B, then fewest false reassurances, then fewest over-flags, then highest accuracy. It chose the epoch-averaged run-B adapter at LoRA scale 0.5 with no escalation (tau 0); see `results/run2/select_final.json`.

4. Predict on the test sets once with the frozen choice:

   ```bash
   python eval.py predict-local --model $M --adapter $W/runs/runB/adapter_avg --lora-scale 0.5 --name homeward_run2 \
     --rows $D/test_independent.jsonl --out preds_test_independent.jsonl --bsz 16 --latency-n 20
   ```

   Our predictions are committed as `results/run2/final_preds_test_independent_tau.jsonl` (and `final_preds_test_tau.jsonl` for the in-distribution set). GPU kernels are not bit-for-bit deterministic, so a retrained adapter can differ by a few facts.

## Reproduce run 4 (the final model)

Published as https://huggingface.co/oktash/homeward-teachback-gemma4-e4b (`main`, LoRA scale 1.0; Apache 2.0). Branch `run2` holds run 2's averaged adapter, which is evaluated at LoRA scale 0.5.

1. Training set. `data/splits_v3/` is committed. It was built with `build_sft_v2.py` over all answer files: Gemma-4-31B-IT teacher labels, DeepSeek-V4-Flash juror labels where present (`--juror`), Clef as one more ordinary vote plus a label-error filter (`--juror-clef`: answers with a "correct" fact that Clef gives P(correct) < 0.2 are dropped), and oversampling of training answers run 2 got wrong (`--mine`, `--mine-probs`). A strict majority wins, else the 31B verdict if it is the unique plurality, else the answer is dropped; a vote never turns a 31B "incorrect" or "missing" into "correct". Clef is Cloudflare's decision model, post-trained from Qwen3.8-27B and served on Token Factory; we could not confirm that its own weights are open. `data/splits_v3/stats.json` records the counts (3,602 unique answers, 3,738 rows) and the contamination check.

2. Train (on the notebook), same recipe as run 2:

   ```bash
   cd /workspace/code; W=/workspace; D=$W/data; M=$W/models/gemma-4-E4B-it
   python train_lora.py --model $M --train $D/v3/sft_train_heardfirst.jsonl --val $D/v3/sft_val_heardfirst.jsonl \
     --out $W/runs/run4 --epochs 2 --bsz 4 --accum 4 --rank 32 --alpha 32 --lr 2e-4 \
     --verdict-weight 3 --ul-lambda 0.5 --max-len 1536
   ```

   468 steps in 58.8 minutes, about 1,045 tokens/s, peak 23.6 GiB; validation loss 0.792 to 0.331 (`results/run4/train_log.json`).

3. Selection (validation only). `run_day2.sh` evaluates the candidates on `val_independent` (the final adapter at scales 1.0 and 0.5, the epoch average at 0.5, a self-consistency variant, and run 2's frozen model) with `select_model.py` and the same declared rule. It chose the final (epoch-2) adapter at LoRA scale 1.0 with escalation threshold tau 0.5: 93.0% on validation, zero false reassurance, 2 over-flags (run 2: 91.5%). See `results/run4/select_final.json` and the candidates table in `results/eval.md`.

4. Predict on the test sets once with the frozen choice, then apply the escalation rule (a fact graded "correct" with P(correct) < tau becomes the most likely other verdict, so the nurse re-checks it):

   ```bash
   A=$W/runs/run4/adapter
   python eval.py predict-local --model $M --adapter $A --lora-scale 1.0 --name homeward_run4 \
     --rows $D/test_independent.jsonl --out preds_test_independent.jsonl --bsz 16 --latency-n 20
   python eval.py verdict-probs --model $M --adapter $A --lora-scale 1.0 --rows $D/test_independent.jsonl \
     --preds preds_test_independent.jsonl --out probs_test_independent.jsonl
   python eval.py escalate --preds preds_test_independent.jsonl --probs probs_test_independent.jsonl --tau 0.5 \
     --out preds_test_independent_final.jsonl
   ```

   Committed as `results/run4/preds_test_independent_final.jsonl` (and `preds_test_final.jsonl` for the in-distribution set); the escalation changed 2 of 341 facts on the independent test.

Notebook stalls: on Oct 17 about half of the model loads after training stalled in a host-to-GPU copy. `HSA_ENABLE_SDMA=0` and loading via CPU (now the default in `train_lora.load_model_and_tokenizer`; `LOAD_VIA_CPU=0` turns it off) did not prevent it. What worked: `run_day2b.sh`, which wraps each model call in a timeout with one retry, runs it under `fhwrap.py` so a stalled load prints its stack, and `sitter.sh`, which kills a load stuck for 2 minutes. Budget about 3 minutes per stalled load. Two planned candidates (epoch average at scale 1.0, scale 0.75) did not run because of these stalls and the GPU window.

## Comparators (API, zero-shot)

Every Token Factory model gets the same grader prompt and the same rows, with no fine-tuning and no examples:

```bash
uv run --with httpx python ml/eval.py predict-api --model Qwen3.8-Flash-Next --reasoning none \
  --rows data/test_independent.jsonl --out ml/results/preds_qwen3_8_flash_next_test_independent.jsonl
```

The per-model settings (`reasoning_effort`, JSON mode) are saved next to each prediction file in `*.meta.json`. Clef is the exception: it is a decision model served on Token Factory's `/systemone` endpoint, so it was asked one multiple-choice question per fact (correct, partial, incorrect, missing) with the grader's criteria, not the free-text grader prompt.

## Score and report

Scoring runs on CPU and needs no key. This reproduces the independent-test table exactly from the committed predictions:

```bash
R=ml/results
python3 ml/eval.py score --rows data/test_independent.jsonl \
  --system e4b_base=$R/preds_base_test_independent.jsonl \
  --system homeward_run2=$R/run2/final_preds_test_independent_tau.jsonl \
  --system homeward_run4=$R/run4/preds_test_independent_final.jsonl \
  --system gemma4_31b=$R/preds_gemma31b_test_independent.jsonl \
  --system deepseek_v4_flash=$R/preds_deepseek_test_independent.jsonl \
  --system qwen3_8_flash_next=$R/preds_qwen3_8_flash_next_test_independent.jsonl \
  --system minicpm_v_4_7_35b_a3b=$R/preds_minicpm_v_4_7_35b_a3b_test_independent.jsonl \
  --system clef=$R/preds_clef_test_independent.jsonl \
  --out /tmp/eval_independent.json --md /tmp/eval_independent.md
```

(In zsh, pass the `--system` flags literally or from an array; zsh does not word-split a string variable.)

Then assemble the summary:

```bash
python3 ml/report.py --dry-run                       # print eval.md, write nothing
python3 ml/report.py --gpu-minutes 529 --ours homeward_run4 --splits-stats data/splits_v3/stats.json
                                                     # write results/eval.{json,md} and web/public/eval.json (as committed)
```

`report.py` flags: `--ours` (system key of the fine-tuned grader, default `homeward_run2`), `--splits-stats` (split stats for the recipe section, default `data/splits_v2/stats.json`, relative to the app root), `--web` (the app's eval file), `--answers-per-discharge` (default 6), `--gpu-minutes`, `--dry-run`.

## Metrics

- Per-fact accuracy over all facts; a fact whose verdict cannot be read counts as wrong. A lenient fallback reads id and verdict with a regex when the JSON is malformed; it is applied identically to every system and changes only DeepSeek-V4-Flash (79.2% strict, 80.4% lenient, on the independent test). JSON validity is reported strictly.
- False reassurance: a fact whose gold label is incorrect or missing, graded "correct" (the unsafe error). Critical misses: the same, restricted to facts flagged critical (108 critical facts with an incorrect or missing gold label on the independent test).
- Over-flag: a fact whose gold label is correct, graded anything else (extra nurse work, not unsafe).
- 95% intervals: case-level bootstrap, 1,000 resamples. Paired comparisons with base E4B: exact McNemar test on discordant facts.
