# Run 5 (Sun Oct 18, GPU window from ~11:20 IST)

Written Sat Oct 17 evening, before any run 5 training. The recipe, the candidates, the selection rule and the cutoffs below are fixed now; nothing in them may change after seeing run 5 numbers.

## Recipe: one change, more labelled data

- **Data: splits_v4** = run 4's pool and run 4's label rule (jury vote of the designed label, Gemma-4-31B-IT, Qwen3.8-Flash-Next or DeepSeek-V4-Flash where present, and Clef; Clef label-error filter; mined and low-confidence upweighting as in run 4) **without the per-case cap of 45**. That cap dropped 1,487 jury-approved answers from splits_v3 only to keep run 4 inside its time budget. Without it: 5,089 unique answers (run 4: 3,602, +41%), 5,243 rows, 83 cases, 23 to 87 answers per case (no case dominates). Contamination check passes (no case, held-out condition or transcript of any evaluation set).
- **Everything else as run 4**: heard-first target, LoRA r32/alpha 32 on the language model, lr 2e-4 cosine, bsz 4 x accum 4 with checkpointing, verdict weight 3, unlikelihood 0.5, max_len 1536, seed 1234. **2 epochs** (about 83 min at run 4's measured 0.476 s per row-epoch), fewer automatically if the launch is late (see cutoffs).
- **Why this change:** the run 2 to run 4 gain came from more and better-labelled data, and this is the largest block of already-labelled data we are not using; it needs no new labels, no new code path in the loss, and no test or val data. The generator-val loss rose after epoch 1 (0.304 to 0.331), so the run keeps 2 epochs at most and checkpoints at epochs 1, 1.5 and 2: more unique data per epoch instead of more repeats, and the selection can pick an earlier or averaged checkpoint if epoch 2 overfits.
- **Optional, tonight after the 21:41 baselines (lead's call):** `DS=1 bash ml/notebook/prep_run5.sh` adds DeepSeek-V4-Flash juror votes on up to 800 contested answers that lack one (about $0.0001 per answer, so under $0.10; spend-gated at $0.70 so the baselines and the sample re-record keep priority; 738 contested answers had no DeepSeek vote at 15:00). Same declared jury rule, just more votes; it can rescue some of the 588 no-majority answers. Without it, `bash ml/notebook/prep_run5.sh` (no API) builds the same splits_v4 as now. Either way, run prep once more on Sunday before launching so splits_v4 matches the label files.

## Candidates (val_independent only), all in the training process

Listed in selection order; the incumbent goes first, so on a full tie run 4 stays.

| Order | Name | Weights | LoRA scale |
|---|---|---|---|
| 0 | run4_final (incumbent, existing val preds/probs, not re-run) | run 4 final adapter | 1.0 |
| 1 | run5_final | epoch-2 adapter | 1.0 |
| 2 | run5_avg | mean of epoch-1 and epoch-2 adapters | 1.0 |
| 3 | run5_final_s075 | epoch-2 adapter | 0.75 |
| 4 | run5_avg_late | mean of epoch-1.5 and epoch-2 adapters | 1.0 |
| 5 | run5_avg_s075 | mean of epoch-1 and epoch-2 adapters | 0.75 |
| 6 | run5_ep1 | epoch-1 adapter | 1.0 |

Each candidate is tau-swept on its own val probabilities by `select_model.py` (unchanged). Candidates are evaluated in this order and the loop stops when the next one would not finish before the pick deadline.

## Selection rule (declared, unchanged, val only)

`select_model.py --base-fr 2`: zero critical false reassurance and total false reassurance no worse than base E4B on val_independent (2), then fewest total false reassurances, then fewest over-flags, then highest accuracy, then the earlier-listed candidate, then the smaller tau. The incumbent is run 4 (val 93.0%, FR 0, critical 0, over-flags 2 at tau 0.5): a run 5 candidate replaces it only if it wins under this rule. If the pick is run 4, the test sets are NOT scored again.

## Speed and stall-proofing

- **One model load per process.** `ml/run5.py` loads E4B once, trains (`train_lora.run_training`), keeps each checkpoint's LoRA weights in GPU memory, and evaluates every candidate by swapping LoRA weights in place: PEFT's own merge into the bf16 base (same arithmetic as eval.py's `load_adapter`), base restored from a GPU copy after each candidate (about 8 GiB extra, fits in 48 GiB). No host-to-GPU model copy after the first load. The frozen pick's test scoring runs in the same process.
- **Batching:** generation at batch 64 (run 4's capacity run: 20.6 GiB, 2.42 answers/s vs 1.55 at batch 16), verdict probabilities at batch 8 (was 4; batch 16 ran out of memory in the GPU smoke, see below). A batch that runs out of memory is split in half and retried. Note: run 5's numbers are produced at batch 64 for every candidate and the test, so all run 5 candidates are compared under one setting; run 4's val numbers stay as frozen on Oct 17 (batch 16).
- **Stalls:** the first load has a faulthandler timer (150 s, exit and stack dump to `logs/fh_run5_<pid>.txt`); `run_day3.sh` restarts run5.py, which resumes from its files (training not repeated once an adapter exists; finished candidates skipped; a written pick is never re-selected). A second watchdog kills run5.py if `logs/run5.log` is silent for 6 min. Post steps (replay, demo, capacity) are separate processes with `fhwrap.py`, timeouts, one retry and `sitter.sh`.
- **Check built in:** the capacity step reloads the frozen pick in a fresh process with eval.py's normal path and logs how many of the 348 in-distribution outputs are identical to the in-process ones (expect all).

## Hard cutoffs (IST, in run_day3.sh; override with env TRAIN_DL / PICK_DL / HARD_DL)

| Cutoff | Default | What happens |
|---|---|---|
| Training | 13:05 | Epochs set at launch so 2 epochs end 4 min before it, else fewer (min 1). During training the cosine horizon shrinks to the measured step time; at 13:05 minus 90 s training stops and saves whatever it has. |
| Pick | 13:40 | No further candidate starts if it would not finish by then (3 min each). If the selection has not been made by 13:40: `final/ABORTED`, run 4 stays, no test scoring. |
| Hard | 14:05 | run5.py is killed by `timeout`; post steps skip anything that cannot finish by then. 10+ min buffer before the 3 h quota ends (~14:20). |

## GPU smoke (Sat Oct 17, 15:40 IST, the day's last 5 quota minutes)

The day3 bundle was uploaded and unpacked on /workspace (pre-staged; Sunday's launch uploads it again), setup.sh ran, then `smoke5.py` used run5.py's own path: one load with the 150 s timer, 3 training steps on splits_v4 rows, the in-process swapper, one batch-64 generation pass on 64 val rows, one verdict-probabilities pass. All in 111 s, no stall.

| Measured on the W7900D | Value |
|---|---|
| Model load (run5 path, via CPU) | 22.8 s; 14.8 GiB allocated |
| Training steps (bsz 4 x accum 4, checkpointing) | 9.5, 7.7, 7.0 s; 910-1,052 tok/s; peak 21.2 GiB (run 4 averaged 7.5 s per step) |
| After the swapper (base copy on the GPU) | 22.7 GiB allocated (copy about 7.9 GiB) |
| Generation, 64 val rows at batch 64 (incl. warm-up) | 42.7 s, 1.65 answers/s, 202 tok/s, peak 28.6 GiB, 64 of 64 replies parse |
| Verdict probabilities, 16 answers at batch 16 | ROCm OOM on a 5.7 GB logits buffer (all-position logits over the 262k vocabulary); the halving fallback finished it (37 slots in 15.4 s, peak 34.5 GiB). Changed to batch 8 for run 5. |

Consequences: a candidate takes about 3 min (generation about 75 s for 120 rows, verdict probabilities about 100 s), so six take about 18 min; the training cutoff moved to 13:05 so a launch at ~11:32 still gets 2 full epochs. The quota resets at ~11:29-11:30 IST Sunday (dashboard at 15:43: "Quota resets in 19h 46m"), so the 3 h end at about 14:30 and HARD_DL 14:05 keeps 25 min of buffer. Also: GNU tar on the notebook cannot chown to the Mac uid (harmless warnings); launch_day3.sh now extracts with `--no-same-owner` and drops `__pycache__`.

## Timeline (measured speeds; launch at 11:32)

| IST | Step |
|---|---|
| ~11:29-11:30 | quota resets; lead opens the notebook (login), confirms 3 h |
| 11:32 | `launch_day3.sh <targetId>`: one bundle upload (2 MB), checksum, start |
| 11:33 | setup.sh (15 s), one model load (23 s; a stall costs about 3 min) |
| 11:34 | tokenise 5,243 rows, step-0 val loss (about 1 min) |
| 11:35-12:58 | train 656 steps (about 7.5 s each); checkpoints about 12:17 (epoch 1), 12:37 (1.5), 12:58 (2) |
| 12:59-13:17 | 6 candidates on val (about 3 min each: swap, generate at 64, verdict probabilities at 8) |
| ~13:17 | frozen pick (seconds) |
| 13:17-13:27 | if run 5 won: test_independent once (incl. 20 batch-1 latency answers), then the in-distribution test |
| 13:27-13:40 | pack results + adapter parts, replay JSON, on-site demo, fresh-process capacity |
| 13:40-13:50 | Mac: `fetch_run5.sh`, `finish_run5.sh`, then `./promote.sh homeward_run5 <gpu_min>` |
| by ~13:30 | frozen pick and its single test score exist (one stalled load adds about 3 min; the 13:40 pick cutoff still holds) |

## Commands

Saturday night (Mac, from `app/`):
```bash
bash ml/notebook/prep_run5.sh                 # no API; or DS=1 ... after the baselines, see above
```
Sunday (Mac). TL = `/private/tmp/claude-502/-Users-naresh-Desktop-Personal-hackathons/014b671e-96ac-4951-83e1-57ca83e83267/scratchpad/ml/tools`:
```bash
cd app && bash ml/notebook/prep_run5.sh       # rebuild splits_v4 from the label files as they are now
bash $TL/launch_day3.sh <targetId> --dry      # builds and checks the bundle, uploads nothing
bash $TL/launch_day3.sh <targetId>            # upload, checksum, start run_day3.sh (nohup)
# watch:
JT=w node $TL/jup.mjs <targetId> run 'tail -5 /workspace/logs/day3.log; grep -E "step [0-9]+/|DEADLINE|candidate|FROZEN|TEST" /workspace/logs/run5.log | tail -4' 30
# after DAY3DONE_RUN5 (or DAY3_KEEP_RUN4 / ABORTED: nothing to fetch beyond the logs):
bash $TL/fetch_run5.sh <targetId>             # results -> app/ml/results/run5/, adapter parts -> lead scratchpad up/run5_adapter (sha256 checked)
bash ml/notebook/finish_run5.sh               # adds homeward_run5 to eval_independent/in_distribution json+md, run 5 vs run 4 paired file
cd .. && ./promote.sh homeward_run5 <total GPU minutes>   # then --deploy
```
Hugging Face: the notebook proxy answers 503 to the Hub, so upload from the Mac (`ml/hf/hf_upload.py`) with the fetched adapter and a new write token (PUBLISH.md section 5). Only for the winner.

Turn the GPU off when done (`hackathon-turnoff` then `hackathon-confirm-turnoff`), or keep it for the W7900D demo clip.

## Tested on the Mac (CPU, tiny random Gemma 4)

Dry runs on the Mac CPU with a tiny random Gemma 4 (same transformers classes, real tokenizer, KV-shared layers) and a torch.cuda shim; scripts and logs in `/private/tmp/claude-502/-Users-naresh-Desktop-Personal-hackathons/7fad4047-14ab-4fda-b39b-18cf1f970800/scratchpad/run5/`:
- `run_day3.sh` end to end with GNU tools: train with checkpoints at epochs 1, 1.5, 2, the six in-process candidates, selection, the frozen pick on both test sets once, pack, replay, capacity; run5.py killed mid-candidates and restarted by the driver, which resumed without retraining and skipped finished candidates; the fresh-process capacity outputs were identical to the in-process ones (3 of 3). The demo failed there only because the fake workspace has no audio (its output is now moved aside on failure).
- In-process swap vs eval.py's fresh load + `load_adapter`: bit-identical merged weights and logits for checkpoints, averages and scales 0.75 and 500; base restored exactly.
- Refactored eval.py vs the previous eval.py: identical predict-local outputs and verdict probabilities (CLI unchanged).
- run 4 as incumbent is kept on a tie and the test is then skipped; pick deadline passed gives `ABORTED`; interrupted training evaluates the latest checkpoint; the training hard stop at the deadline saves the adapter and goes on to candidates.
- `report.py`: run 4 output unchanged (`./promote.sh --dry-run` passes); run 5 sections render from a plumbing test. `ml/add_system.py` reproduces run 4's eval entries exactly. `launch_day3.sh --dry` builds and checks the bundle.

Not testable without the GPU: real step times, ROCm memory at batch 64 with the 8 GiB base copy (an OOM halves the batch), the cosine-horizon shrink (needs 10+ real steps), and the notebook transfer commands in `fetch_run5.sh`.

## Result (Sun Oct 18, written after the run): run 4 stays

- **Launch was late.** The Mac rebooted at about 07:30 IST and wiped the scratchpad tools, so run 5 started at 12:47:47 IST instead of ~11:32, with deadlines moved to train 13:40, pick 14:05, hard 14:20. `run_day3.sh` sized training to the time left: **1.16 epochs** (380 steps, not 2 epochs), so there is no epoch-1.5 checkpoint and `run5_avg_late` was skipped. Data: splits_v4 rebuilt Sunday, 5,073 unique answers / 5,228 rows, without the optional DeepSeek votes.
- **Training:** one model load (23.4 s), no stall, no restart. 380 steps in 47.8 min (2,869.8 s), ~1,038 tok/s, peak 23.9 GiB allocated. Generator-val loss 0.792 (step 0) -> 0.307 (epoch 1) -> 0.313 (final, epoch 1.16). Three transient HIP allocator OOM warnings were retried by the allocator; no step failed.
- **Candidates on val_independent** (in process, ~164 s each), after the tau sweep:

| Candidate | Scale | Accuracy | False reassurance | Critical | Over-flags |
|---|---|---|---|---|---|
| run4_final (incumbent, frozen Oct 17) | 1.0, tau 0.5 | 93.0% | 0 | 0 | 2 |
| run5_final_s075 | 0.75 | 94.0% | 1 | 1 | 1 |
| run5_avg_s075 | 0.75 | 94.0% | 1 | 1 | 1 |
| run5_final | 1.0 | 92.6% | 1 | 1 | 2 |
| run5_avg | 1.0 | 92.6% | 1 | 1 | 2 |
| run5_ep1 | 1.0 | 92.3% | 1 | 1 | 2 |

- **Frozen pick (13:50:28 IST): run4_final**, by the declared rule. Run 5 at scale 0.75 reached the highest val accuracy (94.0% vs 93.0%) but every run 5 candidate passed one critical false reassurance on val, and the rule requires zero, so run 4 was kept. As declared, the test sets were not scored again, so there is no run 5 test number. The 5x-cost rule picks run 4 too.
- Files: `ml/results/run5/` (day3.log, run5.log, train_log.json, select_final.json, pick.json, DONE.txt, cands/list.txt, cands/select_final.log). The run 5 adapters and val predictions stay on the notebook's `/workspace/runs/run5/` (not fetched; not needed).
- GPU: session 12:47 to 13:59:38 IST, 73 min today (dashboard: 1 h 47 min left of 3 h); total across the hackathon 529 + 73 = **602 min**. The 529 in eval.json/report.py covers training and evaluation of the final model; whether to quote 602 is the lead's call.
