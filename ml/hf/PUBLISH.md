# Publishing the adapter to Hugging Face

Status (Oct 17, 2026): published. `main` = run 4 adapter at LoRA scale 1.0 with the card from `build_card.py --system homeward_run4 --license apache-2.0`; branch `run2` = run 2 (scale 0.5). Repo is public. The steps below are kept for reference.

Target: https://huggingface.co/oktash/homeward-teachback-gemma4-e4b (account `oktash`, logged in in the agent Chrome).

Files in this folder:

| File | What |
|---|---|
| `README.md` | the model card. Run-dependent numbers sit between `<!--v:...-->` / `<!--b:...-->` markers and are filled by `build_card.py`; never type them by hand. |
| `build_card.py` | fills the card and writes `homeward_inference.json` from `ml/results` for one eval.json system key. |
| `homeward_inference.json` | generated: LoRA scale, decoding, attention, source files. Uploaded with the card. |
| `hf_upload.py` | uploads adapters (notebook) or card files (Mac). Reads the token from a file, deletes the file, never prints the token. |
| `notebook_push.sh` | on the notebook: selected adapter to `main`, the other candidate to branch `run2` or `run4`. |

Token rules: never print, echo, paste into a terminal command, log or commit the token. It lives only in a mode-600 file, which `hf_upload.py` deletes after reading. Revoke it when publishing is done.

## Timing (why the adapters go up before the final pick)

The GPU session that starts ~11:19 IST ends ~14:19 IST (3 h quota), before the ~14:30 final numbers. Adapters (~140 MB) only exist on the notebook, and `jup.mjs get` fails on files that size. So upload both candidates while the session runs, as soon as `run_day2.sh` prints its `FINAL:` line (validation-only selection, decided before the test scoring, so it does not change later). The selected adapter goes to `main`, the other one to a branch, and the repo stays private until the card is final. After that everything runs from the Mac without a GPU.

## 1. Before the `FINAL:` line (browser, lead)

1. Check the base model's licence: open https://huggingface.co/google/gemma-4-E4B-it and read the `License:` field in the right-hand panel. If it says `apache-2.0`, pass `--license apache-2.0` to `build_card.py` in step 4; if it says `gemma` (Gemma Terms of Use), keep the default.
2. Create the repo: https://huggingface.co/new, owner `oktash`, name `homeward-teachback-gemma4-e4b`, licence as in step 1, **Private**. Create.
3. Create a token: https://huggingface.co/settings/tokens/new?tokenType=fineGrained, name `homeward-upload`, under "Repository permissions" pick `oktash/homeward-teachback-gemma4-e4b` with "Write access to contents/settings". Create.
4. Save it straight to a file, without printing it. For example, read the value of the token field with the CDP helper and redirect the output: `node SP/tools/cdp.mjs T eval '<expression returning the token input value>' > "$SPX/.hf_tok"` (`SPX` = a private scratch folder outside the repo), then `chmod 600 "$SPX/.hf_tok"`. Check only its shape: `head -c 3 "$SPX/.hf_tok"` must print `hf_`, and `wc -c` should be about 37. If the helper wraps the value in quotes, strip them with `sed -i '' 's/"//g'`.

## 2. Upload the adapters from the notebook (right after `FINAL:`)

`T` = the notebook tab's targetId, `SP` = the folder holding the notebook helper scripts (`tools/jup.mjs`, `tools/cdp.mjs`; not in this repo).

```bash
APP=<path to this app checkout>
node SP/tools/jup.mjs T put $APP/ml/hf/hf_upload.py code/hf_upload.py
node SP/tools/jup.mjs T put $APP/ml/hf/notebook_push.sh code/notebook_push.sh
node SP/tools/jup.mjs T put "$SPX/.hf_tok" .hf_tok && rm "$SPX/.hf_tok"
JT=hf node SP/tools/jup.mjs T run 'grep FINAL /workspace/logs/day2.log; cat /workspace/runs/final2/adapter_path.txt /workspace/runs/final2/lora_scale.txt' 30
JT=hf node SP/tools/jup.mjs T run 'nohup bash /workspace/code/notebook_push.sh >/dev/null 2>&1 &' 10
# poll until HF_PUSH_DONE (2 x ~140 MB; a few minutes)
JT=hf node SP/tools/jup.mjs T run 'cat /workspace/logs/hf_push.log; ls -la /workspace/.hf_tok 2>&1' 20
```

The log shows, per adapter: `r`, `lora_alpha`, the original `base_model_name_or_path`, the size and the sha256 of `adapter_model.safetensors`, then the commit URL. `ls` must say the token file is gone. If `run4` used a different adapter folder than `run4/adapter_avg` for the alternate (for example `run4/adapter`), start the script with `RUN4_ADAPTER=/workspace/runs/run4/adapter bash ...`.

What gets uploaded per adapter:

- `adapter_model.safetensors`: unchanged.
- `adapter_config.json`: unchanged except `base_model_name_or_path`, which becomes `google/gemma-4-E4B-it` (it is the notebook's local path otherwise). `lora_alpha` and `r` stay as trained (32/32, scaling 1.0).

The LoRA scale is not in `adapter_config.json`. It is applied at load time (`module.scaling["default"] *= lora_scale` before `merge_and_unload()`, as in `ml/eval.py load_adapter`), and recorded in the card, in the load snippet and in `homeward_inference.json`. Run 2's adapter is evaluated at scale 0.5; whoever loads it with plain PEFT at 1.0 gets a different model, which the card says.

The core of `hf_upload.py`, for reference:

```python
from huggingface_hub import CommitOperationAdd, HfApi
token = Path(token_file).read_text().strip(); os.remove(token_file)
api = HfApi(token=token)
cfg = json.loads((src / "adapter_config.json").read_text())
cfg["base_model_name_or_path"] = "google/gemma-4-E4B-it"
if rev != "main":
    api.create_branch("oktash/homeward-teachback-gemma4-e4b", branch=rev, exist_ok=True)
api.create_commit(repo_id="oktash/homeward-teachback-gemma4-e4b", revision=rev, commit_message=f"Add LoRA adapter from {src}",
                  operations=[CommitOperationAdd("adapter_config.json", (json.dumps(cfg, indent=2) + "\n").encode()),
                              CommitOperationAdd("adapter_model.safetensors", str(src / "adapter_model.safetensors"))])
```

Fallback if the notebook cannot reach the Hub: split and fetch in small parts, then upload from the Mac.

```bash
JT=hf node SP/tools/jup.mjs T run 'cd /workspace/runs/final2 && mkdir -p hfparts && cp $(cat adapter_path.txt)/adapter_config.json hfparts/ && split -b 20m $(cat adapter_path.txt)/adapter_model.safetensors hfparts/part_ && sha256sum $(cat adapter_path.txt)/adapter_model.safetensors && ls hfparts' 120
# get each part (runs/final2/hfparts/part_aa, part_ab, ...) and adapter_config.json with jup.mjs get into $SPX/hfadapter/, then:
cat $SPX/hfadapter/part_* > $SPX/hfadapter/adapter_model.safetensors && shasum -a 256 $SPX/hfadapter/adapter_model.safetensors   # must match
cd $APP && uv run --with huggingface_hub python ml/hf/hf_upload.py --token-file "$SPX/.hf_tok" --adapter $SPX/hfadapter:main
```

## 3. Before building the card (ML agent, after the run)

`build_card.py` reads only local files. For run 4 it needs:

- `ml/results/eval.json` with the system key (for example `homeward_run4`), its `meta.adapter` and `meta.lora_scale` (from `report.py`).
- The run's training log in `ml/results/` (any `*train_log*.json`, for example `ml/results/run4/train_log.json` fetched from `/workspace/runs/run4/train_log.json`). It is matched by `args.out` against the adapter's folder.
- Optional, for the selection paragraph: `ml/results/run4/select_final.json` (from `/workspace/runs/cands2/select_final.json`), matched by adapter and scale.
- `data/splits_v3/stats.json` as uploaded for the run (its row count is checked against the training log; on a mismatch the split counts are left out).

## 4. Build the card (Mac, ~14:30)

```bash
cd <path to this app checkout>
python3 ml/hf/build_card.py --system homeward_run4 --check     # print the values only
python3 ml/hf/build_card.py --system homeward_run4             # add --license apache-2.0 if step 1.1 said so
```

Use the key of the model on `main` (the `FINAL:` pick): `homeward_run2` or `homeward_run4`. Then read the whole card once: the headline bullets, the comparison table (the 31B row is labelled teacher), the data section (run 4 lists Clef as a label vote), the selection paragraph, the load snippet scale. Prose outside the markers is static; if a static sentence no longer fits the run, edit the prose (never a number). Check with `python3 -c "import sys; print(open('ml/hf/README.md').read().count(chr(0x2014)))"` (must be 0).

Comparators missing from the table (late baselines) appear after `report.py` adds them to eval.json; just re-run `build_card.py`. Sizes are checked against `model-sizes.md`; a new model key needs a line in `SYSTEMS` in `build_card.py`.

## 5. Upload the card, promote if needed, publish

```bash
cd <path to this app checkout>
# token file again (step 1.4; the first one was deleted). Then:
uv run --with huggingface_hub python ml/hf/hf_upload.py --token-file "$SPX/.hf_tok" --keep-token-file \
    --card-dir ml/hf --prompt prompts/grader.json
```

If the final pick is not the adapter on `main` (it should be, since both follow the declared selection), copy it from its branch first, from the Mac:

```python
import json
from pathlib import Path
from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi, hf_hub_download
api = HfApi(token=Path(TOKEN_FILE).read_text().strip())
repo, branch = "oktash/homeward-teachback-gemma4-e4b", "run2"   # the branch holding the chosen adapter
cfg = Path(hf_hub_download(repo, "adapter_config.json", revision=branch, token=api.token)).read_bytes()
api.create_commit(repo_id=repo, commit_message=f"Promote the {branch} adapter to main", operations=[
    CommitOperationCopy(src_path_in_repo="adapter_model.safetensors", path_in_repo="adapter_model.safetensors", src_revision=branch),
    CommitOperationAdd(path_in_repo="adapter_config.json", path_or_fileobj=cfg)])
```

Then rebuild the card for that key and upload it again (the command above).

Make it public and finish:

```bash
uv run --with huggingface_hub python ml/hf/hf_upload.py --token-file "$SPX/.hf_tok" --make-public
```

1. Open https://huggingface.co/oktash/homeward-teachback-gemma4-e4b logged out (or in a private window): card renders, licence and base model shown, `Files` has `adapter_model.safetensors`, `adapter_config.json`, `README.md`, `homeward_inference.json`, `grader_prompt.json`.
2. Revoke the `homeward-upload` token at https://huggingface.co/settings/tokens and delete `$SPX/.hf_tok` if it still exists.
3. Optional: delete the alternate branch on the Hub (repo Settings, or `api.delete_branch(repo, branch="run2")`) so only the published model is there.
4. Update the links: STATUS.md live assets, deck slide 12 (HF name), `submission/form.md`, the app README.

Nothing in `ml/hf/` contains a token. Do not commit `homeward_inference.json` from a trial build for the wrong key: rebuild for the published key before committing.
