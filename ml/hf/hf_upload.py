"""Upload Homeward LoRA adapters and the model card to the Hugging Face Hub. See ml/hf/PUBLISH.md.

Runs on the AMD notebook (adapters) and on the Mac (card files). Needs only huggingface_hub
(installed on the notebook with transformers). The token is read from a file and the file is
deleted right after reading; the token is never printed.

  # notebook: the selected adapter to main, the other candidate to a branch
  python hf_upload.py --token-file /workspace/.hf_tok \
      --adapter /workspace/runs/run4/adapter_avg:main --adapter /workspace/runs/runB/adapter_avg:run2
  # Mac: the card, inference settings and prompt to main
  python3 ml/hf/hf_upload.py --token-file <file> --card-dir ml/hf --prompt prompts/grader.json
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = "oktash/homeward-teachback-gemma4-e4b"
BASE_ID = "google/gemma-4-E4B-it"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--keep-token-file", action="store_true", help="do not delete the token file after reading it")
    ap.add_argument("--adapter", action="append", default=[], help="ADAPTER_DIR:REVISION (main or a branch name); repeatable")
    ap.add_argument("--card-dir", default=None, help="folder with README.md and homeward_inference.json (uploaded to main)")
    ap.add_argument("--prompt", default=None, help="prompts/grader.json, uploaded as grader_prompt.json")
    ap.add_argument("--make-public", action="store_true")
    a = ap.parse_args()

    token = Path(a.token_file).read_text().strip()
    if not a.keep_token_file:
        os.remove(a.token_file)
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=token)
    print("authenticated as", api.whoami()["name"])

    for spec in a.adapter:
        src, rev = spec.rsplit(":", 1)
        src = Path(src)
        cfg = json.loads((src / "adapter_config.json").read_text())
        st = src / "adapter_model.safetensors"
        print(f"{src}: r={cfg.get('r')} lora_alpha={cfg.get('lora_alpha')} base={cfg.get('base_model_name_or_path')} "
              f"size={st.stat().st_size / 2**20:.1f} MiB sha256={sha256(st)}")
        # Only metadata changes: the notebook path in base_model_name_or_path becomes the Hub id. Weights untouched.
        cfg["base_model_name_or_path"] = BASE_ID
        if rev != "main":
            api.create_branch(a.repo, branch=rev, exist_ok=True)
        ops = [CommitOperationAdd(path_in_repo="adapter_config.json", path_or_fileobj=(json.dumps(cfg, indent=2) + "\n").encode()),
               CommitOperationAdd(path_in_repo="adapter_model.safetensors", path_or_fileobj=str(st))]
        info = api.create_commit(repo_id=a.repo, operations=ops, revision=rev, commit_message=f"Add LoRA adapter from {src}")
        print("uploaded to", rev, getattr(info, "commit_url", info))

    if a.card_dir or a.prompt:
        files = []
        if a.card_dir:
            d = Path(a.card_dir)
            card = (d / "README.md").read_text()
            assert "\u2014" not in card, "em dash in the card"
            files += [(d / "README.md", "README.md"), (d / "homeward_inference.json", "homeward_inference.json")]
        if a.prompt:
            files.append((Path(a.prompt), "grader_prompt.json"))
        ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)) for src, dst in files]
        info = api.create_commit(repo_id=a.repo, operations=ops, commit_message="Model card, inference settings and grader prompt")
        print("uploaded", [d for _, d in files], getattr(info, "commit_url", info))

    if a.make_public:
        if hasattr(api, "update_repo_settings"):
            api.update_repo_settings(repo_id=a.repo, private=False)
        else:
            api.update_repo_visibility(repo_id=a.repo, private=False)
        print("repo is now public")
    print("files on main:", api.list_repo_files(a.repo))


if __name__ == "__main__":
    main()
