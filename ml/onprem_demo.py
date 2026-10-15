"""On-prem Homeward loop on one AMD Radeon PRO W7900D, for the demo video.

Takes one recorded teach-back answer (16 kHz mono WAV) and its discharge facts, then:
  1. Gemma 4 E4B transcribes the speech (native audio, LoRA adapter off),
  2. the fine-tuned E4B grader (same weights, adapter on) labels every fact,
  3. prints a readable nurse view with timings and GPU memory (rocm-smi).
Nothing leaves the machine.

Usage: python onprem_demo.py --model M --adapter A --manifest manifest_human.jsonl --clip human_h02 --wavdir DIR
"""

import argparse
import json
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import torch

import grader
from asr_eval import ASR_PROMPT, LANG_EN
from train_lora import load_model_and_tokenizer

MARK = {"correct": "OK  ", "partial": "PART", "incorrect": "WRONG", "missing": "MISS"}


def rocm_vram():
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=10).stdout
        tot = used = None
        for line in out.splitlines():
            if "Total Memory" in line:
                tot = int(line.split(":")[-1])
            if "Total Used Memory" in line:
                used = int(line.split(":")[-1])
        return f"rocm-smi: VRAM used {used / 2**30:.1f} GiB of {tot / 2**30:.1f} GiB" if tot else out.strip()[-200:]
    except Exception as e:  # noqa: BLE001
        return f"rocm-smi unavailable: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clip", default="human_h02")
    ap.add_argument("--wavdir", required=True)
    a = ap.parse_args()

    import transformers
    from peft import PeftModel
    m = next(json.loads(l) for l in open(a.manifest) if json.loads(l)["id"] == a.clip)
    row = m["row"]
    print("=" * 78)
    print("Homeward on-prem teach-back check   |   " + torch.cuda.get_device_name(0) + "   |   ROCm, torch " + torch.__version__)
    print("=" * 78)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(a.model)
    model = PeftModel.from_pretrained(model, a.adapter).eval()
    proc = transformers.AutoProcessor.from_pretrained(a.model)
    print(f"Loaded Gemma 4 E4B + Homeward LoRA grader in {time.time() - t0:.1f} s")
    print(rocm_vram())
    print()
    print(f"Question ({row['language_name']}): {row['question']}")
    print("Facts the patient must understand:")
    for f in row["facts"]:
        print(f"  {f['id']} {'[critical] ' if f.get('critical') else ''}{f['statement']}")
    with wave.open(str(Path(a.wavdir) / m["wav"])) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    dur = len(audio) / 16000
    print(f"\nPatient's spoken answer: {m['wav']} ({dur:.1f} s of audio, recorded by a person)")

    # 1. ASR with the adapter off.
    msgs = [{"role": "user", "content": [{"type": "text", "text": ASR_PROMPT.format(lang=LANG_EN.get(row["language"], row["language_name"]))},
                                         {"type": "audio", "audio": audio}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                   add_generation_prompt=True, enable_thinking=False).to(model.device)
    inp["input_features"] = inp["input_features"].to(torch.bfloat16)
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad(), model.disable_adapter():
        out = model.generate(**inp, max_new_tokens=300, do_sample=False)
    torch.cuda.synchronize()
    t_asr = time.time() - t1
    heard = proc.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"\n[1] Gemma 4 E4B speech recognition ({t_asr:.1f} s):\n    {heard}")

    # 2. Grade with the adapter on.
    prompt = grader.load_prompt()
    text = tok.apply_chat_template(grader.messages(dict(row, transcript=heard), prompt), tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    torch.cuda.synchronize()
    t2 = time.time()
    with torch.no_grad():
        g = model.generate(**ids, max_new_tokens=768, do_sample=False)
    torch.cuda.synchronize()
    t_grade = time.time() - t2
    reply = tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    obj = grader.parse_reply(reply) or {"facts": []}
    print(f"\n[2] Fine-tuned E4B grader ({t_grade:.1f} s):")
    gold = {l["id"]: l["verdict"] for l in row["labels"]}
    for f in obj.get("facts", []):
        v = f.get("verdict", "?")
        print(f"  {MARK.get(v, v):5} {f.get('id')}: {v:9} heard: {f.get('heard', '')}")
        if f.get("reteach"):
            print(f"        re-teach: {f['reteach']}")
    crit_bad = [f["id"] for f in obj.get("facts", []) if f.get("verdict") != "correct"
                and any(x["id"] == f["id"] and x.get("critical") for x in row["facts"])]
    print("\nReadiness: " + ("NOT READY, re-teach critical facts " + ", ".join(crit_bad) if crit_bad else "ready"))
    print(f"Designed labels for this clip: {gold}")
    print(f"\nEnd to end on one GPU: {t_asr + t_grade:.1f} s (ASR {t_asr:.1f} s + grading {t_grade:.1f} s); "
          f"peak PyTorch memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print(rocm_vram())
    print("No audio or text left this machine.")


if __name__ == "__main__":
    main()
