"""Grade the web app's three sample patients' recorded answers with the fine-tuned Gemma 4 E4B
on the W7900D and write web/public/samples/<id>.onprem.json (format: web/ONPREM_REPLAY.md).

Messages are built exactly like graderMessages() in web/worker/pipeline.ts, with
prompts/grader.json verbatim. One request per recorded attempt, batch size 1, greedy, timed.

Usage (notebook): python onprem_replay.py --model M --adapter A --lora-scale 0.5 \
    --samples-dir /workspace/data/samples --out-dir /workspace/runs/replay --adapter-ref "oktash/homeward-gemma4-e4b-grader@<rev>"
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import grader
from train_lora import load_model_and_tokenizer

LANG = {"it": "Italian", "es": "Spanish", "hi": "Hindi", "bn": "Bengali", "tl": "Tagalog", "ar": "Arabic",
        "zh": "Mandarin Chinese", "en": "English", "pt": "Portuguese", "ta": "Tamil", "fr": "French", "vi": "Vietnamese"}
SAMPLES = ["heart-failure-it", "anticoagulant-hi", "knee-replacement-es"]


def messages(prompt, facts, question, transcript, lang_name):
    # Mirrors graderMessages() in web/worker/pipeline.ts.
    lines = "\n".join(f"- {f['id']} [{f['category']}{', critical' if f.get('critical') else ''}]: {f['statement']}" for f in facts)
    return [{"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"].replace("{facts}", lines).replace("{question}", question)
                .replace("{language_name}", lang_name).replace("{transcript}", transcript)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--samples-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--adapter-ref", default="local adapter (not yet published)")
    ap.add_argument("--run-name", default="run 2")
    a = ap.parse_args()

    import transformers
    from eval import load_adapter
    prompt = grader.load_prompt()
    model, tok = load_model_and_tokenizer(a.model)
    model = load_adapter(model, a.adapter, a.lora_scale).eval()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rocm = getattr(torch.version, "hip", None) or "unknown"

    # Warm-up so the first timed request is not paying for kernel compilation.
    w = tok.apply_chat_template(messages(prompt, [{"id": "f1", "category": "medication", "statement": "Take it daily."}],
                                         "Tell me.", "I take it daily.", "English"), tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    with torch.no_grad():
        model.generate(**tok(w, return_tensors="pt", add_special_tokens=False).to(model.device), max_new_tokens=32, do_sample=False)

    for sid in SAMPLES:
        p = Path(a.samples_dir) / f"{sid}.json"
        if not p.exists():
            print("missing sample", p)
            continue
        s = json.loads(p.read_text())
        facts = {f["id"]: f for f in s["facts"]}
        topics = {t["id"]: t for t in s["topics"]}
        lang_name = LANG.get(s["version"]["language"], s["version"]["language"])
        out = {"sample_id": sid, "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "model": {"name": f"Gemma 4 E4B, fine-tuned for Homeward ({a.run_name})", "base": "google/gemma-4-e4b-it",
                         "adapter": a.adapter_ref, "lora_scale": a.lora_scale},
               "hardware": {"gpu": "AMD Radeon PRO W7900D (48 GB)", "rocm": rocm,
                            "runtime": f"transformers {transformers.__version__}, torch {torch.__version__}, bf16, batch 1, greedy"},
               "prompt": "app/prompts/grader.json (verbatim)", "answers": {}}
        for tid, attempts in s["answers"].items():
            rows = []
            for i, att in enumerate(attempts):
                fs = [facts[x] for x in att["fact_ids"]]
                text = tok.apply_chat_template(messages(prompt, fs, topics[tid]["question"], att["transcript"], lang_name),
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
                enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=768, do_sample=False)
                torch.cuda.synchronize()
                gen_s = time.time() - t0
                new = g[0, enc["input_ids"].shape[1]:]
                raw = tok.decode(new, skip_special_tokens=True)
                obj = grader.parse_reply(raw)
                verdicts, strict = grader.verdicts_from(obj, att["fact_ids"])
                latency = time.time() - t0
                entry = {"attempt": i, "transcript": att["transcript"], "fact_ids": att["fact_ids"],
                         "latency_ms": round(1000 * latency), "tokens_in": int(enc["input_ids"].shape[1]),
                         "tokens_out": int(new.shape[0]), "tokens_per_s": round(int(new.shape[0]) / gen_s, 1),
                         "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                if all(v is not None for v in verdicts):
                    src = {f.get("id"): f for f in (obj or {}).get("facts", []) if isinstance(f, dict)}
                    entry["grades"] = [{"id": fid, "verdict": v, "heard": str(src.get(fid, {}).get("heard", "")),
                                        "reteach": "" if v == "correct" else str(src.get(fid, {}).get("reteach", ""))}
                                       for fid, v in zip(att["fact_ids"], verdicts)]
                    if not strict:
                        entry["lenient_parse"] = True
                else:
                    entry.update(parse_error=True, raw=raw)
                rows.append(entry)
                print(sid, tid, i, entry.get("grades") and [g_["verdict"] for g_ in entry["grades"]], entry["latency_ms"], "ms", flush=True)
            out["answers"][tid] = rows
        (out_dir / f"{sid}.onprem.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print("wrote", out_dir / f"{sid}.onprem.json")


if __name__ == "__main__":
    main()
