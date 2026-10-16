"""Evaluate teach-back graders on the held-out test set.

Three subcommands:

  predict-local  run Gemma 4 E4B (base, or base + LoRA adapter) with transformers on the
                 notebook GPU (AMD Radeon PRO W7900D, ROCm). Greedy decoding. Batched pass
                 over every test answer for throughput, plus a sequential batch-size-1 pass
                 over the first --latency-n answers for per-teach-back latency.
  predict-api    run a Token Factory model zero-shot with the same prompt (tf.py client).
  score          compare predictions with the designed labels and write results/eval.json
                 and results/eval.md.

All systems get the identical grader prompt (prompts/grader.json). A fact whose verdict
cannot be read from the reply (invalid JSON, missing id, unknown verdict) is scored as
wrong, never as correct.

Usage:
  python eval.py predict-local --model $W/models/gemma-4-E4B-it --rows test.jsonl --out preds_base.jsonl
  python eval.py predict-local --model ... --adapter $W/runs/lora1/adapter --rows test.jsonl --out preds_ft.jsonl
  uv run --with httpx python ml/eval.py predict-api --model Gemma-4-31B-IT --rows data/splits/test.jsonl --out ml/results/preds_gemma31b.jsonl
  python3 ml/eval.py score --rows data/splits/test.jsonl --system "name=path" ... --out ml/results/eval.json
"""

import argparse
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import grader

LANG_NAMES = {}


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)



def load_adapter(model, adapter, scale=1.0):
    """Attach a LoRA adapter, optionally scaled (WiSE-FT-style interpolation toward the base),
    then merge it for fast inference."""
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)
    if scale != 1.0:
        n = 0
        for mod in model.modules():
            if hasattr(mod, "scaling") and isinstance(mod.scaling, dict) and "default" in mod.scaling:
                mod.scaling["default"] *= scale
                n += 1
        print(f"LoRA scale {scale} applied to {n} modules", flush=True)
    return model.merge_and_unload()

# ---------------------------------------------------------------- local (GPU)

def predict_local(args):
    import torch
    from train_lora import load_model_and_tokenizer

    prompt = grader.load_prompt(args.prompt)
    rows = read_jsonl(args.rows)
    if args.limit:
        rows = rows[:args.limit]
    t0 = time.time()
    model, tok = load_model_and_tokenizer(args.model)
    if args.adapter:
        model = load_adapter(model, args.adapter, args.lora_scale)
    model.eval()
    load_s = time.time() - t0
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    texts = [tok.apply_chat_template(grader.messages(r, prompt), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for r in rows]
    enc_all = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]

    def gen(idx):
        L = max(len(enc_all[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            n = len(enc_all[i])
            ids[r, L - n:] = torch.tensor(enc_all[i])
            att[r, L - n:] = 1
        ids, att = ids.cuda(), att.cuda()
        with torch.no_grad():
            if args.temperature > 0:
                out = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=args.max_new,
                                     do_sample=True, temperature=args.temperature, top_p=0.95, pad_token_id=pad)
            else:
                out = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=args.max_new,
                                     do_sample=False, pad_token_id=pad)
        res = []
        eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
                  else [model.generation_config.eos_token_id])
        for r in range(len(idx)):
            new = out[r, L:].tolist()
            n = 0
            for t in new:
                n += 1
                if t in eos or t == pad:
                    break
            res.append((tok.decode(new[:n], skip_special_tokens=True), n))
        return res

    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    # Warm-up (kernel compilation / caches), excluded from timings.
    gen([0])

    # Sequential batch-size-1 latency pass.
    lat = {}
    for i in range(min(args.latency_n, len(rows))):
        torch.cuda.synchronize()
        t = time.time()
        (text, n), = gen([i])
        torch.cuda.synchronize()
        lat[i] = {"latency_s": time.time() - t, "out_tokens": n, "text": text}

    # Batched pass over everything, sorted by prompt length.
    order = sorted(range(len(rows)), key=lambda i: len(enc_all[i]))
    preds = {}
    torch.cuda.synchronize()
    tb = time.time()
    for b in range(0, len(order), args.bsz):
        idx = order[b:b + args.bsz]
        for i, (text, n) in zip(idx, gen(idx)):
            preds[i] = (text, n)
        print(f"batched {min(b + args.bsz, len(order))}/{len(order)}", flush=True)
    torch.cuda.synchronize()
    batch_s = time.time() - tb

    with open(args.out, "w") as fh:
        for i, r in enumerate(rows):
            text, n = preds[i]
            rec = {"key": grader.row_key(r), "text": text, "prompt_tokens": len(enc_all[i]),
                   "completion_tokens": n}
            if i in lat:
                rec["latency_s_bs1"] = lat[i]["latency_s"]
                rec["text_bs1"] = lat[i]["text"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    lats = [v["latency_s"] for v in lat.values()]
    meta = {
        "system": args.name or ("e4b_lora" if args.adapter else "e4b_base"),
        "model": args.model, "adapter": args.adapter, "lora_scale": args.lora_scale, "rows": len(rows),
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "load_s": round(load_s, 1), "decoding": "greedy" if args.temperature <= 0 else f"sampled T={args.temperature} top_p=0.95 seed={args.seed}", "max_new_tokens": args.max_new,
        "batched": {"batch_size": args.bsz, "wall_s": round(batch_s, 1),
                    "completion_tokens": sum(n for _, n in preds.values()),
                    "prompt_tokens": sum(len(e) for e in enc_all),
                    "gen_tok_per_s": sum(n for _, n in preds.values()) / batch_s,
                    "answers_per_s": len(rows) / batch_s},
        "bs1": {"n": len(lats), "p50_s": pct(lats, 0.5), "p95_s": pct(lats, 0.95),
                "mean_s": statistics.mean(lats) if lats else None,
                "gen_tok_per_s": (sum(v["out_tokens"] for v in lat.values()) / sum(lats)) if lats else None},
        "peak_vram_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- verdict probabilities (GPU)

def verdict_probs(args):
    """For every fact slot in a model's own reply, score the four verdicts by teacher forcing:
    the reply prefix up to '"verdict": "' plus each candidate verdict and its closing quote.
    p(verdict) = softmax over the four sequence log-probs. Used for the escalation rule
    (send a 'correct' to the nurse unless p(correct) >= tau)."""
    import re as _re
    import torch
    from train_lora import load_model_and_tokenizer

    prompt = grader.load_prompt(args.prompt)
    rows = {grader.row_key(r): r for r in read_jsonl(args.rows)}
    preds = read_jsonl(args.preds)
    model, tok = load_model_and_tokenizer(args.model)
    if args.adapter:
        model = load_adapter(model, args.adapter, args.lora_scale)
    model.eval()
    jobs = []  # (key, fact_id, candidate, full_text, prefix_text)
    for p in preds:
        r = rows.get(p["key"])
        if not r:
            continue
        ptxt = tok.apply_chat_template(grader.messages(r, prompt), tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        text = p["text"]
        ids = [f["id"] for f in r["facts"]]
        for fid in ids:
            m = _re.search(r'"id"\s*:\s*"' + _re.escape(fid) + r'"[^{}]*?"verdict"\s*:\s*"', text)
            if not m:
                continue
            prefix = ptxt + text[:m.end()]
            for v in grader.VERDICTS:
                jobs.append((p["key"], fid, v, prefix + v + '"', prefix))
    out = defaultdict(dict)
    t0 = time.time()
    for b in range(0, len(jobs), args.bsz):
        chunk = jobs[b:b + args.bsz]
        enc = [tok(j[3], add_special_tokens=False)["input_ids"] for j in chunk]
        pre = [tok(j[4], add_special_tokens=False)["input_ids"] for j in chunk]
        L = max(len(e) for e in enc)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        inp = torch.full((len(chunk), L), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, e in enumerate(enc):
            inp[i, :len(e)] = torch.tensor(e)
            att[i, :len(e)] = 1
        with torch.no_grad():
            logits = model(input_ids=inp.cuda(), attention_mask=att.cuda()).logits
        for i, (e, pr) in enumerate(zip(enc, pre)):
            k = 0
            while k < min(len(e), len(pr)) and e[k] == pr[k]:
                k += 1
            k = min(k, len(pr))  # first token that differs from the bare prefix
            lsm = torch.log_softmax(logits[i, k - 1:len(e) - 1].float(), -1)  # only the scored positions
            lp = sum(lsm[t - k, e[t]].item() for t in range(k, len(e)))
            key, fid, v = chunk[i][0], chunk[i][1], chunk[i][2]
            out[(key, fid)][v] = lp
    with open(args.out, "w") as fh:
        for (key, fid), d in out.items():
            import math
            mx = max(d.values())
            z = sum(math.exp(x - mx) for x in d.values())
            fh.write(json.dumps({"key": key, "id": fid, "p": {v: math.exp(d[v] - mx) / z for v in d}}) + "\n")
    print(f"{len(out)} fact slots scored in {time.time() - t0:.1f}s")


def escalate(args):
    """Escalation rule: a fact graded 'correct' with p(correct) < tau becomes the most likely
    other verdict (so the nurse re-checks it). Writes a new predictions file."""
    probs = {(r["key"], r["id"]): r["p"] for r in read_jsonl(args.probs)}
    n_changed = 0
    with open(args.out, "w") as fh:
        for p in read_jsonl(args.preds):
            obj = grader.parse_reply(p["text"])
            if isinstance(obj, dict) and isinstance(obj.get("facts"), list):
                for f in obj["facts"]:
                    pr = probs.get((p["key"], f.get("id")))
                    if pr and f.get("verdict") == "correct" and pr.get("correct", 1.0) < args.tau:
                        f["verdict"] = max((v for v in grader.VERDICTS if v != "correct"), key=lambda v: pr.get(v, 0))
                        n_changed += 1
                p = dict(p, text=json.dumps(obj, ensure_ascii=False), escalated_tau=args.tau)
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta_src = Path(args.preds).with_suffix(".meta.json")
    if meta_src.exists():
        meta = json.loads(meta_src.read_text())
        meta.update(escalation={"tau": args.tau, "facts_changed": n_changed})
        Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"tau {args.tau}: {n_changed} facts escalated")



def vote(args):
    """Self-consistency: per fact, majority verdict over several prediction files (greedy and/or
    samples); ties go to the less reassuring verdict (missing > incorrect > partial > correct)."""
    order = {"missing": 0, "incorrect": 1, "partial": 2, "correct": 3}
    rows = {grader.row_key(r): r for r in read_jsonl(args.rows)}
    files = [{p["key"]: p for p in read_jsonl(f)} for f in args.preds]
    with open(args.out, "w") as fh:
        for k, r in rows.items():
            ids = [f["id"] for f in r["facts"]]
            votes = [grader.verdicts_from(grader.parse_reply(f[k]["text"]) if k in f else None, ids)[0] for f in files]
            base_obj = grader.parse_reply(files[0][k]["text"]) if k in files[0] else None
            facts = []
            for j, fid in enumerate(ids):
                c = Counter(v[j] for v in votes if v[j])
                if c:
                    top = max(c.values())
                    pick = min((v for v in c if c[v] == top), key=lambda v: order[v])
                else:
                    pick = None
                src = next((f for f in (base_obj or {}).get("facts", []) if isinstance(f, dict) and f.get("id") == fid), {})
                facts.append(dict(src, id=fid, verdict=pick or src.get("verdict")))
            fh.write(json.dumps({"key": k, "text": json.dumps({"facts": facts}, ensure_ascii=False), "votes": len(files)}, ensure_ascii=False) + "\n")
    print("voted", len(rows), "answers over", len(files), "prediction files")

# ---------------------------------------------------------------- Token Factory

def predict_api(args):
    import asyncio
    import httpx
    from tf import chat, content_of

    prompt = grader.load_prompt(args.prompt)
    rows = read_jsonl(args.rows)
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.out)
    done = {r["key"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if grader.row_key(r) not in done]
    print(f"{len(todo)} to run ({len(done)} done)")

    async def one(client, r):
        t = time.time()
        try:
            resp = await chat(client, args.model, grader.messages(r, prompt), max_tokens=args.max_new,
                              temperature=0.0, json_mode=args.json_mode,
                              reasoning_effort=args.reasoning or None)
        except Exception as e:  # noqa: BLE001
            return {"key": grader.row_key(r), "text": "", "error": str(e)[:300]}
        u = resp.get("usage", {})
        return {"key": grader.row_key(r), "text": content_of(resp), "latency_s": time.time() - t,
                "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)}

    async def run():
        async with httpx.AsyncClient() as client:
            with out.open("a") as fh:
                for k, coro in enumerate(asyncio.as_completed([one(client, r) for r in todo]), 1):
                    rec = await coro
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    if k % 20 == 0:
                        print(f"{k}/{len(todo)}")
    asyncio.run(run())
    recs = read_jsonl(out)
    meta = {"system": args.name or args.model, "model": args.model, "rows": len(recs),
            "errors": sum(1 for r in recs if r.get("error")),
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in recs),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in recs),
            "latency_p50_s": pct([r["latency_s"] for r in recs if "latency_s" in r], 0.5),
            "latency_p95_s": pct([r["latency_s"] for r in recs if "latency_s" in r], 0.95),
            "json_mode": args.json_mode, "reasoning_effort": args.reasoning or None,
            "note": "latency includes Token Factory queueing under the shared 20 requests/minute limit"}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- scoring

def score_system(rows, preds):
    gold, pred, lang, crit, cases = [], [], [], [], []
    valid = []
    for r in rows:
        p = preds.get(grader.row_key(r))
        ids = [f["id"] for f in r["facts"]]
        obj = grader.parse_reply(p["text"]) if p else None
        got, strict = grader.verdicts_from(obj, ids)
        valid.append(strict)
        cmap = {f["id"]: bool(f.get("critical")) for f in r["facts"]}
        for lab, g in zip(r["labels"], got):
            gold.append(lab["verdict"])
            pred.append(g or "invalid")
            lang.append(r["language"])
            crit.append(cmap[lab["id"]])
            cases.append(r["case_id"])
    return gold, pred, lang, crit, cases, valid


def metrics(gold, pred, crit):
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n
    f1s = {}
    for v in grader.VERDICTS:
        tp = sum(g == v and p == v for g, p in zip(gold, pred))
        fp = sum(g != v and p == v for g, p in zip(gold, pred))
        fn = sum(g == v and p != v for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s[v] = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                  "support": tp + fn}
    bad = [(p, c) for g, p, c in zip(gold, pred, crit) if g in ("incorrect", "missing")]
    part = [p for g, p in zip(gold, pred) if g == "partial"]
    good = [p for g, p in zip(gold, pred) if g == "correct"]
    crit_bad = [p for p, c in bad if c]
    return {
        "facts": n, "accuracy": acc,
        "macro_f1": sum(x["f1"] for x in f1s.values()) / 4, "per_verdict": f1s,
        "false_reassurance_rate": sum(p == "correct" for p, _ in bad) / len(bad) if bad else None,
        "false_reassurance_n": [sum(p == "correct" for p, _ in bad), len(bad)],
        "false_reassurance_rate_critical": sum(p == "correct" for p in crit_bad) / len(crit_bad) if crit_bad else None,
        "false_reassurance_critical_n": [sum(p == "correct" for p in crit_bad), len(crit_bad)],
        "partial_graded_correct_rate": sum(p == "correct" for p in part) / len(part) if part else None,
        "over_flag_rate": sum(p != "correct" for p in good) / len(good) if good else None,
        "understood_vs_not_accuracy": sum((g == "correct") == (p == "correct") for g, p in zip(gold, pred)) / n,
        "invalid_facts": sum(p == "invalid" for p in pred),
        "confusion": {g: {p: sum(1 for a, b in zip(gold, pred) if a == g and b == p)
                          for p in grader.VERDICTS + ["invalid"]} for g in grader.VERDICTS},
    }


def bootstrap(gold, pred, crit, cases, iters=1000, seed=0):
    """95% intervals by resampling whole cases (facts within a case are correlated)."""
    by_case = defaultdict(list)
    for i, c in enumerate(cases):
        by_case[c].append(i)
    keys = list(by_case)
    rng = random.Random(seed)
    accs, frrs, f1s = [], [], []
    for _ in range(iters):
        idx = [i for k in (rng.choice(keys) for _ in keys) for i in by_case[k]]
        m = metrics([gold[i] for i in idx], [pred[i] for i in idx], [crit[i] for i in idx])
        accs.append(m["accuracy"])
        f1s.append(m["macro_f1"])
        if m["false_reassurance_rate"] is not None:
            frrs.append(m["false_reassurance_rate"])
    ci = lambda xs: [pct(xs, 0.025), pct(xs, 0.975)]  # noqa: E731
    return {"accuracy_95ci": ci(accs), "macro_f1_95ci": ci(f1s), "false_reassurance_rate_95ci": ci(frrs),
            "method": f"case-level bootstrap, {iters} resamples"}


def paired_vs(ref_pred, pred, gold):
    """Facts where one system is right and the other wrong (for a sign/McNemar-style read)."""
    a = sum(p == g and r != g for r, p, g in zip(ref_pred, pred, gold))
    b = sum(p != g and r == g for r, p, g in zip(ref_pred, pred, gold))
    return {"this_right_ref_wrong": a, "this_wrong_ref_right": b}


def score(args):
    rows = read_jsonl(args.rows)
    systems = {}
    for spec in args.system:
        name, path = spec.split("=", 1)
        preds = {p["key"]: p for p in read_jsonl(path)}
        meta_p = Path(path).with_suffix(".meta.json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        gold, pred, lang, crit, cases, valid = score_system(rows, preds)
        m = metrics(gold, pred, crit)
        m.update(bootstrap(gold, pred, crit, cases))
        m["json_valid_rate"] = sum(valid) / len(valid)
        m["answers"] = len(valid)
        m["answers_all_facts_right"] = None
        # answer-level exact match
        k, exact = 0, 0
        for r in rows:
            n = len(r["labels"])
            exact += all(gold[k + j] == pred[k + j] for j in range(n))
            k += n
        m["answer_exact_match_rate"] = exact / len(rows)
        del m["answers_all_facts_right"]
        per_lang = defaultdict(lambda: [0, 0])
        for g, p, l in zip(gold, pred, lang):
            per_lang[l][0] += g == p
            per_lang[l][1] += 1
        m["per_language_accuracy"] = {l: {"accuracy": a / n, "facts": n} for l, (a, n) in sorted(per_lang.items())}
        missing_preds = sum(1 for r in rows if grader.row_key(r) not in preds)
        m["answers_without_prediction"] = missing_preds
        # tokens and cost
        pt = sum(preds[grader.row_key(r)].get("prompt_tokens", 0) for r in rows if grader.row_key(r) in preds)
        ct = sum(preds[grader.row_key(r)].get("completion_tokens", 0) for r in rows if grader.row_key(r) in preds)
        m["tokens"] = {"prompt_per_answer": pt / len(rows), "completion_per_answer": ct / len(rows)}
        systems[name] = {"meta": meta, "metrics": m, "_pred": pred, "_gold": gold}

    ref = args.reference if args.reference in systems else None
    if ref:
        for name, s in systems.items():
            if name != ref:
                s["metrics"]["paired_vs_" + ref] = paired_vs(systems[ref]["_pred"], s["_pred"], s["_gold"])

    gold = next(iter(systems.values()))["_gold"]
    result = {
        "schema": "homeward.eval.v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "test_set": {"path": args.rows, "answers": len(rows), "facts": len(gold),
                     "cases": len({r["case_id"] for r in rows}),
                     "languages": sorted({r["language"] for r in rows}),
                     "gold_verdict_counts": dict(Counter(gold)),
                     "labels": "designed labels from the generator (not verifier-filtered)"},
        "metric_definitions": {
            "accuracy": "share of facts whose predicted verdict equals the designed verdict",
            "macro_f1": "unweighted mean of per-verdict F1 over correct, partial, incorrect, missing; unreadable verdicts count as wrong",
            "false_reassurance_rate": "share of facts designed incorrect or missing that the system graded correct (the unsafe error)",
            "false_reassurance_rate_critical": "same, restricted to facts flagged critical",
            "over_flag_rate": "share of facts designed correct that the system did not grade correct (extra nurse work, not unsafe)",
            "json_valid_rate": "share of answers whose reply parsed as JSON with exactly the expected fact ids in order and known verdicts",
        },
        "systems": {k: {"meta": v["meta"], "metrics": v["metrics"]} for k, v in systems.items()},
    }
    if args.extra:
        for spec in args.extra:
            k, p = spec.split("=", 1)
            result[k] = json.loads(Path(p).read_text())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    md = render_md(result)
    Path(args.md or Path(args.out).with_suffix(".md")).write_text(md)
    print(md)


def f(x, p=1):
    return "n/a" if x is None else f"{100 * x:.{p}f}%"


def render_md(res):
    ts = res["test_set"]
    lines = [f"# Teach-back grader evaluation", "",
             f"Test set: {ts['answers']} answers, {ts['facts']} facts, {ts['cases']} cases, "
             f"{len(ts['languages'])} languages. Gold verdicts: {ts['gold_verdict_counts']}. Generated {res['generated_at']}.", "",
             "| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | FR on critical facts | Over-flag | JSON valid |",
             "|---|---|---|---|---|---|---|"]
    for name, s in res["systems"].items():
        m = s["metrics"]
        a, b = m["accuracy_95ci"], m["false_reassurance_rate_95ci"]
        lines.append(f"| {name} | {f(m['accuracy'])} ({f(a[0])}–{f(a[1])}) | {m['macro_f1']:.3f} | "
                     f"{f(m['false_reassurance_rate'])} ({f(b[0])}–{f(b[1])}) "
                     f"| {f(m['false_reassurance_rate_critical'])} ({m['false_reassurance_critical_n'][0]}/{m['false_reassurance_critical_n'][1]}) "
                     f"| {f(m['over_flag_rate'])} | {f(m['json_valid_rate'])} |")
    langs = sorted({l for s in res["systems"].values() for l in s["metrics"]["per_language_accuracy"]})
    lines += ["", "## Accuracy by language", "", "| System | " + " | ".join(langs) + " |",
              "|---|" + "---|" * len(langs)]
    for name, s in res["systems"].items():
        pl = s["metrics"]["per_language_accuracy"]
        lines.append(f"| {name} | " + " | ".join(f(pl[l]["accuracy"], 0) if l in pl else "" for l in langs) + " |")
    lines += ["", "## Speed and tokens", "", "| System | Where | p50 latency | p95 latency | Gen tok/s | Prompt tok/answer | Output tok/answer |",
              "|---|---|---|---|---|---|---|"]
    for name, s in res["systems"].items():
        me, m = s["meta"], s["metrics"]
        if "bs1" in me:
            where = me.get("device", "local")
            p50, p95, tps = me["bs1"]["p50_s"], me["bs1"]["p95_s"], me["bs1"]["gen_tok_per_s"]
            tps_s = (f"{tps:.1f} (bs1), " if tps else "") + f"{me['batched']['gen_tok_per_s']:.0f} (bs{me['batched']['batch_size']})"
        else:
            where = "AMD Token Factory"
            p50, p95, tps_s = me.get("latency_p50_s"), me.get("latency_p95_s"), "n/a"
        fmt = lambda x: "n/a" if x is None else f"{x:.2f} s"  # noqa: E731
        lines.append(f"| {name} | {where} | {fmt(p50)} | {fmt(p95)} | {tps_s} | "
                     f"{m['tokens']['prompt_per_answer']:.0f} | {m['tokens']['completion_per_answer']:.0f} |")
    for name, s in res["systems"].items():
        lines += ["", f"### Confusion matrix: {name} (rows gold, columns predicted)", "",
                  "| gold \\ pred | " + " | ".join(grader.VERDICTS + ["invalid"]) + " |", "|---|" + "---|" * 5]
        for g, row in s["metrics"]["confusion"].items():
            lines.append(f"| {g} | " + " | ".join(str(row[p]) for p in grader.VERDICTS + ["invalid"]) + " |")
    if "cost" in res:
        lines += ["", "## Cost", "", "```", json.dumps(res["cost"], indent=1), "```"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("predict-local")
    a.add_argument("--model", required=True)
    a.add_argument("--adapter")
    a.add_argument("--rows", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--name")
    a.add_argument("--prompt")
    a.add_argument("--bsz", type=int, default=16)
    a.add_argument("--latency-n", type=int, default=40)
    a.add_argument("--max-new", type=int, default=768)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--lora-scale", type=float, default=1.0)
    a.add_argument("--temperature", type=float, default=0.0, help="> 0 samples (top_p 0.95) instead of greedy")
    a.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("predict-api")
    b.add_argument("--model", required=True)
    b.add_argument("--rows", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--name")
    b.add_argument("--prompt")
    b.add_argument("--max-new", type=int, default=1500)
    b.add_argument("--reasoning", default="")
    b.add_argument("--json-mode", action="store_true")
    b.add_argument("--limit", type=int, default=0)
    v = sub.add_parser("verdict-probs")
    v.add_argument("--model", required=True)
    v.add_argument("--adapter")
    v.add_argument("--rows", required=True)
    v.add_argument("--preds", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--prompt")
    v.add_argument("--bsz", type=int, default=4)
    v.add_argument("--lora-scale", type=float, default=1.0)
    x = sub.add_parser("escalate")
    x.add_argument("--preds", required=True)
    x.add_argument("--probs", required=True)
    x.add_argument("--tau", type=float, required=True)
    x.add_argument("--out", required=True)
    vt = sub.add_parser("vote")
    vt.add_argument("--rows", required=True)
    vt.add_argument("--preds", action="append", required=True)
    vt.add_argument("--out", required=True)
    c = sub.add_parser("score")
    c.add_argument("--rows", required=True)
    c.add_argument("--system", action="append", required=True, help="name=preds.jsonl")
    c.add_argument("--reference", default="e4b_base")
    c.add_argument("--extra", action="append", help="key=json file to embed (verifier summary, train log, cost)")
    c.add_argument("--out", default="ml/results/eval.json")
    c.add_argument("--md")
    args = ap.parse_args()
    {"predict-local": predict_local, "predict-api": predict_api, "verdict-probs": verdict_probs, "escalate": escalate, "vote": vote, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
