"""Clef (Token Factory decision model) as an extra juror. Same request format as clef_baseline.py.

  eval mode:  python clef_label.py eval <rows.jsonl> <out_preds.jsonl>
  train mode: python clef_label.py train <out juror.jsonl> [--until 2026-10-17T10:30]
     builds the training pool exactly like build_sft_v2 (v1, v2 remapped n+1000, v3 + targeted files,
     eligibility and contamination filters), labels rows not yet in the output, re-scans for new files
     (answers_targeted_t2) every 5 min until --until.
Pacing: TF_RPM / TF_CONCURRENCY from the environment (tf.py sliding window).
"""
import asyncio, glob, json, os, sys, time
from datetime import datetime
import httpx

ML = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(ML)
sys.path.insert(0, ML)
from tf import _pace, _sem, API_KEY, BASE_URL  # noqa: E402
import grader  # noqa: E402

CRIT = {"correct": "conveys it accurately enough to act on safely (paraphrase or everyday medicine names are fine if unambiguous)",
        "partial": "conveys part of it but leaves out or is vague about a key element; says nothing wrong",
        "incorrect": "says something that conflicts with it (wrong dose, timing, medicine, action, threshold, date or person)",
        "missing": "does not mention it"}


def rj(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


async def one(c, r, key):
    state = {"question_asked": r["question"], "speaker_answer": r["transcript"], "answer_language": r["language_name"],
             "facts": [{"id": f["id"], "statement": f["statement"]} for f in r["facts"]]}
    qs = {f["id"]: {"type": "choice", "instructions": f"Did the speaker's answer show they understood fact {f['id']}: \"{f['statement']}\"? When unsure between two verdicts, choose the less reassuring one.", "criteria": CRIT} for f in r["facts"]}
    err = ""
    for attempt in range(6):
        async with _sem:
            await _pace()
            t0 = time.time()
            try:
                resp = await c.post(f"{BASE_URL}/systemone", json={"model": "Clef", "state": state, "questions": qs},
                                    headers={"Authorization": f"Bearer {API_KEY}"}, timeout=120)
                if resp.status_code == 200:
                    d = resp.json()
                    ans = d["answers"]
                    facts = [{"id": f["id"], "verdict": ans.get(f["id"], {}).get("choice"),
                              "confidence": ans.get(f["id"], {}).get("confidence"),
                              "probabilities": ans.get(f["id"], {}).get("probabilities")} for f in r["facts"]]
                    return {"key": key, "facts": facts, "latency_s": round(time.time() - t0, 2),
                            "prompt_tokens": d.get("usage", {}).get("input_tokens")}
                err = f"{resp.status_code} {resp.text[:200]}"
            except Exception as e:  # noqa: BLE001
                err = repr(e)[:200]
        await asyncio.sleep(5 * (attempt + 1))
    return {"key": key, "error": err}


def train_pool():
    sys.argv_backup = None
    cases = {c["case_id"]: c for c in rj(f"{APP}/data/cases.jsonl") + rj(f"{APP}/data/cases_v2.jsonl")}
    held = set(json.load(open(f"{APP}/data/splits/holdout.json"))["conditions"])
    evs = [rj(f"{APP}/data/splits/test.jsonl"), rj(f"{APP}/data/splits/val.jsonl")] + \
          [rj(f"{APP}/data/{n}.jsonl") for n in ("test_independent", "val_independent", "dev_independent")]
    from build_sft_v2 import h
    eval_cases = {r["case_id"] for rows in evs for r in rows}
    eval_hashes = {h(r["transcript"]) for rows in evs for r in rows}
    # priority: targeted, v3, v2, v1
    tg = [r for p in sorted(glob.glob(f"{APP}/data/answers_targeted*.jsonl")) for r in rj(p)]
    v3 = [r for p in sorted(glob.glob(f"{APP}/data/answers_v3*.jsonl")) for r in rj(p)]
    v2 = [dict(r, n=r["n"] + 1000 if r["n"] < 1000 else r["n"]) for r in rj(f"{APP}/data/answers_v2.jsonl")]
    v1 = rj(f"{APP}/data/answers.jsonl")
    out = []
    for r in tg + v3 + v2 + v1:
        c = cases.get(r["case_id"])
        if not c or c.get("condition") in held or r["case_id"] in eval_cases or h(r["transcript"]) in eval_hashes:
            continue
        out.append(r)
    return out


async def label(rows, outp, fmt):
    done = {x["key"] for x in rj(outp)}
    todo = [r for r in rows if grader.row_key(r) not in done]
    print(datetime.now().strftime("%H:%M:%S"), len(todo), "to do of", len(rows), flush=True)
    if not todo:
        return 0
    n = 0
    async with httpx.AsyncClient() as c:
        with open(outp, "a") as fh:
            # bounded batches so a long queue does not hold thousands of pending coroutines
            for i in range(0, len(todo), 40):
                for coro in asyncio.as_completed([one(c, r, grader.row_key(r)) for r in todo[i:i + 40]]):
                    res = await coro
                    if "facts" not in res:
                        print("failed", res["key"], res.get("error"), flush=True)
                        continue
                    if fmt == "preds":
                        res = {"key": res["key"], "text": json.dumps({"facts": [{"id": f["id"], "verdict": f["verdict"], "probabilities": f["probabilities"]} for f in res["facts"]]}),
                               "facts": res["facts"], "latency_s": res["latency_s"], "prompt_tokens": res["prompt_tokens"]}
                    else:
                        res["verifier"] = [f["verdict"] for f in res["facts"]]
                        res["model"] = "Clef"
                    fh.write(json.dumps(res, ensure_ascii=False) + "\n"); fh.flush()
                    n += 1
                print(datetime.now().strftime("%H:%M:%S"), "done", n, "/", len(todo), flush=True)
    return n


def main():
    mode = sys.argv[1]
    if mode == "eval":
        asyncio.run(label(rj(sys.argv[2]), sys.argv[3], "preds"))
        return
    outp = sys.argv[2]
    until = sys.argv[sys.argv.index("--until") + 1] if "--until" in sys.argv else "2026-10-17T10:30"
    end = datetime.fromisoformat(until)
    async def loop():  # one event loop: tf.py's semaphore is bound to the first loop it is used in
        while datetime.now() < end:
            n = await label(train_pool(), outp, "juror")
            if n == 0:
                await asyncio.sleep(300)
    asyncio.run(loop())
    print("CLEFDONE", flush=True)


if __name__ == "__main__":
    main()
