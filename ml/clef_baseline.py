"""Clef (Token Factory decision model) zero-shot baseline: one multiple-choice question per fact.

  python clef_baseline.py [rows.jsonl] [out_preds.jsonl]
"""
import asyncio, json, os, sys, time
import httpx
ML = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(ML)
sys.path.insert(0, ML)
from tf import _pace, _sem, API_KEY, BASE_URL
ROWS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(APP, "data/test_independent.jsonl")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ML, "results/preds_clef_test_independent.jsonl")
CRIT = {"correct": "conveys it accurately enough to act on safely (paraphrase or everyday medicine names are fine if unambiguous)",
        "partial": "conveys part of it but leaves out or is vague about a key element; says nothing wrong",
        "incorrect": "says something that conflicts with it (wrong dose, timing, medicine, action, threshold, date or person)",
        "missing": "does not mention it"}
async def one(c, r):
    key = r.get("key") or f"{r['case_id']}/{r['topic_id']}/{r['n']}"
    state = {"question_asked": r["question"], "speaker_answer": r["transcript"], "answer_language": r["language_name"],
             "facts": [{"id": f["id"], "statement": f["statement"]} for f in r["facts"]]}
    qs = {f["id"]: {"type": "choice", "instructions": f"Did the speaker's answer show they understood fact {f['id']}: \"{f['statement']}\"? When unsure between two verdicts, choose the less reassuring one.", "criteria": CRIT} for f in r["facts"]}
    for attempt in range(6):
        async with _sem:
            await _pace()
            t0 = time.time()
            try:
                resp = await c.post(f"{BASE_URL}/systemone", json={"model": "Clef", "state": state, "questions": qs},
                                    headers={"Authorization": f"Bearer {API_KEY}"}, timeout=120)
                if resp.status_code == 200:
                    d = resp.json()
                    facts = [{"id": fid, "verdict": a["choice"], "probabilities": a.get("probabilities")} for fid, a in d["answers"].items()]
                    return {"key": key, "text": json.dumps({"facts": facts}), "latency_s": time.time() - t0, "prompt_tokens": d.get("usage", {}).get("input_tokens")}
                err = f"{resp.status_code} {resp.text[:200]}"
            except Exception as e:
                err = repr(e)[:200]
        await asyncio.sleep(5 * (attempt + 1))
    return {"key": key, "error": err, "text": ""}
async def main():
    rows = [json.loads(l) for l in open(ROWS)]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["key"] for l in open(OUT) if json.loads(l).get("text")}
    todo = [r for r in rows if (r.get("key") or f"{r['case_id']}/{r['topic_id']}/{r['n']}") not in done]
    print(len(todo), "to do", flush=True)
    async with httpx.AsyncClient() as c:
        with open(OUT, "a") as fh:
            for coro in asyncio.as_completed([one(c, r) for r in todo]):
                res = await coro
                if res.get("text"):
                    fh.write(json.dumps(res, ensure_ascii=False) + "\n"); fh.flush()
                else:
                    print("failed", res["key"], res.get("error"), flush=True)
asyncio.run(main())
