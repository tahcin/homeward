"""Blind second-opinion labels for the synthetic teach-back answers.

A different model family from the generator (DeepSeek) re-grades every answer with the
shared grader prompt (prompts/grader.json, used verbatim). It never sees the designed
labels. To stay inside the shared 20 requests/minute limit, several answers from
different cases go into one request: each answer is the verbatim grader user prompt
under its own "### Answer n" heading, and a short wrapper added after the verbatim
system prompt asks for one result per answer.

Output: one JSONL row per answer with the verifier's verdicts next to the designed ones
and per-fact agreement. Resumable (skips keys already in --out).

Usage: uv run --with httpx python ml/verify.py --answers data/answers.jsonl --out data/verify.jsonl
"""

import argparse
import asyncio
import json
import random
import time
from collections import Counter
from pathlib import Path

import httpx

import grader
from tf import chat, content_of, parse_json

WRAPPER = """

This message holds several independent teach-back answers, each under its own heading "### Answer <n>" with its own facts, question and answer. Grade each answer on its own, using only its own facts. Reply with JSON only: {"answers": [{"n": 1, "facts": [...]}, {"n": 2, "facts": [...]}]} with one entry per answer, where each "facts" list has the format described above (every fact id of that answer, in its given order)."""


def batch_messages(rows: list[dict], prompt: dict) -> list[dict]:
    user = "\n\n".join(f"### Answer {i + 1}\n{grader.user_prompt(r, prompt)}" for i, r in enumerate(rows))
    return [{"role": "system", "content": prompt["system"] + WRAPPER},
            {"role": "user", "content": user}]


async def grade_batch(client, rows, prompt, model, reasoning):
    msgs = batch_messages(rows, prompt)
    for attempt in range(3):
        t0 = time.time()
        try:
            resp = await chat(client, model, msgs, max_tokens=900 * len(rows), temperature=0.0,
                              json_mode=True, reasoning_effort=reasoning)
            data = parse_json(content_of(resp))
            answers = {a.get("n"): a for a in data.get("answers", []) if isinstance(a, dict)}
        except Exception as e:  # noqa: BLE001
            print("batch error", str(e)[:200])
            continue
        usage = resp.get("usage", {})
        out = []
        for i, r in enumerate(rows):
            ids = [f["id"] for f in r["facts"]]
            got, strict = grader.verdicts_from(answers.get(i + 1), ids)
            designed = [lab["verdict"] for lab in r["labels"]]
            out.append({
                "key": grader.row_key(r), "case_id": r["case_id"], "language": r["language"],
                "model": model, "verifier": got, "designed": designed, "strict_json": strict,
                "agree": [g == d for g, d in zip(got, designed)],
                "agree_all": all(g == d for g, d in zip(got, designed)),
                "verifier_raw": answers.get(i + 1),
            })
        if all(o["strict_json"] for o in out) or attempt == 2:
            meta = {"prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "latency_s": round(time.time() - t0, 2), "batch": len(rows)}
            for o in out:
                o["batch_meta"] = meta
            return out
        print("batch had invalid items, retrying")
    return []


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in grader.VERDICTS) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else None


def summarize(rows: list[dict]) -> dict:
    pairs = [(d, v) for r in rows for d, v in zip(r["designed"], r["verifier"])]
    by_lang = {}
    for r in rows:
        by_lang.setdefault(r["language"], []).append(r["agree_all"])
    conf = Counter((d, v or "invalid") for d, v in pairs)
    return {
        "answers": len(rows),
        "facts": len(pairs),
        "answers_all_facts_agree": sum(r["agree_all"] for r in rows),
        "answer_agreement_rate": sum(r["agree_all"] for r in rows) / max(len(rows), 1),
        "fact_agreement_rate": sum(d == v for d, v in pairs) / max(len(pairs), 1),
        "cohen_kappa_facts": kappa([(d, v) for d, v in pairs if v]),
        "verifier_invalid_facts": sum(v is None for _, v in pairs),
        "confusion_designed_vs_verifier": {d: {v: conf[(d, v)] for v in grader.VERDICTS + ["invalid"]}
                                           for d in grader.VERDICTS},
        "answer_agreement_by_language": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_lang.items())},
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="data/answers.jsonl")
    ap.add_argument("--out", default="data/verify.jsonl")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--reasoning", default="none")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summary", default="ml/results/verify_summary.json")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    prompt = grader.load_prompt()
    rows = [json.loads(l) for l in Path(args.answers).read_text().splitlines() if l.strip()]
    out = Path(args.out)
    done = {}
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                done[r["key"]] = r
    todo = [r for r in rows if grader.row_key(r) not in done]
    if args.limit:
        todo = todo[:args.limit]
    if not args.summary_only and todo:
        random.Random(5).shuffle(todo)  # mix cases and languages inside each batch
        groups = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
        print(f"{len(todo)} answers to verify in {len(groups)} requests ({len(done)} done)")
        async with httpx.AsyncClient() as client:
            with out.open("a") as fh:
                jobs = [grade_batch(client, g, prompt, args.model, args.reasoning) for g in groups]
                for k, coro in enumerate(asyncio.as_completed(jobs), 1):
                    res = await coro
                    for r in res:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                        done[r["key"]] = r
                    fh.flush()
                    if k % 10 == 0:
                        print(f"{k}/{len(groups)} requests done")
    keys = {grader.row_key(r) for r in rows}
    vrows = [r for k, r in done.items() if k in keys]
    s = summarize(vrows)
    s["model"] = args.model
    s["reasoning_effort"] = args.reasoning
    s["batch_size"] = args.batch
    seen, pt, ct = set(), 0, 0
    for r in vrows:
        m = r.get("batch_meta") or {}
        sig = (m.get("latency_s"), m.get("prompt_tokens"), m.get("completion_tokens"))
        if m and sig not in seen:
            seen.add(sig)
            pt += m.get("prompt_tokens", 0)
            ct += m.get("completion_tokens", 0)
    s["requests_counted"], s["prompt_tokens"], s["completion_tokens"] = len(seen), pt, ct
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(s, indent=1))
    print(json.dumps({k: v for k, v in s.items() if k != "confusion_designed_vs_verifier"}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
