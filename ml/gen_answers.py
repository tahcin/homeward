"""Generate teach-back answers with designed per-fact verdicts.

For each topic of each case, sample a target verdict for every fact, then ask a
model to write what a patient (or caregiver) would actually say, in the
patient's language, as a speech transcript that hits exactly those verdicts.
The designed verdicts are the labels. A separate blind pass (verify.py) checks
them.

Usage: uv run --with httpx python ml/gen_answers.py --cases data/cases.jsonl --k 4 --out data/answers.jsonl
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import httpx

from tf import chat, content_of, parse_json

VERDICT_WEIGHTS = {"correct": 0.42, "partial": 0.2, "incorrect": 0.28, "missing": 0.10}
# v2 "hard" mode: more wrong answers, and subtle errors that a grader is most likely to wave through.
HARD_WEIGHTS = {"correct": 0.32, "partial": 0.2, "incorrect": 0.36, "missing": 0.12}
SUBTLE = {
    "medication": ["near_number", "negation_flip", "wrong_drug", "wrong_timing"],
    "medication_change": ["negation_flip", "wrong_action", "near_number"],
    "warning_sign": ["wrong_threshold", "wrong_symptom", "negation_flip"],
    "follow_up": ["wrong_date", "wrong_who"],
    "self_monitoring": ["near_number", "wrong_threshold"],
    "activity": ["negation_flip", "wrong_threshold"],
    "diet": ["negation_flip", "wrong_threshold"],
    "wound_care": ["negation_flip", "wrong_timing"],
    "device": ["wrong_action", "wrong_timing"],
}
# Per-model request settings: keep thinking off where the model allows it.
MODEL_OPTS = {
    "DeepSeek-V4-Flash": {},
    "Gemma-4-31B-IT": {},
    "Qwen3.8-Flash-Next": {"reasoning_effort": "none"},
    "MiMo-V2.6-Flash": {"reasoning_effort": "none"},
    "GLM-5.3-Flash": {"reasoning_effort": "low"},
}
ERRORS = {
    "medication": ["wrong_number", "wrong_timing", "wrong_drug", "wrong_action"],
    "medication_change": ["wrong_action", "wrong_number", "wrong_timing"],
    "warning_sign": ["wrong_threshold", "wrong_action", "wrong_symptom"],
    "follow_up": ["wrong_date", "wrong_who"],
    "self_monitoring": ["wrong_threshold", "wrong_number", "wrong_timing"],
    "activity": ["wrong_threshold", "wrong_action"],
    "diet": ["wrong_action", "wrong_threshold"],
    "wound_care": ["wrong_action", "wrong_timing"],
    "device": ["wrong_action", "wrong_timing"],
}
ERROR_HINT = {
    "wrong_number": "states a different dose, amount or count",
    "wrong_timing": "states the wrong frequency, time of day or duration",
    "wrong_drug": "attaches this instruction to a different medicine from the list",
    "wrong_action": "plans the opposite action (keeps taking a stopped medicine, stops a medicine that must continue, does an activity that is forbidden)",
    "wrong_threshold": "uses the wrong trigger or limit (wrong weight gain, wrong temperature, waits too long before getting help)",
    "wrong_symptom": "names a different symptom than the real warning sign, or plans to wait it out",
    "wrong_date": "gives the wrong day or week for the appointment",
    "wrong_who": "plans to see the wrong clinician or go to the wrong place",
    "near_number": "gets a number only slightly wrong (20 mg instead of 25 mg, every 8 hours instead of every 6, 3 days instead of 5) while sounding confident",
    "negation_flip": "sounds almost right but flips one negation or condition (takes it with food when it must be without, keeps a medicine that was stopped, calls only if it gets worse)",
}
ASR_STYLES = [
    "speech-recognition errors: a few misheard words, missing punctuation, one medicine name slightly misspelled as heard",
    "two languages mixed mid-sentence as people really speak",
    "an older speaker who repeats themselves and trails off",
]
SPEAKERS = ["the patient", "the patient", "the patient", "an adult child who will care for the patient",
            "the patient's spouse"]
STYLES = [
    "plain conversational speech",
    "hesitant speech with fillers and self-corrections",
    "short answers, mixing in English drug names and numbers",
    "uses everyday names for medicines (the water pill, the blood thinner, the white one) where natural",
    "long rambling answer with an unrelated remark",
]

SYSTEM = """You create realistic, fictional speech transcripts of patients explaining discharge instructions back to a nurse ("teach-back"), for training software that checks understanding. You follow the requested correctness labels exactly. Output one JSON object only."""

PROMPT = """Discharge facts being checked (from a fictional case, condition: {condition}):
{facts}

The nurse asks: "{question}"

Write {k} different answers, each a speech-to-text transcript in {language_name}{script_note}. Speaker and style differ per answer as listed. Each answer must match its per-fact labels exactly:
- correct: conveys the fact accurately enough to act on safely. Paraphrase is fine; an everyday name for a medicine is fine if it is unambiguous.
- partial: conveys part of the fact but leaves out or is vague about a key element (for example knows to take it but not how often). Says nothing wrong about it.
- incorrect: says something that conflicts with the fact, in the stated way.
- missing: does not mention the fact at all.

{plans}

Return JSON: {{"answers": [{{"n": 1, "transcript": "...", "facts": [{{"id": "f1", "verdict": "correct|partial|incorrect|missing", "heard": "short English rendering of what the speaker said about this fact, or empty if missing", "reteach": "one short English sentence the nurse should re-teach, or empty if correct"}}]}}]}}
Keep every fact id of the topic in every answer's "facts" list, in the given order."""


from gen_cases import LANGUAGES


def balance_language(case: dict) -> dict:
    """Assign languages round-robin by case number so every language gets the same share.

    The case generator picked languages at random, which left some with a single case.
    """
    code, name = LANGUAGES[(int(case["case_id"][1:]) - 1) % len(LANGUAGES)]
    return {**case, "language": code, "language_name": name}


def plan_for(rng: random.Random, facts: list[dict], hard: bool = False) -> list[dict]:
    weights = HARD_WEIGHTS if hard else VERDICT_WEIGHTS
    plan = []
    for f in facts:
        v = rng.choices(list(weights), weights=list(weights.values()))[0]
        item = {"id": f["id"], "verdict": v}
        if v == "incorrect":
            pool = (SUBTLE if hard and rng.random() < 0.6 else ERRORS).get(f.get("category"), ["wrong_action"])
            err = rng.choice(pool)
            item["error"] = err
        plan.append(item)
    return plan


def describe_plan(n: int, speaker: str, style: str, plan: list[dict]) -> str:
    parts = []
    for p in plan:
        s = f'{p["id"]}={p["verdict"]}'
        if p.get("error"):
            s += f' ({ERROR_HINT[p["error"]]})'
        parts.append(s)
    return f"Answer {n}: speaker is {speaker}; style: {style}; labels: " + "; ".join(parts)


N_OFFSET = 0


async def one_topic(client, case, topic, k, rng, model, hard=False):
    facts = [f for f in case["facts"] if f["id"] in topic["fact_ids"]]
    facts.sort(key=lambda f: topic["fact_ids"].index(f["id"]))
    plans = []
    for n in range(1, k + 1):
        style = rng.choice(STYLES + (ASR_STYLES * 2 if hard else []))
        plans.append({"n": n, "speaker": rng.choice(SPEAKERS), "style": style,
                      "plan": plan_for(rng, facts, hard)})
    # Guarantee at least one fully correct answer per topic so the grader sees clean passes.
    if not any(all(p["verdict"] == "correct" for p in pl["plan"]) for pl in plans):
        for p in plans[0]["plan"]:
            p["verdict"] = "correct"
            p.pop("error", None)
    script_note = ""
    if case["language"] in ("hi", "bn", "ta"):
        script_note = " (native script, with the code-mixing of English words common in everyday speech)"
    elif case["language"] == "tl":
        script_note = " (Taglish where natural)"
    prompt = PROMPT.format(
        condition=case["condition"],
        facts="\n".join(f'- {f["id"]} [{f["category"]}{", critical" if f.get("critical") else ""}]: {f["statement"]}' for f in facts),
        question=topic["question"], k=k, language_name=case["language_name"], script_note=script_note,
        plans="\n".join(describe_plan(pl["n"], pl["speaker"], pl["style"], pl["plan"]) for pl in plans),
    )
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    for _ in range(3):
        try:
            resp = await chat(client, model, msgs, max_tokens=6000, temperature=0.9, json_mode=True,
                              **MODEL_OPTS.get(model, {}))
            data = parse_json(content_of(resp))
            answers = data["answers"]
        except Exception as e:  # noqa: BLE001
            print(case["case_id"], topic["id"], "error", str(e)[:200])
            continue
        rows = []
        ok = isinstance(answers, list) and len(answers) == k and all(
            isinstance(a, dict) and isinstance(a.get("facts"), list)
            and all(isinstance(f, dict) for f in a["facts"]) and isinstance(a.get("transcript"), str)
            for a in answers)
        if not ok:
            print(case["case_id"], topic["id"], "malformed answers, retrying")
            continue
        for pl, ans in zip(plans, answers):
            got = {f.get("id"): f for f in ans.get("facts", [])}
            if [p["id"] for p in pl["plan"]] != [f.get("id") for f in ans.get("facts", [])]:
                ok = False
                break
            # The designed plan is the label; the model's own labels must agree with it.
            if any(got[p["id"]].get("verdict") != p["verdict"] for p in pl["plan"]):
                ok = False
                break
            rows.append({
                "case_id": case["case_id"], "topic_id": topic["id"], "n": pl["n"] + N_OFFSET,
                "language": case["language"], "language_name": case["language_name"],
                "question": topic["question"], "speaker": pl["speaker"], "style": pl["style"],
                "facts": facts, "transcript": ans["transcript"], "generator": model,
                "labels": [{**p, "heard": got[p["id"]].get("heard", ""),
                            "reteach": got[p["id"]].get("reteach", "")} for p in pl["plan"]],
            })
        if ok:
            return rows
        print(case["case_id"], topic["id"], "label mismatch, retrying")
    return []


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/cases.jsonl")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--model", default="DeepSeek-V4-Flash")
    ap.add_argument("--out", default="data/answers.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--models", default="", help="comma list; rotates per topic (overrides --model)")
    ap.add_argument("--hard", action="store_true", help="more wrong answers and subtle errors")
    ap.add_argument("--n-offset", type=int, default=0, help="added to answer n so keys never collide with earlier runs")
    ap.add_argument("--cases-allow", default="", help="file with allowed case ids, one per line")
    args = ap.parse_args()

    cases = [balance_language(json.loads(l)) for l in Path(args.cases).read_text().splitlines() if l.strip()]
    if args.cases_allow:
        allow = set(Path(args.cases_allow).read_text().split())
        cases = [c for c in cases if c["case_id"] in allow]
    if args.limit:
        cases = cases[:args.limit]
    models = [m for m in args.models.split(",") if m] or [args.model]
    global N_OFFSET
    N_OFFSET = args.n_offset
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(r["case_id"], r["topic_id"]) for r in map(json.loads, out.read_text().splitlines()) if r}
    jobs = []
    for c in cases:
        for t in c["topics"]:
            if (c["case_id"], t["id"]) not in done:
                rng = random.Random(f'{args.seed}-{c["case_id"]}-{t["id"]}')
                model = models[len(jobs) % len(models)]
                jobs.append(one_topic_wrapped(c, t, args.k, rng, model, args.hard))
    print(f"{len(jobs)} topics to generate")
    global _client
    async with httpx.AsyncClient() as client:
        _client = client
        with out.open("a") as fh:
            for coro in asyncio.as_completed(jobs):
                rows = await coro
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                if rows:
                    print("ok", rows[0]["case_id"], rows[0]["topic_id"], len(rows))


_client = None


async def one_topic_wrapped(c, t, k, rng, model, hard=False):
    while _client is None:
        await asyncio.sleep(0.05)
    try:
        return await one_topic(_client, c, t, k, rng, model, hard)
    except Exception as e:  # noqa: BLE001  one bad topic must not stop the run
        print(c["case_id"], t["id"], "failed:", repr(e)[:200])
        return []


if __name__ == "__main__":
    asyncio.run(main())
