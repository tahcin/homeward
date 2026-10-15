"""Generate synthetic hospital discharge cases with structured, checkable facts.

Every case is fictional. Each one has a clinician-style discharge summary, the
atomic facts a patient must understand before going home, and teach-back
questions that group those facts.

Usage: uv run --with httpx python ml/gen_cases.py --n 72 --out data/cases.jsonl
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import httpx

from tf import chat, content_of, parse_json

CONDITIONS = [
    # (condition, typical med count range)
    ("heart failure exacerbation, new diuretic dose and daily weights", (4, 7)),
    ("atrial fibrillation, newly started on an anticoagulant", (3, 5)),
    ("community-acquired pneumonia, oral antibiotics to finish at home", (2, 4)),
    ("COPD exacerbation, steroid taper and new inhaler", (3, 6)),
    ("type 2 diabetes, newly started on basal insulin", (3, 6)),
    ("diabetic ketoacidosis in type 1 diabetes, insulin plan changes", (2, 4)),
    ("laparoscopic appendectomy, wound care and pain control", (2, 4)),
    ("total knee replacement, blood-clot prevention and physiotherapy", (3, 5)),
    ("hip fracture repair in an older adult, fall prevention", (3, 6)),
    ("acute myocardial infarction with stent, dual antiplatelet therapy", (5, 7)),
    ("ischemic stroke, secondary prevention and swallowing advice", (4, 6)),
    ("deep vein thrombosis, anticoagulant started", (2, 4)),
    ("cellulitis of the leg, oral antibiotics and leg elevation", (2, 3)),
    ("kidney stone, pain control and straining urine", (2, 4)),
    ("urinary tract infection with sepsis in an older adult", (2, 4)),
    ("asthma attack, steroid burst and inhaler technique", (2, 4)),
    ("upper gastrointestinal bleed from an ulcer, acid suppression, stopping NSAIDs", (2, 5)),
    ("new seizure, antiseizure medicine started and driving restrictions", (1, 3)),
    ("hypertensive urgency, blood pressure medicine changes", (3, 5)),
    ("acute pancreatitis, alcohol cessation and diet", (2, 4)),
    ("gestational hypertension after delivery, postpartum warning signs", (1, 3)),
    ("chemotherapy-related neutropenic fever, infection precautions", (2, 4)),
    ("cholecystectomy, diet and wound care", (2, 3)),
    ("lithium toxicity, medicine dose change and blood tests", (2, 4)),
]

LANGUAGES = [
    ("it", "Italian"), ("es", "Spanish"), ("hi", "Hindi"), ("bn", "Bengali"),
    ("tl", "Tagalog"), ("ar", "Arabic"), ("zh", "Mandarin Chinese"), ("en", "English"),
    ("pt", "Portuguese"), ("ta", "Tamil"), ("fr", "French"), ("vi", "Vietnamese"),
]

SYSTEM = """You write realistic but entirely fictional hospital discharge material for training and testing patient-education software. Never use real people. Use generic drug names and realistic adult doses. Output one JSON object only."""

PROMPT = """Write one fictional discharge case.

Patient: {age}-year-old {sex}, preferred language {language_name}. Condition: {condition}.
The discharge medication list has {n_meds} items. Include at least one change that patients often get wrong: a {change_kind}.
Health literacy: {literacy}.

Return JSON with exactly these keys:
- "patient_name": a fictional name suited to a {language_name} speaker
- "summary_text": the discharge instructions as a busy clinician writes them in English, 220-420 words, with typical abbreviations (PO, BID, PRN, f/u, s/p), a medication list with doses and frequencies, return precautions, follow-up appointments with fictional dates in November 2026, and activity or diet limits.
- "facts": 7-12 facts the patient must understand before going home. Each fact is an object:
  {{"id": "f1", "category": one of ["medication", "medication_change", "warning_sign", "follow_up", "activity", "diet", "self_monitoring", "wound_care", "device"],
    "critical": true if misunderstanding it could cause serious harm,
    "statement": one plain-English sentence a patient could follow,
    "anchors": a JSON array of strings, the exact tokens that must survive any rewrite, copied character for character from the statement: always the drug name for medication facts, plus numbers with units, times, dates and thresholds}}
  Make facts non-overlapping: one fact per medicine (dose, timing and any change together), one per warning-sign group, one per appointment.
  Cover every medication, every warning sign, every follow-up and every limit in the summary. Do not invent facts that are not in the summary.
- "topics": 3-5 teach-back topics. Each is {{"id": "t1", "question": an open question a nurse asks so the patient explains in their own words (e.g. "Tell me how you will take ..."), "fact_ids": the 1-4 facts it checks}}. Every fact belongs to exactly one topic.
"""

CHANGE_KINDS = [
    "dose change of a medicine the patient already took at home",
    "home medicine the patient must stop taking",
    "medicine that must be tapered over several days",
    "medicine taken only when needed with a maximum daily amount",
    "medicine with a specific timing rule (with food, at bedtime, on an empty stomach)",
    "new medicine with a fixed end date",
]

LITERACY = ["limited, prefers short sentences", "average", "limited, relies on a family caregiver", "good"]


def make_specs(n: int, seed: int, offset: int = 0, exclude: frozenset = frozenset()) -> list[dict]:
    rng = random.Random(seed)
    conds = [c for c in CONDITIONS if c[0] not in exclude]
    specs = []
    for i in range(n):
        cond, (lo, hi) = conds[i % len(conds)]
        code, lang = LANGUAGES[rng.randrange(len(LANGUAGES))]
        specs.append({
            "case_id": f"c{offset + i + 1:03d}",
            "condition": cond,
            "age": rng.randint(24, 88),
            "sex": rng.choice(["woman", "man"]),
            "language": code,
            "language_name": lang,
            "n_meds": rng.randint(lo, hi),
            "change_kind": rng.choice(CHANGE_KINDS),
            "literacy": rng.choice(LITERACY),
        })
    return specs


def problems(case: dict) -> list[str]:
    facts = case.get("facts") or []
    topics = case.get("topics") or []
    ids = [f.get("id") for f in facts]
    covered = [fid for t in topics for fid in t.get("fact_ids", [])]
    out = []
    if not 5 <= len(facts) <= 14:
        out.append(f"{len(facts)} facts")
    if not 2 <= len(topics) <= 6:
        out.append(f"{len(topics)} topics")
    if sorted(covered) != sorted(ids):
        out.append(f"topic coverage {sorted(set(ids) ^ set(covered))} dup={len(covered) != len(set(covered))}")
    for f in facts:
        if not f.get("statement") or not isinstance(f.get("anchors"), list):
            out.append(f"fact {f.get('id')} malformed")
        elif f.get("category", "").startswith("medication") and not f["anchors"]:
            out.append(f"fact {f.get('id')} has no anchors")
    return out


def repair(case: dict) -> None:
    """Fix common shape slips: anchors given as one string, facts left out of topics, duplicates."""
    facts = case.get("facts") or []
    for f in facts:
        if isinstance(f.get("anchors"), str):
            f["anchors"] = [a.strip() for a in f["anchors"].split(",") if a.strip()]
    topics = [t for t in case.get("topics") or [] if isinstance(t, dict)]
    ids = [f.get("id") for f in facts]
    seen = set()
    for t in topics:
        t["fact_ids"] = [i for i in t.get("fact_ids", []) if i in ids and not (i in seen or seen.add(i))]
    topics = [t for t in topics if t["fact_ids"]]
    for i in ids:
        if i not in seen and topics:
            min(topics, key=lambda t: len(t["fact_ids"]))["fact_ids"].append(i)
            seen.add(i)
    case["topics"] = topics


def valid(case: dict) -> bool:
    return not problems(case)


async def one(client: httpx.AsyncClient, spec: dict, model: str) -> dict | None:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(**spec)}]
    for _ in range(3):
        try:
            resp = await chat(client, model, msgs, max_tokens=5000, temperature=0.8, json_mode=True)
            case = parse_json(content_of(resp))
        except Exception as e:  # noqa: BLE001
            print(spec["case_id"], "error", str(e)[:200])
            continue
        repair(case)
        for f in case.get("facts") or []:
            if isinstance(f.get("anchors"), list) and f.get("statement"):
                # Keep only anchors that really occur in the statement, so the fact lock can rely on them.
                f["anchors"] = [a for a in f["anchors"] if isinstance(a, str) and a and a in f["statement"]]
        if valid(case):
            return {**spec, **case}
        print(spec["case_id"], "invalid, retrying:", problems(case)[:4])
    return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=72)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="DeepSeek-V4-Flash")
    ap.add_argument("--out", default="data/cases.jsonl")
    ap.add_argument("--offset", type=int, default=0, help="first case number minus one")
    ap.add_argument("--exclude", default="", help="JSON file with {'conditions': [...]} to leave out (held-out test conditions)")
    ap.add_argument("--models", default="", help="comma list; rotates per case")
    args = ap.parse_args()

    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["case_id"] for l in out.read_text().splitlines() if l.strip()}
    exclude = frozenset(json.loads(Path(args.exclude).read_text())["conditions"]) if args.exclude else frozenset()
    specs = [s for s in make_specs(args.n, args.seed, args.offset, exclude) if s["case_id"] not in done]
    models = [m for m in args.models.split(",") if m] or [args.model]
    print(f"{len(specs)} cases to generate ({len(done)} already done)")

    async with httpx.AsyncClient() as client:
        with out.open("a") as fh:
            jobs = [one(client, s, models[i % len(models)]) for i, s in enumerate(specs)]
            for coro in asyncio.as_completed(jobs):
                case = await coro
                if case:
                    fh.write(json.dumps(case, ensure_ascii=False) + "\n")
                    fh.flush()
                    print("ok", case["case_id"], case["language_name"], len(case["facts"]), "facts")


if __name__ == "__main__":
    asyncio.run(main())
