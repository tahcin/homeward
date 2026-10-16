"""Run-2 training data: teacher-filtered, deduplicated, contamination-checked SFT set.

Pool: v1 answers from training cases + data v2 answers (multi-model hard examples).
Training cases exclude: held-out conditions, in-distribution test cases, generator-val
cases (also the val_independent cases) and the dev_independent cases.

Label rule per answer:
  1. Gemma-4-31B-IT (teacher) agrees with the designed verdict on every fact -> keep,
     with the teacher's heard/reteach text (sequence-level distillation).
  2. Otherwise, if designed, verifier (Qwen3.8-Flash-Next) and teacher all exist: per-fact
     2-of-3 vote; keep only if every fact has a majority; text from the teacher where it
     matches the majority, else the designed text.
  3. No teacher label: keep only if the verifier agreed on every fact (run-1 rule), unless
     --require-teacher.
Then: near-duplicate removal (character 5-gram Jaccard > 0.8 within a case topic), at most
--cap answers per case, oversampling of near-miss pairs (same topic, verdicts differ on
exactly one fact) and of weak languages, and a hash check that no transcript or case from
any evaluation set is in training. Also writes a copy with "heard" before "verdict".

Usage: python3 ml/build_sft_v2.py --out data/splits_v2
"""

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import grader

APP = Path(__file__).resolve().parent.parent


def rj(p):
    p = APP / p if not str(p).startswith("/") else Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def norm_t(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    s = re.sub(r"\b(uh|um|eh|ah|hmm|uhm|mm)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def grams(s, n=5):
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def h(s):
    return hashlib.sha1(norm_t(s).encode()).hexdigest()


def target(a, labels, heard_first=False):
    facts = []
    for lab in labels:
        v = lab["verdict"]
        heard = "" if v == "missing" else (lab.get("heard") or "")
        reteach = "" if v == "correct" else (lab.get("reteach") or "")
        facts.append({"id": lab["id"], "heard": heard, "verdict": v, "reteach": reteach} if heard_first
                     else {"id": lab["id"], "verdict": v, "heard": heard, "reteach": reteach})
    return json.dumps({"facts": facts}, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="data/answers.jsonl")
    ap.add_argument("--answers-v2", default="data/answers_v2.jsonl")
    ap.add_argument("--answers-v3", action="append", default=[], help="further answer files (unique n, no remap), e.g. data/answers_v3.jsonl")
    ap.add_argument("--cases", default="data/cases.jsonl")
    ap.add_argument("--cases-v2", default="data/cases_v2.jsonl")
    ap.add_argument("--verify", default="data/verify.jsonl", help="verifier labels (v1 and, if present, v2)")
    ap.add_argument("--verify-v2", action="append", default=["data/verify_v2.jsonl"])
    ap.add_argument("--teacher", action="append", default=["data/teacher_31b.jsonl"])
    ap.add_argument("--require-teacher", action="store_true")
    ap.add_argument("--cap", type=int, default=30)
    ap.add_argument("--near-miss-x", type=float, default=1.5)
    ap.add_argument("--weak-langs", default="ta,bn,en,ar")
    ap.add_argument("--weak-x", type=float, default=1.25)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--out", default="data/splits_v2")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    prompt = grader.load_prompt()

    cases = {c["case_id"]: c for c in rj(a.cases) + rj(a.cases_v2)}
    held = set(json.loads((APP / "data/splits/holdout.json").read_text())["conditions"])
    eval_sets = {"test": rj("data/splits/test.jsonl"), "val": rj("data/splits/val.jsonl"),
                 "test_independent": rj("data/test_independent.jsonl"),
                 "val_independent": rj("data/val_independent.jsonl"), "dev_independent": rj("data/dev_independent.jsonl")}
    eval_cases = {r["case_id"] for rows in eval_sets.values() for r in rows}
    eval_hashes = {h(r["transcript"]) for rows in eval_sets.values() for r in rows}

    ver = {r["key"]: r for p in [a.verify] + a.verify_v2 for r in rj(p)}
    teach = {r["key"]: r for p in a.teacher for r in rj(p)}
    # answers_v2 reuses v1 case ids with n restarting at 1, so its keys collide with v1;
    # remap v2 to n + 1000 (the teacher labels for v2 are produced on the same remapped copy).
    v2 = [dict(r, n=r["n"] + 1000 if r["n"] < 1000 else r["n"], _src="v2") for r in rj(a.answers_v2)]
    v3 = [dict(r, _src="v3") for p in a.answers_v3 for r in rj(p)]
    pool = [dict(r, _src="v1") for r in rj(a.answers)] + v2 + v3
    keys = [grader.row_key(r) for r in pool]
    assert len(keys) == len(set(keys)), "duplicate answer keys in the pool"

    stats = Counter()
    kept = []
    dropped_examples = []
    for r in pool:
        c = cases.get(r["case_id"], {})
        if c.get("condition") in held or r["case_id"] in eval_cases or not c:
            stats["excluded_eval_or_heldout_case"] += 1
            continue
        if h(r["transcript"]) in eval_hashes:
            stats["excluded_transcript_in_eval"] += 1
            continue
        k = grader.row_key(r)
        designed = [l["verdict"] for l in r["labels"]]
        t, v = teach.get(k), ver.get(k)
        traw = {f.get("id"): f for f in ((t or {}).get("verifier_raw") or {}).get("facts", []) if isinstance(f, dict)}
        if t and t.get("agree_all"):
            labels = [dict(l, heard=traw.get(l["id"], {}).get("heard", l.get("heard", "")),
                           reteach=traw.get(l["id"], {}).get("reteach", l.get("reteach", ""))) for l in r["labels"]]
            stats[f"kept_teacher_agree_{r['_src']}"] += 1
        elif t and v:
            labels, ok = [], True
            for lab, tv, vv in zip(r["labels"], t["verifier"], v["verifier"]):
                votes = Counter([lab["verdict"], tv, vv])
                maj, n = votes.most_common(1)[0]
                if n < 2 or maj is None:
                    ok = False
                    break
                src = traw.get(lab["id"], {}) if tv == maj else lab
                labels.append(dict(lab, verdict=maj, heard=src.get("heard", ""), reteach=src.get("reteach", "")))
            if not ok:
                stats[f"dropped_no_majority_{r['_src']}"] += 1
                dropped_examples.append(k)
                continue
            changed = sum(l["verdict"] != d for l, d in zip(labels, designed))
            stats[f"kept_majority_vote_{r['_src']}"] += 1
            stats["facts_relabelled_by_vote"] += changed
        elif not t and not a.require_teacher and v and v.get("agree_all"):
            labels = r["labels"]
            stats[f"kept_verifier_only_{r['_src']}"] += 1
        else:
            stats[f"dropped_{'teacher_disagree' if t else 'no_teacher'}_{r['_src']}"] += 1
            dropped_examples.append(k)
            continue
        kept.append(dict(r, labels=labels))

    # Near-duplicate removal within a case topic.
    by_topic = defaultdict(list)
    for r in kept:
        by_topic[(r["case_id"], r["topic_id"])].append(r)
    dedup = []
    for rows in by_topic.values():
        seen = []
        for r in rows:
            g = grams(norm_t(r["transcript"]))
            if any(len(g & s) / max(1, len(g | s)) > 0.8 for s in seen):
                stats["dropped_near_duplicate"] += 1
                continue
            seen.append(g)
            dedup.append(r)
    # Cap per case.
    per_case = defaultdict(list)
    for r in dedup:
        per_case[r["case_id"]].append(r)
    final = []
    for cid, rows in per_case.items():
        rng.shuffle(rows)
        stats["dropped_case_cap"] += max(0, len(rows) - a.cap)
        final += rows[:a.cap]

    # Oversampling: near-miss pairs and weak languages.
    weights = {grader.row_key(r): 1.0 for r in final}
    by_topic = defaultdict(list)
    for r in final:
        by_topic[(r["case_id"], r["topic_id"])].append(r)
    for rows in by_topic.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                vi = [l["verdict"] for l in rows[i]["labels"]]
                vj = [l["verdict"] for l in rows[j]["labels"]]
                if len(vi) == len(vj) and sum(x != y for x, y in zip(vi, vj)) == 1:
                    for r in (rows[i], rows[j]):
                        weights[grader.row_key(r)] = max(weights[grader.row_key(r)], float(a.near_miss_x))
    weak = set(a.weak_langs.split(","))
    for r in final:
        if r["language"] in weak:
            weights[grader.row_key(r)] *= a.weak_x
    expanded = []
    for r in final:
        w = weights[grader.row_key(r)]
        n = int(w) + (1 if rng.random() < w - int(w) else 0)
        expanded += [r] * max(1, n)
    rng.shuffle(expanded)

    # Contamination guard (hard fail).
    tr_hash = {h(r["transcript"]) for r in final}
    tr_cases = {r["case_id"] for r in final}
    for name, rows in eval_sets.items():
        assert not ({h(x["transcript"]) for x in rows} & tr_hash), f"transcript overlap with {name}"
        assert not ({x["case_id"] for x in rows} & tr_cases), f"case overlap with {name}"
    assert not ({cases[c]["condition"] for c in tr_cases} & held), "held-out condition in training"

    out = APP / a.out
    out.mkdir(parents=True, exist_ok=True)
    for name, hf in (("sft_train", False), ("sft_train_heardfirst", True)):
        with (out / f"{name}.jsonl").open("w") as fh:
            for r in expanded:
                msgs = grader.messages(r, prompt) + [{"role": "assistant", "content": target(r, r["labels"], hf)}]
                fh.write(json.dumps({"key": grader.row_key(r), "messages": msgs}, ensure_ascii=False) + "\n")
    # Loss-curve val: generator val split (verifier-agreed rows, as in run 1).
    val = eval_sets["val"]
    for name, hf in (("sft_val", False), ("sft_val_heardfirst", True)):
        with (out / f"{name}.jsonl").open("w") as fh:
            for r in val:
                msgs = grader.messages(r, prompt) + [{"role": "assistant", "content": target(r, r["labels"], hf)}]
                fh.write(json.dumps({"key": grader.row_key(r), "messages": msgs}, ensure_ascii=False) + "\n")
    (out / "train_unique.jsonl").write_text("".join(json.dumps({k: v for k, v in r.items() if k != "_src"}, ensure_ascii=False) + "\n" for r in final))
    s = {"pool": len(pool), "unique_kept": len(final), "sft_rows_after_oversampling": len(expanded),
         "facts_unique": sum(len(r["labels"]) for r in final), "counts": dict(stats),
         "teacher_rows_available": len(teach), "verifier_rows_available": len(ver),
         "verdicts_unique": dict(Counter(l["verdict"] for r in final for l in r["labels"])),
         "languages_unique": dict(sorted(Counter(r["language"] for r in final).items())),
         "sources_unique": dict(Counter(r["_src"] for r in final)),
         "cases": len(tr_cases), "contamination_check": "passed (transcript hashes, cases, held-out conditions)",
         "dropped_examples_sample": dropped_examples[:20]}
    (out / "stats.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    print(json.dumps(s, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
