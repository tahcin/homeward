"""Split the synthetic teach-back answers by case into train / val / test and build the
LoRA SFT files.

- Split is by case, never by answer, so no discharge case appears in two splits.
- At least --holdout-conditions whole conditions go only to test (unseen conditions), on
  top of a random share of the remaining cases. Test is grown until it has at least
  --min-test answers.
- Training and validation rows keep only answers where the blind verifier (verify.py)
  agrees with every designed fact label. Test keeps the designed labels for every
  answer; verifier agreement on test is reported as a label-quality figure.
- SFT row: grader system + user prompt (prompts/grader.json, verbatim) -> gold JSON.

Usage: python3 ml/build_sft.py --answers data/answers.jsonl --verify data/verify.jsonl --outdir data/splits
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import grader


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/cases.jsonl")
    ap.add_argument("--answers", default="data/answers.jsonl")
    ap.add_argument("--verify", default="data/verify.jsonl")
    ap.add_argument("--outdir", default="data/splits")
    ap.add_argument("--holdout-conditions", type=int, default=3)
    ap.add_argument("--holdout-file", default="data/splits/holdout.json",
                    help="fixed held-out conditions (shared with the independent test set)")
    ap.add_argument("--independent", default="data/test_independent.jsonl")
    ap.add_argument("--min-test", type=int, default=250)
    ap.add_argument("--val-cases", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-filter", action="store_true", help="train on all answers (ablation)")
    ap.add_argument("--teacher", help="teacher labels (verify.py output with Gemma-4-31B-IT): filter train/val on "
                                      "designed == teacher for every fact and use the teacher's heard/reteach strings")
    ap.add_argument("--extra-answers", action="append", default=[], help="more answers (e.g. data v2), train only")
    ap.add_argument("--extra-cases", action="append", default=[], help="cases files for --extra-answers")
    ap.add_argument("--oversample", type=int, default=1,
                    help="repeat training rows that contain any incorrect, missing or partial fact this many times")
    ap.add_argument("--stats", default="ml/results/split_stats.json")
    args = ap.parse_args()

    prompt = grader.load_prompt()
    answers = read_jsonl(args.answers)
    cases = {c["case_id"]: c for c in read_jsonl(args.cases)}
    ver = {r["key"]: r for r in read_jsonl(args.verify)} if Path(args.verify).exists() else {}

    by_case = defaultdict(list)
    for a in answers:
        by_case[a["case_id"]].append(a)
    cond_of = {cid: (cases.get(cid) or {}).get("condition", "unknown") for cid in by_case}
    by_cond = defaultdict(list)
    for cid in sorted(by_case):
        by_cond[cond_of[cid]].append(cid)

    rng = random.Random(args.seed)
    conds = sorted(by_cond)
    rng.shuffle(conds)
    held = conds[:args.holdout_conditions]
    if args.holdout_file and Path(args.holdout_file).exists():
        held = json.loads(Path(args.holdout_file).read_text())["conditions"]
    test_cases = {cid for c in held for cid in by_cond[c]}
    rest = [cid for cid in sorted(by_case) if cid not in test_cases]
    rng.shuffle(rest)
    # Grow test with random cases from the other conditions, preferring languages that
    # test lacks, until it reaches the minimum answer count.
    all_langs = {a["language"] for a in answers}

    def test_langs():
        return {by_case[c][0]["language"] for c in test_cases}

    while (sum(len(by_case[c]) for c in test_cases) < args.min_test or test_langs() != all_langs) and rest:
        have = Counter(by_case[c][0]["language"] for c in test_cases)
        rest.sort(key=lambda c: have[by_case[c][0]["language"]])
        test_cases.add(rest.pop(0))
    # Validation: cases with distinct languages, so no language is drained from training.
    val_cases, seen = set(), set()
    for c in rest:
        lang = by_case[c][0]["language"]
        if lang not in seen and len(val_cases) < args.val_cases:
            val_cases.add(c)
            seen.add(lang)
    train_cases = set(rest) - val_cases

    stats = {"seed": args.seed, "held_out_conditions": held, "filter": not args.no_filter,
             "filter_source": "teacher (Gemma-4-31B-IT)" if args.teacher else "verifier (Qwen3.8-Flash-Next)",
             "verifier_rows_available": len(ver), "oversample_hard_rows": args.oversample, "splits": {}}
    teacher = {r["key"]: r for r in read_jsonl(args.teacher)} if args.teacher else {}

    def agreed(a):
        if teacher:
            t = teacher.get(grader.row_key(a))
            return bool(t and t["agree_all"])
        v = ver.get(grader.row_key(a))
        return bool(v and v["agree_all"])

    def distilled(a):
        """Designed verdicts with the teacher's heard/reteach text where the teacher agreed."""
        t = teacher.get(grader.row_key(a))
        if not t or not t.get("verifier_raw"):
            return a
        raw = {f.get("id"): f for f in t["verifier_raw"].get("facts", []) if isinstance(f, dict)}
        labels = []
        for lab in a["labels"]:
            f = raw.get(lab["id"], {})
            labels.append(dict(lab, heard=f.get("heard", lab.get("heard", "")), reteach=f.get("reteach", lab.get("reteach", ""))))
        return dict(a, labels=labels)

    # Extra (v2) answers: training only; drop anything from held-out conditions or test/val cases.
    extra_rows, extra_dropped = [], 0
    xcases = {}
    for p in args.extra_cases:
        xcases.update({c["case_id"]: c for c in read_jsonl(p)})
    for p in args.extra_answers:
        for a in read_jsonl(p):
            cond = (xcases.get(a["case_id"]) or cases.get(a["case_id"]) or {}).get("condition")
            if cond in held or a["case_id"] in test_cases or a["case_id"] in val_cases or cond is None:
                extra_dropped += 1
                continue
            extra_rows.append(a)
    stats["extra_answers"] = {"kept_for_train": len(extra_rows), "dropped_heldout_or_eval_cases_or_unknown": extra_dropped}

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, cs in (("train", train_cases), ("val", val_cases), ("test", test_cases)):
        rows = [a for cid in sorted(cs) for a in by_case[cid]]
        if name == "train":
            rows = rows + extra_rows
        if name == "test":
            kept = rows
        else:
            kept = rows if args.no_filter else [a for a in rows if agreed(a)]
            if teacher:
                kept = [distilled(a) for a in kept]
        (out / f"{name}.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in kept))
        if name != "test":
            hard = lambda a: any(l["verdict"] != "correct" for l in a["labels"])  # noqa: E731
            sft_rows = [a for a in kept for _ in range(args.oversample if (name == "train" and hard(a)) else 1)]
            stats.setdefault("sft_rows", {})[name] = len(sft_rows)
            with (out / f"sft_{name}.jsonl").open("w") as fh:
                for a in sft_rows:
                    msgs = grader.messages(a, prompt) + [{"role": "assistant", "content": grader.gold_text(a)}]
                    fh.write(json.dumps({"key": grader.row_key(a), "messages": msgs}, ensure_ascii=False) + "\n")
        verified = [a for a in rows if grader.row_key(a) in ver]
        tch = [a for a in rows if grader.row_key(a) in teacher]
        fact_pairs = [(d, v) for a in verified for d, v in zip(ver[grader.row_key(a)]["designed"], ver[grader.row_key(a)]["verifier"])]
        stats["splits"][name] = {
            "cases": len(cs),
            "conditions": len({cond_of[c] for c in cs}),
            "answers_total": len(rows),
            "answers_verified": len(verified),
            "answers_verifier_agrees": sum(agreed(a) for a in rows),
            "answers_kept": len(kept),
            "answers_teacher_labelled": len(tch),
            "teacher_answer_agreement": (sum(teacher[grader.row_key(a)]["agree_all"] for a in tch) / len(tch)) if tch else None,
            "facts_kept": sum(len(a["labels"]) for a in kept),
            "verifier_answer_agreement": (sum(agreed(a) for a in verified) / len(verified)) if verified else None,
            "verifier_fact_agreement": (sum(d == v for d, v in fact_pairs) / len(fact_pairs)) if fact_pairs else None,
            "languages": dict(sorted(Counter(a["language"] for a in kept).items())),
            "verdicts": dict(Counter(l["verdict"] for a in kept for l in a["labels"])),
        }
    # Independent test set (written by Claude, never seen by the generator): must use only
    # held-out conditions, so none of its facts appear in train or val.
    if Path(args.independent).exists():
        ind = read_jsonl(args.independent)
        bad = [a["case_id"] for a in ind if cond_of.get(a["case_id"], (cases.get(a["case_id"]) or {}).get("condition")) not in held]
        assert not bad, f"independent rows outside held-out conditions: {sorted(set(bad))}"
        stats["splits"]["test_independent"] = {
            "answers_total": len(ind), "facts": sum(len(a["labels"]) for a in ind),
            "cases": len({a["case_id"] for a in ind}),
            "languages": dict(sorted(Counter(a["language"] for a in ind).items())),
            "verdicts": dict(Counter(l["verdict"] for a in ind for l in a["labels"])),
            "critical_facts": sum(1 for a in ind for f in a["facts"] if f.get("critical")),
        }
    # Leakage checks.
    assert not (train_cases & test_cases) and not (val_cases & test_cases) and not (train_cases & val_cases)
    assert not ({cond_of[c] for c in train_cases | val_cases} & set(held))
    stats["leakage_checks"] = "passed: no case in two splits; held-out conditions absent from train and val"
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
