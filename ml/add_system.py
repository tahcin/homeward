"""Add (or replace) one system in an existing eval_*.json without re-scoring the others, then rewrite its .md.

Scoring is eval.py's own (same metrics, same seed-0 case bootstrap per system, paired comparison with the
reference), so the result is what a full `eval.py score` over all systems would give for this system.

Usage:
  python3 ml/add_system.py --eval ml/results/eval_independent.json --rows data/test_independent.jsonl \
      --system homeward_run5=ml/results/run5/preds_test_independent_final.jsonl \
      --reference e4b_base=ml/results/preds_base_test_independent.jsonl --after homeward_run4 [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as ev  # noqa: E402


def score_one(rows, name, path, ref_name, ref_path):
    """Metrics for one system, with paired_vs_<ref> as eval.score() computes it."""
    out = {}
    for n, p in ((ref_name, ref_path), (name, path)):
        preds = {r["key"]: r for r in ev.read_jsonl(p)}
        meta_p = Path(p).with_suffix(".meta.json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        gold, pred, lang, crit, cases, valid = ev.score_system(rows, preds)
        m = ev.metrics(gold, pred, crit)
        m.update(ev.bootstrap(gold, pred, crit, cases))
        m["json_valid_rate"] = sum(valid) / len(valid)
        m["answers"] = len(valid)
        k, exact = 0, 0
        for r in rows:
            n_ = len(r["labels"])
            exact += all(gold[k + j] == pred[k + j] for j in range(n_))
            k += n_
        m["answer_exact_match_rate"] = exact / len(rows)
        per_lang = {}
        for g, p_, l in zip(gold, pred, lang):
            a = per_lang.setdefault(l, [0, 0])
            a[0] += g == p_
            a[1] += 1
        m["per_language_accuracy"] = {l: {"accuracy": a / n2, "facts": n2} for l, (a, n2) in sorted(per_lang.items())}
        m["answers_without_prediction"] = sum(1 for r in rows if ev.grader.row_key(r) not in preds)
        pt = sum(preds[ev.grader.row_key(r)].get("prompt_tokens", 0) for r in rows if ev.grader.row_key(r) in preds)
        ct = sum(preds[ev.grader.row_key(r)].get("completion_tokens", 0) for r in rows if ev.grader.row_key(r) in preds)
        m["tokens"] = {"prompt_per_answer": pt / len(rows), "completion_per_answer": ct / len(rows)}
        out[n] = {"meta": meta, "metrics": m, "_pred": pred, "_gold": gold}
    s = out[name]
    s["metrics"]["paired_vs_" + ref_name] = ev.paired_vs(out[ref_name]["_pred"], s["_pred"], s["_gold"])
    return {"meta": s["meta"], "metrics": s["metrics"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--system", required=True, help="name=preds.jsonl")
    ap.add_argument("--reference", required=True, help="name=preds.jsonl of the paired reference (e4b_base in the committed files)")
    ap.add_argument("--after", default="", help="insert after this system key (default: end)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rows = ev.read_jsonl(a.rows)
    res = json.loads(Path(a.eval).read_text())
    assert res["test_set"]["answers"] == len(rows), "rows do not match the eval file's test set"
    name, path = a.system.split("=", 1)
    rname, rpath = a.reference.split("=", 1)
    new = score_one(rows, name, path, rname, rpath)
    systems = {k: v for k, v in res["systems"].items() if k != name}
    keys = list(systems)
    pos = keys.index(a.after) + 1 if a.after in keys else len(keys)
    res["systems"] = {k: systems[k] for k in keys[:pos]} | {name: new} | {k: systems[k] for k in keys[pos:]}
    m = new["metrics"]
    print(f"{name}: accuracy {100 * m['accuracy']:.1f}% ({100 * m['accuracy_95ci'][0]:.1f}-{100 * m['accuracy_95ci'][1]:.1f}), "
          f"critical {m['false_reassurance_critical_n'][0]}/{m['false_reassurance_critical_n'][1]}, "
          f"FR {m['false_reassurance_n'][0]}/{m['false_reassurance_n'][1]}, over-flag {100 * m['over_flag_rate']:.1f}%, "
          f"vs {rname}: {m['paired_vs_' + rname]}")
    if not a.dry_run:
        Path(a.eval).write_text(json.dumps(res, indent=1, ensure_ascii=False))
        Path(a.eval).with_suffix(".md").write_text(ev.render_md(res))
        print("wrote", a.eval, "and", Path(a.eval).with_suffix(".md"))


if __name__ == "__main__":
    main()
