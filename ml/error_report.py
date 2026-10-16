"""Where does a grader go wrong? Breaks errors on a labelled set down by verdict confusion,
language, fact category, critical flag and answer style, and lists the unsafe misses
(gold incorrect/missing graded correct). For targeting new training data; run it on
selection sets only (val_independent, dev), never on the test sets.

Usage: python3 ml/error_report.py --rows data/val_independent.jsonl --preds P [--out report.json]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import grader


def rj(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    preds = {p["key"]: p for p in rj(a.preds)}
    conf = Counter()
    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    unsafe, examples = [], []
    for r in rj(a.rows):
        k = grader.row_key(r)
        ids = [f["id"] for f in r["facts"]]
        v, _ = grader.verdicts_from(grader.parse_reply(preds[k]["text"]) if k in preds else None, ids)
        for f, lab, pv in zip(r["facts"], r["labels"], v):
            g, pv = lab["verdict"], pv or "invalid"
            conf[(g, pv)] += 1
            ok = g == pv
            for dim, val in (("language", r["language"]), ("category", f.get("category")),
                             ("critical", bool(f.get("critical"))), ("error_type", lab.get("error", "-"))):
                by[dim][val][0] += ok
                by[dim][val][1] += 1
            if not ok:
                item = {"key": k, "language": r["language"], "fact": f["statement"], "gold": g, "pred": pv,
                        "error": lab.get("error"), "transcript": r["transcript"][:300]}
                examples.append(item)
                if g in ("incorrect", "missing") and pv == "correct":
                    unsafe.append(item)
    out = {"confusion": {f"{g}->{p}": n for (g, p), n in conf.most_common() if g != p},
           "accuracy_by": {d: {str(k): {"acc": round(c / n, 3), "n": n} for k, (c, n) in sorted(vals.items(), key=lambda x: x[1][0] / x[1][1])}
                           for d, vals in by.items()},
           "unsafe_misses": unsafe, "errors": examples}
    print("Confusions:", out["confusion"])
    for d, vals in out["accuracy_by"].items():
        print(d, {k: f"{v['acc']:.2f} (n={v['n']})" for k, v in list(vals.items())[:6]})
    print(f"{len(unsafe)} unsafe misses")
    for u in unsafe:
        print(" -", u["language"], u["gold"], "->", u["pred"], "|", u["fact"][:80], "|", u["transcript"][:120])
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
