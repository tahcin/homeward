"""Choose the escalation threshold tau on the VALIDATION split only, then report it.

Rule, fixed before looking at any test probabilities (2026-10-15): pick tau in a grid that
minimises 5 x (false reassurances) + 1 x (over-flags) on validation, i.e. an unsafe miss
costs five unnecessary nurse re-checks. Ties go to the smaller tau (fewer escalations).

Usage: python3 ml/select_tau.py --rows data/splits/val.jsonl --preds P --probs Q [--name e4b_lora]
"""

import argparse
import json
from pathlib import Path

import grader

GRID = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
COST_FR, COST_OVERFLAG = 5, 1


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def sweep(rows, preds, probs):
    preds = {p["key"]: p for p in preds}
    probs = {(r["key"], r["id"]): r["p"] for r in probs}
    out = []
    for tau in GRID:
        fr = bad = over = good = acc = n = crit_fr = crit_bad = 0
        for r in rows:
            k = grader.row_key(r)
            ids = [f["id"] for f in r["facts"]]
            v, _ = grader.verdicts_from(grader.parse_reply(preds[k]["text"]) if k in preds else None, ids)
            crit = {f["id"]: bool(f.get("critical")) for f in r["facts"]}
            for fid, lab, pv in zip(ids, r["labels"], v):
                pr = probs.get((k, fid))
                if pv == "correct" and pr and pr.get("correct", 1) < tau:
                    pv = max((x for x in grader.VERDICTS if x != "correct"), key=lambda x: pr.get(x, 0))
                g = lab["verdict"]
                n += 1
                acc += pv == g
                if g in ("incorrect", "missing"):
                    bad += 1
                    fr += pv == "correct"
                    if crit[fid]:
                        crit_bad += 1
                        crit_fr += pv == "correct"
                if g == "correct":
                    good += 1
                    over += pv != "correct"
        out.append({"tau": tau, "false_reassurance": fr, "bad_facts": bad, "over_flag": over, "correct_facts": good,
                    "accuracy": acc / n, "cost": COST_FR * fr + COST_OVERFLAG * over,
                    "critical_false_reassurance": crit_fr, "critical_bad_facts": crit_bad})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = sweep(read_jsonl(a.rows), read_jsonl(a.preds), read_jsonl(a.probs))
    best = min(res, key=lambda r: (r["cost"], r["tau"]))
    for r in res:
        print(f'{a.name} tau {r["tau"]:<6} FR {r["false_reassurance"]}/{r["bad_facts"]}  over-flag {r["over_flag"]}/{r["correct_facts"]}  acc {100 * r["accuracy"]:.1f}%  cost {r["cost"]}{"  <- chosen" if r is best else ""}')
    if a.out:
        Path(a.out).write_text(json.dumps({"rule": "minimise 5*FR + 1*over-flag on validation; ties to smaller tau",
                                           "chosen_tau": best["tau"], "sweep": res}, indent=1))


if __name__ == "__main__":
    main()
