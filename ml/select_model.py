"""Pick the final grader (adapter, LoRA scale, tau) on val_independent only.

Declared policy (2026-10-16, decided on validation before any test scoring):
SAFETY-FIRST. A missed critical fact is never acceptable; extra re-checks are. Among all
(candidate, tau) pairs: require zero critical false reassurance and total false
reassurance no worse than base E4B on the same set (--base-fr); then fewest total misses;
then fewest over-flags; then highest accuracy; then the simpler candidate (listed first), then the smaller tau.
If no pair qualifies, take the fewest critical misses, then the fewest misses, then fewest over-flags.
For transparency the earlier rule (min 5 x FR + over-flag) is also reported.

Usage: python select_model.py --rows val_independent.jsonl --out final.json \
          --cand NAME:ADAPTER_DIR:SCALE:PREDS:PROBS [--cand ...]
"""

import argparse
import json
from pathlib import Path

from select_tau import read_jsonl, sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--cand", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--as-written", dest="then_fr", action="store_false",
                    help="after the constraints, minimise over-flags first (default: total FR first, then over-flags)")
    ap.set_defaults(then_fr=True)
    ap.add_argument("--base-fr", type=int, default=2, help="base E4B false reassurances on the same set (val_independent: 2/113)")
    a = ap.parse_args()
    rows = read_jsonl(a.rows)
    table, pairs = [], []
    for i, c in enumerate(a.cand):
        name, adapter, scale, preds, probs = c.split(":")
        if not (Path(preds).exists() and Path(probs).exists()):
            print("skip (missing files)", name)
            continue
        sw = sweep(rows, read_jsonl(preds), read_jsonl(probs))
        for r in sw:
            pairs.append(dict(r, order=i, name=name, adapter=adapter, scale=float(scale)))
        best = min(sw, key=lambda r: (r["cost"], -r["accuracy"], r["tau"]))
        at0 = sw[0]
        table.append({"order": i, "name": name, "adapter": adapter, "scale": float(scale), "tau": best["tau"],
                      "cost": best["cost"], "accuracy": best["accuracy"], "false_reassurance": best["false_reassurance"],
                      "over_flag": best["over_flag"], "accuracy_tau0": at0["accuracy"], "fr_tau0": at0["false_reassurance"],
                      "overflag_tau0": at0["over_flag"], "sweep": sw})
    table.sort(key=lambda r: (r["cost"], -r["accuracy"], r["order"]))
    for r in table:
        print(f"{r['name']:<28} scale {r['scale']:<5} tau {r['tau']:<6} cost {r['cost']:<4} acc {100 * r['accuracy']:.1f}% "
              f"FR {r['false_reassurance']} over {r['over_flag']} | at tau 0: acc {100 * r['accuracy_tau0']:.1f}% FR {r['fr_tau0']} over {r['overflag_tau0']}")
    cost_win = table[0]
    cw = [p for p in pairs if p["name"] == cost_win["name"] and p["tau"] == cost_win["tau"]][0]
    cost_win = dict(cost_win, critical_false_reassurance=cw["critical_false_reassurance"])
    ok = [p for p in pairs if p["critical_false_reassurance"] == 0 and p["false_reassurance"] <= a.base_fr]
    if ok:
        key = (lambda p: (p["false_reassurance"], p["over_flag"], -p["accuracy"], p["order"], p["tau"])) if a.then_fr else \
              (lambda p: (p["over_flag"], -p["accuracy"], p["order"], p["tau"]))
        safe = sorted(ok, key=key)[0]
        note = "safety-first constraints met"
    else:
        safe = sorted(pairs, key=lambda p: (p["critical_false_reassurance"], p["false_reassurance"], p["over_flag"], -p["accuracy"], p["order"]))[0]
        note = "no pair met zero critical misses; fell back to fewest critical misses"
    keys = ("name", "adapter", "scale", "tau", "accuracy", "false_reassurance", "critical_false_reassurance", "over_flag", "cost")
    print(f"SAFETY-FIRST pick ({note}):", {k: safe[k] for k in keys})
    print("5x-cost rule pick (reported for transparency):", {k: cost_win.get(k) for k in keys})
    Path(a.out).write_text(json.dumps({
        "policy": "safety-first: zero critical false reassurance and total FR <= base on val_independent, then " + ("fewest total FR, then fewest over-flags" if a.then_fr else "fewest over-flags") + ", then accuracy",
        "policy_note": note, "base_fr": a.base_fr,
        "chosen": {k: safe[k] for k in keys},
        "cost_rule_pick": {k: cost_win.get(k) for k in keys},
        "candidates": table}, indent=1))
    print("CHOSEN", safe["name"], safe["adapter"], safe["scale"], safe["tau"])


if __name__ == "__main__":
    main()
