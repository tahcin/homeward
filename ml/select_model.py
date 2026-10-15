"""Pick the final grader (adapter, LoRA scale, tau) on val_independent only, by the rule
fixed on 2026-10-15: for each candidate, sweep tau and take its lowest cost
5 x false reassurance + 1 x over-flag; the candidate with the lowest cost wins; ties go
to higher accuracy, then to the simpler candidate (listed first).

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
    a = ap.parse_args()
    rows = read_jsonl(a.rows)
    table = []
    for i, c in enumerate(a.cand):
        name, adapter, scale, preds, probs = c.split(":")
        if not (Path(preds).exists() and Path(probs).exists()):
            print("skip (missing files)", name)
            continue
        sw = sweep(rows, read_jsonl(preds), read_jsonl(probs))
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
    win = table[0]
    Path(a.out).write_text(json.dumps({"rule": "min over tau of 5*FR + over-flag on val_independent; ties: accuracy, then order",
                                       "chosen": {k: win[k] for k in ("name", "adapter", "scale", "tau", "cost", "accuracy")},
                                       "candidates": table}, indent=1))
    print("CHOSEN", win["name"], win["adapter"], win["scale"], win["tau"])


if __name__ == "__main__":
    main()
