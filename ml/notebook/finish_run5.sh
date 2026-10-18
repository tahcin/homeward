#!/bin/bash
# Mac side, after fetch_run5.sh: add homeward_run5 to the official score files (others are not re-scored),
# write the paired run 5 vs run 4 file, and print the numbers. Then the lead runs ./promote.sh homeward_run5 <gpu_min>.
# Only for a run 5 that WON on validation: refuses if results/run5/pick.json names the incumbent.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=ml/results; R5=$R/run5
[ -f $R5/pick.json ] || { echo "no $R5/pick.json (fetch first)"; exit 1; }
python3 -c "import json,sys; p=json.load(open('$R5/pick.json')); sys.exit(1 if p['name'] in p.get('incumbent', []) else 0)" \
  || { echo "the frozen pick is the incumbent (run 4): nothing to add"; exit 1; }
[ -f $R5/preds_test_independent_final.jsonl ] || { echo "missing $R5/preds_test_independent_final.jsonl"; exit 1; }
python3 ml/add_system.py --eval $R/eval_independent.json --rows data/test_independent.jsonl \
  --system homeward_run5=$R5/preds_test_independent_final.jsonl --reference e4b_base=$R/preds_base_test_independent.jsonl --after homeward_run4
if [ -f $R5/preds_test_final.jsonl ]; then
  python3 ml/add_system.py --eval $R/eval_in_distribution.json --rows data/splits/test.jsonl \
    --system homeward_run5=$R5/preds_test_final.jsonl --reference e4b_base=$R/preds_base_test.jsonl --after homeward_run4
fi
python3 ml/eval.py score --rows data/test_independent.jsonl --reference homeward_run4 \
  --system homeward_run4=$R/run4/preds_test_independent_final.jsonl --system homeward_run5=$R5/preds_test_independent_final.jsonl \
  --out $R5/eval_independent_vs_run4.json --md $R5/eval_independent_vs_run4.md > /dev/null
python3 - <<'PY'
import json, math
v = json.load(open("ml/results/run5/eval_independent_vs_run4.json"))["systems"]
pv = v["homeward_run5"]["metrics"]["paired_vs_homeward_run4"]
b, c = pv["this_right_ref_wrong"], pv["this_wrong_ref_right"]
n = b + c
p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else None
for k in ("homeward_run4", "homeward_run5"):
    m = v[k]["metrics"]
    print(f"{k}: {100 * m['accuracy']:.1f}% ({100 * m['accuracy_95ci'][0]:.1f}-{100 * m['accuracy_95ci'][1]:.1f}), critical {m['false_reassurance_critical_n'][0]}/{m['false_reassurance_critical_n'][1]}, over-flag {100 * m['over_flag_rate']:.1f}%")
print(f"run 5 vs run 4 paired: +{b} / -{c}, exact McNemar p = {p}")
PY
echo "next: ./promote.sh homeward_run5 <total GPU minutes> (from amd-act-iii-2026/), then --deploy"
