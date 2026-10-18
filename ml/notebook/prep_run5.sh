#!/bin/bash
# Mac side, before the run 5 GPU window: build data/splits_v4 = the run 4 pool and label rule (jury: designed,
# Gemma-4-31B-IT, Qwen3.8-Flash-Next / DeepSeek-V4-Flash where present, Clef) WITHOUT the per-case cap of 45,
# which dropped 1,487 jury-approved answers from splits_v3 only to fit the old time budget.
#
#   bash ml/notebook/prep_run5.sh            # rebuild splits_v4 from the label files as they are now
#   DS=1 bash ml/notebook/prep_run5.sh       # first add DeepSeek juror votes on contested answers (Token Factory;
#                                            # only after the 21:41 baselines are finished, and only while the day's
#                                            # spend is under DS_MAX_SPEND, default $0.70; ~$0.0001 per answer)
# Run from app/. The splits are built only from generator answers labelled by open models on Token Factory;
# build_sft_v2.py hard-fails if any case, held-out condition or transcript of an evaluation set is in training.
set -euo pipefail
cd "$(dirname "$0")/../.."
S=/private/tmp/claude-502/-Users-naresh-Desktop-Personal-hackathons/014b671e-96ac-4951-83e1-57ca83e83267/scratchpad/ml
cat data/answers_v3*.jsonl data/answers_targeted*.jsonl > $S/v3_input.jsonl
if [ "${DS:-0}" = 1 ]; then
  set -a; source /private/tmp/claude-502/-Users-naresh-Desktop-Personal-hackathons/014b671e-96ac-4951-83e1-57ca83e83267/scratchpad/.env.amd; set +a
  SPENT=$(curl -s -H "Authorization: Bearer $TOKENFACTORY_API_KEY" https://developer.amd.com.cn/radeon/api/v1/usage | python3 -c "import json,sys;print(json.load(sys.stdin)['daily_cost_used_usd'])" 2>/dev/null || echo 9)
  if python3 -c "import sys; sys.exit(0 if float('$SPENT') < float('${DS_MAX_SPEND:-0.70}') else 1)"; then
    python3 $S/ds_contested.py $S/v3_input.jsonl $S/ds_input.jsonl
    # verify.py appends and skips keys it already has (resumable); batch 1 because DeepSeek ignores the multi-answer wrapper
    TF_CONCURRENCY=3 TF_RPM=${TF_RPM:-10} uv run -q --with httpx python ml/verify.py --answers $S/ds_input.jsonl --out data/juror_ds_v3.jsonl \
      --model DeepSeek-V4-Flash --reasoning "" --batch 1 --limit ${DS_LIMIT:-800} --summary ml/results/juror_ds_v3_summary.json | tail -2
  else
    echo "skip DeepSeek: today's spend is \$$SPENT (limit \$${DS_MAX_SPEND:-0.70})"
  fi
fi
python3 ml/build_sft_v2.py --answers-v3 $S/v3_input.jsonl --verify-v2 data/verify_v3.jsonl --teacher data/teacher_31b.jsonl \
  --teacher data/teacher_31b_v3.jsonl --juror data/juror_ds_v3.jsonl --juror-clef data/juror_clef.jsonl \
  --mine ml/results/mining_run2_preds.jsonl --mine ml/results/mining_run2_preds_b.jsonl \
  --mine-probs ml/results/mining_probs_a.jsonl --mine-probs ml/results/mining_probs_b.jsonl \
  --weak-x 1 --near-miss-x 1 --cap 999 --out data/splits_v4 > /dev/null
python3 -c "
import json; s=json.load(open('data/splits_v4/stats.json')); c=s['counts']
print('splits_v4:', s['unique_kept'], 'unique answers,', s['sft_rows_after_oversampling'], 'rows,', s['cases'], 'cases; dropped by case cap:', c.get('dropped_case_cap', 0),
      '; DeepSeek rows', s['juror_rows_available'], '; verdicts', s['verdicts_unique'], '; contamination:', s['contamination_check'])
print('est. training minutes at 2 epochs:', round(s['sft_rows_after_oversampling'] * 0.476 * 2 / 60))"
