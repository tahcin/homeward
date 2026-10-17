#!/bin/bash
# Continuation of run_day2.sh after a hung run4_avg load (Oct 17 07:06-07:33 UTC). Same steps and thresholds,
# elapsed measured from the original start (06:01:28 UTC); each model call gets a timeout and one retry.
set -u
W=/workspace; D=$W/data; C=$W/code; L=$W/logs; M=$W/models/gemma-4-E4B-it; R=$W/runs
cd $C; export TOKENIZERS_PARALLELISM=false HF_DEACTIVATE_ASYNC_LOAD=1
mark(){ echo "=== $1 $(date -u +%T)" >> $L/day2.log; }
START=$(date -u -d "2026-10-17 06:01:28" +%s); elapsed(){ echo $(( ($(date +%s) - START) / 60 )); }
mark resume_b
py(){ local t=$1; shift; timeout $t python fhwrap.py "$@" || { echo "retry after rc=$? : $*" >> $L/day2.log; timeout $t python fhwrap.py "$@"; }; }
evalcand(){ # name adapter scale [extra predict args]
  local n=$1 ad=$2 sc=$3; shift 3; local o=$R/cands2; mkdir -p $o
  py 360 eval.py predict-local --model $M --adapter $ad --lora-scale $sc --name $n --rows $D/val_independent.jsonl --out $o/preds_$n.jsonl --bsz 16 --latency-n 0 "$@" > $L/cand2_$n.log 2>&1
  py 300 eval.py verdict-probs --model $M --adapter $ad --lora-scale $sc --rows $D/val_independent.jsonl --preds $o/preds_$n.jsonl --out $o/probs_$n.jsonl >> $L/cand2_$n.log 2>&1
  if [ "$(wc -l < $o/preds_$n.jsonl 2>/dev/null)" = 120 ] && [ -s $o/probs_$n.jsonl ]; then echo "$n:$ad:$sc:$o/preds_$n.jsonl:$o/probs_$n.jsonl" >> $o/list.txt; else mark CAND_FAILED_$n; fi
}
[ -d $R/run4/adapter_avg ] && { evalcand run4_avg $R/run4/adapter_avg 1.0; evalcand run4_avg_s050 $R/run4/adapter_avg 0.5; }
[ -d $R/run4/adapter_avg ] && [ $(elapsed) -lt 110 ] && evalcand run4_avg_s075 $R/run4/adapter_avg 0.75
[ -d $R/run4/adapter_avg_last2 ] && evalcand run4_avgl2 $R/run4/adapter_avg_last2 1.0
[ -d $R/run4/adapter ] && [ $(elapsed) -lt 115 ] && evalcand run4_final_s050 $R/run4/adapter 0.5

sel(){ python select_model.py --rows $D/val_independent.jsonl --out $R/cands2/select_$1.json $(sed 's/^/--cand /' $R/cands2/list.txt) > $L/select2_$1.log 2>&1; grep -E "SAFETY|5x|CHOSEN" $L/select2_$1.log >> $L/day2.log; }
sel pre_sc
read BN BAD BSC <<< $(python -c "import json; c=json.load(open('$R/cands2/select_pre_sc.json'))['chosen']; print(c['name'], c['adapter'], c['scale'])")

# Self-consistency candidate: greedy + 2 samples (T 0.7), per-fact majority, ties to the less reassuring verdict.
if [ $(elapsed) -lt 125 ]; then
  mark sc_$BN
  for s in 1 2; do py 420 eval.py predict-local --model $M --adapter $BAD --lora-scale $BSC --name sc$s --rows $D/val_independent.jsonl --out $R/cands2/sc${s}_$BN.jsonl --bsz 16 --latency-n 0 --temperature 0.7 --seed $s > $L/sc2_$s.log 2>&1; done
  python eval.py vote --rows $D/val_independent.jsonl --preds $R/cands2/preds_$BN.jsonl --preds $R/cands2/sc1_$BN.jsonl --preds $R/cands2/sc2_$BN.jsonl --out $R/cands2/preds_${BN}_sc.jsonl >> $L/day2.log 2>&1
  echo "${BN}_sc:$BAD:$BSC:$R/cands2/preds_${BN}_sc.jsonl:$R/cands2/probs_$BN.jsonl" >> $R/cands2/list.txt
fi
sel final
read FN FAD FSC FTAU <<< $(python -c "import json; c=json.load(open('$R/cands2/select_final.json'))['chosen']; print(c['name'], c['adapter'], c['scale'], c['tau'])")
echo "FINAL: $FN $FAD scale $FSC tau $FTAU" >> $L/day2.log
F=$R/final2; mkdir -p $F; echo "$FAD" > $F/adapter_path.txt; echo "$FSC" > $F/lora_scale.txt; echo "$FN" > $F/name.txt

# Frozen choice on the test sets, once.
for set in test_independent test; do
  mark final2_$set
  py 900 eval.py predict-local --model $M --adapter $FAD --lora-scale $FSC --name homeward_final2 --rows $D/$set.jsonl --out $F/preds_$set.jsonl --bsz 16 --latency-n 20 > $L/final2_$set.log 2>&1
  P=$F/preds_$set.jsonl
  case $FN in *_sc)
    for s in 1 2; do py 900 eval.py predict-local --model $M --adapter $FAD --lora-scale $FSC --name sc$s --rows $D/$set.jsonl --out $F/sc${s}_$set.jsonl --bsz 16 --latency-n 0 --temperature 0.7 --seed $s >> $L/final2_$set.log 2>&1; done
    python eval.py vote --rows $D/$set.jsonl --preds $F/preds_$set.jsonl --preds $F/sc1_$set.jsonl --preds $F/sc2_$set.jsonl --out $F/preds_${set}_sc.jsonl >> $L/final2_$set.log 2>&1
    P=$F/preds_${set}_sc.jsonl;;
  esac
  py 600 eval.py verdict-probs --model $M --adapter $FAD --lora-scale $FSC --rows $D/$set.jsonl --preds $F/preds_$set.jsonl --out $F/probs_$set.jsonl >> $L/final2_$set.log 2>&1
  python eval.py escalate --preds $P --probs $F/probs_$set.jsonl --tau $FTAU --out $F/preds_${set}_final.jsonl >> $L/final2_$set.log 2>&1
done

mark replay
py 900 onprem_replay.py --model $M --adapter $FAD --lora-scale $FSC --samples-dir $D/samples --out-dir $F/replay --run-name "run 4 selection: $FN" > $L/replay.log 2>&1
mark demo
cp $F/adapter_path.txt $R/final/adapter_path.txt; cp $F/lora_scale.txt $R/final/lora_scale.txt
bash $W/demo.sh > $F/onprem_demo.txt 2>&1
mark capacity
py 900 eval.py predict-local --model $M --adapter $FAD --lora-scale $FSC --name homeward_final2 --rows $D/test.jsonl --out $F/cap_bs64.jsonl --bsz 64 --latency-n 0 > $L/final2_cap.log 2>&1
mark DAY2DONE
