#!/bin/bash
# Day-2 GPU window (Oct 17): run 4 = run-B recipe (heard-first, verdict weight 3, unlikelihood 0.5, r32) on splits_v3,
# epochs sized to the time budget (max 3), epoch-adapter averaging, LoRA-scale sweep, a 3-sample self-consistency
# candidate, selection on val_independent (safety-first, then total FR, then over-flags, then accuracy) with run 2's
# final model as a candidate, then the frozen choice on both test sets (the independent test's second use, for a
# different frozen model), on-site replay JSON for the 3 web samples, the on-prem demo, and capacity.
set -u
W=/workspace; D=$W/data; C=$W/code; L=$W/logs; M=$W/models/gemma-4-E4B-it; R=$W/runs
cd $C; export TOKENIZERS_PARALLELISM=false
mark(){ echo "=== $1 $(date -u +%T)" >> $L/day2.log; }
mark start; START=$(date +%s); elapsed(){ echo $(( ($(date +%s) - START) / 60 )); }
bash $W/setup.sh >> $L/day2.log 2>&1
until [ "$(stat -c %s $M/model.safetensors 2>/dev/null)" = 15992595884 ]; do
  mark resume_e4b; (cd $M && curl -sSL --retry 20 -C - -o model.safetensors "https://modelscope.cn/models/google/gemma-4-E4B-it/resolve/master/model.safetensors"); sleep 2; done

TR=$D/v3/sft_train_heardfirst.jsonl; VA=$D/v3/sft_val_heardfirst.jsonl
ROWS=$(wc -l < $TR)
# Measured on Oct 16: 0.476 s per training row per epoch (bsz 4 x accum 4, checkpointing, r32). Budget 85 min of training.
EPOCHS=$(python -c "r=$ROWS; print(max(1, min(3, int(85*60 // (r*0.476)))))")
echo "rows $ROWS -> epochs $EPOCHS (est $(python -c "print(round($ROWS*0.476*$EPOCHS/60))") min)" >> $L/day2.log

evalcand(){ # name adapter scale [extra predict args]
  local n=$1 ad=$2 sc=$3; shift 3; local o=$R/cands2; mkdir -p $o
  python eval.py predict-local --model $M --adapter $ad --lora-scale $sc --name $n --rows $D/val_independent.jsonl --out $o/preds_$n.jsonl --bsz 16 --latency-n 0 "$@" > $L/cand2_$n.log 2>&1
  python eval.py verdict-probs --model $M --adapter $ad --lora-scale $sc --rows $D/val_independent.jsonl --preds $o/preds_$n.jsonl --out $o/probs_$n.jsonl >> $L/cand2_$n.log 2>&1
  echo "$n:$ad:$sc:$o/preds_$n.jsonl:$o/probs_$n.jsonl" >> $o/list.txt
}
mkdir -p $R/cands2; : > $R/cands2/list.txt
# Run 2's frozen model is a candidate (its val preds/probs exist), so run 4 must beat it on val to be chosen.
echo "run2_final:$R/runB/adapter_avg:0.5:$R/cands/preds_runB_avg_s050.jsonl:$R/cands/probs_runB_avg_s050.jsonl" >> $R/cands2/list.txt

mark train_run4
python train_lora.py --model $M --train $TR --val $VA --out $R/run4 --epochs $EPOCHS --bsz 4 --accum 4 --rank 32 --alpha 32 --lr 2e-4 \
  --verdict-weight 3 --ul-lambda 0.5 --max-len 1536 > $L/train_run4.log 2>&1 || mark TRAIN_FAILED_run4
EP=$(ls -d $R/run4/adapter_epoch* 2>/dev/null | sort)
if [ -n "$EP" ]; then python avg_adapters.py $R/run4/adapter_avg $EP $R/run4/adapter >> $L/train_run4.log 2>&1; fi
LAST2=$(ls -d $R/run4/adapter_epoch* 2>/dev/null | sort | tail -1)
[ -n "$LAST2" ] && [ $EPOCHS -ge 3 ] && python avg_adapters.py $R/run4/adapter_avg_last2 $LAST2 $R/run4/adapter >> $L/train_run4.log 2>&1
mark eval_run4
[ -d $R/run4/adapter ] && evalcand run4_final $R/run4/adapter 1.0
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
  for s in 1 2; do python eval.py predict-local --model $M --adapter $BAD --lora-scale $BSC --name sc$s --rows $D/val_independent.jsonl --out $R/cands2/sc${s}_$BN.jsonl --bsz 16 --latency-n 0 --temperature 0.7 --seed $s > $L/sc2_$s.log 2>&1; done
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
  python eval.py predict-local --model $M --adapter $FAD --lora-scale $FSC --name homeward_final2 --rows $D/$set.jsonl --out $F/preds_$set.jsonl --bsz 16 --latency-n 20 > $L/final2_$set.log 2>&1
  P=$F/preds_$set.jsonl
  case $FN in *_sc)
    for s in 1 2; do python eval.py predict-local --model $M --adapter $FAD --lora-scale $FSC --name sc$s --rows $D/$set.jsonl --out $F/sc${s}_$set.jsonl --bsz 16 --latency-n 0 --temperature 0.7 --seed $s >> $L/final2_$set.log 2>&1; done
    python eval.py vote --rows $D/$set.jsonl --preds $F/preds_$set.jsonl --preds $F/sc1_$set.jsonl --preds $F/sc2_$set.jsonl --out $F/preds_${set}_sc.jsonl >> $L/final2_$set.log 2>&1
    P=$F/preds_${set}_sc.jsonl;;
  esac
  python eval.py verdict-probs --model $M --adapter $FAD --lora-scale $FSC --rows $D/$set.jsonl --preds $F/preds_$set.jsonl --out $F/probs_$set.jsonl >> $L/final2_$set.log 2>&1
  python eval.py escalate --preds $P --probs $F/probs_$set.jsonl --tau $FTAU --out $F/preds_${set}_final.jsonl >> $L/final2_$set.log 2>&1
done

mark replay
python onprem_replay.py --model $M --adapter $FAD --lora-scale $FSC --samples-dir $D/samples --out-dir $F/replay --run-name "run 4 selection: $FN" > $L/replay.log 2>&1
mark demo
cp $F/adapter_path.txt $R/final/adapter_path.txt; cp $F/lora_scale.txt $R/final/lora_scale.txt
bash $W/demo.sh > $F/onprem_demo.txt 2>&1
mark capacity
python eval.py predict-local --model $M --adapter $FAD --lora-scale $FSC --name homeward_final2 --rows $D/test.jsonl --out $F/cap_bs64.jsonl --bsz 64 --latency-n 0 > $L/final2_cap.log 2>&1
mark DAY2DONE
