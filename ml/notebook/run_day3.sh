#!/bin/bash
# Day-3 GPU window (Sun Oct 18): run 5. One process (run5.py) loads Gemma 4 E4B once, trains on splits_v4
# (the same jury-labelled pool as run 4 without the per-case cap), evaluates the in-process candidates on
# val_independent, makes the frozen pick with the declared rule (run 4 is the incumbent and listed first), and
# scores the pick once on the test sets. Then, only if run 5 won: transfer parts, replay JSON, on-prem demo,
# fresh-process capacity. Hard cutoffs (IST): training ends by TRAIN_DL, no pick by PICK_DL = keep run 4,
# everything stops by HARD_DL. Stage markers in /workspace/logs/day3.log.
set -u
W=${W:-/workspace}; D=$W/data; C=$W/code; L=$W/logs; M=$W/models/gemma-4-E4B-it; R=$W/runs; O=$R/run5; F=$O/final
TRAIN_DL=${TRAIN_DL:-13:05}; PICK_DL=${PICK_DL:-13:40}; HARD_DL=${HARD_DL:-14:05}
mkdir -p $L $O; cd $C
export TOKENIZERS_PARALLELISM=false HF_DEACTIVATE_ASYNC_LOAD=1 HSA_ENABLE_SDMA=0 PYTHONUNBUFFERED=1 FH_DIR=$L
ist(){ TZ=Asia/Kolkata date -d "today $1" +%s; }       # HH:MM IST today -> unix time
left(){ echo $(( $(ist $1) - $(date +%s) )); }          # seconds until HH:MM IST
mark(){ echo "=== $1 $(TZ=Asia/Kolkata date +%T) IST" >> $L/day3.log; }
mark start
bash $W/setup.sh >> $L/day3.log 2>&1
until [ "$(stat -c %s $M/model.safetensors 2>/dev/null)" = "${MODEL_BYTES:-15992595884}" ]; do
  mark resume_e4b; (cd $M && curl -sSL --retry 20 -C - -o model.safetensors "https://modelscope.cn/models/google/gemma-4-E4B-it/resolve/master/model.safetensors"); sleep 2; done

TR=$D/v4/sft_train_heardfirst.jsonl; VA=$D/v4/sft_val_heardfirst.jsonl
ROWS=$(wc -l < $TR)
# 0.476 s per training row per epoch (run 4: 3,738 rows x 2 epochs in 58.8 min, bsz 4 x accum 4, checkpointing, r32).
# Two epochs when they fit before TRAIN_DL with 4 min to spare (load, tokenising, val loss), else what fits (min 1).
EPOCHS=$(python -c "r=$ROWS; b=($(left $TRAIN_DL)-240)/(r*0.476); print(round(max(1.0, min(2.0, b)), 2))")
echo "rows $ROWS -> epochs $EPOCHS (est $(python -c "print(round($ROWS*0.476*$EPOCHS/60))") min); deadlines train $TRAIN_DL pick $PICK_DL hard $HARD_DL IST" >> $L/day3.log

INC="run4_final:$R/run4/adapter:1.0:$R/cands2/preds_run4_final.jsonl:$R/cands2/probs_run4_final.jsonl"
RARGS=(--model $M --train $TR --val-loss $VA --out $O --logs $L --val-rows $D/val_independent.jsonl
  --test test_independent=$D/test_independent.jsonl --test test=$D/test.jsonl --incumbent "$INC" --base-fr 2
  --epochs $EPOCHS --bsz 4 --accum 4 --rank 32 --alpha 32 --lr 2e-4 --verdict-weight 3 --ul-lambda 0.5 --max-len 1536
  --save-at 1.5 --train-deadline $TRAIN_DL --pick-deadline $PICK_DL --gen-bsz 64 --probs-bsz 8 --latency-n 20
  --load-timeout 150 ${RUN5_EXTRA:-})   # RUN5_EXTRA: only for dry runs (e.g. --max-steps 4)
# Watchdog fallback (beyond the in-process load timeout): kill run5.py if its log is silent for 6 min.
stale(){ while sleep 30; do P=$(pgrep -f "python run5.py" | head -1); [ -z "$P" ] && continue
  if [ $(( $(date +%s) - $(stat -c %Y $L/run5.log) )) -gt 360 ]; then echo "$(TZ=Asia/Kolkata date +%T) IST kill silent run5 pid $P" >> $L/day3.log; kill $P; sleep 5; kill -9 $P 2>/dev/null; fi; done; }
touch $L/run5.log; stale & WD=$!

n=0
until [ -f $F/DONE ] || [ -f $F/ABORTED ]; do
  T=$(left $HARD_DL); [ $T -lt 120 ] && { mark HARD_DEADLINE_before_done; break; }
  n=$((n+1)); [ $n -gt 6 ] && { mark TOO_MANY_RESTARTS; break; }
  mark run5_attempt_$n
  timeout $T python run5.py "${RARGS[@]}" >> $L/run5.log 2>&1
  rc=$?; mark "run5_exit_rc=$rc"
  grep -E "FROZEN PICK|TEST |ABORT|DEADLINE|candidate run|incumbent kept|RUN5 DONE" $L/run5.log | tail -12 > $L/run5_milestones.txt
done
kill $WD 2>/dev/null
cat $L/run5_milestones.txt >> $L/day3.log 2>/dev/null
[ -f $F/DONE ] && cat $F/DONE >> $L/day3.log
[ -f $F/ABORTED ] && { cat $F/ABORTED >> $L/day3.log; mark DAY3_KEEP_RUN4; }

# Everything below only if run 5 won (a final test file exists for the frozen pick).
if [ -f $F/preds_test_independent_final.jsonl ]; then
  AD=$(cat $F/adapter_path.txt); SC=$(cat $F/lora_scale.txt); NM=$(cat $F/name.txt)
  mark pack
  mkdir -p $F/hfparts && cp $AD/adapter_config.json $F/hfparts/ && split -b 20m $AD/adapter_model.safetensors $F/hfparts/part_ \
    && sha256sum $AD/adapter_model.safetensors | tee $F/hfparts/SHA256 >> $L/day3.log
  pk(){ (cd $O && tar czf $F/results_run5.tgz train_log.json select_final.json cands final/*.json final/*.jsonl final/*.txt \
         -C $L day3.log run5.log 2>/dev/null); }
  pk
  py(){ local t=$1; shift; local s=$(left $HARD_DL); [ $s -lt $((t+60)) ] && { echo "skip (time): $*" >> $L/day3.log; return 1; }
        timeout $t python fhwrap.py "$@" || { echo "retry after rc=$? : $1" >> $L/day3.log; s=$(left $HARD_DL); [ $s -lt $((t+60)) ] && return 1; timeout $t python fhwrap.py "$@"; }; }
  bash $C/sitter.sh run_day3.sh day3.log & SIT=$!
  mark replay
  py 300 onprem_replay.py --model $M --adapter $AD --lora-scale $SC --samples-dir $D/samples --out-dir $F/replay --run-name "run 5 selection: $NM" --adapter-ref "homeward run 5" > $L/replay5.log 2>&1
  mark demo
  py 300 onprem_demo.py --model $M --adapter $AD --lora-scale $SC --manifest $D/asr/manifest_human.jsonl --wavdir $D/asr --clip human_h02 > $F/onprem_demo_run5.txt 2>$L/demo5.err \
    || mv $F/onprem_demo_run5.txt $L/onprem_demo_run5.failed.txt   # a failed demo must not reach slide 7
  mark capacity
  # Fresh process, same frozen model, bs64 on the in-distribution test: deployable peak memory and a check that the
  # in-process merge gives the same outputs as eval.py's own load + merge path.
  py 420 eval.py predict-local --model $M --adapter $AD --lora-scale $SC --name homeward_run5 --rows $D/test.jsonl --out $F/cap_bs64.jsonl --bsz 64 --latency-n 0 > $L/cap5.log 2>&1 \
    && python -c "
import json
a={json.loads(l)['key']:json.loads(l)['text'] for l in open('$F/cap_bs64.jsonl')}
b={json.loads(l)['key']:json.loads(l)['text'] for l in open('$F/preds_test.jsonl')}
print('fresh-process vs in-process outputs identical on', sum(a[k]==b.get(k) for k in a), 'of', len(a), 'answers')" >> $L/day3.log 2>&1
  kill $SIT 2>/dev/null; pk
  mark DAY3DONE_RUN5
else
  mark DAY3DONE_NO_RUN5_TEST
fi
