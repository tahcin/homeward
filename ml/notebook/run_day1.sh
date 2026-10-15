#!/bin/bash
# Day-1 GPU window (Oct 16): runs A and B (E4B, sft_v2, verdict weight + unlikelihood; B = heard-first),
# candidate evals on val_independent, run C chosen automatically, LoRA-scale sweep, model selection by the
# fixed rule, then the frozen choice on both test sets once. Stage markers in /workspace/logs/day1.log.
set -u
W=/workspace; D=$W/data; C=$W/code; L=$W/logs; M=$W/models/gemma-4-E4B-it
cd $C; export TOKENIZERS_PARALLELISM=false
mark(){ echo "=== $1 $(date -u +%T)" >> $L/day1.log; }
mark start
START=$(date +%s); elapsed(){ echo $(( ($(date +%s) - START) / 60 )); }
bash $W/setup.sh >> $L/day1.log 2>&1
# E4B lives on persistent /workspace; finish (resume) the download if it is incomplete. HF speed varies 5-75 MB/s.
mkdir -p $M; cd $M
for f in config.json generation_config.json processor_config.json tokenizer.json tokenizer_config.json chat_template.jinja; do [ -s $f ] || curl -sSL --retry 20 -o $f https://huggingface.co/google/gemma-4-E4B-it/resolve/main/$f; done
until [ "$(stat -c %s model.safetensors 2>/dev/null)" = 15992595884 ]; do
  # ModelScope mirror ran at ~30 MB/s vs ~1 MB/s from huggingface.co on 2026-10-15 evening; identical sha256
  # (cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503) on both.
  mark resume_e4b_download
  curl -sSL --retry 20 -C - -o model.safetensors "https://modelscope.cn/models/google/gemma-4-E4B-it/resolve/master/model.safetensors"     || curl -sSL --retry 20 -C - -o model.safetensors https://huggingface.co/google/gemma-4-E4B-it/resolve/main/model.safetensors
  sleep 2
done
cd $C; mark e4b_ready

COMMON="--bsz ${BSZ:-4} --accum ${ACCUM:-4} --rank 32 --alpha 32 --lr 2e-4 --verdict-weight 3 --max-len 1536"
# Smoke (first GPU test of the new loss code): no checkpointing at bsz 8, verdict weight 3, unlikelihood 0.5.
SAFE=0
smoke(){ python train_lora.py --model $M --train $D/v2/sft_train.jsonl --val $D/v2/sft_val.jsonl --out $W/runs/smoke_$1 --max-steps 4 --rank 32 --alpha 32 --verdict-weight 3 --max-len 1536 "${@:2}" > $L/smoke_$1.log 2>&1; }
mark smoke_nockpt
if smoke nockpt --bsz 8 --accum 2 --no-ckpt --ul-lambda 0.5; then
  COMMON="--bsz 8 --accum 2 --no-ckpt --rank 32 --alpha 32 --lr 2e-4 --verdict-weight 3 --max-len 1536"; echo "no-ckpt OK" >> $L/day1.log
elif smoke ckpt --bsz 4 --accum 4 --ul-lambda 0.5; then
  echo "no-ckpt failed (see smoke_nockpt.log), keeping checkpointing" >> $L/day1.log
else
  # New loss code broke: fall back to the run-1 loss so the window is not wasted.
  SAFE=1; COMMON="--bsz 4 --accum 4 --rank 32 --alpha 32 --lr 2e-4 --max-len 1536"; echo "SAFE MODE: new loss failed, plain CE" >> $L/day1.log
fi
grep -E "unlikelihood|step [0-9]+/|peak|Error|Traceback" $L/smoke_nockpt.log | tail -6 >> $L/day1.log
python avg_adapters.py $W/runs/smoke_avgtest $W/runs/smoke_nockpt/adapter $W/runs/smoke_nockpt/adapter >> $L/day1.log 2>&1 || echo "avg_adapters FAILED" >> $L/day1.log

evalcand(){ # name adapter scale
  local n=$1 ad=$2 sc=$3 o=$W/runs/cands; mkdir -p $o
  python eval.py predict-local --model $M --adapter $ad --lora-scale $sc --name $n --rows $D/val_independent.jsonl --out $o/preds_$n.jsonl --bsz 16 --latency-n 0 > $L/cand_$n.log 2>&1
  python eval.py verdict-probs --model $M --adapter $ad --lora-scale $sc --rows $D/val_independent.jsonl --preds $o/preds_$n.jsonl --out $o/probs_$n.jsonl >> $L/cand_$n.log 2>&1
  echo "$n:$ad:$sc:$o/preds_$n.jsonl:$o/probs_$n.jsonl" >> $W/runs/cands/list.txt
}
mkdir -p $W/runs/cands; : > $W/runs/cands/list.txt
# run-1 adapter stays a candidate (its val_independent preds/probs already exist)
echo "run1:$W/runs/lora1/adapter:1.0:$W/runs/lora1/preds_ft_valind.jsonl:$W/runs/lora1/probs_ft_valind.jsonl" >> $W/runs/cands/list.txt

trainrun(){ # name trainfile valfile extra-args...
  local n=$1 tr=$2 va=$3; shift 3
  mark train_$n
  local args=("$@"); [ $SAFE = 1 ] && args=($(printf '%s
' "$@" | paste -sd' ' | sed -E 's/--ul-lambda [0-9.]+//g'))
  python train_lora.py --model $M --train $tr --val $va --out $W/runs/$n $COMMON "${args[@]}" > $L/train_$n.log 2>&1 || { mark TRAIN_FAILED_$n; return 1; }
  [ -d $W/runs/$n/adapter_epoch1 ] && python avg_adapters.py $W/runs/$n/adapter_avg $W/runs/$n/adapter_epoch1 $W/runs/$n/adapter >> $L/train_$n.log 2>&1
  mark eval_$n
  evalcand ${n}_final $W/runs/$n/adapter 1.0
  [ -d $W/runs/$n/adapter_avg ] && evalcand ${n}_avg $W/runs/$n/adapter_avg 1.0
}
select_now(){ python select_model.py --rows $D/val_independent.jsonl --out $W/runs/cands/select_$1.json $(sed 's/^/--cand /' $W/runs/cands/list.txt) > $L/select_$1.log 2>&1; cat $L/select_$1.log >> $L/day1.log; }

trainrun runA $D/v2/sft_train.jsonl $D/v2/sft_val.jsonl --epochs 2 --ul-lambda 0.5
trainrun runB $D/v2/sft_train_heardfirst.jsonl $D/v2/sft_val_heardfirst.jsonl --epochs 2 --ul-lambda 0.5
select_now AB

# Run C: refine the better of A/B. If its val FR (at tau 0) is above run 1's, raise lambda to 1.0; else 3 epochs.
read CHOICE CFR R1FR <<< $(python - <<'PY'
import json
s = json.load(open("/workspace/runs/cands/select_AB.json"))
c = [x for x in s["candidates"] if x["name"] != "run1"]
best = sorted(c, key=lambda r: (r["cost"], -r["accuracy"]))[0]
r1 = [x for x in s["candidates"] if x["name"] == "run1"][0]
print(best["name"], best["fr_tau0"], r1["fr_tau0"])
PY
)
echo "best of A/B: $CHOICE (val FR $CFR vs run1 $R1FR)" >> $L/day1.log
case $CHOICE in runB*) TR=$D/v2/sft_train_heardfirst.jsonl; VA=$D/v2/sft_val_heardfirst.jsonl;; *) TR=$D/v2/sft_train.jsonl; VA=$D/v2/sft_val.jsonl;; esac
if [ $(elapsed) -gt 85 ]; then echo "skip run C (elapsed $(elapsed) min)" >> $L/day1.log
elif [ "$CFR" -gt "$R1FR" ]; then trainrun runC $TR $VA --epochs 2 --ul-lambda 1.0; else trainrun runC $TR $VA --epochs 3 --ul-lambda 0.5; fi

# LoRA-scale sweep on the current best non-run1 candidate.
select_now ABC
read BN BAD <<< $(python -c "
import json; s=json.load(open('/workspace/runs/cands/select_ABC.json')); c=[x for x in s['candidates'] if x['name']!='run1']; b=sorted(c,key=lambda r:(r['cost'],-r['accuracy']))[0]; print(b['name'], b['adapter'])")
if [ $(elapsed) -lt 140 ]; then mark scale_sweep_$BN; evalcand ${BN}_s075 $BAD 0.75; evalcand ${BN}_s050 $BAD 0.5; else echo "skip scale sweep" >> $L/day1.log; fi
select_now final
read FN FAD FSC FTAU <<< $(python -c "
import json; c=json.load(open('/workspace/runs/cands/select_final.json'))['chosen']; print(c['name'], c['adapter'], c['scale'], c['tau'])")
echo "FINAL: $FN $FAD scale $FSC tau $FTAU" >> $L/day1.log

# Frozen choice on the test sets, once.
F=$W/runs/final; mkdir -p $F
for set in test_independent test; do
  mark final_$set
  python eval.py predict-local --model $M --adapter $FAD --lora-scale $FSC --name homeward_final --rows $D/$set.jsonl --out $F/preds_$set.jsonl --bsz 16 --latency-n 20 > $L/final_$set.log 2>&1
  python eval.py verdict-probs --model $M --adapter $FAD --lora-scale $FSC --rows $D/$set.jsonl --preds $F/preds_$set.jsonl --out $F/probs_$set.jsonl >> $L/final_$set.log 2>&1
  python eval.py escalate --preds $F/preds_$set.jsonl --probs $F/probs_$set.jsonl --tau $FTAU --out $F/preds_${set}_tau.jsonl >> $L/final_$set.log 2>&1
done
mark capacity; python eval.py predict-local --model $M --adapter $FAD --lora-scale $FSC --name homeward_final --rows $D/test.jsonl --out $F/cap_bs64.jsonl --bsz 64 --latency-n 0 > $L/final_cap.log 2>&1
echo "$FAD" > $W/runs/final/adapter_path.txt
mark DAY1DONE
