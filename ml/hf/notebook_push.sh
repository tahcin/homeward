#!/bin/bash
# On the AMD notebook: push the selected adapter to main and the other candidate to a branch of
# oktash/homeward-teachback-gemma4-e4b. Needs /workspace/.hf_tok (deleted by hf_upload.py after reading).
# Log: /workspace/logs/hf_push.log. See ml/hf/PUBLISH.md.
set -u
R=/workspace/runs; L=/workspace/logs/hf_push.log
FAD=$(cat $R/final2/adapter_path.txt 2>/dev/null || cat $R/final/adapter_path.txt)
RUN2=$R/runB/adapter_avg
RUN4=${RUN4_ADAPTER:-$R/run4/adapter_avg}
if [ "$FAD" = "$RUN2" ]; then ALT="$RUN4:run4"; else ALT="$RUN2:run2"; fi
ARGS=(--token-file /workspace/.hf_tok --adapter "$FAD:main")
[ -f "${ALT%:*}/adapter_model.safetensors" ] && ARGS+=(--adapter "$ALT")
echo "selected: $FAD (scale $(cat $R/final2/lora_scale.txt 2>/dev/null)); alt: $ALT" > $L
cd /workspace/code && python hf_upload.py "${ARGS[@]}" >> $L 2>&1
echo "HF_PUSH_DONE rc=$?" >> $L
