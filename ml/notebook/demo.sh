#!/bin/bash
# One-command on-prem demo: Gemma 4 E4B ASR + fine-tuned E4B grading of a real Hindi clip on the W7900D.
bash /workspace/setup.sh >/dev/null 2>&1
M=/workspace/models/gemma-4-E4B-it
AD=$(cat /workspace/runs/final/adapter_path.txt 2>/dev/null || echo /workspace/runs/lora1/adapter)
SC=$(cat /workspace/runs/final/lora_scale.txt 2>/dev/null || echo 1.0)
cd /workspace/code && python onprem_demo.py --model $M --adapter $AD --lora-scale $SC --manifest /workspace/data/asr/manifest_human.jsonl --wavdir /workspace/data/asr --clip ${1:-human_h02} 2>/dev/null
