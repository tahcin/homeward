# On-site grader replay: data format

The results board of each sample patient can switch to "Grade with the on-site model (recorded on the AMD Radeon PRO W7900D)". The switch appears only when this file exists:

```
web/public/samples/<sample id>.onprem.json      # heart-failure-it, anticoagulant-hi, knee-replacement-es
```

Produce it on the W7900D by grading the sample's own recorded answers with the fine-tuned Gemma 4 E4B, using `app/prompts/grader.json` verbatim (the same messages the web app builds: `graderMessages()` in `web/worker/pipeline.ts`). Grade every recorded attempt in `web/public/samples/<id>.json` → `answers[<topic id>][<attempt index>]`, with that attempt's `transcript`, its `fact_ids` (only those facts), the topic's English `question` and the sample's language name.

```json
{
  "sample_id": "anticoagulant-hi",
  "recorded_at": "2026-10-16T08:00:00Z",
  "model": {
    "name": "Gemma 4 E4B, fine-tuned for Homeward (run 2)",
    "base": "google/gemma-4-e4b-it",
    "adapter": "oktash/homeward-gemma4-e4b-grader@<revision>"
  },
  "hardware": {
    "gpu": "AMD Radeon PRO W7900D (48 GB)",
    "rocm": "6.x.y",
    "runtime": "transformers 4.x, bf16, batch 1"
  },
  "prompt": "app/prompts/grader.json (verbatim)",
  "answers": {
    "t1": [
      {
        "attempt": 0,
        "transcript": "<exactly the transcript in the sample file>",
        "fact_ids": ["f1", "f2", "f3", "f4", "f5"],
        "grades": [
          { "id": "f1", "verdict": "incorrect", "heard": "Apixaban once in the morning", "reteach": "Take Apixaban 5 mg twice a day, at 9 am and 9 pm." }
        ],
        "latency_ms": 4892,
        "tokens_in": 512,
        "tokens_out": 140,
        "tokens_per_s": 28.6,
        "peak_vram_gib": 17.2
      }
    ]
  }
}
```

Rules:

- One entry per recorded attempt, in the same order, with `transcript` and `fact_ids` identical to the sample file (the app matches on them and ignores anything that differs).
- `grades`: every requested fact id, `verdict` one of `correct | partial | incorrect | missing`, as parsed from the model's JSON. If the output can't be parsed, keep the entry with `"parse_error": true`, `"raw": "<model output>"` and no `grades`; the board then keeps the hosted grade for those facts and says so.
- Timings are measured on the card for that single request: `latency_ms` is wall time from request to parsed output, `tokens_per_s` is output tokens divided by generation time, `peak_vram_gib` is the peak from `torch.cuda.max_memory_allocated()` or `rocm-smi` during the run.
- Never edit numbers by hand. Re-generate the file if the sample is re-recorded (the transcripts change).

What the app shows: the per-fact verdicts and "Understanding confirmed / Re-teach needed" recomputed from the on-site grades, and an on-site audit table (model, card, latency, tokens in and out, tokens per second, peak VRAM) next to the hosted audit trail.
