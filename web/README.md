# Homeward web app

The hosted demo of Homeward: a discharge nurse pastes a discharge note, confirms the facts the patient must understand, gets a patient version in the patient's language with every fact verified, runs a spoken teach-back with the patient, and gets a readiness board.

Live: https://homeward.tahcin.tech (fallback: https://homeward.tahcin49.workers.dev). Synthetic data only; no login.

## Architecture

```
browser (React, Vite)                      Cloudflare Worker (worker/)                       AMD Token Factory
---------------------                      ---------------------------                       -----------------
note, facts editor, patient mode   --->    /api/extract    Gemma 4 31B                --->   AMD GPUs
speech: browser recognition                /api/prepare    write -> lock -> repair (NDJSON stream)
or WAV to /api/transcribe                  /api/grade      grader prompt (prompts/grader.json)
readiness board, audit trail               /api/ocr        MinerU 2.5 Pro
                                           Guard (Durable Object, SQLite): per-IP and daily limits
```

- `worker/pipeline.ts`: the prompts and steps. Gemma 4 31B extracts facts, writes the patient version and grades. DeepSeek V4 Flash is the independent checker (quotes each fact, back-translates, compares meaning) and the automatic fallback when Gemma fails; the audit trail marks every fallback.
- `worker/lock.ts`: the fact lock in plain code. A fact passes only if the quoted sentence is verbatim in the patient text and contains every anchor: medicine names (Latin script, as on the box label), numbers after normalising Devanagari, Bengali, Arabic-Indic and Persian digits, medical units, dates by day number plus the month in the patient's language (from the runtime's ICU data), and 12-hour times in 24-hour form. In non-Latin scripts, any Latin word that is not a medicine or person name or a medical unit ("Oct", "am", "min") fails the fact. Then the checker model translates every sentence back word for word, told to keep errors, never guess what was meant, and list wrong or nonsensical words with a severity; one "wrong" word fails the fact. It also compares meaning (negation, frequency, dose, timing, action, any missing symptom or condition), and critical warning signs get a second, fact-aware reading. Failing facts get up to two targeted repairs (only their sentences are rewritten and spliced in by code), then are flagged for the nurse. A checker failure is retried, never "repaired". Qwen3.8-Flash-Next was dropped as a checker because it misreads Devanagari vowel signs. Tests: `pnpm test`.
- `worker/structured.ts`: checks decided in code from fields the checker fills in. Medicine facts are compared field by field (dose, unit, doses per day where "twice a day" = "every 12 hours" = "9 am and 9 pm", time of day, take vs stop, only when needed, maximum per day, duration) between the English fact and the literal back-translation of its sentences; any mismatch fails the fact with the field named. Warning-sign facts are split into items and every item must appear, with the quoted words really present in the back-translation. Also: local-script unit names ("40 毫克", "40 मिलीग्राम") accepted right after a number, ordinary words removed from anchors, and "furosemide (Furosemide)" tidied to one name.
- `worker/rules.ts`: clinical rules in code. Medicine facts naming a high-risk class (anticoagulants, antiplatelets, beta-blockers, insulin and other glucose-lowering medicines, opioids, diuretics, digoxin, lithium, methotrexate, potassium, anticonvulsants) are critical by default; the nurse can untick them.
- Localisation: the writer puts the local medicine name first with the box-label name in brackets ("paracetamolo (paracetamol)"), uses the local decimal comma and doctors' titles ("Dott.ssa"); the lock accepts all three.
- `ONPREM_REPLAY.md`: the data format for replaying our fine-tuned Gemma 4 E4B's grades (recorded on the AMD Radeon PRO W7900D) on a sample's results board; the switch appears only when `public/samples/<id>.onprem.json` exists.
- `scripts/lock-bench.mjs`: the fact-lock benchmark (planted faults across languages, clean controls); writes `ml/results/lock_bench.json` and `.md`.
- `worker/tf.ts`: Token Factory client with a model chain, patient waiting on rate and concurrency limits (the team shares one key), cost and latency per call.
- `worker/index.ts`: routes, input limits, and the `Guard` Durable Object (free tier, SQLite) that stores only a salted IP hash with a timestamp and the day's spend. No patient text is stored or logged (`observability` is off).
- `src/`: the React app, one warm light theme (WCAG AA text contrast). Patient mode uses large type, one question per screen, a big microphone button with a timer, read-aloud on every screen, and right-to-left layout for Arabic. Session state lives in `sessionStorage`; answer audio stays in memory for the nurse to replay.
- `public/samples/*.json`: three fictional patients recorded once through the real pipeline by `scripts/build-samples.mjs` (notes in `scripts/notes/`, nurse-reviewed facts in `scripts/runs/`, scripted teach-back answers in `scripts/answers/`). Two runs put a deliberate error into the first draft (a reversed instruction, a doubled dose) so the replay shows the lock catching it. The answers are written by us; every grade in the files comes from the real grader.
- `public/eval.json`: evaluation numbers, written by the ML pipeline. Nulls render as Pending.
- `src/i18n.json`: patient-mode button labels in 12 languages, translated once with Gemma 4 31B.

## Setup

```bash
pnpm install
pnpm test                      # fact lock unit tests
pnpm build                     # typecheck + vite build into dist/
npx wrangler dev --env-file /path/outside/repo/.env   # .env holds TOKENFACTORY_API_KEY=...
npx wrangler deploy
```

Secrets (never in the repo): `wrangler secret put TOKENFACTORY_API_KEY` and `wrangler secret put IP_SALT`.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `TOKENFACTORY_API_KEY` | (secret) | AMD Token Factory key |
| `IP_SALT` | (secret) | salt for the rate-limit IP hash |
| `LIVE_ENABLED` | `true` | `false` stops all live model calls; samples keep working |
| `LIMIT_PER_IP_HOUR` | `40` | live API calls per IP per hour |
| `LIMIT_USD_PER_DAY` | `0.5` | share of the team's $1/day Token Factory cap this site may spend |
| `MIN_UPSTREAM_USD` | `0.05` | close live calls when the shared key has less than this left today |
| `ONPREM_GRADER_URL` | unset | OpenAI-compatible base URL (ending in `/v1`) serving the fine-tuned Gemma 4 E4B grader; used first when healthy, labelled in the audit trail |
| `ONPREM_GRADER_MODEL`, `ONPREM_GRADER_KEY`, `ONPREM_GRADER_LABEL` | | model id, bearer key (secret), label shown in the audit trail |
| `ONPREM_ASR_URL` | unset | OpenAI-compatible base URL for on-site transcription |
| `ONPREM_ASR_KIND` | `chat` | `chat`: `/chat/completions` with an `input_audio` part (Gemma 4 E4B); `whisper`: `/audio/transcriptions` |
| `ONPREM_ASR_MODEL`, `ONPREM_ASR_KEY` | | model id and key for transcription |
| `MIMO_AUDIO` | unset | `true` tries MiMo-V2.6-Flash audio input on Token Factory (documented, but rejected upstream as of 2026-10-15) |

## Known limits

- Without a server transcriber the hosted demo uses the browser's speech recognition (Chrome, Edge, Safari; not Firefox, which falls back to typing). The patient confirms the transcript before it is graded.
- The shared Token Factory key has a $1/day cap, 20 requests a minute and 8 concurrent requests for the whole team, so a live patient version can take one to two minutes when other work is running. Samples replay instantly.
- Read-aloud uses the device's own voices; some languages (often Tagalog, Bengali, Tamil) have no voice on some devices, and the button is hidden.

## Licences

Code: MIT. Satoshi font (public/fonts) by Indian Type Foundry, under the ITF Free Font License, which allows web use. Noto Sans Devanagari, Bengali, Tamil, Arabic and SC via Fontsource, under the SIL Open Font License. Icons drawn after Lucide (ISC).
