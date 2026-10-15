# Speech recognition and on-prem loop

not measured: Token Factory lists it as text+image only and its backend rejected input_audio on 2026-10-15.
TTS clips are macOS `say` voices reading the independent test transcripts written by Claude: clean studio speech, so they flatter ASR compared with real patients.

| System | Source | Language | Clips | WER (edits/words) | CER (edits/chars) | WER normalised | CER normalised |
|---|---|---|---|---|---|---|---|
| e4b | human | ALL | 15 | 20.2% (48/238) | 21.9% (213/974) | 20.2% | 21.9% |
| e4b | human | bn | 5 | 33.3% (23/69) | 31.3% (99/316) | 33.3% | 31.3% |
| e4b | human | en | 5 | 6.2% (5/81) | 3.6% (12/334) | 6.2% | 3.6% |
| e4b | human | hi | 5 | 22.7% (20/88) | 31.5% (102/324) | 22.7% | 31.5% |
| e4b | tts | ALL | 110 | 27.5% (804/2928) | 21.4% (2681/12500) | 25.6% | 20.6% |
| e4b | tts | ar | 10 | 44.5% (109/245) | 18.0% (202/1120) | 22.9% | 13.6% |
| e4b | tts | bn | 10 | 36.4% (76/209) | 19.4% (179/925) | 36.4% | 19.4% |
| e4b | tts | en | 10 | 19.5% (69/353) | 17.3% (236/1368) | 19.5% | 17.3% |
| e4b | tts | es | 10 | 13.1% (44/335) | 11.4% (147/1292) | 13.1% | 11.4% |
| e4b | tts | fr | 10 | 32.8% (96/293) | 25.0% (291/1164) | 32.8% | 25.0% |
| e4b | tts | hi | 10 | 28.9% (81/280) | 36.2% (375/1036) | 28.9% | 36.2% |
| e4b | tts | it | 10 | 9.5% (34/357) | 11.5% (178/1552) | 9.5% | 11.5% |
| e4b | tts | pt | 10 | 14.8% (38/257) | 14.1% (155/1100) | 14.8% | 14.1% |
| e4b | tts | ta | 10 | 53.8% (114/212) | 39.6% (537/1357) | 53.8% | 39.6% |
| e4b | tts | vi | 10 | 26.7% (89/333) | 21.5% (251/1166) | 26.7% | 21.5% |
| e4b | tts | zh | 10 | 100.0% (54/54) | 31.0% (130/420) | 98.1% | 17.1% |
| whisper | human | ALL | 15 | 38.2% (91/238) | 33.1% (322/974) | 38.2% | 33.1% |
| whisper | human | bn | 5 | 91.3% (63/69) | 66.5% (210/316) | 91.3% | 66.5% |
| whisper | human | en | 5 | 8.6% (7/81) | 4.8% (16/334) | 8.6% | 4.8% |
| whisper | human | hi | 5 | 23.9% (21/88) | 29.6% (96/324) | 23.9% | 29.6% |
| whisper | tts | ALL | 110 | 27.9% (817/2928) | 19.8% (2472/12500) | 27.8% | 19.7% |
| whisper | tts | ar | 10 | 21.2% (52/245) | 5.3% (59/1120) | 20.0% | 4.4% |
| whisper | tts | bn | 10 | 87.6% (183/209) | 52.2% (483/925) | 87.6% | 52.2% |
| whisper | tts | en | 10 | 14.7% (52/353) | 14.8% (202/1368) | 14.7% | 14.8% |
| whisper | tts | es | 10 | 8.1% (27/335) | 8.6% (111/1292) | 8.1% | 8.6% |
| whisper | tts | fr | 10 | 17.7% (52/293) | 14.1% (164/1164) | 17.7% | 14.1% |
| whisper | tts | hi | 10 | 33.2% (93/280) | 36.7% (380/1036) | 33.2% | 36.7% |
| whisper | tts | it | 10 | 11.8% (42/357) | 11.7% (182/1552) | 11.8% | 11.7% |
| whisper | tts | pt | 10 | 13.6% (35/257) | 12.0% (132/1100) | 13.6% | 12.0% |
| whisper | tts | ta | 10 | 64.6% (137/212) | 34.5% (468/1357) | 64.6% | 34.5% |
| whisper | tts | vi | 10 | 27.0% (90/333) | 19.8% (231/1166) | 27.0% | 19.8% |
| whisper | tts | zh | 10 | 100.0% (54/54) | 14.3% (60/420) | 100.0% | 14.3% |

## Grading from transcripts (fine-tuned E4B)

| Transcript / source | Answers | Facts | Same grade as from reference | Accuracy vs labels | False reassurance | Critical FR |
|---|---|---|---|---|---|---|
| ref/tts | 110 | 261 | 261/261 | 229/261 | 7/103 | 6/83 |
| ref/human | 15 | 24 | 24/24 | 24/24 | 0/9 | 0/7 |
| e4b/tts | 110 | 261 | 217/261 | 212/261 | 2/103 | 2/83 |
| e4b/human | 15 | 24 | 24/24 | 24/24 | 0/9 | 0/7 |
| whisper/tts | 110 | 261 | 222/261 | 208/261 | 6/103 | 5/83 |
| whisper/human | 15 | 24 | 22/24 | 22/24 | 0/9 | 0/7 |

## On-prem loop (W7900D)

25 clips: end-to-end p50 10.9 s, p95 14.9 s (ASR p50 1.8 s, grading p50 8.7 s, audio p50 6.9 s); peak VRAM 15.1 GiB.
