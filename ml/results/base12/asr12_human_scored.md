# Speech recognition and on-prem loop

not measured: Token Factory lists it as text+image only and its backend rejected input_audio on 2026-10-15.
TTS clips are macOS `say` voices reading the independent test transcripts written by Claude: clean studio speech, so they flatter ASR compared with real patients.

| System | Source | Language | Clips | WER (edits/words) | CER (edits/chars) | WER normalised | CER normalised |
|---|---|---|---|---|---|---|---|
| e4b | human | ALL | 15 | 41.2% (98/238) | 47.1% (459/974) | 41.2% | 47.1% |
| e4b | human | bn | 5 | 101.4% (70/69) | 105.1% (332/316) | 101.4% | 105.1% |
| e4b | human | en | 5 | 11.1% (9/81) | 7.2% (24/334) | 11.1% | 7.2% |
| e4b | human | hi | 5 | 21.6% (19/88) | 31.8% (103/324) | 21.6% | 31.8% |
