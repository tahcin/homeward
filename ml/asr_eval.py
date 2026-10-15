"""Speech recognition check for Homeward: Gemma 4 E4B (native audio) vs Whisper
large-v3-turbo, both on the AMD Radeon PRO W7900D, plus the full on-prem loop
(E4B transcribes, the fine-tuned E4B grades) and whether grades change when the grader
reads the ASR transcript instead of the reference text.

MiMo-V2.6-Flash was the planned hosted baseline, but on 2026-10-15 Token Factory listed
it as text+image only and its backend rejected every audio content format we tried, so
the baseline is Whisper large-v3-turbo (MIT) on the same GPU.

Subcommands:
  synth     (Mac) macOS `say` -> 16 kHz mono WAV for up to --per-lang rows per language
            from a rows file (e.g. the independent test set); clips over 29.5 s are skipped
            (Gemma 4 audio limit is 30 s).
  human     (Mac) convert a folder of recordings (m4a/wav) listed in a refs JSONL
            (fields: clip, audio, language, language_name, question, facts, transcript, labels).
  gpu       (notebook) transcribe every clip with E4B and Whisper, grade from reference and
            from each ASR transcript with base E4B + LoRA adapter, time the on-prem loop.
  score     (Mac) WER/CER per language, system and source; grade agreement and accuracy.

Manifest row: {"id", "wav", "source": "tts"|"human", "voice", "language", "language_name",
               "ref", "duration_s", "row": {grader row with facts, question, labels}}
"""

import argparse
import json
import re
import statistics
import subprocess
import time
import unicodedata
import wave
from collections import defaultdict
from pathlib import Path

import grader

VOICES = {"it": "Alice", "es": "Mónica", "hi": "Lekha", "zh": "Tingting", "ar": "Majed",
          "pt": "Luciana", "fr": "Jacques", "en": "Samantha", "vi": "Linh", "bn": "Piya", "ta": "Vani"}
LANG_EN = {"it": "Italian", "es": "Spanish", "hi": "Hindi", "zh": "Mandarin Chinese", "ar": "Arabic",
           "pt": "Portuguese", "fr": "French", "en": "English", "vi": "Vietnamese", "bn": "Bengali",
           "ta": "Tamil", "tl": "Tagalog"}
WHISPER_LANG = {"zh": "chinese", "tl": "tagalog"}
CHAR_LANGS = {"zh"}  # no spaces between words, so CER is the meaningful number

ASR_PROMPT = ("Transcribe the following speech segment in {lang} into {lang} text.\n\n"
              "Follow these specific instructions for formatting the answer:\n"
              "* Only output the transcription, with no newlines.\n"
              "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, "
              "and write 3 instead of three.")


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def wav_seconds(p):
    with wave.open(str(p)) as w:
        return w.getnframes() / w.getframerate()


# ---------------------------------------------------------------- Mac side

def synth(args):
    rows = read_jsonl(args.rows)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    per = defaultdict(int)
    man = []
    for r in rows:
        lang = r["language"]
        if lang not in VOICES or per[lang] >= args.per_lang:
            continue
        cid = f'{lang}_{grader.row_key(r).replace("/", "_")}'
        aiff, wav = out / f"{cid}.aiff", out / f"{cid}.wav"
        subprocess.run(["say", "-v", VOICES[lang], "-o", str(aiff), r["transcript"]], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)
        aiff.unlink()
        dur = wav_seconds(wav)
        if dur > 29.5:
            wav.unlink()
            continue
        per[lang] += 1
        man.append({"id": cid, "wav": wav.name, "source": "tts", "voice": f"macOS say {VOICES[lang]}",
                    "language": lang, "language_name": r["language_name"], "ref": r["transcript"],
                    "duration_s": round(dur, 2), "row": r})
    Path(args.manifest).write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in man))
    print(dict(per), len(man), "clips")


def human(args):
    refs = read_jsonl(args.refs)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    man, missing = [], []
    for r in refs:
        src = Path(str(r["audio"]).replace("~", str(Path.home()), 1))
        if not src.exists():
            missing.append(str(src))
            continue
        wav = out / f'human_{r["clip"]}.wav'
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(src), str(wav)], check=True)
        row = {"case_id": "owner", "topic_id": r["clip"], "n": 0, "language": r["language"],
               "language_name": r["language_name"], "question": r["question"], "facts": r["facts"],
               "transcript": r["transcript"], "labels": r["labels"], "source": r.get("source", "owner-voice")}
        man.append({"id": f'human_{r["clip"]}', "wav": wav.name, "source": "human", "voice": "owner",
                    "language": r["language"], "language_name": r["language_name"], "ref": r["transcript"],
                    "duration_s": round(wav_seconds(wav), 2), "row": row})
    Path(args.manifest).write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in man))
    print(len(man), "clips;", len(missing), "missing", missing[:3])


# ---------------------------------------------------------------- GPU side

def load_wav(p):
    import numpy as np
    with wave.open(str(p)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return x


def gpu(args):
    import torch
    import transformers
    from peft import PeftModel

    man = read_jsonl(args.manifest)
    if args.limit:
        man = man[:args.limit]
    wavdir = Path(args.wavdir)
    prompt = grader.load_prompt(args.prompt)
    proc = transformers.AutoProcessor.from_pretrained(args.model)
    tok = proc.tokenizer
    torch.cuda.reset_peak_memory_stats()
    from train_lora import load_model_and_tokenizer
    model, _ = load_model_and_tokenizer(args.model)  # text eager, audio tower SDPA (see there)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    import contextlib
    no_adapter = model.disable_adapter if args.adapter else contextlib.nullcontext
    vram_loaded = torch.cuda.memory_allocated() / 2**30

    def sync_time():
        torch.cuda.synchronize()
        return time.time()

    def e4b_asr(audio, lang):
        msgs = [{"role": "user", "content": [{"type": "text", "text": ASR_PROMPT.format(lang=LANG_EN.get(lang, lang))},
                                             {"type": "audio", "audio": audio}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                       add_generation_prompt=True, enable_thinking=False).to(model.device)
        if "input_features" in inp:
            inp["input_features"] = inp["input_features"].to(torch.bfloat16)
        with torch.no_grad(), no_adapter():
            out = model.generate(**inp, max_new_tokens=300, do_sample=False)
        return tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def grade(row, transcript):
        r = dict(row, transcript=transcript)
        text = tok.apply_chat_template(grader.messages(r, prompt), tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=768, do_sample=False)
        new = out[0, ids["input_ids"].shape[1]:]
        return tok.decode(new, skip_special_tokens=True), int(new.shape[0])

    do_grade = bool(args.adapter) and args.grade
    # warm-up
    e4b_asr(load_wav(wavdir / man[0]["wav"]), man[0]["language"])
    if do_grade:
        grade(man[0]["row"], man[0]["ref"])

    res = []
    for k, m in enumerate(man):
        audio = load_wav(wavdir / m["wav"])
        t0 = sync_time()
        hyp = e4b_asr(audio, m["language"])
        t1 = sync_time()
        rec = {"id": m["id"], "e4b_hyp": hyp, "e4b_asr_s": t1 - t0, "audio_s": m["duration_s"]}
        t2 = t1
        if do_grade:
            g_hyp, n_hyp = grade(m["row"], hyp)
            t2 = sync_time()
            rec.update({"grade_from_e4b_hyp": g_hyp, "grade_from_e4b_hyp_s": t2 - t1, "grade_tokens": n_hyp,
                        "loop_s": t2 - t0})
        res.append(rec)
        print(f"{k + 1}/{len(man)} {m['id']} asr {t1 - t0:.1f}s grade {t2 - t1:.1f}s | {hyp[:80]}", flush=True)
    peak_e4b = torch.cuda.max_memory_allocated() / 2**30

    if args.skip_whisper:
        man_w = []
    else:
        man_w = man
    # Whisper baseline on the same GPU.
    if not man_w:
        wh_proc = wh = None
    else:
      wh_proc = transformers.AutoProcessor.from_pretrained(args.whisper)
      wh = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
        args.whisper, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()

    def whisper_asr(audio, lang):
        feats = wh_proc(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.bfloat16)
        with torch.no_grad():
            ids = wh.generate(feats, language=WHISPER_LANG.get(lang, lang), task="transcribe", max_new_tokens=300)
        return wh_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    if man_w:
        whisper_asr(load_wav(wavdir / man[0]["wav"]), man[0]["language"])
    for k, (m, r) in enumerate(zip(man_w, res)):
        audio = load_wav(wavdir / m["wav"])
        t0 = sync_time()
        hyp = whisper_asr(audio, m["language"])
        t1 = sync_time()
        r.update({"whisper_hyp": hyp, "whisper_asr_s": t1 - t0})
        print(f"whisper {k + 1}/{len(man)} {t1 - t0:.1f}s | {hyp[:80]}", flush=True)

    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res))
    meta = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": transformers.__version__, "e4b": args.model, "adapter": args.adapter,
            "whisper": args.whisper, "clips": len(res),
            "vram_after_load_e4b_plus_adapter_gib": vram_loaded,
            "peak_vram_e4b_asr_plus_grading_gib": peak_e4b,
            "peak_vram_with_whisper_also_loaded_gib": torch.cuda.max_memory_allocated() / 2**30,
            "note": "one E4B copy in memory: ASR runs with the LoRA adapter disabled, grading with it enabled; greedy decoding, batch size 1"}
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- scoring

def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in s)
    return re.sub(r"\s+", " ", s).strip()


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


_T2S = None


def lenient(s, lang):
    """Orthography-only normalisation used for the second ("normalised") score: Traditional to
    Simplified Chinese (OpenCC t2s) and the usual Arabic ASR normalisation (alef variants to
    bare alef, ta marbuta to ha, alef maqsura to ya, diacritics removed)."""
    global _T2S
    if lang == "zh":
        if _T2S is None:
            try:
                import opencc
                _T2S = opencc.OpenCC("t2s").convert
            except ImportError:
                _T2S = False
        if _T2S:
            s = _T2S(s)
    if lang == "ar":
        s = re.sub(r"[\u064B-\u0652\u0670]", "", s)
        s = re.sub(r"[\u0622\u0623\u0625]", "\u0627", s).replace("\u0629", "\u0647").replace("\u0649", "\u064A")
    return s


def err_counts(ref, hyp, lang=None):
    if lang:
        ref, hyp = lenient(ref, lang), lenient(hyp, lang)
    r, h = norm(ref), norm(hyp)
    rw, hw = r.split(), h.split()
    rc, hc = r.replace(" ", ""), h.replace(" ", "")
    return edits(rw, hw), len(rw), edits(list(rc), list(hc)), len(rc)


def pctl(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def rows_cmd(args):
    """Grader rows whose transcript is the reference or one system's ASR output (for eval.py predict-local)."""
    man = [m for p in args.manifest for m in read_jsonl(p)]
    res = {r["id"]: r for p in args.results for r in read_jsonl(p)}
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for src in ("ref", "e4b", "whisper"):
        lines = []
        for m in man:
            r = res.get(m["id"])
            text = m["ref"] if src == "ref" else (r or {}).get(f"{src}_hyp")
            if text is None:
                continue
            lines.append(json.dumps(dict(m["row"], transcript=text, clip_id=m["id"]), ensure_ascii=False))
        (out / f"rows_{src}.jsonl").write_text("\n".join(lines) + "\n")
        print(src, len(lines))


def score(args):
    man = {m["id"]: m for p in args.manifest for m in read_jsonl(p)}
    res = [r for p in args.results for r in read_jsonl(p)]
    meta = {}
    for p in args.results:
        mp = Path(p).with_suffix(".meta.json")
        if mp.exists():
            meta[p] = json.loads(mp.read_text())
    agg = defaultdict(lambda: [0, 0, 0, 0, 0])  # word edits, words, char edits, chars, clips
    aggn = defaultdict(lambda: [0, 0, 0, 0, 0])  # same after orthographic normalisation
    lat = defaultdict(list)
    per_clip = []
    for r in res:
        m = man[r["id"]]
        src, lang = m["source"], m["language"]
        clip = {"id": r["id"], "source": src, "language": lang, "ref": m["ref"], "duration_s": m["duration_s"]}
        for sysname in ("e4b", "whisper"):
            hyp = r.get(f"{sysname}_hyp")
            if hyp is None:
                continue
            we, wn, ce, cn = err_counts(m["ref"], hyp)
            nwe, nwn, nce, ncn = err_counts(m["ref"], hyp, lang)
            for key in ((sysname, src, lang), (sysname, src, "ALL")):
                a = agg[key]
                a[0] += we; a[1] += wn; a[2] += ce; a[3] += cn; a[4] += 1
                b = aggn[key]
                b[0] += nwe; b[1] += nwn; b[2] += nce; b[3] += ncn; b[4] += 1
            lat[(sysname, src)].append(r[f"{sysname}_asr_s"])
            clip[f"{sysname}_hyp"] = hyp
            clip[f"{sysname}_wer"] = we / wn if wn else None
            clip[f"{sysname}_cer"] = ce / cn if cn else None
        per_clip.append(clip)

    # Grading from transcripts: preds from eval.py predict-local over rows_<src>.jsonl.
    grading = {}
    if args.grades:
        preds = {}
        for spec in args.grades:
            name, path = spec.split("=", 1)
            preds[name] = {}
            rows = read_jsonl(Path(path).parent / f"rows_{name}.jsonl") if (Path(path).parent / f"rows_{name}.jsonl").exists() else None
            for p in read_jsonl(path):
                preds[name][p["key"]] = p
        ref_rows = {grader.row_key(m["row"]): m for m in man.values()}
        ref_v = {}
        for k, m in ref_rows.items():
            ids = [f["id"] for f in m["row"]["facts"]]
            ref_v[k] = grader.verdicts_from(grader.parse_reply(preds["ref"][k]["text"]), ids)[0] if k in preds.get("ref", {}) else None
        for name, pr in preds.items():
            for src in ("tts", "human"):
                g = {"answers": 0, "facts": 0, "same_as_ref_grade": 0, "correct_vs_gold": 0,
                     "false_reassurance": 0, "bad_facts": 0, "critical_false_reassurance": 0, "critical_bad_facts": 0,
                     "invalid": 0, "answers_with_changed_grade": 0}
                for k, m in ref_rows.items():
                    if m["source"] != src or k not in pr:
                        continue
                    ids = [f["id"] for f in m["row"]["facts"]]
                    v = grader.verdicts_from(grader.parse_reply(pr[k]["text"]), ids)[0]
                    rv = ref_v.get(k) or [None] * len(ids)
                    g["answers"] += 1
                    g["answers_with_changed_grade"] += any(a != b for a, b in zip(v, rv))
                    for f, lab, hv, r0 in zip(m["row"]["facts"], m["row"]["labels"], v, rv):
                        gv = lab["verdict"]
                        g["facts"] += 1
                        g["same_as_ref_grade"] += hv == r0
                        g["correct_vs_gold"] += hv == gv
                        g["invalid"] += hv is None
                        if gv in ("incorrect", "missing"):
                            g["bad_facts"] += 1
                            g["false_reassurance"] += hv == "correct"
                            if f.get("critical"):
                                g["critical_bad_facts"] += 1
                                g["critical_false_reassurance"] += hv == "correct"
                if g["answers"]:
                    grading[f"{name}/{src}"] = g

    asr = {}
    for (sysname, src, lang), (we, wn, ce, cn, n) in sorted(agg.items()):
        asr.setdefault(sysname, {}).setdefault(src, {})[lang] = {
            "clips": n, "words": wn, "word_edits": we, "wer": we / wn if wn else None,
            "chars": cn, "char_edits": ce, "cer": ce / cn if cn else None,
            "primary": "cer" if lang in CHAR_LANGS else "wer"}
        nwe, nwn, nce, ncn, _ = aggn[(sysname, src, lang)]
        asr[sysname][src][lang]["normalised"] = {"wer": nwe / nwn if nwn else None, "word_edits": nwe, "words": nwn,
                                                 "cer": nce / ncn if ncn else None, "char_edits": nce, "chars": ncn}
    out = {
        "schema": "homeward.asr.v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "clips": {"tts": sum(1 for m in man.values() if m["source"] == "tts"),
                  "human": sum(1 for m in man.values() if m["source"] == "human")},
        "normalised_score": "second score per row after orthography-only normalisation: Traditional to Simplified Chinese (OpenCC t2s) and standard Arabic letter normalisation; number words vs digits and script choice in code-mixed speech still count as errors",
        "normalisation": "NFKC, lowercase, punctuation and symbols removed, whitespace collapsed; WER over whitespace tokens, CER over characters without spaces. Script choice counts as an error (e.g. an English drug name written in Devanagari).",
        "tts_note": "TTS clips are macOS `say` voices reading the independent test transcripts written by Claude: clean studio speech, so they flatter ASR compared with real patients.",
        "mimo_v2_6_flash": "not measured: Token Factory lists it as text+image only and its backend rejected input_audio on 2026-10-15",
        "asr": asr,
        "asr_latency_s": {f"{s}/{src}": {"p50": pctl(v, .5), "p95": pctl(v, .95), "n": len(v)} for (s, src), v in lat.items()},
        "grading_from_transcripts": grading,
        "meta": meta,
        "per_clip": per_clip,
    }
    loops = [r for p in (args.loop or []) for r in read_jsonl(p) if "loop_s" in r]
    if loops:
        lm = json.loads(Path(args.loop[0]).with_suffix(".meta.json").read_text())
        out["onprem_loop"] = {
            "what": "one teach-back: E4B transcribes the audio (adapter off), then the fine-tuned E4B grades the transcript (adapter on); one model copy on the W7900D, batch size 1, greedy",
            "clips": len(loops), "loop_p50_s": pctl([r["loop_s"] for r in loops], .5),
            "loop_p95_s": pctl([r["loop_s"] for r in loops], .95),
            "asr_p50_s": pctl([r["e4b_asr_s"] for r in loops], .5),
            "grade_p50_s": pctl([r["grade_from_e4b_hyp_s"] for r in loops], .5),
            "audio_p50_s": pctl([r["audio_s"] for r in loops], .5),
            "peak_vram_gib": lm.get("peak_vram_e4b_asr_plus_grading_gib"),
            "vram_after_load_gib": lm.get("vram_after_load_e4b_plus_adapter_gib")}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    lines = ["# Speech recognition and on-prem loop", "", out["mimo_v2_6_flash"] + ".", out["tts_note"], "",
             "| System | Source | Language | Clips | WER (edits/words) | CER (edits/chars) | WER normalised | CER normalised |", "|---|---|---|---|---|---|---|---|"]
    for s_, d in asr.items():
        for src, langs in d.items():
            for lang, v in langs.items():
                lines.append(f"| {s_} | {src} | {lang} | {v['clips']} | "
                             f"{100 * v['wer']:.1f}% ({v['word_edits']}/{v['words']}) | {100 * v['cer']:.1f}% ({v['char_edits']}/{v['chars']}) | "
                             f"{100 * v['normalised']['wer']:.1f}% | {100 * v['normalised']['cer']:.1f}% |")
    if grading:
        lines += ["", "## Grading from transcripts (fine-tuned E4B)", "",
                  "| Transcript / source | Answers | Facts | Same grade as from reference | Accuracy vs labels | False reassurance | Critical FR |",
                  "|---|---|---|---|---|---|---|"]
        for k, v in grading.items():
            lines.append(f"| {k} | {v['answers']} | {v['facts']} | {v['same_as_ref_grade']}/{v['facts']} | {v['correct_vs_gold']}/{v['facts']} | "
                         f"{v['false_reassurance']}/{v['bad_facts']} | {v['critical_false_reassurance']}/{v['critical_bad_facts']} |")
    if "onprem_loop" in out:
        o = out["onprem_loop"]
        lines += ["", "## On-prem loop (W7900D)", "",
                  f"{o['clips']} clips: end-to-end p50 {o['loop_p50_s']:.1f} s, p95 {o['loop_p95_s']:.1f} s "
                  f"(ASR p50 {o['asr_p50_s']:.1f} s, grading p50 {o['grade_p50_s']:.1f} s, audio p50 {o['audio_p50_s']:.1f} s); "
                  f"peak VRAM {o['peak_vram_gib']:.1f} GiB."]
    Path(args.out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("synth")
    a.add_argument("--rows", required=True)
    a.add_argument("--outdir", required=True)
    a.add_argument("--manifest", required=True)
    a.add_argument("--per-lang", type=int, default=10)
    b = sub.add_parser("human")
    b.add_argument("--refs", default="data/voice/owner_refs.jsonl")
    b.add_argument("--outdir", required=True)
    b.add_argument("--manifest", required=True)
    c = sub.add_parser("gpu")
    c.add_argument("--manifest", required=True)
    c.add_argument("--wavdir", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--adapter")
    c.add_argument("--whisper")
    c.add_argument("--grade", action="store_true", help="time the on-prem loop: grade the E4B transcript with the adapter")
    c.add_argument("--skip-whisper", action="store_true")
    c.add_argument("--out", required=True)
    c.add_argument("--prompt")
    c.add_argument("--limit", type=int, default=0)
    e = sub.add_parser("rows")
    e.add_argument("--manifest", action="append", required=True)
    e.add_argument("--results", action="append", required=True)
    e.add_argument("--outdir", required=True)
    d = sub.add_parser("score")
    d.add_argument("--manifest", action="append", required=True)
    d.add_argument("--results", action="append", required=True)
    d.add_argument("--grades", action="append", help="ref=preds.jsonl, e4b=..., whisper=... from eval.py predict-local")
    d.add_argument("--loop", action="append", help="results file from gpu --grade (on-prem loop timing)")
    d.add_argument("--out", default="ml/results/asr.json")
    args = ap.parse_args()
    {"synth": synth, "human": human, "gpu": gpu, "rows": rows_cmd, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
