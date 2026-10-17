"""Fill the Hugging Face model card (ml/hf/README.md) from the result files, for one system.

Every run-dependent number in README.md sits between markers and is rewritten by this script:
  inline  <!--v:NAME-->value<!--/v-->
  block   <!--b:NAME-->  ...  <!--/b:NAME-->
The YAML front matter is regenerated as a whole. Prose outside markers is kept as written, so
edit wording in README.md directly, but never type a number inside a marker by hand.

Sources (nothing typed in by hand):
  ml/results/eval.json                 test metrics, system meta (adapter path, LoRA scale), speech
  ml/results/**/*train_log*.json       the training log whose args.out matches the adapter's run dir
  ml/results/**/select*.json           the validation selection that chose this adapter (optional)
  data/splits_v*/stats.json            training-set counts (the split named in the train log's args)
  ../model-sizes.md                    verified model sizes (each size below is checked against it)

Also writes ml/hf/homeward_inference.json (LoRA scale, decoding, attention) for the upload.

Usage (from app/):
  python3 ml/hf/build_card.py --system homeward_run2
  python3 ml/hf/build_card.py --system homeward_run4 --license apache-2.0
  python3 ml/hf/build_card.py --system homeward_run4 --check     # print values, write nothing
"""

import argparse
import json
import re
import sys
from pathlib import Path

HF = Path(__file__).resolve().parent
ML = HF.parent
R = ML / "results"
APP = ML.parent
REPO_ID = "oktash/homeward-teachback-gemma4-e4b"
GITHUB = "https://github.com/tahcin/homeward"
LIVE = "https://homeward.tahcin.tech"
LANGS = {"ar": "Arabic", "bn": "Bengali", "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
         "it": "Italian", "pt": "Portuguese", "ta": "Tamil", "tl": "Tagalog", "vi": "Vietnamese", "zh": "Chinese"}

# Comparators: display name, model-sizes.md row (first column), short size, total params in B (for ordering),
# and the substrings that must appear in that model-sizes.md row (so a stale size fails loudly).
SYSTEMS = {
    "e4b_base": ("Gemma 4 E4B, base (no adapter)", "Gemma 4 E4B", "8B total, 4.5B effective", 8, ["8B", "4.5B"]),
    "gemma4_31b": ("Gemma 4 31B (teacher)", "Gemma-4-31B-IT", "30.7B dense", 30.7, ["30.7B"]),
    "deepseek_v4_flash": ("DeepSeek V4 Flash", "DeepSeek-V4-Flash", "284B total, 13B active", 284, ["284B", "13B"]),
    "qwen3_8_flash_next": ("Qwen3.8 Flash Next", "Qwen3.8-Flash-Next", "180B total, 6B active", 180, ["180B", "6B"]),
    "glm_5_3_flash": ("GLM-5.3 Flash", "GLM-5.3-Flash", "320B total, 18B active", 320, ["320B", "18B"]),
    "mimo_v2_6_flash": ("MiMo-V2.6 Flash", "MiMo-V2.6-Flash", "309B total, 15B active", 309, ["309B", "15B"]),
    "qwen3_8_27b": ("Qwen3.8 27B", "Qwen3.8-27B", "27B dense", 27, ["27B"]),
    "minicpm_v_4_7_35b_a3b": ("MiniCPM-V 4.7 35B-A3B", "MiniCPM-V-4.7-35B-A3B", "35B total, about 3B active", 35, ["35B", "3B"]),
    "clef": ("Cloudflare Clef", "Clef", "27B dense", 27, ["27B"]),
}
OURS_SIZE = ("8B total, 4.5B effective, plus {lora}M LoRA parameters", 8)
TEACHER_KEY, BASE_KEY = "gemma4_31b", "e4b_base"


def die(msg):
    sys.exit(f"build_card: {msg}")


def load(p):
    return json.loads(Path(p).read_text())


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def ci(lo_hi):
    return f"{100 * lo_hi[0]:.1f}–{100 * lo_hi[1]:.1f}"


def num(n):
    return f"{n:,}"


def pval(p):
    m, e = f"{p:.1e}".split("e")
    return f"{m}e{int(e)}" if int(e) < -3 else f"{p:.2g}"


def size_rows(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("| ") and not line.startswith("| :") and not line.startswith("| Token Factory"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows[cells[0]] = line
    return rows


def find_train_log(adapter, explicit):
    cands = [Path(explicit)] if explicit else [R / "train_log.json"] + sorted(R.glob("**/*train_log*.json"))
    run_dir = str(Path(adapter).parent)
    for c in cands:
        if c.exists():
            t = load(c)
            if str(Path(t.get("args", {}).get("out", "/none"))) == run_dir:
                return c, t
    die(f"no training log with args.out == {run_dir}; copy the run's train_log.json into ml/results/ or pass --train-log")


def find_selection(adapter, scale, explicit):
    cands = [Path(explicit)] if explicit else sorted(R.glob("**/select*.json"))
    for c in cands:
        try:
            s = load(c)
        except (json.JSONDecodeError, OSError):
            continue
        ch = s.get("chosen") if isinstance(s, dict) else None
        if ch and ch.get("adapter") == adapter and abs(float(ch.get("scale", -1)) - scale) < 1e-9:
            return c, s
    return None, None


def averaging_text(adapter, epochs):
    b = Path(adapter).name
    if b == "adapter_avg":
        return (f"the published weights are the equal-weight average of the adapters saved at the end of each of the {epochs} epochs"
                if epochs != "1" else "none (one epoch)")
    if b == "adapter_avg_last2":
        return "the published weights are the equal-weight average of the adapters saved at the end of the last two epochs"
    if b == "adapter":
        return "none: the published weights are the adapter at the end of training"
    return f"the published weights are `{b}` (see the training log)"


def rel(p):
    try:
        return Path(p).resolve().relative_to(APP).as_posix()
    except ValueError:
        return str(p)


def join_and(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, help="eval.json system key of the published adapter, e.g. homeward_run2")
    ap.add_argument("--eval", default=str(R / "eval.json"))
    ap.add_argument("--sizes", default=str(APP.parent / "model-sizes.md"))
    ap.add_argument("--train-log", default=None)
    ap.add_argument("--select", default=None)
    ap.add_argument("--license", default="gemma", choices=["gemma", "apache-2.0"],
                    help="must match the license field on huggingface.co/google/gemma-4-E4B-it")
    ap.add_argument("--card", default=str(HF / "README.md"))
    ap.add_argument("--check", action="store_true", help="print the values and write nothing")
    a = ap.parse_args()

    ev = load(a.eval)
    ind = ev["test_sets"]["independent"]
    systems = ind["systems"]
    if a.system not in systems:
        die(f"{a.system} not in eval.json independent systems: {list(systems)}")
    me, m = systems[a.system]["meta"], systems[a.system]["metrics"]
    adapter, scale = me.get("adapter"), float(me.get("lora_scale", 1.0))
    if not adapter:
        die(f"{a.system} has no adapter in its meta")
    tlog_path, tl = find_train_log(adapter, a.train_log)
    ta = tl["args"]
    sel_path, sel = find_selection(adapter, scale, a.select)
    chosen = (sel or {}).get("chosen", {})
    sc_vote = str(chosen.get("name", "")).endswith("_sc")
    tau = float(chosen.get("tau", (me.get("escalation") or {}).get("tau", 0.0)))
    split_dir = re.search(r"/data/(v\d+)/", ta["train"])
    split_name = {"v2": "splits_v2", "v3": "splits_v3"}.get(split_dir.group(1) if split_dir else "", None)
    stats = load(APP / "data" / split_name / "stats.json") if split_name else None
    # The local split must be the one the run trained on (rows may differ only by examples dropped for length).
    stats_ok = bool(stats) and 0 <= stats["sft_rows_after_oversampling"] - tl["train_examples"] <= tl.get("dropped_over_max_len", 0)
    if stats and not stats_ok:
        print(f"warning: data/{split_name}/stats.json has {stats['sft_rows_after_oversampling']} rows but the run trained on "
              f"{tl['train_examples']}; the local split was rebuilt after upload, so its counts are left out of the card", file=sys.stderr)
    run_m = re.search(r"_run(\d+)$", a.system)
    run_label = f"run {run_m.group(1)}" if run_m else a.system

    sizes = size_rows(a.sizes)
    for k, (_, tf, _, _, must) in SYSTEMS.items():
        if k in systems:
            row = sizes.get(tf) or die(f"model-sizes.md has no row named {tf!r}")
            for s in must:
                if s not in row:
                    die(f"model-sizes.md row {tf!r} no longer contains {s!r}; update SYSTEMS in build_card.py")

    base = systems.get(BASE_KEY, {}).get("metrics")
    teach = systems.get(TEACHER_KEY, {}).get("metrics")
    pv = m.get("paired_vs_e4b_base") or {}
    crit = m["false_reassurance_critical_n"]
    lora_m = tl["trainable_params"] / 1e6
    test = ind["test_set"]
    langs = sorted(test["languages"])

    V = {
        "run": run_label, "system": f"`{a.system}`", "reqs": f"`transformers=={tl['transformers']}`, `peft=={tl['peft']}`",
        "acc": pct(m["accuracy"]), "acc_ci": ci(m["accuracy_95ci"]), "f1": f"{m['macro_f1']:.3f}",
        "fr": pct(m["false_reassurance_rate"]), "fr_ci": ci(m["false_reassurance_rate_95ci"]),
        "fr_n": f"{m['false_reassurance_n'][0]} of {m['false_reassurance_n'][1]}",
        "crit": f"{crit[0]} of {crit[1]}", "over": pct(m["over_flag_rate"]), "json": pct(m["json_valid_rate"]),
        "answers": str(test["answers"]), "facts": str(test["facts"]), "cases": str(test["cases"]), "nlang": str(len(langs)),
        "base_acc": pct(base["accuracy"]) if base else "n/a",
        "base_crit": f"{base['false_reassurance_critical_n'][0]} of {base['false_reassurance_critical_n'][1]}" if base else "n/a",
        "fixes": str(pv.get("this_right_ref_wrong", "n/a")), "breaks": str(pv.get("this_wrong_ref_right", "n/a")),
        "p": pval(pv["mcnemar_exact_p"]) if pv.get("mcnemar_exact_p") is not None else "n/a",
        "teacher_acc": pct(teach["accuracy"]) if teach else "n/a",
        "teacher_crit": f"{teach['false_reassurance_critical_n'][0]} of {teach['false_reassurance_critical_n'][1]}" if teach else "n/a",
        "keep": pct(m["accuracy"] / teach["accuracy"]) if teach else "n/a",
        "scale": f"{scale:g}", "tau": f"{tau:g}",
        "rank": str(ta["rank"]), "alpha": str(ta["alpha"]), "dropout": f"{ta['dropout']:g}", "lr": f"{ta['lr']:g}",
        "epochs": f"{ta['epochs']:g}", "bsz": str(ta["bsz"]), "accum": str(ta["accum"]), "eff_bsz": str(ta["bsz"] * ta["accum"]),
        "max_len": str(ta["max_len"]), "warmup": f"{ta['warmup']:g}", "seed": str(ta["seed"]),
        "vw": f"{ta.get('verdict_weight', 1.0):g}", "ul": f"{ta.get('ul_lambda', 0.0):g}",
        "ckpt": "off" if ta.get("no_ckpt") else "on",
        "rows": num(tl["train_examples"]), "steps": num(tl["optimizer_steps"]),
        "wall": f"{tl['train_wall_s'] / 60:.1f}", "tokps": num(round(tl["mean_tok_per_s"])),
        "peak": f"{tl['peak_vram_allocated_gib']:.1f}", "trainable": num(tl["trainable_params"]),
        "total_params": num(tl["total_params"]), "lora_mods": str(tl["lora_modules"]), "lora_m": f"{lora_m:.1f}",
        "torch": tl["torch"], "transformers": tl["transformers"], "peft": tl["peft"],
        "val0": f"{tl['val'][0]['loss']:.3f}", "val1": f"{tl['val'][-1]['loss']:.3f}",
        "avg": averaging_text(adapter, f"{ta['epochs']:g}"),
        "bs1": f"{me['bs1']['p50_s']:.1f} s" if me.get("bs1") else "n/a",
        "aps": f"{me['batched']['answers_per_s']:.2f}" if me.get("batched") else "n/a",
        "bsz_inf": str(me["batched"]["batch_size"]) if me.get("batched") else "n/a",
        "vram_inf": f"{me['peak_vram_allocated_gib']:.1f}" if me.get("peak_vram_allocated_gib") else "n/a",
        "decoding": ("greedy plus two samples at temperature 0.7, per-fact majority vote (ties go to the less reassuring verdict)"
                     if sc_vote else "greedy decoding, one pass"),
    }
    if a.license == "gemma":
        V["license_text"] = ("This adapter is a derivative of Gemma 4 and is distributed under the Gemma Terms of Use. "
                             "Gemma is provided under and subject to the Gemma Terms of Use found at "
                             "[ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms). Use of this adapter must also follow the "
                             "[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).")
    else:
        V["license_text"] = ("This adapter is released under the Apache 2.0 licence, the same licence as the base model "
                             "[google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it). Check the base model card for its terms.")

    # Speech (base E4B with the adapter off; not run-dependent, but still read from the file).
    asr = (ev.get("asr") or {}).get("asr", {})
    eh, wh = asr.get("e4b", {}).get("human", {}), asr.get("whisper", {}).get("human", {})
    if eh.get("ALL") and wh.get("ALL"):
        V.update({"wer_e4b": pct(eh["ALL"]["wer"]), "wer_whisper": pct(wh["ALL"]["wer"]),
                  "wer_bn_e4b": pct(eh["bn"]["wer"]), "wer_bn_whisper": pct(wh["bn"]["wer"]),
                  "asr_clips": str(eh["ALL"]["clips"]), "asr_words": str(eh["ALL"]["words"])})

    if scale != 1.0:
        V["scale_matters"] = " The scale matters: see [How to load](#how-to-load)."
        B = {"scale_limit": f"- Scored at LoRA scale {V['scale']}. Loading the adapter at the default scale of 1.0 gives a different model that this card does not describe.",
             "load_intro": (f"Apply the LoRA scale ({V['scale']}) before merging: the adapter's own `adapter_config.json` is the training config, "
                            "which PEFT loads at scale 1.0. The scale is also stored in `homeward_inference.json`.")}
    else:
        V["scale_matters"] = ""
        B = {"scale_limit": "", "load_intro": "The adapter is used at its trained scale (1.0); `homeward_inference.json` records the inference settings."}
    # Headline bullets.
    comps = [(k, systems[k]["metrics"]) for k in SYSTEMS if k in systems and k not in (BASE_KEY, TEACHER_KEY)]
    beaten = [k for k, cm in comps if cm["accuracy"] < m["accuracy"]]
    not_beaten = [k for k, cm in comps if cm["accuracy"] >= m["accuracy"]]
    def overlap(cm):
        return not (cm["accuracy_95ci"][1] < m["accuracy_95ci"][0] or cm["accuracy_95ci"][0] > m["accuracy_95ci"][1])
    lines = [f"- Per-fact accuracy {V['acc']} (95% CI {V['acc_ci']}) on an independent test set of {V['answers']} answers and "
             f"{V['facts']} facts in {V['nlang']} languages. Critical facts the patient got wrong or left out but that were graded "
             f"\"understood\": {V['crit']}."]
    if base:
        lines.append(f"- Against the same model without the adapter: {V['base_acc']} accuracy and {V['base_crit']} critical misses. "
                     f"The adapter fixes {V['fixes']} fact grades and breaks {V['breaks']} (exact McNemar p = {V['p']}).")
    if teach:
        verb = "matches or beats" if m["accuracy"] >= teach["accuracy"] else f"keeps {V['keep']} of the accuracy of"
        txt = (f"- It {verb} its teacher, Gemma 4 31B ({V['teacher_acc']}, {V['teacher_crit']} critical misses), "
               f"at about a quarter of the size (8B against 30.7B parameters).")
        if m["accuracy"] < teach["accuracy"]:
            ours_c, teach_c = m["false_reassurance_critical_n"][0], teach["false_reassurance_critical_n"][0]
            rel_c = "ties it on" if ours_c == teach_c else ("has fewer" if ours_c < teach_c else "has more")
            txt += f" It {rel_c} critical misses but is below it on accuracy."
        lines.append(txt)
    if beaten:
        big = max(beaten, key=lambda k: SYSTEMS[k][3])
        names = join_and([f"{SYSTEMS[k][0]} ({SYSTEMS[k][2]}, {pct(systems[k]['metrics']['accuracy'])})" for k in sorted(beaten, key=lambda k: -SYSTEMS[k][3])])
        ov = [SYSTEMS[k][0] for k in beaten if overlap(systems[k]["metrics"])]
        txt = f"- Higher point accuracy than {names}. The largest of these is {SYSTEMS[big][0]} ({SYSTEMS[big][2]})."
        if ov:
            txt += f" The 95% intervals overlap for {join_and(ov)}, so {'that gap is' if len(ov) == 1 else 'those gaps are'} not conclusive."
        lines.append(txt)
    if not_beaten:
        lines.append("- Lower than or equal to: " + ", ".join(f"{SYSTEMS[k][0]} ({pct(systems[k]['metrics']['accuracy'])})" for k in not_beaten) + ".")
    B["headline"] = "\n".join(lines)

    # Comparison table, sorted by accuracy.
    rows = [(a.system, m, f"**Homeward adapter ({run_label})**", OURS_SIZE[0].format(lora=V["lora_m"]), "AMD Radeon PRO W7900D")]
    for k in SYSTEMS:
        if k in systems:
            where = "AMD Radeon PRO W7900D" if k == BASE_KEY else "AMD Token Factory"
            rows.append((k, systems[k]["metrics"], SYSTEMS[k][0], SYSTEMS[k][2], where))
    skipped = [k for k, s in systems.items() if k not in SYSTEMS and k != a.system]
    rows.sort(key=lambda r: -r[1]["accuracy"])
    T = ["| System | Size | Ran on | Accuracy (95% CI) | Macro-F1 | Critical misses passed | False reassurance, all facts | Over-flag | Valid JSON |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, cm, name, size, where in rows:
        c = cm["false_reassurance_critical_n"]
        cell = lambda s: f"**{s}**" if k == a.system else s
        T.append(f"| {name} | {size} | {where} | {cell(pct(cm['accuracy']) + ' (' + ci(cm['accuracy_95ci']) + ')')} | {cm['macro_f1']:.3f} | "
                 f"{cell(f'{c[0]} of {c[1]}')} | {pct(cm['false_reassurance_rate'])} | {pct(cm['over_flag_rate'])} | {pct(cm['json_valid_rate'])} |")
    B["results_table"] = "\n".join(T)
    if skipped:
        print(f"note: not in the comparison table (other adapters or unknown keys): {skipped}", file=sys.stderr)

    # Per-language accuracy.
    pl = m.get("per_language_accuracy", {})
    B["per_language"] = "\n".join(["| Language | Facts | Accuracy |", "|---|---|---|"] +
                                  [f"| {LANGS.get(l, l)} | {pl[l]['facts']} | {pct(pl[l]['accuracy'])} |" for l in sorted(pl, key=lambda l: LANGS.get(l, l))])

    # Confusion matrix.
    cf = m["confusion"]
    vs = ["correct", "partial", "incorrect", "missing"]
    rank = {"correct": 3, "partial": 2, "incorrect": 1, "missing": 1}
    less = sum(cf[g][p] for g in vs for p in vs if rank[p] < rank[g])
    more = sum(cf[g][p] for g in vs for p in vs if rank[p] > rank[g])
    other = sum(cf[g][p] for g in vs for p in vs if p != g and rank[p] == rank[g]) + sum(cf[g].get("invalid", 0) for g in vs)
    to_correct = sum(cf[g]["correct"] for g in vs if g != "correct")
    B["error_direction"] = (f"- Of its {less + more + other} wrong fact grades, {more} are more reassuring than the gold verdict and {less} less reassuring"
                            + (f" ({other} confuse incorrect with missing or are unreadable)" if other else "")
                            + f". {to_correct} of the more reassuring ones reach \"correct\"; the rest stop at \"partial\" (for example "
                            "\"incorrect\" graded \"partial\"), which in Homeward still triggers a re-teach but understates the error to the nurse.")
    B["confusion"] = "\n".join(["| Gold \\ predicted | " + " | ".join(vs) + " | unreadable |", "|---" * 6 + "|"] +
                               [f"| {g} | " + " | ".join(str(cf[g][p]) for p in vs) + f" | {cf[g].get('invalid', 0)} |" for g in vs])

    # Training data.
    D = []
    if stats and stats_ok:
        c = stats["counts"]
        D.append(f"- {num(stats['unique_kept'])} unique teach-back answers ({V['rows']} training rows after oversampling) from fictional discharge cases "
                 f"in {len(stats['languages_unique'])} languages. Unique answers by source batch: "
                 + ", ".join(f"{k} {num(v)}" for k, v in stats["sources_unique"].items()) + ". Verdicts: "
                 + ", ".join(f"{k} {num(v)}" for k, v in stats["verdicts_unique"].items()) + " facts.")
        if split_name == "splits_v2":
            D.append("- Labels: each answer was written to a designed per-fact plan. Gemma-4-31B-IT re-graded every answer. When it agreed with the "
                     "design on every fact, the answer was kept with the 31B's wording for the `heard` and `reteach` fields. Otherwise each fact went "
                     "to a 2-of-3 vote of the designed label, the 31B and Qwen3.8-Flash-Next; answers with any fact lacking a majority were dropped. "
                     f"Kept on full teacher agreement: {num(c['kept_teacher_agree_v1'] + c['kept_teacher_agree_v2'])}; kept by vote: "
                     f"{num(c['kept_majority_vote_v1'] + c['kept_majority_vote_v2'])} ({num(c['facts_relabelled_by_vote'])} facts relabelled); dropped without "
                     f"a majority: {num(c['dropped_no_majority_v1'] + c['dropped_no_majority_v2'])}; dropped by a per-case cap: {num(c['dropped_case_cap'])}.")
            D.append("- Oversampling: near-miss answer pairs x1.5 and weaker languages x1.25.")
        elif split_name == "splits_v3":
            kept = sum(v for k, v in c.items() if k.startswith("kept_clefjury_"))
            nomaj = sum(v for k, v in c.items() if k.startswith("dropped_clefjury_no_majority_"))
            lerr = sum(v for k, v in c.items() if k.startswith("dropped_clef_label_error_"))
            D.append("- Labels: each answer was written to a designed per-fact plan. Every fact then went to a vote of the designed label, "
                     "Gemma-4-31B-IT (the teacher), Qwen3.8-Flash-Next and DeepSeek-V4-Flash where they had labelled it, and Cloudflare Clef "
                     "(a decision model post-trained from Qwen3.8-27B, served on Token Factory). A strict majority wins; otherwise the 31B's "
                     "verdict if it is the unique plurality; otherwise the answer is dropped. A vote never turns a 31B \"incorrect\" or "
                     "\"missing\" into \"correct\". Answers with a \"correct\" fact that Clef gives P(correct) < 0.2 are dropped as likely label errors. "
                     f"Kept by the vote: {num(kept)}; dropped without a majority: {num(nomaj)}; dropped as label errors: {num(lerr)}; "
                     f"dropped by a per-case cap: {num(c['dropped_case_cap'])}.")
            D.append(f"- Oversampling: {c.get('mined_answers_upweighted', 0)} hard answers x2 (training-side answers where the previous adapter "
                     f"disagreed with agreed labels) and {c.get('lowconf_correct_answers_upweighted', 0)} low-confidence correct answers x1.5.")
        D.append(f"- Contamination check: {stats['contamination_check']}.")
    else:
        D.append(f"- {V['rows']} training rows (see `{rel(tlog_path)}` and the split statistics in the GitHub repo).")
    B["data"] = "\n".join(D)
    if split_name == "splits_v3":
        B["data_intro"] = ("All training data is synthetic and fictional. Cases and patient answers were written only by open models served on "
                           "AMD Token Factory; the labels come from those open models plus Cloudflare Clef, also served on Token Factory, as one "
                           "extra vote. Clef is post-trained from Qwen3.8-27B; we could not confirm that its own weights are open. "
                           "No Claude output and no real patient data was used for training.")
    else:
        B["data_intro"] = ("All training data is synthetic and fictional: discharge cases, patient answers and labels were generated and labelled "
                           "only by open models served on AMD Token Factory. No Claude output and no real patient data was used for training.")

    # Selection.
    S = []
    if sel:
        S.append(f"- Declared policy, fixed before any test scoring: {sel.get('policy', 'see the repo')}.")
        names = [c_.get("name") for c_ in sel.get("candidates", [])]
        S.append(f"- {len(set(names))} candidate settings (adapters, LoRA scales, escalation thresholds) were compared on a validation set "
                 f"written by Claude (`val_independent`, cases separate from the test set). Chosen: `{chosen.get('name')}` at LoRA scale "
                 f"{V['scale']}, escalation threshold {V['tau']}.")
    else:
        S.append(f"- Chosen on a Claude-written validation set (`val_independent`) with a safety-first rule; LoRA scale {V['scale']}.")
    S.append(f"- The frozen choice was then scored once on the independent test set.")
    B["selection"] = "\n".join(S)

    # Load snippet.
    B["load_snippet"] = LOAD_SNIPPET.replace("@SCALE@", V["scale"]).replace("@REPO@", REPO_ID).replace(
        "@DECODE@", "do_sample=False" if not sc_vote else "do_sample=False  # the evaluated setting also votes over 2 samples at T 0.7, see the card")

    SYSTEM_RUN = lambda k: "run " + k.rsplit("run", 1)[1]
    # In-distribution (optional).
    idd = ev["test_sets"].get("in_distribution", {}).get("systems", {})
    if a.system in idd:
        im = idd[a.system]["metrics"]
        t31 = idd.get(TEACHER_KEY, {}).get("metrics")
        B["in_dist"] = (f"On a second, generator-labelled test split ({ev['test_sets']['in_distribution']['test_set']['answers']} answers from "
                        f"held-out cases, labels written by the same models that wrote the training data, so noisier) the adapter scores "
                        f"{pct(im['accuracy'])} ({ci(im['accuracy_95ci'])})" +
                        (f" and the 31B teacher {pct(t31['accuracy'])}" if t31 else "") +
                        "".join(f"; the {SYSTEM_RUN(k)} adapter scores {pct(idd[k]['metrics']['accuracy'])}" for k in idd
                                if k.startswith("homeward_run") and k != a.system) + "." +
                        " The independent set above is the headline because its labels do not come from the training-data pipeline.")
    else:
        B["in_dist"] = "The adapter was not scored on the generator-labelled split; the independent set above is the headline."

    fm = FRONT.replace("@LICENSE@", a.license + ("\nlicense_link: https://ai.google.dev/gemma/terms" if a.license == "gemma" else ""))
    fm = fm.replace("@LANGS@", "\n".join(f"- {l}" for l in langs))

    inference = {"base_model": "google/gemma-4-E4B-it", "adapter_repo": REPO_ID, "system_key": a.system, "run": run_label,
                 "source_adapter_path": adapter, "lora_scale": scale,
                 "lora_scale_note": "multiply every LoRA module's scaling by this factor before merging (peft: module.scaling['default'] *= lora_scale)",
                 "decoding": V["decoding"], "max_new_tokens": me.get("max_new_tokens", 768), "escalation_tau": tau,
                 "chat_template_kwargs": {"enable_thinking": False},
                 "attn_implementation_gfx1100": {"text_config": "eager", "audio_config": "sdpa", "vision_config": "eager"},
                 "prompt_file": "grader_prompt.json", "train_log": rel(tlog_path),
                 "selection_file": rel(sel_path) if sel_path else None}

    if a.check:
        for k, v in V.items():
            print(f"{k}: {v}")
        print(json.dumps(inference, indent=1))
        return

    card = Path(a.card).read_text()
    card = re.sub(r"\A---\n.*?\n---\n", lambda _: fm, card, count=1, flags=re.S)
    used = set()
    def sub_v(mo):
        k = mo.group(1)
        if k not in V:
            die(f"card uses unknown inline value {k!r}")
        used.add(k)
        return f"<!--v:{k}-->{V[k]}<!--/v-->"
    def sub_b(mo):
        k = mo.group(1)
        if k not in B:
            die(f"card uses unknown block {k!r}")
        return f"<!--b:{k}-->\n{B[k]}\n<!--/b:{k}-->"
    card = re.sub(r"<!--v:(\w+)-->.*?<!--/v-->", sub_v, card, flags=re.S)
    card = re.sub(r"<!--b:(\w+)-->.*?<!--/b:\1-->", sub_b, card, flags=re.S)
    if "\u2014" in card:
        die("em dash in the card; rewrite it")
    Path(a.card).write_text(card)
    (Path(a.card).parent / "homeward_inference.json").write_text(json.dumps(inference, indent=1) + "\n")
    print(f"wrote {a.card} for {a.system} ({run_label}): accuracy {V['acc']} ({V['acc_ci']}), critical {V['crit']}, LoRA scale {V['scale']}; "
          f"train log {rel(tlog_path)}; selection {rel(sel_path) if sel_path else 'not found'}; license {a.license}")


FRONT = """---
base_model: google/gemma-4-E4B-it
library_name: peft
license: @LICENSE@
pipeline_tag: text-generation
language:
@LANGS@
tags:
- lora
- peft
- gemma4
- healthcare
- medical
- patient-education
- teach-back
- multilingual
- llm-as-judge
- rocm
- amd
---
"""

LOAD_SNIPPET = '''```python
import json
import torch
import transformers
from huggingface_hub import hf_hub_download
from peft import PeftModel

BASE = "google/gemma-4-E4B-it"
ADAPTER = "@REPO@"
LORA_SCALE = @SCALE@  # the evaluated setting; also in homeward_inference.json

tok = transformers.AutoProcessor.from_pretrained(BASE).tokenizer
cls = getattr(transformers, "AutoModelForMultimodalLM", None) or transformers.AutoModelForImageTextToText
# On AMD gfx1100 (Radeon PRO W7900 / W7900D, RX 7900) SDPA in the text model gave NaN logits in bf16,
# so the text model uses eager attention; the audio tower needs SDPA. Elsewhere the default is fine.
attn = {"text_config": "eager", "audio_config": "sdpa", "vision_config": "eager"}
model = cls.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto", attn_implementation=attn)

model = PeftModel.from_pretrained(model, ADAPTER)
for module in model.modules():  # apply the LoRA scale before merging
    if hasattr(module, "scaling") and isinstance(module.scaling, dict) and "default" in module.scaling:
        module.scaling["default"] *= LORA_SCALE
model = model.merge_and_unload().eval()

prompt = json.load(open(hf_hub_download(ADAPTER, "grader_prompt.json")))
facts = [  # one line per fact: id, category, critical flag, statement (fictional example)
    {"id": "f1", "category": "medication", "critical": True, "statement": "Take furosemide 40 mg once every morning."},
    {"id": "f2", "category": "warning sign", "critical": True, "statement": "Call the clinic if your weight goes up by 2 kg in 3 days."},
]
fact_lines = "\\n".join(f"- {f['id']} [{f['category']}{', critical' if f['critical'] else ''}]: {f['statement']}" for f in facts)
user = prompt["user"].format(facts=fact_lines, question="How will you take your water tablet, and when do you call us?",
                             language_name="Spanish",
                             transcript="la pastilla del agua la tomo por la mañana, una, y si me hincho llamo")
messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(text, add_special_tokens=False, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=768, @DECODE@)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# -> {"facts": [{"id": "f1", "verdict": "...", "heard": "...", "reteach": "..."}, ...]}
```'''


if __name__ == "__main__":
    main()
