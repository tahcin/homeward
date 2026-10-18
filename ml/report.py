import os
"""Assemble ml/results/eval.json + eval.md (and the web app's public/eval.json) from the
per-set score files, the training log, verifier summary, ASR results, capacity runs and
Token Factory prices. Pure reporting: no number is typed in by hand.

Usage: python3 ml/report.py [--web web/public/eval.json] [--ours homeward_run2]
                            [--splits-stats data/splits_v2/stats.json] [--dry-run]
"""

import argparse
import json
import math
import time
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
APP = R.parent.parent


def load(p):
    p = R / p if not str(p).startswith("/") else Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def mcnemar_p(b, c):
    """Exact two-sided binomial test on discordant pairs."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def prices():
    m = load(os.environ.get("TF_MODELS_JSON", str(Path(__file__).parent / "results" / "models.json"))) or {}
    return {x["id"]: {"prompt_per_token": float(x["pricing"]["prompt"]), "completion_per_token": float(x["pricing"]["completion"])}
            for x in m.get("data", []) if "pricing" in x}


RUNS = {
    "homeward_run2": {"label": "run 2", "train": "train_log.json", "cap16": "run2/final_preds_test.meta.json",
                      "cap64": "run2/final_cap_bs64.meta.json", "select": "run2/select_final.json"},
    "homeward_run4": {"label": "run 4", "train": "run4/train_log.json", "cap16": "run4/preds_test.meta.json",
                      "cap64": "run4/cap_bs64.meta.json", "select": "run4/select_final.json",
                      "splits": "splits_v3", "prev": "homeward_run2", "vs": "run4/eval_independent_vs_run2.json",
                      "data_note": "near-miss and weak-language oversampling off to keep the run inside the GPU window",
                      "cand_note": "Run 2's frozen model was a candidate, so run 4 had to beat it on validation. Two planned candidates did not run: "
                                   "the epoch-averaged adapter at scale 1.0 (its model load stalled twice on a ROCm host-to-GPU copy) and scale 0.75 (out of time). "
                                   "The self-consistency candidate (greedy plus 2 samples at T 0.7) tied on accuracy with one more over-flag at 3x the generation."},
    # Run 5 (Sun Oct 18): files written by ml/notebook/run_day3.sh and unpacked by ml/notebook/finish_run5.sh.
    # Capacity comes from a fresh process at batch 64 (cap_bs64); there is no batch-16 run.
    "homeward_run5": {"label": "run 5", "train": "run5/train_log.json", "cap16": "run5/cap_bs16.meta.json",
                      "cap64": "run5/cap_bs64.meta.json", "select": "run5/select_final.json",
                      "splits": "splits_v4", "prev": "homeward_run4", "vs": "run5/eval_independent_vs_run4.json",
                      "data_note": "the same jury-labelled pool and label rule as run 4 without run 4's cap of 45 answers per case; "
                                   "near-miss and weak-language oversampling off",
                      "cand_note": "Run 4's frozen model was the incumbent and listed first, so run 5 had to beat it on validation under the same rule. "
                                   "All run 5 candidates were evaluated in the training process by swapping LoRA weights in place (one model load)."},
}


# How the selection names read in the recipe section.
WEIGHTS_TEXT = {"run4_final": "the final (epoch-2) adapter", "run5_final": "the final adapter", "run5_final_s075": "the final adapter",
                "run5_avg": "the average of the epoch-1 and final adapters", "run5_avg_s075": "the average of the epoch-1 and final adapters",
                "run5_avg_late": "the average of the epoch-1.5 and final adapters", "run5_ep1": "the epoch-1 adapter"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", default=str(APP / "web" / "public" / "eval.json"))
    ap.add_argument("--answers-per-discharge", type=float, default=6.0,
                    help="graded teach-back answers per discharge: 4 questions plus about 2 re-asks")
    ap.add_argument("--gpu-minutes", type=float, default=None)
    ap.add_argument("--ours", default="homeward_run2",
                    help="system key of the fine-tuned grader in eval_independent.json, mapped to the web app's gemma-4-e4b-homeward row")
    ap.add_argument("--splits-stats", default=str(APP / "data" / "splits_v2" / "stats.json"),
                    help="stats.json of the SFT split used for the recipe section (relative paths resolve against the app root)")
    ap.add_argument("--dry-run", action="store_true", help="print eval.md to stdout and write no files")
    a = ap.parse_args()

    ind, ind_d = load("eval_independent.json"), load("eval_in_distribution.json")
    run = RUNS.get(a.ours, RUNS["homeward_run2"])
    train = load(run["train"])
    ver = load("verify_summary.json")
    asr = load("asr.json")
    split = load("split_stats.json")
    tau = {"e4b_lora": load("escalation/tau_ft_dev.json"), "e4b_base": load("escalation/tau_base_dev.json"),
           "e4b_lora_valsplit": load("escalation/tau_ft.json")}
    cap = {}
    for bs in (16, 64):
        m = load(run["cap16"] if bs == 16 else run["cap64"])
        if m:
            cap[bs] = {"answers_per_s": m["batched"]["answers_per_s"], "gen_tok_per_s": m["batched"]["gen_tok_per_s"],
                       "peak_vram_gib": m["peak_vram_allocated_gib"], "answers": m["rows"]}

    for d in (ind, ind_d):
        for k, s in d["systems"].items():
            pv = s["metrics"].get("paired_vs_e4b_base")
            if pv and "mcnemar_exact_p" not in pv:
                pv["mcnemar_exact_p"] = mcnemar_p(pv["this_right_ref_wrong"], pv["this_wrong_ref_right"])

    # Cost per discharge from measured tokens and listed prices.
    pr = prices()
    api_model = {"gemma4_31b": "Gemma-4-31B-IT", "deepseek_v4_flash": "DeepSeek-V4-Flash"}
    cost = {"answers_per_discharge_assumption": a.answers_per_discharge, "per_system": {}}
    for k, s in ind["systems"].items():
        t = s["metrics"]["tokens"]
        if k in api_model and api_model[k] in pr:
            p = pr[api_model[k]]
            per = t["prompt_per_answer"] * p["prompt_per_token"] + t["completion_per_answer"] * p["completion_per_token"]
            cost["per_system"][k] = {"usd_per_answer": per, "usd_per_discharge": per * a.answers_per_discharge,
                                     "basis": "independent test set mean tokens x Token Factory list price"}
    asr_rtf = None
    if asr:
        e4b_lat = [c for c in asr.get("per_clip", []) if c.get("source") == "human"]
    loop = (asr or {}).get("onprem_loop")
    best = cap.get(64) or cap.get(16)
    if best and loop:
        grade_s = 1 / best["answers_per_s"]
        asr_s = loop["asr_p50_s"]
        per_discharge = a.answers_per_discharge * (grade_s + asr_s)
        cost["onprem_w7900d"] = {
            "grading_s_per_answer_batched": grade_s, "asr_s_per_answer_bs1_p50": asr_s,
            "gpu_seconds_per_discharge": per_discharge,
            "discharges_per_hour_per_card": 3600 / per_discharge,
            "grading_only_discharges_per_hour_per_card": 3600 / (a.answers_per_discharge * grade_s),
            "assumptions": f"{a.answers_per_discharge:g} graded answers per discharge; grading batched at {64 if 64 in cap else 16}; ASR unbatched at batch 1 (p50 of the on-prem loop clips); one card; no other load",
        }

    out = {
        "schema": "homeward.eval.v2",
        "ours": {"system_key": a.ours, "label": run["label"]},
        "generated_at": time.strftime("%Y-%m-%d %H:%M %Z"),
        "headline": "independent test set (written by Claude, a different model family, never seen by the generator, held-out conditions)",
        "test_sets": {"independent": ind, "in_distribution": ind_d},
        "training": {k: train[k] for k in ("train_examples", "val_examples", "trainable_params", "total_params", "lora_modules",
                                            "optimizer_steps", "train_wall_s", "mean_tok_per_s", "peak_vram_allocated_gib",
                                            "peak_vram_reserved_gib", "device", "torch", "transformers", "peft", "val")} if train else None,
        "training_args": train.get("args") if train else None,
        "splits": split,
        "label_quality": {"verifier": ver, "note": "blind second labels from Qwen3.8-Flash-Next; training kept only answers where it agreed with every designed label; test keeps designed labels"},
        "escalation": {"rule": "a fact graded correct with p(correct) < tau is escalated to the nurse; tau chosen on selection sets only by minimising 5*FR + over-flags",
                       "chosen": {k: (v or {}).get("chosen_tau") for k, v in tau.items()},
                       "finding": "both selection sets (generator val split and a 60-answer independent dev set written by Claude) had zero false reassurance at every tau, so the rule chose tau = 0 (no escalation); the threshold could not be calibrated and test numbers are unchanged",
                       "sweeps": {k: (v or {}).get("sweep") for k, v in tau.items()}},
        "capacity": cap,
        "cost": cost,
        "asr": {k: asr[k] for k in ("clips", "asr", "asr_latency_s", "grading_from_transcripts", "onprem_loop", "mimo_v2_6_flash", "normalisation", "normalised_score", "tts_note")} if asr else None,
        "gpu_minutes_used": a.gpu_minutes,
    }
    if not a.dry_run:
        (R / "eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # Markdown summary.
    def row(name, m):
        a_, b_ = m["accuracy_95ci"], m["false_reassurance_rate_95ci"]
        return (f"| {name} | {100 * m['accuracy']:.1f}% ({100 * a_[0]:.1f}–{100 * a_[1]:.1f}) | {m['macro_f1']:.3f} | "
                f"{100 * m['false_reassurance_rate']:.1f}% ({100 * b_[0]:.1f}–{100 * b_[1]:.1f}) | "
                f"{m['false_reassurance_critical_n'][0]}/{m['false_reassurance_critical_n'][1]} | {100 * m['over_flag_rate']:.1f}% | {100 * m['json_valid_rate']:.1f}% |")
    L = [f"# Homeward grader: evaluation ({run['label']}, final)", "", f"Generated {out['generated_at']}. All numbers come from the files in ml/results; 95% intervals are case-level bootstrap (1,000 resamples).", ""]
    L += ["## Scoring", "", "Facts whose verdict cannot be read count as wrong. Since 2026-10-16 the parser has a lenient fallback for malformed JSON (a dropped colon or quote): "
          "each fact's id and verdict are read with a regex. It is applied identically to every system and every set; JSON validity is still reported strictly. "
          "It was added after model selection, so it did not influence which model was chosen. On the test sets it changes only DeepSeek-V4-Flash (79.2% to 80.4% independent).", ""]
    for key, title in (("independent", "Independent test set (headline)"), ("in_distribution", "In-distribution test set (generator-labelled)")):
        d = out["test_sets"][key]
        ts = d["test_set"]
        L += [f"## {title}", "", f"{ts['answers']} answers, {ts['facts']} facts, {ts['cases']} cases, {len(ts['languages'])} languages; gold {ts['gold_verdict_counts']}.", "",
              "| System | Fact accuracy (95% CI) | Macro-F1 | False reassurance (95% CI) | Critical facts graded correct when wrong/missing | Over-flag | JSON valid |",
              "|---|---|---|---|---|---|---|"]
        for k, s in d["systems"].items():
            L.append(row(k, s["metrics"]))
        L += [""]
        for k, s in d["systems"].items():
            pv = s["metrics"].get("paired_vs_e4b_base")
            if pv:
                L.append(f"- {k} vs base E4B: right where base is wrong on {pv['this_right_ref_wrong']} facts, wrong where base is right on {pv['this_wrong_ref_right']} (exact McNemar p = {pv['mcnemar_exact_p']:.2g}).")
        L += [""]
    if train:
        L += ["## Training (AMD Radeon PRO W7900D, ROCm)", "",
              f"LoRA r={train['args']['rank']}, alpha={train['args']['alpha']}, lr {train['args']['lr']}, {train.get('epochs_completed', train['args']['epochs'])} epochs, "
              f"{train['train_examples']} training rows, {train['optimizer_steps']} steps; {train['trainable_params']:,} trainable of {train['total_params']:,} params; "
              f"{train['train_wall_s'] / 60:.1f} min wall, {train['mean_tok_per_s']:.0f} tokens/s, peak {train['peak_vram_allocated_gib']:.1f} GiB allocated; "
              f"val loss {train['val'][0]['loss']:.3f} -> {train['val'][-1]['loss']:.3f}.", ""]
    sel = load(run["select"])
    if sel:
        out["selection"] = sel
        c, k = sel["chosen"], sel["cost_rule_pick"]
        L += ["## Model selection (validation only)", "",
              "Declared policy, fixed on validation before any test scoring: a missed critical fact is never acceptable; extra re-checks are. "
              "Among all candidates (adapters, LoRA scales) and escalation thresholds tau on val_independent, require zero critical false reassurance "
              "and total false reassurance no worse than base E4B on the same set; then minimise total false reassurance, then over-flags, then maximise accuracy.", "",
              f"Chosen: {c['name']} (LoRA scale {c['scale']}, tau {c['tau']}): val accuracy {100 * c['accuracy']:.1f}%, false reassurance {c['false_reassurance']}, "
              f"critical {c['critical_false_reassurance']}, over-flags {c['over_flag']}.",
              f"For transparency, the earlier rule (minimise 5 x false reassurance + over-flags) would have picked {k['name']} at tau {k['tau']}: "
              f"accuracy {100 * k['accuracy']:.1f}%, false reassurance {k['false_reassurance']}, critical {k['critical_false_reassurance']}, over-flags {k['over_flag']}.", ""]
    else:
        L += ["## Safety threshold", "", out["escalation"]["finding"] + ".", ""]
    sp = Path(a.splits_stats)
    v2 = load(sp if sp.is_absolute() else APP / sp)
    if train and v2:
        ta = train["args"]
        if a.ours in ("homeward_run4", "homeward_run5"):
            c_ = v2["counts"]
            ch = (sel or {}).get("chosen", {})
            w = WEIGHTS_TEXT.get(ch.get("name"), "the selected adapter")
            tau_ = ch.get("tau", 0.0)
            ep = train.get("epochs_completed", ta["epochs"])
            L += [f"## Run-{run['label'].split()[-1]} recipe", "",
                  f"- Data ({run['splits']}): {v2['unique_kept']} unique teach-back answers ({v2['sft_rows_after_oversampling']} rows; {c_.get('mined_answers_upweighted', 0)} answers where run 2 disagreed with agreed labels x2, "
                  f"{c_.get('lowconf_correct_answers_upweighted', 0)} low-confidence correct answers x1.5; {run['data_note']}). "
                  f"Sources {v2['sources_unique']}. All answers generated and labelled by open models on AMD Token Factory. Labels: per-fact vote of the designed label, Gemma-4-31B-IT, "
                  f"Qwen3.8-Flash-Next or DeepSeek-V4-Flash where present, and Clef (Cloudflare's grading model) as one ordinary vote; strict majority, else the 31B verdict if it is the unique plurality, else the answer is dropped; "
                  f"a vote never turns a 31B incorrect/missing into correct; answers with a final 'correct' fact that Clef gives P(correct) < 0.2 are dropped as likely label errors.",
                  f"- Contamination check: {v2['contamination_check']}; dev, val_independent and test cases excluded from training.",
                  f"- Target order: 'heard' before 'verdict'. LoRA r={ta['rank']}, alpha={ta['alpha']}, lr {ta['lr']}, {ep} epochs, "
                  f"loss weight {ta.get('verdict_weight', 1)} on verdict tokens, unlikelihood {ta.get('ul_lambda', 0)} on P('correct') where gold is incorrect/missing (half for partial).",
                  f"- Inference: {w} at LoRA scale {ch.get('scale')}, greedy decoding; "
                  + (f"a fact graded correct with P(correct) < {tau_} goes to the nurse (tau {tau_}, chosen on validation)." if tau_ else "no escalation (tau 0, chosen on validation)."), ""]
        else:
          L += ["## Run-2 recipe", "",
              f"- Data: {v2['unique_kept']} unique teach-back answers ({v2['sft_rows_after_oversampling']} rows after oversampling near-miss pairs x1.5 and weaker languages x1.25), "
              f"v1 generator answers plus multi-model v2 answers. Labels: kept when Gemma-4-31B-IT agreed with the designed label on every fact (heard/reteach text distilled from 31B), "
              f"otherwise a per-fact 2-of-3 vote of designed, 31B and Qwen3.8-Flash-Next labels; answers without a majority dropped. Counts: {v2['counts']}.",
              f"- Contamination check: {v2['contamination_check']}; dev, val_independent and test cases excluded from training.",
              f"- Target order: 'heard' before 'verdict' (run B). LoRA r={ta['rank']}, alpha={ta['alpha']}, lr {ta['lr']}, {ta['epochs']} epochs, "
              f"loss weight {ta.get('verdict_weight', 1)} on verdict tokens, unlikelihood {ta.get('ul_lambda', 0)} on P('correct') where gold is incorrect/missing (half for partial).",
              "- Inference: average of the epoch-1 and epoch-2 adapters, applied at LoRA scale 0.5 (chosen on validation), greedy decoding, no escalation (tau 0).", ""]
    sc = load("run2/eval_sc_valind.json") if a.ours == "homeward_run2" else None
    if sel and a.ours != "homeward_run2":
        L += ["## Candidates on validation (val_independent)", "", "| Candidate | LoRA scale | tau | Accuracy | False reassurance | Critical | Over-flags |", "|---|---|---|---|---|---|---|"]
        for c_ in sel.get("candidates", []):
            L.append(f"| {c_['name']} | {c_['scale']} | {c_['tau']} | {100 * c_['accuracy']:.1f}% | {c_['false_reassurance']} | {next((w['critical_false_reassurance'] for w in c_.get('sweep', []) if w['tau'] == c_['tau']), 'n/a')} | {c_['over_flag']} |")
        L += ["", run.get("cand_note", ""), ""]
    if sc:
        L += ["## Self-consistency (validation only)", ""]
        for k, s_ in sc["systems"].items():
            m = s_["metrics"]
            L.append(f"- {k}: accuracy {100 * m['accuracy']:.1f}%, false reassurance {m['false_reassurance_n'][0]}/{m['false_reassurance_n'][1]}, over-flag {100 * m['over_flag_rate']:.1f}%")
        L += ["- With the lenient parser (see Scoring), the 3-sample vote matches greedy (91.5%); its only effect is fewer broken outputs. Not worth 3x the generation for run 2.", ""]
    vr2 = load(run["vs"]) if run.get("vs") else None
    if vr2 and a.ours in vr2.get("systems", {}):
        prev = run["prev"]
        pl, ol = RUNS[prev]["label"], run["label"]
        pv = vr2["systems"][a.ours]["metrics"]["paired_vs_" + prev]
        b_, c_ = pv["this_right_ref_wrong"], pv["this_wrong_ref_right"]
        p_ = mcnemar_p(b_, c_)
        mo, mp = vr2["systems"][a.ours]["metrics"], vr2["systems"][prev]["metrics"]
        co, cp = mo["false_reassurance_critical_n"], mp["false_reassurance_critical_n"]
        n_ans = vr2["test_set"]["answers"]
        if p_ is not None and p_ < 0.05:
            tail = f"the accuracy difference is statistically significant; critical misses {co[0]} vs {cp[0]} of {co[1]}."
        elif mo["accuracy"] > mp["accuracy"] and co[0] < cp[0]:
            tail = f"neither the accuracy gain nor the critical-miss difference ({co[0]} vs {cp[0]} of {co[1]}) is statistically significant on {n_ans} answers."
        else:
            tail = f"the accuracy difference is not statistically significant on {n_ans} answers; critical misses {co[0]} vs {cp[0]} of {co[1]}."
        L += [f"## {ol.capitalize()} vs {pl} (independent test, paired)", "",
              f"{ol.capitalize()} is right where {pl} is wrong on {b_} facts and wrong where {pl} is right on {c_} (exact McNemar p = {'n/a' if p_ is None else format(p_, '.2g')}): " + tail, ""]
    v31 = load("run2/eval_valind_with31b.json")
    if v31:
        m = v31["systems"]["gemma4_31b"]["metrics"]
        L += ["## Gemma-4-31B on the validation set", "", f"val_independent: accuracy {100 * m['accuracy']:.1f}%, false reassurance {m['false_reassurance_n'][0]}/{m['false_reassurance_n'][1]}, over-flag {100 * m['over_flag_rate']:.1f}%.", ""]
    s12 = load("base12/eval_val_independent.json")
    if s12:
        m = s12["systems"]["gemma4_12b_base"]["metrics"]
        L += ["## Gemma 4 12B screen (not trained)", "", f"Base 12B on val_independent: {100 * m['accuracy']:.1f}%, critical misses {m['false_reassurance_critical_n'][0]}/{m['false_reassurance_critical_n'][1]}, JSON valid {100 * m['json_valid_rate']:.1f}%; below the bar for training it, and its ASR on the recorded voices was worse (WER 41% vs 20%).", ""]
    if cap:
        L += ["## Capacity on one W7900D (fine-tuned grader)", "", "| Batch | Teach-backs/s | Teach-backs/hour | Gen tokens/s | Peak GiB |", "|---|---|---|---|---|"]
        for bs, c in cap.items():
            L.append(f"| {bs} | {c['answers_per_s']:.2f} | {3600 * c['answers_per_s']:.0f} | {c['gen_tok_per_s']:.0f} | {c['peak_vram_gib']:.1f} |")
        if "onprem_w7900d" in cost:
            o = cost["onprem_w7900d"]
            L += ["", f"Discharges per hour per card, ASR plus grading: {o['discharges_per_hour_per_card']:.0f} ({o['assumptions']}). Grading only: {o['grading_only_discharges_per_hour_per_card']:.0f}.", ""]
    if cost["per_system"]:
        L += ["## Hosted cost per discharge (Token Factory list prices)", ""]
        for k, c in cost["per_system"].items():
            L.append(f"- {k}: ${c['usd_per_discharge']:.5f} per discharge (${c['usd_per_answer']:.6f} per answer)")
        L += [""]
    if loop:
        L += ["## On-prem loop", "", f"{loop['clips']} clips (timed with the run-1 adapter; same model size and code path): E4B ASR then fine-tuned E4B grading on one card, p50 {loop['loop_p50_s']:.1f} s, p95 {loop['loop_p95_s']:.1f} s, peak {loop['peak_vram_gib']:.1f} GiB.", ""]
    if not a.dry_run:
        (R / "eval.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # Web app file: keep its schema, fill the nulls.
    web = Path(a.web)
    if web.exists() and not a.dry_run:
        w = json.loads(web.read_text())
        ts = ind["test_set"]
        w["grader"]["test_set"].update({"description": "Independent teach-back answers written by Claude (a different model family) (never seen by the data generator), held-out conditions",
                                        "cases": ts["cases"], "answers": ts["answers"], "facts_graded": ts["facts"], "languages": len(ts["languages"])})
        ids = {"gemma-4-e4b-base": "e4b_base", "gemma-4-e4b-homeward": a.ours, "gemma-4-31b": "gemma4_31b"}
        for mrow in w["grader"]["models"]:
            s = ind["systems"].get(ids.get(mrow["id"], ""))
            if not s:
                continue
            m, me = s["metrics"], s["meta"]
            crit = m["false_reassurance_critical_n"]
            mrow["accuracy"] = round(100 * m["accuracy"], 1)
            mrow["critical_recall"] = round(100 * (1 - crit[0] / crit[1]), 1)
            mrow["false_reassurance"] = round(100 * m["false_reassurance_rate"], 1)
            lat = (me.get("bs1") or {}).get("p50_s") if "bs1" in me else None
            mrow["latency_p50_ms"] = round(1000 * lat) if lat else None
            mrow["ci95"] = {"accuracy": [round(100 * x, 1) for x in m["accuracy_95ci"]],
                            "false_reassurance": [round(100 * x, 1) for x in m["false_reassurance_rate_95ci"]]}
        if train:
            w["training"].update({"examples": train["train_examples"], "epochs": train["args"]["epochs"],
                                  "minutes": round(train["train_wall_s"] / 60, 1)})
        w["status"] = "final"
        w["updated"] = time.strftime("%Y-%m-%d")
        web.write_text(json.dumps(w, indent=2, ensure_ascii=False) + "\n")
        print("wrote", web)


if __name__ == "__main__":
    main()
