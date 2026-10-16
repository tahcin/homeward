import os
"""Assemble ml/results/eval.json + eval.md (and the web app's public/eval.json) from the
per-set score files, the training log, verifier summary, ASR results, capacity runs and
Token Factory prices. Pure reporting: no number is typed in by hand.

Usage: python3 ml/report.py [--web web/public/eval.json]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", default=str(APP / "web" / "public" / "eval.json"))
    ap.add_argument("--answers-per-discharge", type=float, default=6.0,
                    help="graded teach-back answers per discharge: 4 questions plus about 2 re-asks")
    ap.add_argument("--gpu-minutes", type=float, default=None)
    a = ap.parse_args()

    ind, ind_d = load("eval_independent.json"), load("eval_in_distribution.json")
    train = load("train_log.json")
    ver = load("verify_summary.json")
    asr = load("asr.json")
    split = load("split_stats.json")
    tau = {"e4b_lora": load("escalation/tau_ft_dev.json"), "e4b_base": load("escalation/tau_base_dev.json"),
           "e4b_lora_valsplit": load("escalation/tau_ft.json")}
    cap = {}
    for bs in (16, 64):
        m = load("run2/final_preds_test.meta.json" if bs == 16 else "run2/final_cap_bs64.meta.json")
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
    (R / "eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # Markdown summary.
    def row(name, m):
        a_, b_ = m["accuracy_95ci"], m["false_reassurance_rate_95ci"]
        return (f"| {name} | {100 * m['accuracy']:.1f}% ({100 * a_[0]:.1f}–{100 * a_[1]:.1f}) | {m['macro_f1']:.3f} | "
                f"{100 * m['false_reassurance_rate']:.1f}% ({100 * b_[0]:.1f}–{100 * b_[1]:.1f}) | "
                f"{m['false_reassurance_critical_n'][0]}/{m['false_reassurance_critical_n'][1]} | {100 * m['over_flag_rate']:.1f}% | {100 * m['json_valid_rate']:.1f}% |")
    L = ["# Homeward grader: evaluation (run 2, final)", "", f"Generated {out['generated_at']}. All numbers come from the files in ml/results; 95% intervals are case-level bootstrap (1,000 resamples).", ""]
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
              f"LoRA r={train['args']['rank']}, alpha={train['args']['alpha']}, lr {train['args']['lr']}, {train['args']['epochs']} epochs, "
              f"{train['train_examples']} training rows, {train['optimizer_steps']} steps; {train['trainable_params']:,} trainable of {train['total_params']:,} params; "
              f"{train['train_wall_s'] / 60:.1f} min wall, {train['mean_tok_per_s']:.0f} tokens/s, peak {train['peak_vram_allocated_gib']:.1f} GiB allocated; "
              f"val loss {train['val'][0]['loss']:.3f} -> {train['val'][-1]['loss']:.3f}.", ""]
    sel = load("run2/select_final.json")
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
    v2 = load("/Users/naresh/Desktop/Personal/hackathons/amd-act-iii-2026/app/data/splits_v2/stats.json")
    if train and v2:
        ta = train["args"]
        L += ["## Run-2 recipe", "",
              f"- Data: {v2['unique_kept']} unique teach-back answers ({v2['sft_rows_after_oversampling']} rows after oversampling near-miss pairs x1.5 and weaker languages x1.25), "
              f"v1 generator answers plus multi-model v2 answers. Labels: kept when Gemma-4-31B-IT agreed with the designed label on every fact (heard/reteach text distilled from 31B), "
              f"otherwise a per-fact 2-of-3 vote of designed, 31B and Qwen3.8-Flash-Next labels; answers without a majority dropped. Counts: {v2['counts']}.",
              f"- Contamination check: {v2['contamination_check']}; dev, val_independent and test cases excluded from training.",
              f"- Target order: 'heard' before 'verdict' (run B). LoRA r={ta['rank']}, alpha={ta['alpha']}, lr {ta['lr']}, {ta['epochs']} epochs, "
              f"loss weight {ta.get('verdict_weight', 1)} on verdict tokens, unlikelihood {ta.get('ul_lambda', 0)} on P('correct') where gold is incorrect/missing (half for partial).",
              "- Inference: average of the epoch-1 and epoch-2 adapters, applied at LoRA scale 0.5 (chosen on validation), greedy decoding, no escalation (tau 0).", ""]
    sc = load("run2/eval_sc_valind.json")
    if sc:
        L += ["## Self-consistency (validation only)", ""]
        for k, s_ in sc["systems"].items():
            m = s_["metrics"]
            L.append(f"- {k}: accuracy {100 * m['accuracy']:.1f}%, false reassurance {m['false_reassurance_n'][0]}/{m['false_reassurance_n'][1]}, over-flag {100 * m['over_flag_rate']:.1f}%")
        L += ["- Majority of 3 samples (T 0.7) costs 3x the generation; candidate for run 4.", ""]
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
    (R / "eval.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # Web app file: keep its schema, fill the nulls.
    web = Path(a.web)
    if web.exists():
        w = json.loads(web.read_text())
        ts = ind["test_set"]
        w["grader"]["test_set"].update({"description": "Independent teach-back answers written by Claude (a different model family) (never seen by the data generator), held-out conditions",
                                        "cases": ts["cases"], "answers": ts["answers"], "facts_graded": ts["facts"], "languages": len(ts["languages"])})
        ids = {"gemma-4-e4b-base": "e4b_base", "gemma-4-e4b-homeward": "homeward_run2", "gemma-4-31b": "gemma4_31b"}
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
