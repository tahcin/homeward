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
    for bs in (16, 32, 64):
        m = load("preds_ft_test.meta.json" if bs == 16 else f"cap_bs{bs}.meta.json")
        if m:
            cap[bs] = {"answers_per_s": m["batched"]["answers_per_s"], "gen_tok_per_s": m["batched"]["gen_tok_per_s"],
                       "peak_vram_gib": m["peak_vram_allocated_gib"], "answers": m["rows"]}

    for d in (ind, ind_d):
        for k, s in d["systems"].items():
            pv = s["metrics"].get("paired_vs_e4b_base")
            if pv:
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
    L = ["# Homeward grader: evaluation (run 1)", "", f"Generated {out['generated_at']}. All numbers come from the files in ml/results; 95% intervals are case-level bootstrap (1,000 resamples).", ""]
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
              f"{train['train_examples']} verifier-agreed examples, {train['optimizer_steps']} steps; {train['trainable_params']:,} trainable of {train['total_params']:,} params; "
              f"{train['train_wall_s'] / 60:.1f} min wall, {train['mean_tok_per_s']:.0f} tokens/s, peak {train['peak_vram_allocated_gib']:.1f} GiB allocated; "
              f"val loss {train['val'][0]['loss']:.3f} -> {train['val'][-1]['loss']:.3f}.", ""]
    L += ["## Safety threshold", "", out["escalation"]["finding"] + ".", ""]
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
        L += ["## On-prem loop", "", f"{loop['clips']} clips: E4B ASR then fine-tuned E4B grading on one card, p50 {loop['loop_p50_s']:.1f} s, p95 {loop['loop_p95_s']:.1f} s, peak {loop['peak_vram_gib']:.1f} GiB.", ""]
    (R / "eval.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # Web app file: keep its schema, fill the nulls.
    web = Path(a.web)
    if web.exists():
        w = json.loads(web.read_text())
        ts = ind["test_set"]
        w["grader"]["test_set"].update({"description": "Independent teach-back answers written by Claude (a different model family) (never seen by the data generator), held-out conditions",
                                        "cases": ts["cases"], "answers": ts["answers"], "facts_graded": ts["facts"], "languages": len(ts["languages"])})
        ids = {"gemma-4-e4b-base": "e4b_base", "gemma-4-e4b-homeward": "e4b_lora", "gemma-4-31b": "gemma4_31b"}
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
