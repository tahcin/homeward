"""Run 5 in ONE process: load Gemma 4 E4B once, train the LoRA adapter, then evaluate every candidate on
val_independent by swapping LoRA weights in place (checkpoints, averages, LoRA scales) without reloading the
base model, select with the declared rule (select_model.py, unchanged, val only), and score the frozen pick
once on the test sets. Every phase writes its files as it goes, so a restarted process resumes where the
last one stopped (run_day3.sh restarts it after a stalled load).

Why: on the W7900D notebook about half of all model loads stalled in a host-to-GPU copy (Oct 17). One load
per process (plus the GPU-side weight swaps here) removes nearly all of them; a stalled load is killed by
faulthandler (exit status 1, stack in <logs>/fh_run5_<pid>.txt) after --load-timeout seconds and the driver
starts a new process.

Candidate inference uses the same arithmetic as eval.py (PEFT's own merge of fp32 LoRA weights into the bf16
base, scale applied to module.scaling before the merge); the base weights are restored from a GPU copy after
each candidate, so nothing drifts. Generation and verdict probabilities come from eval.predict_rows and
eval.verdict_probs_rows, the same code as the eval.py CLI.

Integrity: training reads only --train/--val-loss (the jury-labelled splits). val_independent is used only for
selection; the test sets are scored once, for the frozen pick only, and never if the incumbent is kept.
"""

import argparse
import faulthandler
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval as ev  # noqa: E402
import grader  # noqa: E402
import train_lora as tl  # noqa: E402

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(p, obj):
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, p)


def ts(s):
    """'HH:MM' in IST today, or a unix time, or '' -> 0."""
    if not s:
        return 0.0
    if ":" not in str(s):
        return float(s)
    import datetime as dt
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    now = dt.datetime.now(ist)
    h, m = map(int, str(s).split(":"))
    return now.replace(hour=h, minute=m, second=0, microsecond=0).timestamp()


def hm(t):
    import datetime as dt
    return dt.datetime.fromtimestamp(t, dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime("%H:%M:%S IST")


# ---------------------------------------------------------------- candidates

# (name, weights, scale). Weights: an adapter folder name under --out, or "avg:A+B" (parameter mean, as
# avg_adapters.py). Listed in evaluation priority; the incumbent is listed before all of them in the
# selection, so on a full tie the incumbent stays. Fixed before training (Sat Oct 17 evening).
CANDIDATES = [
    ("run5_final", "adapter", 1.0),
    ("run5_avg", "avg:adapter_epoch1+adapter", 1.0),
    ("run5_final_s075", "adapter", 0.75),
    ("run5_avg_late", "avg:LATE+adapter", 1.0),
    ("run5_avg_s075", "avg:adapter_epoch1+adapter", 0.75),
    ("run5_ep1", "adapter_epoch1", 1.0),
]


class Swapper:
    """Holds the trained PeftModel and installs candidate LoRA weights + scale by merging into the base
    weights in place, restoring the base from a GPU copy afterwards."""

    def __init__(self, pm, out):
        import torch
        from peft.tuners.lora import LoraLayer
        self.torch, self.pm, self.out = torch, pm, Path(out)
        self.layers = [(n, m) for n, m in pm.named_modules() if isinstance(m, LoraLayer) and hasattr(m, "base_layer")]
        self.backup = {n: m.base_layer.weight.data.clone() for n, m in self.layers}
        self.scaling0 = {n: dict(m.scaling) for n, m in self.layers}
        self.snap = {}  # adapter name -> state dict on the GPU
        gib = sum(t.numel() * t.element_size() for t in self.backup.values()) / 2**30
        log(f"swapper: {len(self.layers)} LoRA layers, base backup {gib:.2f} GiB on the GPU")

    def snapshot(self, name):
        from peft import get_peft_model_state_dict
        self.snap[name] = {k: v.detach().clone() for k, v in get_peft_model_state_dict(self.pm).items()}

    def state(self, spec):
        """State dict for a weights spec (from memory, else from disk)."""
        torch = self.torch
        if spec.startswith("avg:"):
            parts = [self.state(p) for p in spec[4:].split("+")]
            return {k: (sum(p[k].float() for p in parts) / len(parts)).to(parts[0][k].dtype) for k in parts[0]}
        if spec not in self.snap:
            from safetensors.torch import load_file
            dev = next(self.pm.parameters()).device
            sd = load_file(str(self.out / spec / "adapter_model.safetensors"))
            self.snap[spec] = {k: v.to(dev) for k, v in sd.items()}
            log(f"loaded {spec} from disk ({len(sd)} tensors)")
        return self.snap[spec]

    def save_dir(self, spec):
        """A PEFT adapter folder for the spec (averages are written to <out>/<avg name>)."""
        if not spec.startswith("avg:"):
            return self.out / spec
        from safetensors.torch import save_file
        d = self.out / ("adapter_" + spec[4:].replace("adapter_", "").replace("+", "_plus_"))
        if not (d / "adapter_model.safetensors").exists():
            d.mkdir(parents=True, exist_ok=True)
            first = spec[4:].split("+")[0]
            shutil.copy(self.out / first / "adapter_config.json", d / "adapter_config.json")
            save_file({k: v.detach().cpu().contiguous() for k, v in self.state(spec).items()}, str(d / "adapter_model.safetensors"))
        return d

    def install(self, spec, scale):
        from peft import set_peft_model_state_dict
        self.restore()
        res = set_peft_model_state_dict(self.pm, self.state(spec))
        bad = [k for k in getattr(res, "unexpected_keys", []) if "lora" in k]
        assert not bad, f"unexpected adapter keys: {bad[:3]}"
        with self.torch.no_grad():
            for n, m in self.layers:
                for a in m.scaling:
                    m.scaling[a] = self.scaling0[n][a] * scale
                m.merge(adapter_names=["default"])
        self.pm.eval()

    def restore(self):
        with self.torch.no_grad():
            for n, m in self.layers:
                if m.merged_adapters:
                    m.base_layer.weight.data.copy_(self.backup[n])
                    m.merged_adapters.clear()
                m.scaling.update(self.scaling0[n])

    @property
    def model(self):
        return self.pm.get_base_model()


def prepare_for_inference(pm, use_cache0):
    """Undo the training-only settings so generation matches a freshly loaded model."""
    base = pm.get_base_model()
    try:
        base.gradient_checkpointing_disable()
    except Exception as e:  # noqa: BLE001
        log("gradient_checkpointing_disable:", e)
    try:
        base.disable_input_require_grads()
    except Exception as e:  # noqa: BLE001
        log("disable_input_require_grads:", e)
    for cfg, v in use_cache0:
        if v == "<absent>":
            cfg.__dict__.pop("use_cache", None)
        else:
            cfg.use_cache = v
    for p in pm.parameters():
        p.requires_grad_(False)
        p.grad = None
    pm.eval()
    gc.collect()
    pm_dev = next(pm.parameters()).device
    if pm_dev.type == "cuda":
        import torch
        torch.cuda.empty_cache()


def cache_flags(model):
    """use_cache as loaded, on the top-level and text configs (training sets it to False)."""
    cfgs = [model.config] + [c for c in [getattr(model.config, "text_config", None)] if c is not None]
    return [(c, c.__dict__.get("use_cache", "<absent>")) for c in cfgs]


# ---------------------------------------------------------------- phases

def load_base(a):
    fh = open(Path(a.logs) / f"fh_run5_{os.getpid()}.txt", "w")
    faulthandler.dump_traceback_later(a.load_timeout, exit=True, file=fh)
    t0 = time.time()
    model, tok = tl.load_model_and_tokenizer(a.model)
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(600, repeat=True, file=fh)  # periodic stack dumps for diagnosis only
    log(f"base model loaded in {time.time() - t0:.1f}s")
    return model, tok, time.time() - t0


def build_peft_for_eval(model, out):
    """Resume path: wrap the fresh base with the saved adapter's config (weights swapped in later)."""
    from peft import PeftModel
    return PeftModel.from_pretrained(model, str(Path(out) / "adapter"))


def train_args(a):
    p = tl.build_parser()
    argv = ["--model", a.model, "--train", a.train, "--out", a.out, "--epochs", str(a.epochs), "--bsz", str(a.bsz),
            "--accum", str(a.accum), "--rank", str(a.rank), "--alpha", str(a.alpha), "--lr", str(a.lr),
            "--verdict-weight", str(a.verdict_weight), "--ul-lambda", str(a.ul_lambda), "--max-len", str(a.max_len),
            "--save-at", a.save_at, "--deadline-ts", str(a.train_deadline_ts), "--deadline-reserve-s", str(a.deadline_reserve_s)]
    if a.val_loss:
        argv += ["--val", a.val_loss]
    if a.max_steps:
        argv += ["--max-steps", str(a.max_steps)]
    return p.parse_args(argv)


def late_ckpt(out):
    """The checkpoint used for the 'late' average: the latest adapter_step<N> past epoch 1, else None."""
    tlog = json.loads((Path(out) / "train_log.json").read_text())
    steps = [c for c in tlog.get("checkpoints", []) if c["name"].startswith("adapter_step") and c["epoch"] > 1.0]
    return steps[-1]["name"] if steps else None


def candidate_list(out):
    o = Path(out)
    late = late_ckpt(out) if (o / "train_log.json").exists() else None
    res = []
    for name, spec, scale in CANDIDATES:
        spec = spec.replace("LATE", late) if late else spec
        parts = spec[4:].split("+") if spec.startswith("avg:") else [spec]
        if "LATE" in spec or not all((o / p / "adapter_model.safetensors").exists() for p in parts):
            log(f"candidate {name}: weights not available ({spec}), skipped")
            continue
        if spec.startswith("avg:") and len(set(parts)) < 2:
            continue
        res.append((name, spec, scale))
    return res


def run_select(a, cands_dir, tag):
    lst = (cands_dir / "list.txt").read_text().split()
    cmd = [sys.executable, str(HERE / "select_model.py"), "--rows", a.val_rows, "--out", str(cands_dir / f"select_{tag}.json"),
           "--base-fr", str(a.base_fr)]
    for c in lst:
        cmd += ["--cand", c]
    r = subprocess.run(cmd, capture_output=True, text=True)
    (cands_dir / f"select_{tag}.log").write_text(r.stdout + r.stderr)
    log(f"selection ({tag}):\n" + r.stdout)
    if r.returncode != 0:
        raise RuntimeError("select_model.py failed: " + r.stderr[-800:])
    return json.loads((cands_dir / f"select_{tag}.json").read_text())


def quick_metrics(rows, preds):
    pmap = {p["key"]: p for p in preds}
    gold, pred, _, crit, _, _ = ev.score_system(rows, pmap)
    m = ev.metrics(gold, pred, crit)
    return {"accuracy": m["accuracy"], "false_reassurance_n": m["false_reassurance_n"],
            "critical_n": m["false_reassurance_critical_n"], "over_flag_rate": m["over_flag_rate"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val-loss", default="", help="generator val split for the loss curve only")
    ap.add_argument("--out", required=True, help="run folder, e.g. /workspace/runs/run5")
    ap.add_argument("--logs", default="/workspace/logs")
    ap.add_argument("--val-rows", required=True, help="val_independent.jsonl (selection only)")
    ap.add_argument("--test", action="append", default=[], help="NAME=rows.jsonl, scored once for the frozen pick (first is the headline)")
    ap.add_argument("--incumbent", action="append", default=[], help="NAME:ADAPTER:SCALE:PREDS:PROBS with existing val preds/probs")
    ap.add_argument("--base-fr", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--verdict-weight", type=float, default=3.0)
    ap.add_argument("--ul-lambda", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--save-at", default="1.5")
    ap.add_argument("--train-deadline", default="", help="HH:MM IST (or unix time): training ends by then (cosine horizon shrinks to fit)")
    ap.add_argument("--deadline-reserve-s", type=float, default=90)
    ap.add_argument("--pick-deadline", default="", help="HH:MM IST: no frozen pick by then -> keep the incumbent, no test")
    ap.add_argument("--cand-minutes", type=float, default=3.0, help="budget per candidate when deciding whether another fits before the pick deadline")
    ap.add_argument("--gen-bsz", type=int, default=64)
    ap.add_argument("--probs-bsz", type=int, default=16)
    ap.add_argument("--latency-n", type=int, default=20, help="batch-1 latency pass on the headline test set")
    ap.add_argument("--load-timeout", type=float, default=150)
    ap.add_argument("--max-new", type=int, default=768)
    a = ap.parse_args()
    a.train_deadline_ts = ts(a.train_deadline)
    a.pick_deadline_ts = ts(a.pick_deadline)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(a.logs).mkdir(parents=True, exist_ok=True)
    cands_dir = out / "cands"
    cands_dir.mkdir(exist_ok=True)
    fin = out / "final"
    fin.mkdir(exist_ok=True)
    prompt = grader.load_prompt()
    val_rows = ev.read_jsonl(a.val_rows)
    log(f"run5 start pid {os.getpid()}; train deadline {hm(a.train_deadline_ts) if a.train_deadline_ts else 'none'}, "
        f"pick deadline {hm(a.pick_deadline_ts) if a.pick_deadline_ts else 'none'}")

    # A frozen pick, once written, is final: a restart never re-selects.
    pick_f = fin / "pick.json"
    if (fin / "DONE").exists():
        log("final/DONE exists: nothing to do")
        return
    if (fin / "ABORTED").exists():
        log("final/ABORTED exists: run 4 stays; nothing to do")
        return

    trained = (out / "train_log.json").exists() and (out / "adapter" / "adapter_model.safetensors").exists()
    pick = json.loads(pick_f.read_text()) if pick_f.exists() else None
    model, tok, load_s = load_base(a)
    use_cache0 = cache_flags(model)

    # ------------------------------------------------ train
    sw = None
    partial = sorted((d for d in out.glob("adapter_*") if (d / "adapter_model.safetensors").exists()
                      and (d.name.startswith("adapter_epoch") or d.name.startswith("adapter_step"))), key=lambda d: d.stat().st_mtime)
    if not trained and partial:
        log(f"training was interrupted after {partial[-1].name}: evaluating the checkpoints that exist, no retraining")
    if not trained and not partial:
        if a.train_deadline_ts and time.time() > a.train_deadline_ts - 300:
            log("too late to start training (less than 5 min before the training deadline)")
            if not any((out / d / "adapter_model.safetensors").exists() for d in ("adapter", "adapter_epoch1")):
                (fin / "ABORTED").write_text("no adapter and no time to train; run 4 stays\n")
                return
        else:
            snaps = {}

            def on_save(name, pm):
                from peft import get_peft_model_state_dict
                snaps[name] = {k: v.detach().clone() for k, v in get_peft_model_state_dict(pm).items()}
                log(f"checkpoint {name} saved and kept in GPU memory")

            for d in out.glob("adapter*"):
                shutil.rmtree(d, ignore_errors=True)  # a half-finished earlier attempt
            pm = tl.run_training(model, tok, train_args(a), load_s=load_s, on_save=on_save, log_fn=log)
            prepare_for_inference(pm, use_cache0)
            sw = Swapper(pm, out)
            sw.snap.update(snaps)
            trained = True
    if sw is None:
        if not (out / "adapter" / "adapter_model.safetensors").exists():
            # training died before its end; the latest checkpoint stands in for the final adapter
            if partial:
                shutil.copytree(partial[-1], out / "adapter")
                log(f"no final adapter: using {partial[-1].name} as 'adapter'")
            else:
                (fin / "ABORTED").write_text("training did not produce an adapter; run 4 stays\n")
                return
        pm = build_peft_for_eval(model, out)
        prepare_for_inference(pm, use_cache0)
        sw = Swapper(pm, out)

    # ------------------------------------------------ candidates on val_independent
    lst = cands_dir / "list.txt"
    if not lst.exists():
        lst.write_text("".join(c + "\n" for c in a.incumbent))
    if pick is None:
        for name, spec, scale in candidate_list(a.out):
            pf, qf = cands_dir / f"preds_{name}.jsonl", cands_dir / f"probs_{name}.jsonl"
            line = f"{name}:{sw.save_dir(spec)}:{scale}:{pf}:{qf}"
            if line in lst.read_text().split():
                continue
            if a.pick_deadline_ts and time.time() + 60 * a.cand_minutes > a.pick_deadline_ts:
                log(f"pick deadline: no time for {name}; stopping candidate evaluation")
                break
            t0 = time.time()
            sw.install(spec, scale)
            recs, meta = ev.predict_rows(sw.model, tok, val_rows, prompt, bsz=a.gen_bsz, max_new=a.max_new, log=log)
            ev.write_preds(pf, recs, dict(system=name, model=a.model, adapter=str(sw.save_dir(spec)), weights=spec, lora_scale=scale,
                                          rows=len(val_rows), in_process=True, **meta))
            probs = ev.verdict_probs_rows(sw.model, tok, val_rows, recs, prompt, bsz=a.probs_bsz, log=log)
            ev.write_jsonl(qf, probs)
            with lst.open("a") as fh:
                fh.write(line + "\n")
            log(f"candidate {name} ({spec} x{scale}) in {time.time() - t0:.0f}s: {quick_metrics(val_rows, recs)}")

        # ------------------------------------------------ frozen pick (val only)
        if a.pick_deadline_ts and time.time() > a.pick_deadline_ts:
            (fin / "ABORTED").write_text(f"no frozen pick by {hm(a.pick_deadline_ts)}; run 4 stays\n")
            log("ABORT: pick deadline passed before selection; run 4 stays")
            return
        sel = run_select(a, cands_dir, "final")
        pick = dict(sel["chosen"], incumbent=[c.split(":")[0] for c in a.incumbent], picked_at=hm(time.time()))
        spec = next((s for n, s, _ in candidate_list(a.out) if n == pick["name"]), None)
        pick["weights"] = spec
        jdump(pick_f, pick)
        shutil.copy(cands_dir / "select_final.json", out / "select_final.json")
        log(f"FROZEN PICK: {pick['name']} scale {pick['scale']} tau {pick['tau']} (val acc {100 * pick['accuracy']:.1f}%, "
            f"FR {pick['false_reassurance']}, critical {pick['critical_false_reassurance']}, over {pick['over_flag']})")

    if pick["name"] in pick.get("incumbent", []):
        (fin / "DONE").write_text(f"incumbent {pick['name']} kept by the declared rule; test sets not scored again\n")
        log("incumbent kept: no test scoring")
        return

    # ------------------------------------------------ the frozen pick on the test sets, once
    (fin / "adapter_path.txt").write_text(str(sw.save_dir(pick["weights"])) + "\n")
    (fin / "lora_scale.txt").write_text(f"{pick['scale']}\n")
    (fin / "tau.txt").write_text(f"{pick['tau']}\n")
    (fin / "name.txt").write_text(pick["name"] + "\n")
    sw.install(pick["weights"], pick["scale"])
    summary = {"pick": pick, "tests": {}}
    for i, spec_t in enumerate(a.test):
        tname, path = spec_t.split("=", 1)
        pf = fin / f"preds_{tname}.jsonl"
        ff = fin / f"preds_{tname}_final.jsonl"
        if ff.exists():
            summary["tests"][tname] = quick_metrics(ev.read_jsonl(path), ev.read_jsonl(ff))
            continue
        rows = ev.read_jsonl(path)
        t0 = time.time()
        recs, meta = ev.predict_rows(sw.model, tok, rows, prompt, bsz=a.gen_bsz, max_new=a.max_new,
                                     latency_n=a.latency_n if i == 0 else 0, log=log)
        ev.write_preds(pf, recs, dict(system="homeward_run5", model=a.model, adapter=str(sw.save_dir(pick["weights"])),
                                      weights=pick["weights"], lora_scale=pick["scale"], rows=len(rows), in_process=True,
                                      load_s=round(load_s, 1), **meta))
        probs = ev.verdict_probs_rows(sw.model, tok, rows, recs, prompt, bsz=a.probs_bsz, log=log)
        ev.write_jsonl(fin / f"probs_{tname}.jsonl", probs)
        ev.escalate(argparse.Namespace(preds=str(pf), probs=str(fin / f"probs_{tname}.jsonl"), tau=pick["tau"], out=str(ff)))
        q = quick_metrics(rows, ev.read_jsonl(ff))
        summary["tests"][tname] = q
        log(f"TEST {tname} (once, frozen pick {pick['name']}) in {time.time() - t0:.0f}s: {q}")
        jdump(fin / "summary.json", summary)
    (fin / "DONE").write_text(f"{pick['name']} scored once on {[t.split('=')[0] for t in a.test]} at {hm(time.time())}\n")
    log("RUN5 DONE")


if __name__ == "__main__":
    main()
