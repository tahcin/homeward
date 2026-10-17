"""LoRA SFT of google/gemma-4-E4B-it as the teach-back grader, on ROCm (AMD Radeon PRO W7900D).

Text-only fine-tuning of the language model: LoRA adapters go on the attention and MLP
projections of the language model only (vision and audio towers stay frozen and untouched).
Loss is computed on the assistant reply tokens only (the gold JSON plus the end-of-turn
marker). Logs tokens/s, peak VRAM and wall time to <out>/train_log.json.

Input: JSONL rows {"key": ..., "messages": [system, user, assistant]} from build_sft.py.

Usage (on the notebook):
  python train_lora.py --model $W/models/gemma-4-E4B-it --train sft_train.jsonl \
      --val sft_val.jsonl --out $W/runs/lora1 --epochs 2
  python train_lora.py ... --max-steps 20      # smoke test
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch

TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def load_model_and_tokenizer(model_path: str, dtype=torch.bfloat16, attn: str | None = None):
    """Attention backends measured on gfx1100 (W7900D, ROCm 10, torch 2.13, transformers 5.19),
    2026-10-15: SDPA in the language model returns NaN logits in bf16, so the text model uses
    eager; the audio tower's eager path garbles speech (also in fp32) while its SDPA path
    transcribes correctly, so the audio tower uses SDPA. ATTN_IMPL=eager|sdpa overrides all."""
    import transformers
    from transformers import AutoProcessor, AutoTokenizer
    try:
        tok = AutoProcessor.from_pretrained(model_path).tokenizer
    except Exception:  # noqa: BLE001
        tok = AutoTokenizer.from_pretrained(model_path)
    cls = getattr(transformers, "AutoModelForMultimodalLM", None) or transformers.AutoModelForImageTextToText
    attn = attn or os.environ.get("ATTN_IMPL") or {"text_config": "eager", "audio_config": "sdpa",
                                                    "vision_config": "eager"}
    # Default (LOAD_VIA_CPU unset or 1): materialise the weights in host memory first, then move them to the GPU in one go.
    # On Oct 17 the direct load (device_map to GPU) stalled at random points in transformers'
    # _materialize_copy (host-to-GPU copy from the memory-mapped safetensors) in 3 of 5 loads.
    via_cpu = os.environ.get("LOAD_VIA_CPU", "1") == "1"
    dmap = None if via_cpu else {"": 0}
    try:
        model = cls.from_pretrained(model_path, dtype=dtype, device_map=dmap, attn_implementation=attn)
    except (ValueError, KeyError, TypeError) as e:  # e.g. a sub-config name the model lacks
        print("attention config fallback to eager:", str(e)[:200], flush=True)
        model = cls.from_pretrained(model_path, dtype=dtype, device_map=dmap, attn_implementation="eager")
    if via_cpu:
        model = model.to("cuda:0")
    return model, tok


def verdict_mask(tok, a_ids: list[int], answer_text: str) -> list[int]:
    """1 for assistant tokens that spell a verdict value ("correct", "partial", ...), else 0."""
    import re
    spans = [(m.start(1), m.end(1)) for m in re.finditer(r'"verdict"\s*:\s*"([a-z]+)"', answer_text)]
    if not spans:
        return [0] * len(a_ids)
    enc = tok(answer_text, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] != a_ids:
        return [0] * len(a_ids)
    out = []
    for (s0, e0) in enc["offset_mapping"]:
        out.append(int(any(s0 < e and e0 > s for s, e in spans)))
    return out


BAD_WEIGHT = {"incorrect": 1.0, "missing": 1.0, "partial": 0.5}


def first_verdict_bad(tok, a_ids: list[int], answer_text: str) -> list[float]:
    """Weight at the FIRST token of each verdict value whose gold verdict is not 'correct'
    (1.0 incorrect/missing, 0.5 partial): where the unlikelihood term pushes P('correct') down."""
    import re
    spans = [(m.start(1), m.group(1)) for m in re.finditer(r'"verdict"\s*:\s*"([a-z]+)"', answer_text)]
    out = [0.0] * len(a_ids)
    if not spans:
        return out
    enc = tok(answer_text, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] != a_ids:
        return out
    for start, v in spans:
        w = BAD_WEIGHT.get(v, 0.0)
        if not w:
            continue
        for k, (s0, e0) in enumerate(enc["offset_mapping"]):
            if e0 > start:  # first token reaching into the verdict value (may include the opening quote)
                out[k] = w
                break
    return out


def correct_token_id(tok) -> int:
    """Token id that starts the value 'correct' after '"verdict": "' (same merge pattern as training text)."""
    a = tok('{"id": "f1", "verdict": "correct"', add_special_tokens=False, return_offsets_mapping=True)
    start = len('{"id": "f1", "verdict": "')
    for tid, (s0, e0) in zip(a["input_ids"], a["offset_mapping"]):
        if e0 > start:
            return tid
    raise ValueError("could not locate the 'correct' token")


def encode(tok, msgs: list[dict], max_len: int):
    """Token ids and labels with everything except the assistant reply masked to -100."""
    prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
    full_text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template: prompt is not a prefix of the full conversation")
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    a_ids = tok(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
    ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + a_ids
    if len(ids) > max_len:
        return None
    ans = full_text[len(prompt_text):]
    vm = verdict_mask(tok, a_ids, ans)
    bad = first_verdict_bad(tok, a_ids, ans)
    return {"input_ids": ids, "labels": labels, "verdict": [0] * len(p_ids) + vm, "bad": [0.0] * len(p_ids) + bad}


def batches(examples, bsz, shuffle, seed):
    """Length-grouped batches (sort within shuffled chunks) to cut padding."""
    idx = list(range(len(examples)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(idx)
    chunk = bsz * 16
    out = []
    for i in range(0, len(idx), chunk):
        part = sorted(idx[i:i + chunk], key=lambda j: len(examples[j]["input_ids"]))
        out += [part[k:k + bsz] for k in range(0, len(part), bsz)]
    if shuffle:
        rng.shuffle(out)
    return out


VERDICT_WEIGHT = 1.0


def collate(examples, ids, pad_id, device):
    L = max(len(examples[i]["input_ids"]) for i in ids)
    inp = torch.full((len(ids), L), pad_id, dtype=torch.long)
    lab = torch.full((len(ids), L), -100, dtype=torch.long)
    att = torch.zeros((len(ids), L), dtype=torch.long)
    wts = torch.ones((len(ids), L), dtype=torch.float)
    bad = torch.zeros((len(ids), L), dtype=torch.float)
    for r, i in enumerate(ids):
        e = examples[i]
        n = len(e["input_ids"])
        inp[r, :n] = torch.tensor(e["input_ids"])
        lab[r, :n] = torch.tensor(e["labels"])
        att[r, :n] = 1
        if VERDICT_WEIGHT != 1.0 and "verdict" in e:
            wts[r, :n] = 1.0 + (VERDICT_WEIGHT - 1.0) * torch.tensor(e["verdict"], dtype=torch.float)
        if "bad" in e:
            bad[r, :n] = torch.tensor(e["bad"], dtype=torch.float)
    return inp.to(device), lab.to(device), att.to(device), (wts.to(device), bad.to(device))


UL_LAMBDA = 0.0
CORRECT_ID = None
UL_STATS = {"terms": 0, "sum": 0.0}


def masked_loss(model, inp, lab, att, extra=None):
    """Summed token cross-entropy over assistant tokens; verdict-value tokens can carry a
    larger weight (--verdict-weight), plus an unlikelihood penalty -log(1 - P('correct')) at
    the first verdict token wherever the gold verdict is incorrect/missing (weight 1) or
    partial (0.5), scaled by --ul-lambda. Returns (loss sum, weighted token count)."""
    wts, bad = extra if extra is not None else (None, None)
    out = model(input_ids=inp, attention_mask=att)
    tgt = lab[:, 1:]
    m = tgt != -100
    logits = out.logits[:, :-1]
    sel = logits[m].float()  # only supervised positions, keeps fp32 logits small
    ce = torch.nn.functional.cross_entropy(sel, tgt[m], reduction="none")
    w = wts[:, 1:][m] if wts is not None else torch.ones_like(ce)
    loss = (ce * w).sum()
    if UL_LAMBDA > 0 and bad is not None:
        b = bad[:, 1:]
        pos = b > 0
        if pos.any():
            p_cor = torch.softmax(logits[pos].float(), -1)[:, CORRECT_ID]
            ul = -(torch.log1p(-p_cor.clamp(max=0.999)) * b[pos]).sum()
            loss = loss + UL_LAMBDA * ul
            UL_STATS["terms"] += int(pos.sum())
            UL_STATS["sum"] += float(ul)
    return loss, w.sum()


@torch.no_grad()
def evaluate(model, examples, bsz, pad_id, device):
    model.eval()
    tot, n = 0.0, 0
    for ids in batches(examples, bsz, False, 0):
        inp, lab, att, _ = collate(examples, ids, pad_id, device)
        loss, cnt = masked_loss(model, inp, lab, att)
        tot += loss.item()
        n += cnt.item()
    model.train()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-ckpt", action="store_true", help="disable gradient checkpointing")
    ap.add_argument("--verdict-weight", type=float, default=1.0, help="loss weight on verdict-value tokens")
    ap.add_argument("--ul-lambda", type=float, default=0.0, help="unlikelihood weight on P(correct) for non-correct gold verdicts")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    global VERDICT_WEIGHT, UL_LAMBDA, CORRECT_ID
    VERDICT_WEIGHT = args.verdict_weight
    UL_LAMBDA = args.ul_lambda
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    t_start = time.time()

    t0 = time.time()
    model, tok = load_model_and_tokenizer(args.model)
    load_s = time.time() - t0
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    CORRECT_ID = correct_token_id(tok)
    print("unlikelihood target token:", repr(tok.decode([CORRECT_ID])), CORRECT_ID, "lambda", args.ul_lambda, flush=True)

    def load(path):
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        ex, dropped = [], 0
        for r in rows:
            e = encode(tok, r["messages"], args.max_len)
            if e is None:
                dropped += 1
            else:
                ex.append(e)
        return ex, dropped

    train, drop_tr = load(args.train)
    val, drop_va = load(args.val) if args.val else ([], 0)
    e0 = train[0]
    sup = [t for t, l in zip(e0["input_ids"], e0["labels"]) if l != -100]
    print("supervised span of example 0:", repr(tok.decode(sup))[:400], flush=True)
    vt = [t for t, v in zip(e0["input_ids"], e0["verdict"]) if v]
    print("verdict tokens of example 0:", repr(tok.decode(vt)), f"(weight {args.verdict_weight})", flush=True)
    for e in train[:3]:
        bt = [(tok.decode([t]), b) for t, b in zip(e["input_ids"], e["bad"]) if b]
        print("unlikelihood positions:", bt, flush=True)
    print(f"train {len(train)} (dropped {drop_tr} > max_len), val {len(val)} (dropped {drop_va})", flush=True)

    model.config.use_cache = False
    if not args.no_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                     target_modules=TARGET_REGEX, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    n_lora_modules = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    print(f"trainable params {n_train:,} of {n_all:,} ({100 * n_train / n_all:.3f}%), lora modules {n_lora_modules}", flush=True)

    steps_per_epoch = math.ceil(len(batches(train, args.bsz, True, 0)) / args.accum)
    total = args.max_steps or math.ceil(steps_per_epoch * args.epochs)
    warm = max(1, int(total * args.warmup))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            weight_decay=0.0, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    log = {"args": vars(args), "load_s": round(load_s, 1), "train_examples": len(train),
           "val_examples": len(val), "dropped_over_max_len": drop_tr + drop_va,
           "trainable_params": n_train, "total_params": n_all, "lora_modules": n_lora_modules,
           "steps": [], "val": [], "device": torch.cuda.get_device_name(0),
           "torch": torch.__version__}
    import transformers, peft
    log["transformers"], log["peft"] = transformers.__version__, peft.__version__
    if val:
        vl = evaluate(model, val, args.bsz, pad_id, dev)
        log["val"].append({"step": 0, "loss": vl})
        print(f"step 0 val_loss {vl:.4f}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    step, epoch, tok_seen, sup_seen = 0, 0, 0, 0
    train_t0 = time.time()
    done = False
    while not done:
        bl = batches(train, args.bsz, True, args.seed + epoch)
        for g in range(0, len(bl), args.accum):
            group = bl[g:g + args.accum]
            t_step = time.time()
            n_sup = sum(sum((1.0 + (VERDICT_WEIGHT - 1.0) * v) for l, v in zip(train[i]["labels"][1:], train[i]["verdict"][1:]) if l != -100)
                        for ids in group for i in ids)
            loss_sum, ntok = 0.0, 0
            for ids in group:
                inp, lab, att, extra = collate(train, ids, pad_id, dev)
                loss, cnt = masked_loss(model, inp, lab, att, extra)
                (loss / n_sup).backward()
                loss_sum += loss.item()
                ntok += int(att.sum().item())
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.time() - t_step
            step += 1
            tok_seen += ntok
            sup_seen += sum(sum(1 for l in train[i]["labels"][1:] if l != -100) for ids in group for i in ids)
            ul_now = UL_STATS["sum"] / max(UL_STATS["terms"], 1)
            UL_STATS.update(terms=0, sum=0.0)
            rec = {"ul_mean": ul_now, "step": step, "epoch": round(epoch + g / len(bl), 3), "loss": loss_sum / n_sup,
                   "grad_norm": float(gn), "lr": sched.get_last_lr()[0], "tokens": ntok,
                   "tok_per_s": ntok / dt, "step_s": dt}
            log["steps"].append(rec)
            if step % 5 == 0 or step == 1:
                print(f'step {step}/{total} loss {rec["loss"]:.4f} gn {rec["grad_norm"]:.2f} '
                      f'lr {rec["lr"]:.2e} {rec["tok_per_s"]:.0f} tok/s '
                      f'peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB', flush=True)
            if step >= total:
                done = True
                break
        epoch += 1
        if val and not done:
            vl = evaluate(model, val, args.bsz, pad_id, dev)
            log["val"].append({"step": step, "loss": vl})
            print(f"epoch {epoch} step {step} val_loss {vl:.4f}", flush=True)
            model.save_pretrained(out / f"adapter_epoch{epoch}")
    train_s = time.time() - train_t0
    if val:
        vl = evaluate(model, val, args.bsz, pad_id, dev)
        log["val"].append({"step": step, "loss": vl})
        print(f"final val_loss {vl:.4f}", flush=True)
    model.save_pretrained(out / "adapter")
    log.update({
        "optimizer_steps": step, "train_wall_s": round(train_s, 1),
        "total_wall_s": round(time.time() - t_start, 1),
        "tokens_processed": tok_seen, "supervised_tokens": sup_seen,
        "mean_tok_per_s": tok_seen / train_s,
        "peak_vram_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    })
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k not in ("steps", "args")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
