"""OpenAI-compatible server for the on-prem Homeward models on one GPU (AMD Radeon PRO W7900D).

One Gemma 4 E4B copy plus the LoRA grader adapter:
  POST /v1/chat/completions      model "homeward-grader" (adapter on) or "gemma-4-e4b" (adapter off);
                                 text messages only, greedy, non-streaming.
  POST /v1/audio/transcriptions  multipart form with "file" (16 kHz mono WAV) and optional "language"
                                 (ISO code or English name); E4B native ASR with the adapter off.
  GET  /v1/models, GET /health

Standard library HTTP server, one request at a time (the GPU is the bottleneck anyway).
Usage: python serve.py --model /workspace/models/gemma-4-E4B-it --adapter /workspace/runs/lora1/adapter --port 8000
"""

import argparse
import contextlib
import io
import json
import threading
import time
import uuid
import wave
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

from asr_eval import ASR_PROMPT, LANG_EN
from train_lora import load_model_and_tokenizer

LOCK = threading.Lock()
STATE = {}


def wav_to_array(data: bytes):
    with wave.open(io.BytesIO(data)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("send 16 kHz mono 16-bit WAV")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def adapter_ctx(on: bool):
    model = STATE["model"]
    if STATE["has_adapter"] and not on:
        return model.disable_adapter()
    return contextlib.nullcontext()


def chat(messages, use_adapter, max_tokens):
    tok, model = STATE["tok"], STATE["model"]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with LOCK, torch.no_grad(), adapter_ctx(use_adapter):
        out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
    new = out[0, ids["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True), int(ids["input_ids"].shape[1]), int(new.shape[0])


def transcribe(audio, language):
    proc, model = STATE["proc"], STATE["model"]
    lang = LANG_EN.get(language, language or "the spoken language")
    msgs = [{"role": "user", "content": [{"type": "text", "text": ASR_PROMPT.format(lang=lang)},
                                         {"type": "audio", "audio": audio}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                   add_generation_prompt=True, enable_thinking=False).to(model.device)
    if "input_features" in inp:
        inp["input_features"] = inp["input_features"].to(torch.bfloat16)
    with LOCK, torch.no_grad(), adapter_ctx(False):
        out = model.generate(**inp, max_new_tokens=300, do_sample=False)
    return proc.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/").endswith("/health"):
            return self._send(200, {"ok": True, "device": torch.cuda.get_device_name(0)})
        if self.path.rstrip("/").endswith("/models"):
            return self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in ("homeward-grader", "gemma-4-e4b")]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        t0 = time.time()
        try:
            if self.path.endswith("/chat/completions"):
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n))
                msgs = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
                         else "".join(p.get("text", "") for p in m["content"])} for m in req["messages"]]
                use_adapter = req.get("model", "homeward-grader") != "gemma-4-e4b"
                text, pt, ct = chat(msgs, use_adapter, int(req.get("max_tokens", 768)))
                return self._send(200, {
                    "id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion", "created": int(time.time()),
                    "model": "homeward-grader" if use_adapter else "gemma-4-e4b",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
                    "timing": {"seconds": round(time.time() - t0, 3), "device": torch.cuda.get_device_name(0)}})
            if self.path.endswith("/audio/transcriptions"):
                n = int(self.headers.get("Content-Length", 0))
                raw = b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + self.rfile.read(n)
                parts = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
                         for p in BytesParser(policy=policy.default).parsebytes(raw).iter_parts()}
                audio = wav_to_array(parts["file"])
                lang = (parts.get("language") or b"").decode()
                text = transcribe(audio, lang)
                return self._send(200, {"text": text, "timing": {"seconds": round(time.time() - t0, 3)}})
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": {"message": str(e)[:500]}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import transformers
    model, tok = load_model_and_tokenizer(args.model)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    STATE.update(model=model, tok=tok, proc=transformers.AutoProcessor.from_pretrained(args.model),
                 has_adapter=bool(args.adapter))
    print(f"serving on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
