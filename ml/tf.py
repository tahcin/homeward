"""Small async client for AMD Token Factory (OpenAI-compatible chat completions).

Reads TOKENFACTORY_API_KEY from the environment. Paces requests under the
per-account limit (20 requests a minute, 8 concurrent) and retries 429/5xx
with backoff.
"""

import asyncio
import json
import os
import random
import re
import time

import httpx

BASE_URL = os.environ.get("TOKENFACTORY_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1")
API_KEY = os.environ["TOKENFACTORY_API_KEY"]

RPM = int(os.environ.get("TF_RPM", "19"))
_sem = asyncio.Semaphore(int(os.environ.get("TF_CONCURRENCY", "6")))
_starts: list[float] = []
_lock = asyncio.Lock()


async def _pace() -> None:
    """Block until a new request fits inside the sliding 60 s window."""
    while True:
        async with _lock:
            now = time.monotonic()
            while _starts and now - _starts[0] > 60:
                _starts.pop(0)
            if len(_starts) < RPM:
                _starts.append(now)
                return
            wait = 60 - (now - _starts[0]) + 0.2
        await asyncio.sleep(wait)


async def chat(client: httpx.AsyncClient, model: str, messages: list[dict], *,
               max_tokens: int = 4000, temperature: float = 0.7,
               json_mode: bool = False, reasoning_effort: str | None = None,
               retries: int = 6) -> dict:
    body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens,
                  "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    delay = 4.0
    for attempt in range(retries):
        async with _sem:
            await _pace()
            try:
                r = await client.post(f"{BASE_URL}/chat/completions", json=body,
                                      headers={"Authorization": f"Bearer {API_KEY}"},
                                      timeout=600)
            except httpx.HTTPError as e:
                err = f"transport: {e}"
            else:
                if r.status_code == 200:
                    return r.json()
                err = f"{r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 401, 403, 404, 422):
                    raise RuntimeError(err)
                ra = r.headers.get("retry-after")
                if ra and ra.isdigit():
                    delay = max(delay, float(ra))
        await asyncio.sleep(delay + random.random() * 2)
        delay = min(delay * 2, 90)
    raise RuntimeError(f"gave up after {retries} tries: {err}")


def content_of(resp: dict) -> str:
    return resp["choices"][0]["message"].get("content") or ""


def parse_json(text: str):
    """Parse a JSON object from model output, tolerating code fences and stray text."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise
