// Homeward Worker: serves the built site from the asset store and runs /api/* on AMD Token Factory.
// Patient text passes through memory only. Nothing about a request is stored except a salted IP hash and a
// timestamp (for the per-IP limit) and the day's spend total.

import { DurableObject } from "cloudflare:workers";
import { FAULT_KINDS, LANGUAGES, type Fact, type FaultKind, type Level, type PrepareEvent, type Topic } from "../shared/types.ts";
import { extract, grade, labelFacts, ocr, prepare } from "./pipeline.ts";
import { BudgetError, chat, Meter, onpremGraderUp, onpremLabel, TF_HOST, type Env } from "./tf.ts";

// Token Factory's spend period resets at 16:00 UTC (21:30 IST).
const dayKey = (now: number) => Math.floor((now - 16 * 3600_000) / 86_400_000);

interface DayState {
  day: number;
  usd: number;
  calls: number;
}

export class Guard extends DurableObject<Env> {
  sql: SqlStorage;
  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.sql = ctx.storage.sql;
    this.sql.exec("CREATE TABLE IF NOT EXISTS hits (k TEXT NOT NULL, ts INTEGER NOT NULL)");
    this.sql.exec("CREATE INDEX IF NOT EXISTS hits_k ON hits (k)");
  }

  private async state(): Promise<DayState> {
    const now = dayKey(Date.now());
    const s = await this.ctx.storage.get<DayState>("day");
    return s && s.day === now ? s : { day: now, usd: 0, calls: 0 };
  }

  async admit(key: string, perHour: number, usdCap: number): Promise<"ok" | "ip" | "budget"> {
    const now = Date.now();
    this.sql.exec("DELETE FROM hits WHERE ts < ?", now - 3600_000);
    const n = this.sql.exec<{ n: number }>("SELECT COUNT(*) AS n FROM hits WHERE k = ?", key).one().n;
    const s = await this.state();
    if (s.usd >= usdCap) return "budget";
    if (n >= perHour) return "ip";
    this.sql.exec("INSERT INTO hits (k, ts) VALUES (?, ?)", key, now);
    await this.ctx.storage.put("day", { ...s, calls: s.calls + 1 });
    return "ok";
  }

  async record(usd: number, upstreamRemaining: number | null, minUpstream: number): Promise<void> {
    const s = await this.state();
    // If the shared Token Factory key is nearly spent (other team work uses it too), close live calls for the day.
    const closed = upstreamRemaining !== null && upstreamRemaining < minUpstream;
    await this.ctx.storage.put("day", { ...s, usd: closed ? Math.max(s.usd + usd, 1e9) : s.usd + usd });
  }

  async today(): Promise<DayState> {
    return this.state();
  }
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });

const MESSAGES = {
  ip: "You have made a lot of live requests in the last hour. Please wait a little, or try one of the sample patients, which work without live calls.",
  budget:
    "Today's live budget for this demo is used up. It resets at 21:30 IST (16:00 UTC). The three sample patients still work end to end, from a recorded run.",
  off: "Live generation is switched off right now. The three sample patients still work end to end, from a recorded run.",
  upstream: "The AMD model service did not answer just now. Please try again in a minute.",
};

async function ipKey(req: Request, env: Env): Promise<string> {
  const ip = req.headers.get("cf-connecting-ip") ?? "local";
  const data = new TextEncoder().encode((env.IP_SALT ?? "homeward") + ip);
  const hash = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(hash).slice(0, 12)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

const guard = (env: Env) => env.GUARD.get(env.GUARD.idFromName("global"));

async function gate(req: Request, env: Env): Promise<Response | null> {
  if (env.LIVE_ENABLED === "false") return json({ error: "off", message: MESSAGES.off }, 503);
  const origin = req.headers.get("origin");
  if (origin && new URL(origin).host !== new URL(req.url).host) return json({ error: "forbidden", message: "Use Homeward from its own website." }, 403);
  const verdict = await guard(env).admit(await ipKey(req, env), Number(env.LIMIT_PER_IP_HOUR ?? 40), Number(env.LIMIT_USD_PER_DAY ?? 0.5));
  if (verdict === "ok") return null;
  return json({ error: verdict, message: MESSAGES[verdict] }, 429);
}

async function settle(env: Env, meter: Meter, ctx: ExecutionContext) {
  ctx.waitUntil(guard(env).record(meter.usd, meter.upstreamRemaining, Number(env.MIN_UPSTREAM_USD ?? 0.05)));
}

function failure(e: unknown): Response {
  if (e instanceof BudgetError) return json({ error: "budget", message: MESSAGES.budget }, 429);
  return json({ error: "upstream", message: MESSAGES.upstream, detail: String((e as Error)?.message ?? e).slice(0, 300) }, 502);
}

async function body<T>(req: Request, max: number): Promise<T | null> {
  const text = await req.text();
  if (text.length > max) return null;
  try {
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
}

const isFacts = (x: unknown): x is Fact[] =>
  Array.isArray(x) && x.length > 0 && x.length <= 24 && x.every((f: any) => typeof f?.id === "string" && typeof f?.statement === "string" && Array.isArray(f?.anchors));
const isTopics = (x: unknown): x is Topic[] => Array.isArray(x) && x.length <= 10 && x.every((t: any) => typeof t?.id === "string" && Array.isArray(t?.fact_ids));
const LEVEL_IDS: Level[] = ["very_simple", "simple", "standard"];

async function api(req: Request, env: Env, ctx: ExecutionContext, path: string): Promise<Response> {
  if (path === "/api/config" && req.method === "GET") {
    const day = await guard(env).today();
    const cap = Number(env.LIMIT_USD_PER_DAY ?? 0.5);
    return json({
      live: env.LIVE_ENABLED !== "false" && day.usd < cap,
      onprem_grader: await onpremGraderUp(env),
      onprem_label: onpremLabel(env),
      asr: env.ONPREM_ASR_URL ? "onprem" : env.MIMO_AUDIO === "true" ? "mimo" : "browser",
      host: TF_HOST,
    });
  }
  if (req.method !== "POST") return json({ error: "not_found", message: "Not found" }, 404);

  if (path === "/api/extract") {
    const b = await body<{ note: string }>(req, 40_000);
    const note = b?.note?.trim() ?? "";
    if (note.length < 40) return json({ error: "bad_request", message: "Paste a discharge note of at least a few lines." }, 400);
    if (note.length > 20_000) return json({ error: "bad_request", message: "That note is too long. Paste up to about 20,000 characters." }, 400);
    const blocked = await gate(req, env);
    if (blocked) return blocked;
    const meter = new Meter();
    try {
      const out = await extract(env, meter, note);
      return json({ ...out, steps: meter.steps });
    } catch (e) {
      return failure(e);
    } finally {
      await settle(env, meter, ctx);
    }
  }

  if (path === "/api/labels") {
    const b = await body<{ facts: Fact[] }>(req, 40_000);
    if (!b || !isFacts(b.facts)) return json({ error: "bad_request", message: "Something is missing from the request." }, 400);
    const blocked = await gate(req, env);
    if (blocked) return blocked;
    const meter = new Meter();
    try {
      return json({ labels: await labelFacts(env, meter, b.facts), steps: meter.steps });
    } catch (e) {
      return failure(e);
    } finally {
      await settle(env, meter, ctx);
    }
  }

  if (path === "/api/ocr") {
    const b = await body<{ data_url: string }>(req, 8_000_000);
    const url = b?.data_url ?? "";
    if (!/^data:(application\/pdf|image\/(png|jpe?g|webp));base64,/.test(url)) return json({ error: "bad_request", message: "Upload a PDF, PNG, JPEG or WebP file under 5 MB." }, 400);
    const blocked = await gate(req, env);
    if (blocked) return blocked;
    const meter = new Meter();
    try {
      const text = await ocr(env, meter, url);
      if (!text) return json({ error: "empty", message: "No text was found in that file. Try a clearer scan, or paste the text." }, 422);
      return json({ text, steps: meter.steps });
    } catch (e) {
      return failure(e);
    } finally {
      await settle(env, meter, ctx);
    }
  }

  if (path === "/api/prepare") {
    const b = await body<{ facts: Fact[]; topics: Topic[]; language: string; level: Level; fault?: FaultKind }>(req, 60_000);
    if (!b || !isFacts(b.facts) || !isTopics(b.topics) || !LANGUAGES.some((l) => l.code === b.language) || !LEVEL_IDS.includes(b.level)) {
      return json({ error: "bad_request", message: "Something is missing from the request. Go back to the facts and try again." }, 400);
    }
    const blocked = await gate(req, env);
    if (blocked) return blocked;
    const meter = new Meter();
    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();
    const enc = new TextEncoder();
    const emit = (e: PrepareEvent) => writer.write(enc.encode(JSON.stringify(e) + "\n")).catch(() => {});
    meter.onStep = (step) => emit({ type: "step", step });
    const heartbeat = setInterval(() => emit({ type: "ping" }), 15_000);
    const work = (async () => {
      try {
        const version = await prepare(env, meter, b.facts, b.topics, b.language, b.level, emit, b.fault && (FAULT_KINDS as readonly string[]).includes(b.fault) ? b.fault : undefined);
        emit({ type: "done", version });
      } catch (e) {
        emit({
          type: "error",
          code: e instanceof BudgetError ? "budget" : "upstream",
          message: e instanceof BudgetError ? MESSAGES.budget : MESSAGES.upstream,
          detail: String((e as Error)?.message ?? e).slice(0, 300),
        });
      } finally {
        clearInterval(heartbeat);
        await settle(env, meter, ctx);
        await writer.close().catch(() => {});
      }
    })();
    ctx.waitUntil(work);
    return new Response(readable, { headers: { "content-type": "application/x-ndjson; charset=utf-8", "cache-control": "no-store" } });
  }

  if (path === "/api/grade") {
    const b = await body<{ facts: Fact[]; question: string; transcript: string; language: string }>(req, 40_000);
    const transcript = b?.transcript?.trim() ?? "";
    if (!b || !isFacts(b.facts) || typeof b.question !== "string" || !LANGUAGES.some((l) => l.code === b.language)) {
      return json({ error: "bad_request", message: "Something is missing from the request." }, 400);
    }
    if (!transcript || transcript.length > 4000) return json({ error: "bad_request", message: "Say or type an answer first (up to about 4,000 characters)." }, 400);
    const blocked = await gate(req, env);
    if (blocked) return blocked;
    const meter = new Meter();
    try {
      const out = await grade(env, meter, b.facts, b.question, transcript, b.language);
      return json({ ...out, steps: meter.steps });
    } catch (e) {
      return failure(e);
    } finally {
      await settle(env, meter, ctx);
    }
  }

  if (path === "/api/transcribe") {
    const b = await body<{ audio_b64: string; language: string }>(req, 4_500_000);
    if (!b?.audio_b64) return json({ error: "bad_request", message: "No audio was received." }, 400);
    if (!env.ONPREM_ASR_URL && env.MIMO_AUDIO !== "true") {
      return json({ error: "asr_unavailable", message: "Server transcription is not configured; the browser's own speech recognition is used." }, 501);
    }
    const blocked = await gate(req, env);
    if (blocked) return blocked;
    const meter = new Meter();
    const started = Date.now();
    const lang = LANGUAGES.find((l) => l.code === b.language);
    if (!env.ONPREM_ASR_URL) {
      // MiMo-V2.6-Flash audio input, documented by Token Factory but rejected upstream as of 2026-10-15, so it is off by default.
      try {
        const r = await chat(env, meter, {
          step: "Transcribe the answer",
          models: ["MiMo-V2.6-Flash"],
          maxTokens: 800,
          temperature: 0,
          messages: [
            { role: "system", content: `Transcribe the speech exactly as spoken, in ${lang?.name ?? "the language spoken"}, in its own script. Output only the transcript.` },
            { role: "user", content: [{ type: "text", text: "Transcribe this answer." }, { type: "input_audio", input_audio: { data: b.audio_b64, format: "wav" } }] },
          ],
        });
        return json({ text: r.text.trim(), steps: meter.steps });
      } catch (e) {
        return json({ error: "asr_failed", message: "Transcription failed; type the answer instead.", detail: String(e).slice(0, 200) }, 502);
      } finally {
        await settle(env, meter, ctx);
      }
    }
    try {
      if (env.ONPREM_ASR_KIND !== "whisper") {
        // Chat-completions with an input_audio part, as served for Gemma 4 E4B by an OpenAI-compatible server.
        const r = await fetch(env.ONPREM_ASR_URL.replace(/\/$/, "") + "/chat/completions", {
          method: "POST",
          headers: { "content-type": "application/json", ...(env.ONPREM_ASR_KEY ? { authorization: `Bearer ${env.ONPREM_ASR_KEY}` } : {}) },
          body: JSON.stringify({
            model: env.ONPREM_ASR_MODEL || "gemma-4-e4b",
            temperature: 0,
            max_tokens: 800,
            messages: [
              {
                role: "user",
                content: [
                  { type: "text", text: `Transcribe this speech exactly as spoken, in ${lang?.name ?? "the language spoken"}, in its own script. Output only the transcript.` },
                  { type: "input_audio", input_audio: { data: b.audio_b64, format: "wav" } },
                ],
              },
            ],
          }),
          signal: AbortSignal.timeout(60_000),
        });
        if (!r.ok) throw new Error(`ASR ${r.status}`);
        const out: any = await r.json();
        meter.add({ step: "Transcribe the answer", model: env.ONPREM_ASR_MODEL || "gemma-4-e4b", host: "Self-hosted, on site", ms: Date.now() - started });
        return json({ text: String(out?.choices?.[0]?.message?.content ?? "").trim(), steps: meter.steps });
      }
      const wav = Uint8Array.from(atob(b.audio_b64), (c) => c.charCodeAt(0));
      const form = new FormData();
      form.append("file", new Blob([wav], { type: "audio/wav" }), "answer.wav");
      form.append("model", env.ONPREM_ASR_MODEL || "whisper");
      if (lang) form.append("language", lang.code === "tl" ? "tl" : lang.code);
      const r = await fetch(env.ONPREM_ASR_URL.replace(/\/$/, "") + "/audio/transcriptions", {
        method: "POST",
        body: form,
        headers: env.ONPREM_ASR_KEY ? { authorization: `Bearer ${env.ONPREM_ASR_KEY}` } : {},
        signal: AbortSignal.timeout(60_000),
      });
      if (!r.ok) throw new Error(`ASR ${r.status}`);
      const out: any = await r.json();
      meter.add({ step: "Transcribe the answer", model: env.ONPREM_ASR_MODEL || "whisper", host: "Self-hosted speech recognition", ms: Date.now() - started });
      return json({ text: String(out?.text ?? "").trim(), steps: meter.steps });
    } catch (e) {
      return json({ error: "asr_failed", message: "Transcription failed; type the answer or use the browser's speech recognition.", detail: String(e).slice(0, 200) }, 502);
    }
  }

  return json({ error: "not_found", message: "Not found" }, 404);
}

export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname.startsWith("/api/")) {
      try {
        return await api(req, env, ctx, url.pathname);
      } catch (e) {
        return json({ error: "server", message: "Something went wrong on our side. Please try again.", detail: String(e).slice(0, 200) }, 500);
      }
    }
    if (url.pathname.startsWith("/assets/")) {
      // A missing hashed file (an old page after a deploy) must be a real 404, not the app's HTML, so the page can reload.
      const res = await env.ASSETS.fetch(req);
      if (!res.ok || (res.headers.get("content-type") ?? "").includes("text/html")) return new Response("Not found", { status: 404 });
      return res;
    }
    return env.ASSETS.fetch(req);
  },
} satisfies ExportedHandler<Env>;
