// A small client for AMD Token Factory (OpenAI-compatible) and an optional self-hosted grader.
// Every call walks a model chain: if a model is saturated (503 no_available_workers), removed, or returns
// unusable output, the next model answers and the audit trail records which one actually served.

import type { Step } from "../shared/types.ts";

export interface Env {
  ASSETS: Fetcher;
  GUARD: DurableObjectNamespace<import("./index.ts").Guard>;
  TOKENFACTORY_API_KEY: string;
  TF_BASE_URL?: string;
  IP_SALT?: string;
  LIVE_ENABLED?: string;
  LIMIT_PER_IP_HOUR?: string;
  LIMIT_USD_PER_DAY?: string;
  MIN_UPSTREAM_USD?: string;
  ONPREM_GRADER_URL?: string;
  ONPREM_GRADER_MODEL?: string;
  ONPREM_GRADER_KEY?: string;
  ONPREM_GRADER_LABEL?: string;
  MIMO_AUDIO?: string;
  ONPREM_ASR_URL?: string;
  ONPREM_ASR_KIND?: string;
  ONPREM_ASR_MODEL?: string;
  ONPREM_ASR_KEY?: string;
}

// The Token Factory manual says the shared models run "on AMD GPUs"; it names no GPU model, so neither do we.
export const TF_HOST = "AMD GPUs via AMD Token Factory";
const DEFAULT_BASE = "https://developer.amd.com.cn/radeon/api/v1";

export class BudgetError extends Error {
  code = "budget";
}
export class UpstreamError extends Error {
  code = "upstream";
}

export type ChatContent = string | Array<Record<string, unknown>>;
export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: ChatContent;
}

export type ModelRef = string | { onprem: true };

export interface ChatOpts {
  step: string;
  models: ModelRef[];
  messages: ChatMessage[];
  json?: boolean;
  maxTokens?: number;
  temperature?: number;
  /** Validate parsed JSON; return false to move on to the next model. */
  accept?: (parsed: any) => boolean;
}

export interface ChatResult<T = any> {
  text: string;
  data: T;
  step: Step;
}

/** Collects the audit trail and spend for one request. */
export class Meter {
  steps: Step[] = [];
  usd = 0;
  upstreamRemaining: number | null = null;
  onStep?: (s: Step) => void;
  add(s: Step) {
    this.steps.push(s);
    this.usd += s.cost_usd ?? 0;
    this.onStep?.(s);
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function parseJsonLoose(text: string): any {
  let t = text.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
  try {
    return JSON.parse(t);
  } catch {
    const a = t.indexOf("{");
    const b = t.lastIndexOf("}");
    if (a >= 0 && b > a) return JSON.parse(t.slice(a, b + 1));
    throw new Error("no JSON object in model output");
  }
}

function bodyFor(model: string, o: ChatOpts): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model,
    messages: o.messages,
    max_tokens: o.maxTokens ?? 3000,
    temperature: o.temperature ?? 0.2,
  };
  if (o.json) body.response_format = { type: "json_object" };
  // Thinking is off by default on DeepSeek and Gemma; MiMo and Qwen think unless told not to.
  if (/^(MiMo|Qwen3\.8-Flash)/.test(model)) body.reasoning_effort = "none";
  return body;
}

let onpremHealthy: { ok: boolean; at: number } | null = null;
export async function onpremGraderUp(env: Env): Promise<boolean> {
  if (!env.ONPREM_GRADER_URL) return false;
  if (onpremHealthy && Date.now() - onpremHealthy.at < 60_000) return onpremHealthy.ok;
  let ok = false;
  try {
    const r = await fetch(env.ONPREM_GRADER_URL.replace(/\/$/, "") + "/models", {
      headers: env.ONPREM_GRADER_KEY ? { authorization: `Bearer ${env.ONPREM_GRADER_KEY}` } : {},
      signal: AbortSignal.timeout(2500),
    });
    ok = r.ok;
  } catch {
    ok = false;
  }
  onpremHealthy = { ok, at: Date.now() };
  return ok;
}
export const onpremLabel = (env: Env) => env.ONPREM_GRADER_LABEL || "Fine-tuned Gemma 4 E4B, self-hosted";

interface Target {
  url: string;
  key: string;
  model: string;
  host: string;
  timeoutMs: number;
}

function target(env: Env, ref: ModelRef): Target | null {
  if (typeof ref === "string") {
    return { url: (env.TF_BASE_URL || DEFAULT_BASE) + "/chat/completions", key: env.TOKENFACTORY_API_KEY, model: ref, host: TF_HOST, timeoutMs: 240_000 };
  }
  if (!env.ONPREM_GRADER_URL) return null;
  return {
    url: env.ONPREM_GRADER_URL.replace(/\/$/, "") + "/chat/completions",
    key: env.ONPREM_GRADER_KEY ?? "",
    model: env.ONPREM_GRADER_MODEL || "homeward-grader",
    host: onpremLabel(env),
    timeoutMs: 30_000,
  };
}

export async function chat<T = any>(env: Env, meter: Meter, o: ChatOpts): Promise<ChatResult<T>> {
  const failures: string[] = [];
  for (const ref of o.models) {
    if (typeof ref !== "string" && !(await onpremGraderUp(env))) continue;
    const t = target(env, ref);
    if (!t) continue;
    let attempts = 0;
    let waits = 0;
    while (attempts < 3) {
      attempts++;
      const started = Date.now();
      let res: Response;
      try {
        res = await fetch(t.url, {
          method: "POST",
          headers: { "content-type": "application/json", ...(t.key ? { authorization: `Bearer ${t.key}` } : {}) },
          body: JSON.stringify(bodyFor(t.model, o)),
          signal: AbortSignal.timeout(t.timeoutMs),
        });
      } catch (e) {
        failures.push(`${t.model}: ${(e as Error).name}`);
        break; // timeout or network: try the next model
      }
      const rem = res.headers.get("x-ratelimit-remaining-user-daily-usd");
      if (rem !== null && t.host === TF_HOST) meter.upstreamRemaining = Number(rem);
      if (res.status === 429) {
        const text = await res.text();
        if (/daily|spend|quota|budget|cost/i.test(text) && !/per.?minute|rpm|concurren/i.test(text)) {
          throw new BudgetError("Token Factory daily budget reached");
        }
        // Rate and concurrency limits are shared with the rest of the team: wait our turn rather than fail.
        if (++waits > 10) {
          failures.push(`${t.model}: still rate limited`);
          break;
        }
        attempts--;
        const ra = Number(res.headers.get("retry-after") ?? "2");
        await sleep(Math.min(Math.max(ra, 1, 2 ** Math.min(waits, 3)), 20) * 1000 + Math.random() * 1000);
        continue;
      }
      if (res.status === 401 || res.status === 403) throw new UpstreamError(`Token Factory refused the key (${res.status})`);
      if (!res.ok) {
        const text = (await res.text()).slice(0, 200);
        failures.push(`${t.model}: ${res.status} ${text}`);
        if (res.status >= 500 && attempts < 2 && !/no_available_workers/.test(text)) {
          await sleep(1200);
          continue;
        }
        break;
      }
      const body: any = await res.json();
      const text: string = body?.choices?.[0]?.message?.content ?? "";
      const usage = body?.usage ?? {};
      const step: Step = {
        step: o.step,
        model: t.model,
        host: t.host,
        ms: Date.now() - started,
        tokens_in: usage.prompt_tokens,
        tokens_out: usage.completion_tokens,
        cost_usd: typeof usage.cost === "number" ? usage.cost : undefined,
      };
      let data: any = text;
      if (o.json) {
        try {
          data = parseJsonLoose(text);
          if (o.accept && !o.accept(data)) throw new Error("output failed validation");
        } catch (e) {
          failures.push(`${t.model}: ${(e as Error).message}`);
          meter.add({ ...step, note: "output rejected, trying the next model" });
          break;
        }
      }
      const first = o.models[0];
      const firstName = typeof first === "string" ? first : onpremLabel(env);
      if (failures.length && firstName !== t.model) step.fallback_from = firstName;
      meter.add(step);
      return { text, data, step };
    }
  }
  throw new UpstreamError(`No model could answer (${failures.join(" | ").slice(0, 400)})`);
}
