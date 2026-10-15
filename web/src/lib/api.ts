import type { Fact, FactGrade, Level, PatientVersion, PrepareEvent, Step, Topic } from "../../shared/types.ts";

export class ApiError extends Error {
  code: string;
  constructor(code: string, message: string) {
    super(message);
    this.code = code;
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  } catch {
    throw new ApiError("network", "Could not reach Homeward. Check your connection and try again.");
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(data?.error ?? "server", data?.message ?? `Request failed (${res.status}).`);
  return data as T;
}

export interface Config {
  live: boolean;
  onprem_grader: boolean;
  onprem_label: string;
  asr: "onprem" | "mimo" | "browser";
  host: string;
}

export async function getConfig(): Promise<Config | null> {
  try {
    const r = await fetch("/api/config");
    return r.ok ? ((await r.json()) as Config) : null;
  } catch {
    return null;
  }
}

export const extractFacts = (note: string) =>
  post<{ facts: Fact[]; topics: Topic[]; nurse_checks: string[]; steps: Step[] }>("/api/extract", { note });

export const ocrFile = (data_url: string) => post<{ text: string; steps: Step[] }>("/api/ocr", { data_url });

export const gradeAnswer = (facts: Fact[], question: string, transcript: string, language: string) =>
  post<{ grades: FactGrade[]; transcript_en: string; steps: Step[] }>("/api/grade", { facts, question, transcript, language });

export const transcribe = (audio_b64: string, language: string) => post<{ text: string; steps: Step[] }>("/api/transcribe", { audio_b64, language });

/** Streams /api/prepare. Calls onEvent for each NDJSON line and resolves with the finished version. */
export async function preparePatientVersion(
  req: { facts: Fact[]; topics: Topic[]; language: string; level: Level; fault?: "dose" | "negation" },
  onEvent: (e: PrepareEvent) => void,
): Promise<PatientVersion> {
  let res: Response;
  try {
    res = await fetch("/api/prepare", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(req) });
  } catch {
    throw new ApiError("network", "Could not reach Homeward. Check your connection and try again.");
  }
  if (!res.ok || !res.body) {
    const data = await res.json().catch(() => null);
    throw new ApiError(data?.error ?? "server", data?.message ?? `Request failed (${res.status}).`);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let version: PatientVersion | null = null;
  for (;;) {
    const { done, value } = await reader.read();
    if (value) buf += dec.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      const e = JSON.parse(line) as PrepareEvent;
      if (e.type === "error") throw new ApiError(e.code ?? "upstream", e.message);
      if (e.type === "done") version = e.version;
      onEvent(e);
    }
    if (done) break;
  }
  if (!version) throw new ApiError("upstream", "The patient version did not finish. Please try again.");
  return version;
}
