// One discharge check, held in this browser tab only (sessionStorage). Audio clips stay in memory and are
// never uploaded unless a server transcriber is configured.

import type { Attempt, Fact, FactGrade, Level, PatientVersion, Step, Topic } from "../../shared/types.ts";

export type Stage = "note" | "facts" | "version" | "teach" | "board";
export const STAGES: { id: Stage; label: string }[] = [
  { id: "note", label: "Discharge note" },
  { id: "facts", label: "Facts to check" },
  { id: "version", label: "Patient version" },
  { id: "teach", label: "Teach-back" },
  { id: "board", label: "Results" },
];

export type AuditStep = Step & { replayed?: boolean; scope?: string }; // scope: `${topic}:${try}` for teach-back steps

export interface SampleAnswer {
  transcript: string;
  fact_ids: string[];
  grades: FactGrade[];
  transcript_en: string;
  steps: Step[];
}

export interface Sample {
  id: string;
  patient: string;
  condition: string;
  recorded_at: string;
  note: string;
  facts: Fact[];
  topics: Topic[];
  nurse_checks: string[];
  version: PatientVersion;
  steps: Step[];
  answers: Record<string, SampleAnswer[]>;
}

export interface Session {
  source: "sample" | "own";
  sample?: { id: string; patient: string; condition: string; recorded_at: string; signature: string };
  note: string;
  facts: Fact[];
  topics: Topic[];
  nurse_checks: string[];
  factsConfirmed: boolean;
  language: string;
  level: Level;
  version?: PatientVersion;
  versionReplayed?: boolean;
  acknowledged: string[]; // flagged facts the nurse reviewed and accepted
  attempts: Attempt[];
  steps: AuditStep[];
  times: { started: number; factsShown?: number; factsConfirmed?: number; decided?: number };
  stage: Stage;
  teach?: { idx: number; attempt: number }; // where the teach-back is, so a reload resumes there
}

export const newSession = (): Session => ({
  source: "own",
  note: "",
  facts: [],
  topics: [],
  nurse_checks: [],
  factsConfirmed: false,
  language: "it",
  level: "simple",
  acknowledged: [],
  attempts: [],
  steps: [],
  times: { started: Date.now() },
  stage: "note",
});

/** Facts and topics as a string, to tell whether the nurse changed a sample's facts (then the version is regenerated live). */
export const signature = (facts: Fact[], topics: Topic[]) => JSON.stringify([facts, topics.map((t) => [t.id, t.fact_ids])]);

export function sessionFromSample(s: Sample): Session {
  return {
    ...newSession(),
    source: "sample",
    sample: { id: s.id, patient: s.patient, condition: s.condition, recorded_at: s.recorded_at, signature: signature(s.facts, s.topics) },
    note: s.note,
    facts: s.facts,
    topics: s.topics,
    nurse_checks: s.nurse_checks,
    language: s.version.language,
    level: s.version.level,
    steps: s.steps.filter((x) => x.step === "Find the facts").map((x) => ({ ...x, replayed: true })),
    times: { started: Date.now(), factsShown: Date.now() },
    stage: "facts",
  };
}

const KEY = "homeward.session.v1";
export function loadSession(): Session | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}
export function saveSession(s: Session | null) {
  try {
    if (s) sessionStorage.setItem(KEY, JSON.stringify(s));
    else sessionStorage.removeItem(KEY);
  } catch {
    /* private mode: the check still works, it just won't survive a reload */
  }
}

const sampleCache = new Map<string, Sample>();
export async function loadSample(id: string): Promise<Sample> {
  const hit = sampleCache.get(id);
  if (hit) return hit;
  const r = await fetch(`/samples/${id}.json`);
  if (!r.ok) throw new Error("Could not load the sample.");
  const s = (await r.json()) as Sample;
  sampleCache.set(id, s);
  return s;
}

/** Audio of each answer, for the nurse to replay. Memory only. */
export const clips = new Map<string, string>(); // key `${topic}:${attempt}` -> object URL

/** Verdict after all attempts for each fact: the latest grade wins. */
export function finalGrades(s: Session): Map<string, { grade: FactGrade; history: FactGrade[]; attempts: Attempt[] }> {
  const out = new Map<string, { grade: FactGrade; history: FactGrade[]; attempts: Attempt[] }>();
  for (const a of s.attempts) {
    for (const g of a.grades) {
      const cur = out.get(g.id) ?? { grade: g, history: [], attempts: [] };
      cur.history.push(g);
      cur.attempts.push(a);
      cur.grade = g;
      out.set(g.id, cur);
    }
  }
  return out;
}

export function fmtDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${String(s % 60).padStart(2, "0")} s`;
}
