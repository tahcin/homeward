// Types shared by the Worker and the browser. The fact and topic shapes match data/cases.jsonl.

export const CATEGORIES = [
  "medication",
  "medication_change",
  "warning_sign",
  "follow_up",
  "activity",
  "diet",
  "self_monitoring",
  "wound_care",
  "device",
] as const;
export type Category = (typeof CATEGORIES)[number];

export interface Fact {
  id: string;
  category: Category;
  critical: boolean;
  statement: string;
  anchors: string[];
  label?: string; // 2-7 word summary for the nurse's board, e.g. "Apixaban: twice a day, 9 am and 9 pm"
}

export interface Topic {
  id: string;
  question: string;
  fact_ids: string[];
}

/** One model call, as shown in the audit trail. */
export interface Step {
  step: string;
  model: string;
  host: string; // "AMD Instinct via Token Factory", or the on-prem label
  ms: number;
  tokens_in?: number;
  tokens_out?: number;
  cost_usd?: number;
  fallback_from?: string; // the model that was tried first and failed
  note?: string;
}

export type MeaningIssue = "negation" | "frequency" | "dose" | "timing" | "action" | "missing_detail" | "none";

export interface AnchorProblem {
  kind: "quote" | "anchor" | "meaning";
  anchor: string;
  reason: string;
}

export interface LockResult {
  id: string;
  status: "locked" | "repaired" | "flagged";
  quote: string;
  quote_en?: string;
  problems: AnchorProblem[];
  rounds: number; // repair rounds this fact needed
  history: { round: number; problems: AnchorProblem[] }[]; // what each failed check found, before repair
}

export const FAULT_KINDS = ["dose", "dose_half", "negation", "frequency", "drop_negation", "date", "drop_symptom", "mistranslate"] as const;
export type FaultKind = (typeof FAULT_KINDS)[number]; // "negation" reverses an action (stop becomes continue)

/** A deliberate error put into the first draft so the lock can be seen catching it. */
export interface Fault {
  kind: FaultKind;
  fact_id: string;
  before: string;
  after: string;
}

export interface PatientVersion {
  language: string;
  level: Level;
  title: string;
  paragraphs: string[];
  back_translation: string[];
  questions: Record<string, string>; // topic id -> question in the patient's language
  lock: LockResult[];
  fault?: Fault;
}

export type Verdict = "correct" | "partial" | "incorrect" | "missing";

export interface FactGrade {
  id: string;
  verdict: Verdict;
  heard: string;
  reteach: string; // English, for the nurse
  reteach_patient?: string; // in the patient's language, checked for invented numbers
}

export interface Attempt {
  topic_id: string;
  transcript: string;
  transcript_en: string;
  input: "voice" | "typed" | "sample";
  grades: FactGrade[];
  replayed?: boolean;
}

export type Level = "very_simple" | "simple" | "standard";

export const LEVELS: { id: Level; label: string; hint: string }[] = [
  { id: "very_simple", label: "Very simple", hint: "Short sentences and everyday words" },
  { id: "simple", label: "Simple", hint: "Plain language, a little more detail" },
  { id: "standard", label: "Standard", hint: "Full detail for confident readers" },
];

export interface Language {
  code: string;
  name: string; // English name
  native: string;
  bcp47: string; // for speech synthesis and recognition
  rtl?: boolean;
}

export const LANGUAGES: Language[] = [
  { code: "it", name: "Italian", native: "Italiano", bcp47: "it-IT" },
  { code: "es", name: "Spanish", native: "Español", bcp47: "es-ES" },
  { code: "hi", name: "Hindi", native: "हिन्दी", bcp47: "hi-IN" },
  { code: "bn", name: "Bengali", native: "বাংলা", bcp47: "bn-IN" },
  { code: "tl", name: "Tagalog", native: "Tagalog", bcp47: "fil-PH" },
  { code: "ar", name: "Arabic", native: "العربية", bcp47: "ar-SA", rtl: true },
  { code: "zh", name: "Mandarin Chinese", native: "中文", bcp47: "zh-CN" },
  { code: "pt", name: "Portuguese", native: "Português", bcp47: "pt-BR" },
  { code: "fr", name: "French", native: "Français", bcp47: "fr-FR" },
  { code: "ta", name: "Tamil", native: "தமிழ்", bcp47: "ta-IN" },
  { code: "vi", name: "Vietnamese", native: "Tiếng Việt", bcp47: "vi-VN" },
  { code: "en", name: "English", native: "English", bcp47: "en-US" },
];

export const langByCode = (code: string): Language => LANGUAGES.find((l) => l.code === code) ?? LANGUAGES[LANGUAGES.length - 1];

/** Events streamed by /api/prepare as NDJSON, one per line. */
export type PrepareEvent =
  | { type: "progress"; stage: "write" | "lock" | "repair" | "translate"; message: string }
  | { type: "step"; step: Step }
  | { type: "ping" } // heartbeat so long, quiet stretches do not look like a dead connection
  | { type: "done"; version: PatientVersion }
  | { type: "error"; message: string; code?: string; detail?: string };

export interface ApiError {
  error: string;
  message: string;
}
