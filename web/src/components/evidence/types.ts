// Shapes of the two JSON files the Evidence page reads at runtime.

export interface SystemResult {
  id: string;
  role: "ours" | "base" | "onsite" | "cloud";
  label: string;
  model: string | null;
  run: string | null;
  paired_vs_base: { base: string; fixes: number; breaks: number; p: number } | null;
  latency_bs1_p50_s: number | null;
  usd_per_discharge: number | null;
  runs_on: string | null;
  accuracy: number | null;
  accuracy_ci: [number, number] | null;
  macro_f1: number | null;
  false_reassurance: number | null;
  false_reassurance_ci: [number, number] | null;
  false_reassurance_n: [number, number] | null;
  critical_false_reassurance_n: [number, number] | null;
  over_flag: number | null;
  json_valid: number | null;
  ci_method: string | null;
}
export interface TestSet {
  answers: number | null;
  facts: number | null;
  cases: number | null;
  languages: string[];
  labels: string | null;
  systems: SystemResult[];
}
export interface WerCell {
  clips: number;
  words: number;
  word_edits: number;
  wer: number | null;
  wer_normalised: number | null;
  cer: number | null;
  cer_normalised?: number | null;
}
export interface GradeFromTranscript {
  answers: number;
  facts: number;
  correct_vs_gold: number;
  false_reassurance: number;
  bad_facts: number;
  critical_false_reassurance: number;
  critical_bad_facts: number;
}
export interface Ours {
  system_key: string;
  label: string;
  model: string;
  run: string | null;
  description: string;
  base_key: string | null;
  lora: { rank: number; alpha: number; epochs: number } | null;
  asr_system: string;
  asr_label: string;
}
export interface Results {
  generated_at: string | null;
  ours: Ours;
  grader: { independent: TestSet | null; in_distribution: TestSet | null };
  asr: {
    clips: { tts: number; human: number } | null;
    tts_note: string | null;
    normalisation: string | null;
    wer: Record<string, Record<string, Record<string, WerCell>>>;
    latency_s: Record<string, { p50: number; p95: number; n: number }> | null;
    grading_from_transcripts: Record<string, GradeFromTranscript> | null;
  };
  onprem_loop: { clips: number | null; p50_s: number | null; p95_s: number | null; asr_p50_s: number | null; grade_p50_s: number | null; peak_vram_gib: number | null };
  capacity: {
    by_batch: { batch: number; per_hour: number; peak_vram_gib: number | null }[];
    discharges_per_hour_per_card: number | null;
    grading_only_discharges_per_hour_per_card: number | null;
    answers_per_discharge: number | null;
    assumptions: string | null;
  };
  training: { device: string; examples: number; minutes: number | null; peak_vram_gib: number | null } | null;
}

export interface Source {
  title: string;
  publisher: string;
  url: string;
  accessed: string;
}
export interface BenchRow {
  model: string;
  family: "gemma4" | "gemma3" | "other";
  params?: string;
  value: number;
  source: string;
  note?: string;
}
export interface BenchPanel {
  id: string;
  benchmark: string;
  setting: string;
  metric: string;
  higher_is_better: boolean;
  size_class: string;
  rows: BenchRow[];
  note?: string;
  wide?: boolean;
}
export interface FleursRow {
  lang: string;
  metric: "WER" | "CER";
  whisper_large_v2: number | null;
  whisper_large_v3_chart: number | null;
  gemma4_e4b: number | null;
  whisper_turbo_open_asr: number | null;
  note?: string;
}
export interface Benchmarks {
  updated: string;
  note: string;
  llm: { title: string; note: string; panels: BenchPanel[] };
  whisper_fleurs: {
    title: string;
    setting: string;
    note: string;
    v3_note: string;
    sources: Record<"whisper_large_v2" | "whisper_large_v3_chart" | "gemma4_e4b" | "whisper_turbo_open_asr", string>;
    rows: FleursRow[];
  };
  gemma_asr?: { model: string; benchmark: string; setting: string; metric: string; value: number; source: string; note?: string }[];
  hardware: { name: string; facts: { label: string; value: string }[]; source: string; note?: string }[];
  sources: Record<string, Source>;
}

export const LANG: Record<string, string> = {
  ALL: "All languages",
  hi: "Hindi",
  bn: "Bengali",
  es: "Spanish",
  it: "Italian",
  ar: "Arabic",
  zh: "Chinese",
  pt: "Portuguese",
  fr: "French",
  ta: "Tamil",
  vi: "Vietnamese",
  tl: "Tagalog",
  en: "English",
};
