// The Homeward pipeline: extract facts, write the patient version, lock every fact, repair, grade teach-back.
// Gemma 4 31B does the writing and grading. An independent model (DeepSeek V4 Flash) does the checking, so the
// writer never marks its own work. Models propose; the code in lock.ts and here decides what counts as carried.

import grader from "../../prompts/grader.json";
import { CATEGORIES, langByCode } from "../shared/types.ts";
import type { AnchorProblem, Fact, FactGrade, Fault, Level, LockResult, MeaningIssue, PatientVersion, PrepareEvent, Topic, Verdict } from "../shared/types.ts";
import { backTranslationProblem, bracketNames, inventedNumbers, numberTokens, splitLeaflet, strayEnglish, verifyFact } from "./lock.ts";
import { HIGH_RISK_HINT, highRiskGroup } from "./rules.ts";
import { cleanAnchors, compareMed, missingItems, notEnglishComplaint, tidyDuplicateBrackets, type ItemCheck, type MedTuple } from "./structured.ts";
import { chat, TF_HOST, type Env, type Meter, type ModelRef } from "./tf.ts";

const GEMMA = "Gemma-4-31B-IT";
const DEEPSEEK = "DeepSeek-V4-Flash";
const WRITER: ModelRef[] = [GEMMA, DEEPSEEK]; // DeepSeek only answers if Gemma fails; the audit trail says so
// The checker never falls back to the writer (a checker that is the writer is no check). Qwen3.8-Flash-Next was a
// fallback here until it proved to misread Devanagari vowel signs; if DeepSeek cannot answer, the check fails and the
// fact goes to the nurse.
const CHECKER: ModelRef[] = [DEEPSEEK];
const GRADER: ModelRef[] = [{ onprem: true }, GEMMA, DEEPSEEK];

const MAX_REPAIRS = 2;

// ---------------------------------------------------------------- extraction

const EXTRACT_SYSTEM = `You help a discharge nurse. From a clinician's discharge note, list the facts the patient must understand before going home. You only restate what the note says: never add advice, doses, reasons or warnings that are not in the note. Reply with one JSON object only.`;

const EXTRACT_USER = `Discharge note:
"""
{note}
"""

Return JSON with exactly these keys:
- "facts": 5-14 facts. Each is {"id": "f1", "category": one of ${JSON.stringify(CATEGORIES)}, "critical": true if misunderstanding it could cause serious harm (always true for any new, changed or continued ${HIGH_RISK_HINT}), "statement": one plain-English sentence the patient could follow, "label": a 3-9 word summary a nurse can scan: the medicine or topic first, then the action (start, stop, change, call) and the key dose, times or threshold (for example "Apixaban: twice a day, 9 am and 9 pm" or "Call 112: bleeding, head injury"), "anchors": tokens that must survive any translation, copied character for character from the statement}.
  Anchors are only: medicine names (always, for medicine facts), numbers with their units (doses, weights, thresholds, durations), times, dates, and names of people. Never use ordinary words, body parts, symptoms, frequency words ("twice a day") or clinic names as anchors.
  One fact per medicine (dose, timing and any change together; say clearly if it is new, changed, paused or stopped, and for a dose change give both the old and the new dose), one per group of warning signs (with what to do and whom to call), one per appointment or test, one per limit or self-check.
  Cover every medicine, warning sign, follow-up and limit in the note. Write statements with no abbreviations (BID = twice a day, PRN = only when needed, SOB = shortness of breath, HF = heart failure).
- "topics": 2-5 teach-back topics. Each is {"id": "t1", "question": an open question in plain words that a nurse asks so the patient explains in their own words, "fact_ids": the facts it checks}. Every fact belongs to exactly one topic. Group related facts (medicines together, warning signs together, appointments together).
  A question must never give the answer away: no doses, numbers, times, dates, thresholds or medicine names in it. Naming the topic in everyday words is fine, for example "Tell me how you will take your medicines from now on.", "What will you do if your weight goes up quickly?", "When is your next appointment, and what is it for?"
- "nurse_checks": up to 3 short notes about anything in the note that is unclear, contradictory or missing (for example a medicine with no dose). Empty list if none.`;

export interface Extracted {
  facts: Fact[];
  topics: Topic[];
  nurse_checks: string[];
}

const SAFE_QUESTIONS: Partial<Record<string, string>> = {
  medication: "Tell me how you will take your medicines from now on.",
  medication_change: "Tell me how you will take your medicines from now on.",
  warning_sign: "Tell me what you will do if you feel unwell at home.",
  follow_up: "Tell me about your next appointments and tests.",
};

/** A question that contains a number or a medicine name gives the answer away; swap it for a neutral one. */
export function safeQuestion(question: string, facts: Fact[]): string {
  const names = facts.flatMap((f) => f.anchors).filter((a) => !/\d/.test(a) && a.length > 3);
  const leaks = /\d/.test(question) || names.some((n) => question.toLowerCase().includes(n.toLowerCase()));
  if (!leaks) return question;
  return SAFE_QUESTIONS[facts[0]?.category ?? ""] ?? "Tell me in your own words what you need to remember about this when you get home.";
}

/** Make model output safe to show: valid categories, anchors that really occur in the statement, full topic coverage. */
export function tidyCase(raw: any): Extracted {
  const facts: Fact[] = [];
  const seen = new Set<string>();
  for (const f of Array.isArray(raw?.facts) ? raw.facts : []) {
    const statement = String(f?.statement ?? "").trim();
    if (!statement) continue;
    let id = String(f?.id ?? "").trim() || `f${facts.length + 1}`;
    while (seen.has(id)) id = `f${facts.length + 1}_${seen.size}`;
    seen.add(id);
    const category = (CATEGORIES as readonly string[]).includes(f?.category) ? f.category : "medication";
    const lower = statement.toLowerCase();
    const rawAnchors: string[] = (Array.isArray(f?.anchors) ? f.anchors : []).map((a: unknown) => String(a).trim());
    const anchors = cleanAnchors(statement, [...new Set(rawAnchors)].filter((a) => a && lower.includes(a.toLowerCase())));
    const label = typeof f?.label === "string" && f.label.trim() ? f.label.trim().slice(0, 80) : undefined;
    // High-risk medicine classes are critical by default, whatever the model said; the nurse can still untick them.
    const critical = Boolean(f?.critical) || highRiskGroup(category, statement) !== null;
    facts.push({ id, category, critical, statement, anchors, ...(label ? { label } : {}) });
  }
  const byId = new Map(facts.map((f) => [f.id, f]));
  const used = new Set<string>();
  const topics: Topic[] = [];
  for (const t of Array.isArray(raw?.topics) ? raw.topics : []) {
    const fact_ids = (Array.isArray(t?.fact_ids) ? t.fact_ids : []).map(String).filter((id: string) => byId.has(id) && !used.has(id));
    const question = String(t?.question ?? "").trim();
    if (!fact_ids.length || !question) continue;
    fact_ids.forEach((id: string) => used.add(id));
    topics.push({ id: `t${topics.length + 1}`, question: safeQuestion(question, fact_ids.map((id: string) => byId.get(id)!)), fact_ids });
  }
  const orphans = facts.filter((f) => !used.has(f.id)).map((f) => f.id);
  if (orphans.length) topics.push({ id: `t${topics.length + 1}`, question: "Tell me what else you need to remember when you get home.", fact_ids: orphans });
  const nurse_checks = (Array.isArray(raw?.nurse_checks) ? raw.nurse_checks : []).map(String).filter(Boolean).slice(0, 3);
  return { facts, topics, nurse_checks };
}

export async function extract(env: Env, meter: Meter, note: string): Promise<Extracted> {
  const r = await chat(env, meter, {
    step: "Find the facts",
    models: [GEMMA, DEEPSEEK],
    json: true,
    maxTokens: 4000,
    temperature: 0.1,
    messages: [
      { role: "system", content: EXTRACT_SYSTEM },
      { role: "user", content: EXTRACT_USER.replace("{note}", note) },
    ],
    accept: (d) => Array.isArray(d?.facts) && d.facts.length > 0,
  });
  return tidyCase(r.data);
}

/** Short nurse labels for facts that have none (used for facts written before labels existed). */
export async function labelFacts(env: Env, meter: Meter, facts: Fact[]): Promise<Record<string, string>> {
  const r = await chat(env, meter, {
    step: "Label the facts",
    models: [GEMMA, DEEPSEEK],
    json: true,
    maxTokens: 1200,
    temperature: 0,
    messages: [
      {
        role: "system",
        content: `For each discharge fact, write a 3-9 word label a busy nurse can scan: the medicine or topic first, then the action (start, stop, change, call) and the key dose, times or threshold, using only words and numbers from the fact (for example "Apixaban: twice a day, 9 am and 9 pm", "Call 112: bleeding, head injury"). Reply with JSON only: {"f1": "..."}`,
      },
      { role: "user", content: facts.map((f) => `- ${f.id}: ${f.statement}`).join("\n") },
    ],
    accept: (d) => d && typeof d === "object",
  });
  const out: Record<string, string> = {};
  for (const f of facts) if (typeof r.data?.[f.id] === "string" && inventedNumbers(r.data[f.id], f).length === 0) out[f.id] = r.data[f.id].trim().slice(0, 80);
  return out;
}

// ---------------------------------------------------------------- patient version

const LEVEL_RULES: Record<Level, string> = {
  very_simple: "Very short sentences of about 10 words. Only everyday words a 9-year-old knows. No medical terms except the medicine names.",
  simple: "Short sentences and everyday words. If you must use a medical word, explain it in a few words.",
  standard: "Clear adult language with full detail.",
};

const isDate = (a: string) => /\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\b/i.test(a) && /\d/.test(a);
const UNIT_WORDS = /^(mg|mcg|µg|g|kg|ml|l|mmhg|mmol|iu|lb|lbs|cm|mm|°c|°f|%)$/i;
/** Tells the writer which part of an anchor must stay as written and which words it should translate. */
function anchorHint(a: string): string {
  if (isDate(a)) return " (a date: keep the numbers, write the month in the patient's language)";
  if (!/\d/.test(a)) return "";
  const words = a.split(/[\s/]+/).filter((w) => /[a-z]/i.test(w) && !UNIT_WORDS.test(w.replace(/^\d+/, "")));
  return words.length ? ` (keep the numbers and units; translate ${words.map((w) => `"${w}"`).join(", ")})` : "";
}

const factLines = (facts: Fact[]) =>
  facts
    .map(
      (f) =>
        `- ${f.id}${f.critical ? " (critical)" : ""}: ${f.statement}\n  anchors: ${
          f.anchors.length ? f.anchors.map((a) => JSON.stringify(a) + anchorHint(a)).join(", ") : "(none)"
        }`,
    )
    .join("\n");

function writerSystem(langName: string, level: Level): string {
  return `You write hospital discharge instructions for a patient in ${langName}. You only restate the facts you are given; you never add advice, reasons, doses or warnings of your own.
Rules:
1. Write in ${langName}, speaking to the patient as "you". ${LEVEL_RULES[level]}
2. Every fact must be stated in its own sentence (or two), complete enough to act on.
3. Inside the sentence for each fact, keep its anchors safe:
   - Medicine names: if ${langName} has its own name for the medicine, write that first and the name from the box label in brackets, in Latin letters exactly as given (for example "paracetamolo (paracetamol)"); if the names are the same (ignoring capital letters), write it once, as given, with no brackets.
   - People: keep their names as written; use the usual ${langName} title for a doctor (for example "Dott.ssa" or "Dott." in Italian, "Dra." or "Dr." in Spanish).
   - Numbers: digits 0-9 with the same value (40, 10:30), written with the decimal separator ${langName} uses (2,5 in Italian, Spanish, Portuguese, French and Vietnamese; 2.5 otherwise).
   - Units exactly as written (mg, ml, L, g, kg, mmHg, °C).
   Translate every other word into ${langName}, including words like days, weeks, daily, minutes, morning, am/pm, clinic names and month names (2 pm may become 14:00).
4. For a change (new, stopped, paused, higher or lower dose) say clearly what changes.
5. Group sentences into short paragraphs of at most 3 sentences, ordered by theme: medicines, warning signs and what to do, appointments and tests, daily life. Start with one short sentence saying these are the instructions for going home.
6. Also translate each teach-back question into ${langName} for the patient, keeping it open: do not add doses, numbers, times or medicine names to it.
Reply with JSON only: {"title": "short title in ${langName}", "paragraphs": ["...", "..."], "questions": {"t1": "..."}}`;
}

interface Draft {
  title: string;
  paragraphs: string[];
}
const acceptDraft = (d: any) => typeof d?.title === "string" && Array.isArray(d?.paragraphs) && d.paragraphs.length > 0 && d.paragraphs.every((p: unknown) => typeof p === "string");
const tidyDraft = (d: any): Draft => ({ title: String(d.title).trim(), paragraphs: d.paragraphs.map((p: string) => tidyDuplicateBrackets(p.trim())).filter(Boolean) });

async function write(env: Env, meter: Meter, facts: Fact[], topics: Topic[], langName: string, level: Level) {
  const r = await chat(env, meter, {
    step: "Write the patient version",
    models: WRITER,
    json: true,
    maxTokens: 5000,
    temperature: 0.3,
    messages: [
      { role: "system", content: writerSystem(langName, level) },
      { role: "user", content: `Facts:\n${factLines(facts)}\n\nTeach-back questions:\n${JSON.stringify(Object.fromEntries(topics.map((t) => [t.id, t.question])))}` },
    ],
    accept: acceptDraft,
  });
  const questions: Record<string, string> = {};
  for (const t of topics) questions[t.id] = typeof r.data?.questions?.[t.id] === "string" ? r.data.questions[t.id].trim() : "";
  return { draft: tidyDraft(r.data), questions };
}

const LOCK_SYSTEM = `You check a patient leaflet against a list of facts. The leaflet is given as numbered sentences. For each fact, list the ids of all the sentences that together tell the patient that fact. A list of warning signs, or a dose and its timing, is often split over two or three sentences: include every one of them. If no sentence tells the patient that fact, return an empty list. Reply with JSON only: {"quotes": [{"id": "f1", "sentences": ["s2"]}]} with every fact id in the given order.`;

async function lockQuotes(env: Env, meter: Meter, facts: Fact[], sentences: { id: string; text: string }[], label: string) {
  const r = await chat(env, meter, {
    step: label,
    models: CHECKER,
    json: true,
    maxTokens: 1500,
    temperature: 0,
    messages: [
      { role: "system", content: LOCK_SYSTEM },
      { role: "user", content: `Facts:\n${factLines(facts)}\n\nLeaflet:\n${sentences.map((x) => `[${x.id}] ${x.text}`).join("\n")}` },
    ],
    accept: (d) => Array.isArray(d?.quotes),
  });
  const valid = new Set(sentences.map((x) => x.id));
  const out: Record<string, string[]> = {};
  for (const q of r.data.quotes) {
    if (q && typeof q.id === "string") out[q.id] = (Array.isArray(q.sentences) ? q.sentences : []).map(String).filter((id: string) => valid.has(id));
  }
  return out;
}

const ISSUES: MeaningIssue[] = ["negation", "frequency", "dose", "timing", "action", "missing_detail", "none"];

const COMPARE_SYSTEM = `You compare sentences for a discharge nurse. For each item you get the clinician's fact and a literal English back-translation of the sentence the patient will read. Decide whether a patient who follows the patient sentence would do exactly what the fact says. Simpler words, paraphrase, extra explanation and extra details that do not contradict the fact (exact times, the previous dose) are fine. Judge each item only against its own fact.
It is a mismatch when the patient sentence:
- negation: reverses the instruction (take vs do not take, stop vs continue)
- frequency: changes how often (twice a day vs once a day, every 12 hours vs with meals)
- dose: changes the amount or the unit
- timing: changes when (morning vs evening, the date or time of an appointment)
- action: changes what to do or whom to call
- missing_detail: leaves out something needed to act safely: a threshold, a duration, a maximum, what to do, any one symptom from a list of warning signs, or a condition such as "before any procedure"
Reply with JSON only: {"results": [{"id": "f1", "match": true, "issue": "none", "explain": ""}]}, every id in the given order; when match is false, issue is one of the kinds above and explain is one sentence of at most 20 words. If you conclude it matches, match must be true.`;

async function compareMeaning(env: Env, meter: Meter, items: { fact: Fact; quote_en: string }[], label: string) {
  if (!items.length) return {} as Record<string, { match: boolean; issue: MeaningIssue; explain: string }>;
  const r = await chat(env, meter, {
    step: label,
    models: CHECKER,
    json: true,
    maxTokens: 2500,
    temperature: 0,
    messages: [
      { role: "system", content: COMPARE_SYSTEM },
      { role: "user", content: items.map(({ fact, quote_en }) => `- ${fact.id}\n  fact: ${fact.statement}\n  patient sentence (back-translated): ${quote_en || "(none)"}`).join("\n") },
    ],
    accept: (d) => Array.isArray(d?.results),
  });
  const out: Record<string, { match: boolean; issue: MeaningIssue; explain: string }> = {};
  for (const x of r.data.results) {
    if (!x || typeof x.id !== "string") continue;
    const issue: MeaningIssue = ISSUES.includes(x.issue) ? x.issue : "missing_detail";
    out[x.id] = { match: x.match === true, issue: x.match === true ? "none" : issue, explain: String(x.explain ?? "") };
  }
  return out;
}

type Sentence = { id: string; para: number; text: string };

/**
 * Targeted repair: Gemma rewrites only the sentences of the failing facts, and code splices the new text in place,
 * so no other sentence can change. Sentences a passing fact relies on are never removed.
 */
async function repair(
  env: Env,
  meter: Meter,
  draft: Draft,
  sentences: Sentence[],
  picked: Record<string, string[]>,
  failing: { fact: Fact; reasons: string[] }[],
  passingIds: string[],
  langName: string,
  level: Level,
  round: number,
): Promise<Draft> {
  const byId = new Map(sentences.map((x) => [x.id, x]));
  const list = failing
    .map(({ fact, reasons }) => {
      const cur = (picked[fact.id] ?? []).map((id) => byId.get(id)?.text).filter(Boolean);
      return `- ${fact.id}: ${fact.statement}\n  anchors: ${fact.anchors.map((a) => JSON.stringify(a)).join(", ") || "(none)"}\n  current sentence: ${cur.length ? cur.join(" ") : "(none: the leaflet does not say this yet)"}\n  problem: ${reasons.join("; ")}`;
    })
    .join("\n");
  const r = await chat(env, meter, {
    step: `Repair round ${round}`,
    models: WRITER,
    json: true,
    maxTokens: 2000,
    temperature: 0.2,
    messages: [
      { role: "system", content: writerSystem(langName, level) },
      {
        role: "user",
        content: `Leaflet:\n${sentences.map((x) => `[${x.id}] ${x.text}`).join("\n")}\n\nThese facts are not carried correctly:\n${list}\n\nFor each of these facts, write the ${langName} sentence or two that will replace its current sentence: it must say exactly what the fact says, contain every one of its anchors, and read naturally in place. Reply with JSON only: {"edits": [{"id": "f1", "text": "..."}]}`,
      },
    ],
    accept: (d) => Array.isArray(d?.edits) && d.edits.every((e: any) => typeof e?.id === "string" && typeof e?.text === "string"),
  });
  const keep = new Set(passingIds.flatMap((id) => picked[id] ?? []));
  const replaced = new Map<string, string>(); // sentence id -> new text ("" removes it)
  const appended: string[] = [];
  for (const e of r.data.edits as { id: string; text: string }[]) {
    const text = e.text.trim();
    if (!text || !failing.some((f) => f.fact.id === e.id)) continue;
    const own = (picked[e.id] ?? []).filter((id) => !keep.has(id) && !replaced.has(id));
    if (own.length) {
      replaced.set(own[0], text);
      own.slice(1).forEach((id) => replaced.set(id, ""));
    } else {
      const shared = (picked[e.id] ?? []).at(-1);
      // Never add a sentence that is already in the leaflet: the fact will simply be checked again.
      const already = splitLeaflet([text]).every((x) => sentences.some((y) => y.text.trim() === x.text.trim()));
      if (already) continue;
      if (shared) replaced.set(shared, `${replaced.get(shared) ?? byId.get(shared)!.text} ${text}`);
      else appended.push(text);
    }
  }
  const paragraphs = draft.paragraphs.map((orig, i) => {
    const mine = sentences.filter((x) => x.para === i);
    if (!mine.some((x) => replaced.has(x.id))) return orig; // untouched paragraphs stay byte for byte
    const sep = /[\u3000-\u9fff]/.test(orig) ? "" : " ";
    return mine
      .map((x) => (replaced.has(x.id) ? replaced.get(x.id)! : x.text))
      .filter(Boolean)
      .join(sep);
  });
  return { title: draft.title, paragraphs: [...paragraphs.filter(Boolean), ...appended] };
}

type Suspect = { word: string; why: string };
/** A "wrong" item whose own explanation says the wording is fine is not a finding. */
const selfCleared = (why: string) => /\b(no error|is correct|it is correct|here it is correct|is fine|is acceptable|not wrong|no problem)\b/i.test(why);

const BACK_SYSTEM = (langName: string) => `You are an adversarial literal translator checking a ${langName} patient leaflet for a discharge nurse. Translate each sentence into English word for word.
- Preserve every error exactly. If a word is wrong, misspelled, nonsensical or means something else, translate what it actually says, not what the writer probably meant (a word meaning "addiction" stays "addiction" even if "stool" was intended).
- Never infer, smooth over or correct. Keep every number, negation and time exactly as written.
- Words in the leaflet's own script and language are expected: never flag a word for not being English or for how it would be transliterated.
- You do not have the original document. Never mark a word wrong because you suspect the original meant something else (a different number, frequency or duration); judge only whether each word is wrong or nonsensical in the ${langName} sentence itself.
- Judge as a careful native speaker: everyday idiomatic expressions are correct even when a dictionary gloss differs; translate them by their idiomatic meaning.
- List as "suspicious" any word or phrase that is wrong, nonsensical, the wrong medical term, or likely to mislead a patient. Give each a severity: "wrong" when it changes the meaning or makes no sense, "awkward" when it is understandable but unusual.
Reply with JSON only: {"s1": {"en": "literal English", "suspicious": [{"word": "the ${langName} word", "severity": "wrong|awkward", "why": "what it actually means"}]}}, one entry per id, with an empty list when nothing is wrong.`;

/**
 * Back-translates sentences literally and collects suspicious words. Sentences already translated (before a repair)
 * are reused. Long lists are sent in small chunks: with 20+ sentences the model dropped and shifted entries.
 */
async function backTranslate(
  env: Env,
  meter: Meter,
  sentences: { id: string; text: string }[],
  lang: { code: string; name: string },
  facts: Fact[],
  known: Map<string, string>,
  suspects: Map<string, Suspect[]>,
  label: string,
): Promise<boolean> {
  const langName = lang.name;
  const system = BACK_SYSTEM(langName);
  const todo = [...new Set(sentences.map((x) => x.text).filter((t) => !known.has(t)))];
  const CHUNK = 4;
  const chunks: string[][] = [];
  for (let i = 0; i < todo.length; i += CHUNK) chunks.push(todo.slice(i, i + CHUNK));
  const results = await Promise.all(
    chunks.map(async (chunk, ci) => {
      // Keep whatever comes back aligned; ask again only for the sentences that are still missing.
      let missing = chunk;
      for (let tryNo = 0; tryNo < 4 && missing.length; tryNo++) {
        try {
          const input = Object.fromEntries(missing.map((t, i) => [`s${i + 1}`, t]));
          const r = await chat(env, meter, {
            step: chunks.length > 1 ? `${label} (${ci + 1} of ${chunks.length})` : label,
            models: CHECKER,
            json: true,
            maxTokens: 3000,
            // The same request at temperature 0 tends to repeat the same failure: vary it a little on later tries.
            temperature: tryNo === 0 ? 0 : 0.3,
            messages: [
              { role: "system", content: system },
              { role: "user", content: JSON.stringify(input) },
            ],
            accept: (d) => d && typeof d === "object" && !Array.isArray(d),
          });
          const still: string[] = [];
          missing.forEach((t, i) => {
            const v = r.data[`s${i + 1}`];
            const en = typeof v === "string" ? v : typeof v?.en === "string" ? v.en : null;
            // An entry whose numbers differ from its source sentence is misaligned (or altered): ask again.
            const same = (x: string) => numberTokens(x).map(Number).sort((p, q) => p - q).join(",");
            if (!en || same(en) !== same(t)) {
              still.push(t);
              return;
            }
            // Its own instructions, the source sentence handed back or a half translation is the checker failing:
            // ask again, and never let a fact pass on it.
            const bad = backTranslationProblem(en, t, lang.code, system, bracketNames(t, facts));
            if (bad) {
              meter.add({ step: `${label}: back-translation rejected`, model: DEEPSEEK, host: TF_HOST, ms: 0, note: `${bad}; asked again` });
              still.push(t);
              return;
            }
            known.set(t, en);
            const sus = Array.isArray(v?.suspicious) ? v.suspicious : [];
            suspects.set(
              t,
              sus
                .filter((x: any) => x && typeof x.word === "string" && x.word.trim() && x.severity === "wrong" && !selfCleared(String(x.why ?? "")) && !notEnglishComplaint(String(x.why ?? "")))
                .map((x: any) => ({ word: String(x.word).trim(), why: String(x.why ?? "").trim() })),
            );
          });
          missing = still;
        } catch (e) {
          // Recorded so the audit trail says why a check could not run.
          meter.add({ step: `${label}: attempt failed`, model: DEEPSEEK, host: TF_HOST, ms: 0, note: String((e as Error)?.message ?? e).slice(0, 300) });
        }
      }
      return missing.length === 0;
    }),
  );
  return results.every(Boolean);
}

/**
 * A second, independent literal reading for critical warning signs, by a different model from both the writer and the
 * first checker. Returns, per fact, the words it found wrong or misleading.
 */
async function secondReading(env: Env, meter: Meter, items: { fact: Fact; quote: string }[], langName: string, label: string): Promise<Record<string, Suspect[]>> {
  if (!items.length) return {};
  const r = await chat(env, meter, {
    step: label,
    // A separate pass that sees the fact as well as the sentence. Qwen3.8-Flash-Next was tried here but misreads
    // Devanagari vowel signs (it reported correct Hindi as misspelled), so DeepSeek does this pass too.
    models: [DEEPSEEK],
    json: true,
    maxTokens: 2000,
    temperature: 0,
    messages: [
      {
        role: "system",
        content: `You check ${langName} warning-sign sentences in a patient leaflet. For each item, read the ${langName} sentence literally, word by word, and list every word or phrase that is wrong, nonsensical, the wrong medical term, or that would make a patient misunderstand the fact, and anything from the fact that the sentence leaves out. Judge as a careful native speaker: everyday idiomatic expressions are correct. Do not report spelling unless a misspelling turns the word into a different word or into nonsense; if you are not sure, do not list it. Give each a severity: "wrong" when it changes the meaning, makes no sense or leaves out part of the fact, "awkward" when it is understandable but unusual. Reply with JSON only: {"f1": {"literal": "word-for-word English", "suspicious": [{"word": "...", "severity": "wrong|awkward", "why": "..."}]}}`,
      },
      { role: "user", content: items.map(({ fact, quote }) => `- ${fact.id}\n  fact: ${fact.statement}\n  ${langName} sentence: ${quote}`).join("\n") },
    ],
    accept: (d) => d && typeof d === "object",
  });
  const out: Record<string, Suspect[]> = {};
  for (const { fact } of items) {
    const sus = r.data?.[fact.id]?.suspicious;
    out[fact.id] = Array.isArray(sus) ? sus.filter((x: any) => x && typeof x.word === "string" && x.word.trim() && x.severity === "wrong" && !selfCleared(String(x.why ?? "")) && !notEnglishComplaint(String(x.why ?? ""))).map((x: any) => ({ word: String(x.word).trim(), why: String(x.why ?? "").trim() })) : [];
  }
  return out;
}

// ---------------------------------------------------------------- structured checks (compared in code)

const MED_FIELDS = `{"drug": "...", "dose_value": number or null, "dose_unit": "mg" or null, "times_per_day": number or null, "every_n_hours": number or null, "time_of_day": ["morning", "9 pm", ...], "duration_days": number or null, "action": "take|start|continue|stop|avoid|hold" or null, "negated": true or false, "as_needed": true or false, "max_per_day": number or null}`;

interface FactStructure {
  meds: Record<string, MedTuple>;
  warnings: Record<string, string[]>;
}

/** Once per patient version, from the confirmed English facts: medicine fields and warning-sign items. */
async function factStructure(env: Env, meter: Meter, facts: Fact[]): Promise<FactStructure | null> {
  const meds = facts.filter((f) => f.category.startsWith("medication"));
  const warns = facts.filter((f) => f.category === "warning_sign");
  if (!meds.length && !warns.length) return { meds: {}, warnings: {} };
  try {
    const r = await chat(env, meter, {
      step: "Structure the facts (medicine fields, warning-sign items)",
      models: CHECKER,
      json: true,
      maxTokens: 3000,
      temperature: 0,
      messages: [
        {
          role: "system",
          content: `You turn discharge facts into fields for an automatic check. Use only what each fact says; null when it does not say. For each medicine fact give ${MED_FIELDS}. For each warning-sign fact list every separate symptom, sign or threshold it names, as short phrases (not the action to take). Reply with JSON only: {"meds": {"f1": {...}}, "warnings": {"f6": ["...", "..."]}}`,
        },
        {
          role: "user",
          content: `Medicine facts:\n${meds.map((f) => `- ${f.id}: ${f.statement}`).join("\n") || "(none)"}\n\nWarning-sign facts:\n${warns.map((f) => `- ${f.id}: ${f.statement}`).join("\n") || "(none)"}`,
        },
      ],
      accept: (d) => d && typeof d === "object",
    });
    const out: FactStructure = { meds: {}, warnings: {} };
    for (const f of meds) if (r.data?.meds?.[f.id] && typeof r.data.meds[f.id] === "object") out.meds[f.id] = r.data.meds[f.id];
    for (const f of warns) if (Array.isArray(r.data?.warnings?.[f.id])) out.warnings[f.id] = r.data.warnings[f.id].map(String).filter(Boolean);
    return out;
  } catch {
    return null;
  }
}

/**
 * From the literal back-translation of each fact's sentences: the same medicine fields, and which warning-sign
 * items appear (with the exact words). Code then compares field by field and requires every item.
 */
async function patientStructure(
  env: Env,
  meter: Meter,
  meds: { id: string; text: string }[],
  warns: { id: string; items: string[]; text: string }[],
  label: string,
): Promise<{ meds: Record<string, MedTuple>; warnings: Record<string, ItemCheck[]> } | null> {
  if (!meds.length && !warns.length) return { meds: {}, warnings: {} };
  try {
    const r = await chat(env, meter, {
      step: label,
      models: CHECKER,
      json: true,
      maxTokens: 3000,
      temperature: 0,
      messages: [
        {
          role: "system",
          content: `You read sentences from a patient leaflet (already translated into English word for word) for an automatic check. Use only what each sentence itself says, never what it probably meant; null when it does not say. For each medicine item give ${MED_FIELDS}. For each warning item, say for every listed symptom whether the sentence mentions it, and copy the exact words from the sentence that mention it. Reply with JSON only: {"meds": {"f1": {...}}, "warnings": {"f6": [{"item": "...", "present": true, "span": "exact words"}]}}`,
        },
        {
          role: "user",
          content: `Medicine items:\n${meds.map((m) => `- ${m.id}: ${m.text}`).join("\n") || "(none)"}\n\nWarning items:\n${warns.map((w) => `- ${w.id}: ${w.text}\n  symptoms to look for: ${JSON.stringify(w.items)}`).join("\n") || "(none)"}`,
        },
      ],
      accept: (d) => d && typeof d === "object",
    });
    const out = { meds: {} as Record<string, MedTuple>, warnings: {} as Record<string, ItemCheck[]> };
    for (const m of meds) if (r.data?.meds?.[m.id] && typeof r.data.meds[m.id] === "object") out.meds[m.id] = r.data.meds[m.id];
    for (const w of warns) if (Array.isArray(r.data?.warnings?.[w.id])) out.warnings[w.id] = r.data.warnings[w.id];
    return out;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------- deliberate faults (to watch the lock work)

const splitSentences = (p: string) => p.split(/(?<=[.!?।。！？])\s+/u);

function findSentence(draft: Draft, test: (s: string) => boolean): { pi: number; sentence: string } | null {
  for (let pi = 0; pi < draft.paragraphs.length; pi++) {
    for (const s of splitSentences(draft.paragraphs[pi])) if (test(s)) return { pi, sentence: s };
  }
  return null;
}

async function injectFault(env: Env, meter: Meter, draft: Draft, facts: Fact[], kind: Fault["kind"], langName: string): Promise<{ draft: Draft; fault: Fault } | null> {
  const drugOf = (f: Fact) => f.anchors.find((a) => !/\d/.test(a));
  if (kind !== "dose" && kind !== "dose_half" && kind !== "negation") return sabotage(env, meter, draft, facts, kind, langName);
  if (kind === "dose" || kind === "dose_half") {
    for (const f of facts.filter((x) => x.category.startsWith("medication") && x.critical)) {
      const drug = drugOf(f);
      const dose = f.anchors.find((a) => /\d/.test(a) && /mg|ml|g\b/i.test(a));
      if (!drug || !dose) continue;
      const n = numberTokens(dose)[0];
      // Prefer the sentence naming the medicine with its dose; otherwise any sentence that carries the dose number and unit.
      const unit = dose.replace(/[\d.,\s]+/g, "").toLowerCase();
      const hit =
        findSentence(draft, (s) => s.toLowerCase().includes(drug.toLowerCase()) && numberTokens(s).includes(n)) ??
        findSentence(draft, (s) => numberTokens(s).includes(n) && (!unit || s.toLowerCase().includes(unit)));
      if (!hit) continue;
      const wrong = String(kind === "dose_half" ? Number(n) / 2 : Number(n) * 2);
      const after = hit.sentence.replace(new RegExp(`(^|[^\\d.,])${n.replace(".", "\\.")}(?![\\d])`), `$1${wrong}`);
      if (after === hit.sentence) continue;
      const paragraphs = [...draft.paragraphs];
      paragraphs[hit.pi] = paragraphs[hit.pi].replace(hit.sentence, after);
      meter.add({ step: `Fault injected on purpose (dose ${kind === "dose_half" ? "halved" : "doubled"})`, model: "code", host: "Homeward Worker", ms: 0, note: `${drug}: ${n} changed to ${wrong}` });
      return { draft: { ...draft, paragraphs }, fault: { kind, fact_id: f.id, before: hit.sentence, after } };
    }
    return null;
  }
  for (const f of facts.filter((x) => /\b(stop|do not|don't|avoid|hold)\b/i.test(x.statement))) {
    const drug = drugOf(f);
    if (!drug) continue;
    const hit = findSentence(draft, (s) => s.toLowerCase().includes(drug.toLowerCase()));
    if (!hit) continue;
    const r = await chat(env, meter, {
      step: "Fault injected on purpose (instruction reversed)",
      models: [GEMMA],
      json: true,
      maxTokens: 400,
      temperature: 0,
      messages: [
        {
          role: "system",
          content: `This is a safety test of a checking system. Rewrite the given ${langName} sentence so that it tells the patient the opposite: to keep taking or keep doing what the sentence says to stop or avoid. Keep the medicine name, numbers and every other word as they are. Reply with JSON only: {"sentence": "..."}`,
        },
        { role: "user", content: hit.sentence },
      ],
      accept: (d) => typeof d?.sentence === "string" && d.sentence.trim().length > 0,
    });
    const after = String(r.data.sentence).trim();
    const paragraphs = [...draft.paragraphs];
    paragraphs[hit.pi] = paragraphs[hit.pi].replace(hit.sentence, after);
    return { draft: { ...draft, paragraphs }, fault: { kind, fact_id: f.id, before: hit.sentence, after } };
  }
  return null;
}

const SABOTAGE: Partial<Record<Fault["kind"], { label: string; pick: (f: Fact) => boolean; instr: (lang: string) => string }>> = {
  frequency: {
    label: "wrong frequency",
    pick: (f) => f.category.startsWith("medication") && /(once|twice|three times|every \d+|daily|a day|per day|bedtime|morning|evening)/i.test(f.statement),
    instr: () => "Change how often or when the medicine is taken (for example twice a day becomes once a day, every 12 hours becomes every 6 hours). Keep the medicine name and the dose exactly as they are.",
  },
  drop_negation: {
    label: "negation dropped",
    pick: (f) => /\b(do not|don't|never|avoid|no)\b/i.test(f.statement),
    instr: () => "Delete the negation so the sentence now tells the patient to do what it currently tells them not to do. Change nothing else.",
  },
  date: {
    label: "wrong date",
    pick: (f) => f.category === "follow_up" && /\d/.test(f.statement),
    instr: () => "Change the day number of the date (or, if there is no date, the time) to a different value. Change nothing else.",
  },
  drop_symptom: {
    label: "warning symptom dropped",
    pick: (f) => f.category === "warning_sign" && /,|\bor\b/.test(f.statement),
    instr: () => "Delete exactly one of the symptoms or warning signs the sentence lists. Change nothing else.",
  },
  mistranslate: {
    label: "word mistranslated",
    pick: (f) => f.critical,
    instr: (lang) =>
      `Replace exactly one important everyday or medical word with a different real ${lang} word that looks or sounds similar but means something unrelated, as a careless translator might. Keep numbers and medicine names unchanged.`,
  },
};

/** Benchmark faults: Gemma finds the sentence that carries a target fact and damages it in one specific way. */
async function sabotage(env: Env, meter: Meter, draft: Draft, facts: Fact[], kind: Fault["kind"], langName: string): Promise<{ draft: Draft; fault: Fault } | null> {
  const spec = SABOTAGE[kind];
  if (!spec) return null;
  const targets = [...facts.filter((f) => f.critical && spec.pick(f)), ...facts.filter((f) => !f.critical && spec.pick(f))];
  const sentences = splitLeaflet(draft.paragraphs);
  for (const f of targets.slice(0, 3)) {
    const r = await chat(env, meter, {
      step: `Fault injected on purpose (${spec.label})`,
      models: [GEMMA],
      json: true,
      maxTokens: 600,
      temperature: 0,
      messages: [
        {
          role: "system",
          content: `This is a safety test of a checking system. Find the one sentence of the ${langName} leaflet that tells the patient the given fact, then rewrite only that sentence: ${spec.instr(langName)} Write the new sentence in ${langName}. Reply with JSON only: {"sentence_id": "s1", "new": "..."}`,
        },
        { role: "user", content: `Fact: ${f.statement}\n\nLeaflet:\n${sentences.map((x) => `[${x.id}] ${x.text}`).join("\n")}` },
      ],
      accept: (d) => (typeof d?.sentence_id === "string" || typeof d?.sentence_id === "number") && typeof d?.new === "string" && d.new.trim().length > 0,
    }).catch(() => null);
    const why = (note: string) => meter.add({ step: `Fault not injected for ${f.id}`, model: "code", host: "Homeward Worker", ms: 0, note });
    if (!r) {
      why("the model gave no usable answer");
      continue;
    }
    // Models answer "s3", "3" or 3: accept any of them.
    const sid = String(r.data.sentence_id).trim();
    const hit = sentences.find((x) => x.id === sid || x.id === `s${sid.replace(/^s/i, "")}`);
    const after = String(r.data.new).trim();
    if (!hit) {
      why(`sentence id ${sid} not in the leaflet`);
      continue;
    }
    if (after === hit.text) {
      why("the rewritten sentence was unchanged");
      continue;
    }
    const paragraphs = [...draft.paragraphs];
    paragraphs[hit.para] = paragraphs[hit.para].replace(hit.text, after);
    if (paragraphs[hit.para] === draft.paragraphs[hit.para]) {
      why("the sentence could not be replaced in its paragraph");
      continue;
    }
    return { draft: { ...draft, paragraphs }, fault: { kind, fact_id: f.id, before: hit.text, after } };
  }
  return null;
}

// ---------------------------------------------------------------- prepare

export async function prepare(
  env: Env,
  meter: Meter,
  facts: Fact[],
  topics: Topic[],
  langCode: string,
  level: Level,
  emit: (e: PrepareEvent) => void,
  faultKind?: Fault["kind"],
): Promise<PatientVersion> {
  const lang = langByCode(langCode);
  emit({ type: "progress", stage: "write", message: `Writing the instructions in ${lang.name}` });
  // The facts' own fields are needed only for checking, so they are worked out while the writer writes.
  const structureP = factStructure(env, meter, facts);
  const written = await write(env, meter, facts, topics, lang.name, level);
  let draft = written.draft;

  let fault: Fault | undefined;
  if (faultKind) {
    const injected = await injectFault(env, meter, draft, facts, faultKind, lang.name).catch(() => null);
    if (injected) {
      draft = injected.draft;
      fault = injected.fault;
    } else {
      meter.add({ step: "Fault requested but not injected", model: "code", host: "Homeward Worker", ms: 0, note: "no matching sentence was found in the first draft" });
    }
  }

  const rounds: Record<string, number> = {};
  const history: Record<string, { round: number; problems: AnchorProblem[] }[]> = {};
  const english = new Map<string, string>(); // sentence text -> literal English, by the checker model
  const suspects = new Map<string, Suspect[]>(); // sentence text -> words the literal translator found wrong
  const passedWith: Record<string, string[]> = {}; // fact id -> the sentence texts it passed with
  const uncheckable = new Set<string>(); // facts the checker could not check this round (not repairable)
  let results: Record<string, { quote: string; quote_en: string; problems: AnchorProblem[] }> = {};
  let sentences = splitLeaflet(draft.paragraphs);
  let backOk = true;
  for (let round = 0; round <= MAX_REPAIRS; round++) {
    emit({ type: "progress", stage: "lock", message: round === 0 ? "Checking that every fact survived" : `Checking the repaired version (round ${round})` });
    const suffix = round === 0 ? "" : ` (after repair ${round})`;
    // Pointing at sentences and translating them back are independent, so they run side by side.
    // After a repair, a fact that already passed keeps its pass while its exact sentences are still in the text;
    // only failing facts and facts whose sentences changed are checked again (fewer calls, no flip-flopping).
    const texts = new Set(sentences.map((x) => x.text));
    const carried = facts.filter((f) => passedWith[f.id] && passedWith[f.id].every((t) => texts.has(t)));
    const toCheck = facts.filter((f) => !carried.includes(f));
    const [picked, ok] = await Promise.all([
      lockQuotes(env, meter, toCheck, sentences, `Fact lock: find each fact's sentences${suffix}`),
      backTranslate(env, meter, sentences, lang, facts, english, suspects, round === 0 ? "Back-translate every sentence" : `Back-translate the repaired sentences${suffix}`),
    ]);
    // The checker sometimes points at nothing for a fact whose sentence is plainly there: ask once more for those
    // facts alone before calling the fact missing (a "missing" fact would otherwise be written again by the repair).
    const empty = toCheck.filter((f) => !(picked[f.id] ?? []).length);
    if (empty.length) {
      const again = await lockQuotes(env, meter, empty, sentences, `Fact lock: find the sentences for facts with none${suffix}`).catch(() => ({}) as Record<string, string[]>);
      for (const f of empty) if (again[f.id]?.length) picked[f.id] = again[f.id];
    }
    backOk = ok;
    // A failed back-translation is the checker's problem, not the text's: try the missing sentences again before judging.
    for (let extra = 0; extra < 2 && !backOk; extra++) {
      backOk = await backTranslate(env, meter, sentences, lang, facts, english, suspects, `Back-translate the remaining sentences${suffix}`);
    }
    const byId = new Map(sentences.map((x) => [x.id, x]));
    const text = draft.paragraphs.join("\n\n");
    const prev = results;
    results = {};
    for (const f of carried) {
      results[f.id] = prev[f.id];
      for (const t of passedWith[f.id]) {
        const sid = sentences.find((x) => x.text === t)?.id;
        if (sid) picked[f.id] = [...(picked[f.id] ?? []), sid];
      }
    }
    for (const f of toCheck) {
      const chosen = (picked[f.id] ?? []).map((id) => byId.get(id)!).sort((x, y) => Number(x.id.slice(1)) - Number(y.id.slice(1)));
      const quote = chosen.map((x) => x.text).join(" ");
      const quote_en = chosen.map((x) => english.get(x.text) ?? "").join(" ").trim();
      const problems = quote ? verifyFact(f, quote, quote, lang.code) : verifyFact(f, "", text, lang.code);
      // English left in a non-Latin-script text ("Oct", "am", "min") fails the fact; the repair translates it.
      if (!problems.length && quote) {
        const stray = strayEnglish(quote, lang.code, facts);
        if (stray.length) problems.push({ kind: "anchor", anchor: stray.join(", "), reason: `English left untranslated: ${stray.map((w) => `"${w}"`).join(", ")}; write it in ${lang.name}` });
      }
      // A wrong or nonsensical word fails the fact even when every number and name is present.
      if (!problems.length) {
        for (const x of chosen.flatMap((c) => suspects.get(c.text) ?? [])) {
          problems.push({ kind: "meaning", anchor: x.word, reason: `suspicious wording "${x.word}"${x.why ? `: ${x.why}` : ""}` });
        }
      }
      results[f.id] = { quote, quote_en, problems };
    }
    // Anchors present is not enough ("take" vs "do not take"): compare meaning on everything that passed the code checks.
    const passed = toCheck.filter((f) => results[f.id].problems.length === 0);
    // A fact whose sentences have no accepted back-translation cannot be checked. The checker failed, not the text:
    // a repair cannot help, so such facts go straight to the nurse. Facts with every sentence translated go on.
    const unchecked = backOk ? [] : passed.filter((f) => (picked[f.id] ?? []).some((id) => !english.has(byId.get(id)!.text)));
    for (const f of unchecked) {
      results[f.id].problems.push({ kind: "meaning", anchor: "", reason: "the back-translation failed, so the meaning could not be checked" });
      uncheckable.add(f.id);
    }
    {
      const checkable = toCheck.filter((f) => results[f.id].problems.length === 0);
      const critical = checkable.filter((f) => f.critical && f.category === "warning_sign");
      const fs = await structureP;
      const medItems = fs ? checkable.filter((f) => fs.meds[f.id] && results[f.id].quote_en).map((f) => ({ id: f.id, text: results[f.id].quote_en })) : [];
      const warnItems = fs
        ? checkable.filter((f) => (fs.warnings[f.id]?.length ?? 0) > 1 && results[f.id].quote_en).map((f) => ({ id: f.id, items: fs.warnings[f.id], text: results[f.id].quote_en }))
        : [];
      const [meaning, second, ps] = await Promise.all([
        compareMeaning(env, meter, checkable.map((f) => ({ fact: f, quote_en: results[f.id].quote_en })), `Fact lock: compare meaning${suffix}`),
        secondReading(env, meter, critical.map((f) => ({ fact: f, quote: results[f.id].quote })), lang.name, `Fact lock: second literal reading of critical warning signs${suffix}`).catch(
          () => null,
        ),
        patientStructure(env, meter, medItems, warnItems, `Fact lock: compare fields and warning-sign items${suffix}`),
      ]);
      // Field-by-field medicine check and item-by-item warning-sign check, decided in code.
      if (fs && !ps && (medItems.length || warnItems.length)) {
        // The field check could not run: that is the checker's failure, so these facts go to the nurse rather than pass.
        for (const it of [...medItems, ...warnItems]) {
          results[it.id].problems.push({ kind: "meaning", anchor: "", reason: "the field check gave no answer for this fact" });
          uncheckable.add(it.id);
        }
      }
      if (fs && ps) {
        for (const m of medItems) {
          const pt = ps.meds[m.id];
          if (!pt) continue;
          for (const issue of compareMed(fs.meds[m.id], pt)) results[m.id].problems.push({ kind: "meaning", anchor: "", reason: `field check: ${issue}` });
        }
        for (const w of warnItems) {
          const miss = missingItems(w.items, ps.warnings[w.id] ?? [], w.text);
          for (const item of miss) results[w.id].problems.push({ kind: "meaning", anchor: item, reason: `missing: ${item}` });
        }
      }
      for (const f of critical) {
        if (second === null) results[f.id].problems.push({ kind: "meaning", anchor: "", reason: "the second reading of this critical warning sign failed" });
        else for (const x of second[f.id] ?? []) results[f.id].problems.push({ kind: "meaning", anchor: x.word, reason: `second reading found "${x.word}"${x.why ? `: ${x.why}` : ""}` });
      }
      for (const f of checkable) {
        const m = meaning[f.id];
        if (!m) results[f.id].problems.push({ kind: "meaning", anchor: "", reason: "the meaning check gave no answer for this fact" });
        else if (!m.match) results[f.id].problems.push({ kind: "meaning", anchor: m.issue, reason: `meaning differs (${m.issue.replace("_", " ")})${m.explain ? `: ${m.explain}` : ""}` });
      }
    }
    for (const f of passed) {
      if (results[f.id].problems.length === 0) passedWith[f.id] = (picked[f.id] ?? []).map((id) => byId.get(id)!.text);
    }
    const failing = facts.filter((f) => results[f.id].problems.length > 0);
    failing.forEach((f) => {
      delete passedWith[f.id];
      (history[f.id] ??= []).push({ round, problems: results[f.id].problems });
    });
    const repairable = failing.filter((f) => !uncheckable.has(f.id));
    uncheckable.clear();
    if (!repairable.length || round === MAX_REPAIRS) break;
    repairable.forEach((f) => (rounds[f.id] = round + 1));
    emit({ type: "progress", stage: "repair", message: `${repairable.length} ${repairable.length === 1 ? "fact needs" : "facts need"} a repair` });
    draft = await repair(
      env,
      meter,
      draft,
      sentences,
      picked,
      repairable.map((f) => ({ fact: f, reasons: results[f.id].problems.map((p) => p.reason) })),
      facts.filter((f) => !failing.includes(f)).map((f) => f.id),
      lang.name,
      level,
      round + 1,
    );
    sentences = splitLeaflet(draft.paragraphs);
  }

  const back = backOk ? draft.paragraphs.map((_, i) => sentences.filter((x) => x.para === i).map((x) => english.get(x.text) ?? "").join(" ")) : [];

  const lock: LockResult[] = facts.map((f) => {
    const r = results[f.id];
    const ok = r.problems.length === 0;
    return {
      id: f.id,
      status: ok ? (rounds[f.id] ? "repaired" : "locked") : "flagged",
      quote: r.quote,
      quote_en: r.quote_en || undefined,
      problems: r.problems,
      rounds: rounds[f.id] ?? 0,
      history: history[f.id] ?? [],
    };
  });
  const questions: Record<string, string> = {};
  for (const t of topics) questions[t.id] = written.questions[t.id] || (lang.code === "en" ? t.question : "");
  return { language: lang.code, level, title: draft.title, paragraphs: draft.paragraphs, back_translation: back, questions, lock, fault };
}

// ---------------------------------------------------------------- teach-back grading

const VERDICTS: Verdict[] = ["correct", "partial", "incorrect", "missing"];

export function graderMessages(facts: Fact[], question: string, transcript: string, langName: string) {
  const lines = facts.map((f) => `- ${f.id} [${f.category}${f.critical ? ", critical" : ""}]: ${f.statement}`).join("\n");
  return [
    { role: "system" as const, content: grader.system },
    {
      role: "user" as const,
      content: grader.user.replace("{facts}", lines).replace("{question}", question).replace("{language_name}", langName).replace("{transcript}", transcript),
    },
  ];
}

export async function grade(env: Env, meter: Meter, facts: Fact[], question: string, transcript: string, langCode: string) {
  const lang = langByCode(langCode);
  const ids = facts.map((f) => f.id);

  // The English rendering of the answer only needs the transcript, so it runs while the grader works.
  const englishP =
    lang.code === "en"
      ? Promise.resolve(transcript)
      : chat(env, meter, {
          step: "Render the answer in English",
          models: [GEMMA, DEEPSEEK],
          json: true,
          maxTokens: 1200,
          temperature: 0,
          messages: [
            { role: "system", content: `You translate a patient's spoken ${lang.name} into English for a nurse, literally, keeping their mistakes. Reply with JSON only: {"english": "..."}` },
            { role: "user", content: transcript },
          ],
          accept: (d) => typeof d?.english === "string",
        })
          .then((r) => String(r.data.english))
          .catch(() => "");

  const r = await chat(env, meter, {
    step: "Grade the teach-back",
    models: GRADER,
    json: true,
    maxTokens: 2000,
    temperature: 0,
    messages: graderMessages(facts, question, transcript, lang.name),
    accept: (d) => Array.isArray(d?.facts) && ids.every((id) => d.facts.some((g: any) => g?.id === id && VERDICTS.includes(g?.verdict))),
  });
  const grades: FactGrade[] = ids.map((id) => {
    const g = r.data.facts.find((x: any) => x?.id === id);
    return { id, verdict: g.verdict, heard: String(g.heard ?? ""), reteach: g.verdict === "correct" ? "" : String(g.reteach ?? "") };
  });

  // Re-teach lines in the patient's language, only for facts that were missed.
  const need = grades.filter((g) => g.verdict !== "correct");
  if (need.length && lang.code !== "en") {
    try {
      const t = await chat(env, meter, {
        step: "Translate the re-teach",
        models: [GEMMA, DEEPSEEK],
        json: true,
        maxTokens: 1000,
        temperature: 0.1,
        messages: [
          {
            role: "system",
            content: `You translate short re-teach sentences into ${lang.name} for a patient: simple words, the polite form of address, medicine names in Latin letters, numbers as digits. Never add advice, numbers or doses. Reply with JSON only: {"f1": "..."}`,
          },
          { role: "user", content: JSON.stringify(Object.fromEntries(need.map((g) => [g.id, g.reteach || facts.find((f) => f.id === g.id)!.statement]))) },
        ],
        accept: (d) => d && typeof d === "object",
      });
      for (const g of need) {
        const line = String(t.data?.[g.id] ?? "").trim();
        const fact = facts.find((f) => f.id === g.id)!;
        // A re-teach line that introduces a number the fact does not contain is dropped; the locked sentence is shown instead.
        if (line && inventedNumbers(line, fact).length === 0) g.reteach_patient = line;
      }
    } catch {
      /* without the translation the patient still sees the locked sentence for each missed fact */
    }
  } else if (need.length) {
    for (const g of need) if (g.reteach && inventedNumbers(g.reteach, facts.find((f) => f.id === g.id)!).length === 0) g.reteach_patient = g.reteach;
  }
  return { grades, transcript_en: await englishP };
}

// ---------------------------------------------------------------- OCR

export async function ocr(env: Env, meter: Meter, dataUrl: string): Promise<string> {
  const started = Date.now();
  let res: Response | null = null;
  for (let tryNo = 0; tryNo < 8; tryNo++) {
    res = await fetch((env.TF_BASE_URL || "https://developer.amd.com.cn/radeon/api/v1") + "/ocr", {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${env.TOKENFACTORY_API_KEY}` },
      body: JSON.stringify({ model: "MinerU2.5-Pro", document: { type: "document_url", document_url: dataUrl }, pages: "0-3" }),
      signal: AbortSignal.timeout(180_000),
    });
    if (res.status !== 429) break;
    // Concurrency is shared with the rest of the team; wait our turn.
    await new Promise((r) => setTimeout(r, Math.min(2 ** tryNo, 10) * 1000 + Math.random() * 800));
  }
  if (!res || !res.ok) throw new Error(`OCR failed (${res?.status})`);
  const body: any = await res.json();
  const pages: any[] = body?.pages ?? [];
  meter.add({ step: "Read the document (OCR)", model: "MinerU2.5-Pro", host: TF_HOST, ms: Date.now() - started, note: `${body?.usage_info?.pages_processed ?? pages.length} page(s)` });
  return pages.map((p) => String(p?.markdown ?? "")).join("\n\n").trim();
}
