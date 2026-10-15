// Fact lock: the model proposes a quote for each fact, this code decides whether the quote is real and
// carries every anchor. No model output is trusted here; everything is plain string checks.

import type { AnchorProblem, Fact } from "../shared/types.ts";

/** Unicode decimal digit blocks whose zero we map to ASCII. NFKC already folds full-width digits. */
const DIGIT_ZEROS = [0x0660, 0x06f0, 0x07c0, 0x0966, 0x09e6, 0x0a66, 0x0ae6, 0x0b66, 0x0be6, 0x0c66, 0x0ce6, 0x0d66, 0x0e50, 0x0ed0, 0x0f20, 0x1040, 0x17e0, 0x1810];

export function asciiDigits(s: string): string {
  let out = "";
  for (const ch of s) {
    const cp = ch.codePointAt(0)!;
    const zero = DIGIT_ZEROS.find((z) => cp >= z && cp <= z + 9);
    out += zero === undefined ? ch : String(cp - zero);
  }
  return out;
}

/** Whitespace, quote marks and dashes made uniform. Letters and marks are kept, so a quote must still be verbatim. */
export function normSpace(s: string): string {
  return asciiDigits(s.normalize("NFKC"))
    .replace(/[‘’‚‛′`´]/g, "'")
    .replace(/[“”„‟«»″「」]/g, '"')
    .replace(/[‐-―−]/g, "-")
    .replace(/…/g, "...")
    .replace(/[\s​-‍﻿]+/g, " ")
    .trim();
}

/** For anchor matching: also case-folded and with Latin diacritics removed (Indic vowel signs are kept). */
export function normLoose(s: string): string {
  return normSpace(s)
    .toLowerCase()
    .normalize("NFD")
    .replace(/([a-z])[̀-ͯ]+/g, "$1")
    .normalize("NFC");
}

/** Trim stray quote marks and full stops that models add around a quote. */
export function cleanQuote(q: string): string {
  return normSpace(q).replace(/^["'\s]+|["'\s]+$/g, "");
}

export function quoteIsVerbatim(patientText: string, quote: string): boolean {
  const q = cleanQuote(quote);
  if (q.length < 6 || q.includes("...")) return false;
  const hay = normSpace(patientText);
  if (hay.includes(q)) return true;
  // Allow the model to drop or add the closing full stop (also the Devanagari danda and CJK stop).
  const bare = q.replace(/[.。।!?！？]+$/u, "");
  return bare.length >= 6 && hay.includes(bare);
}

/** Numbers as comparable tokens: "2,5" -> "2.5", "1,000" -> "1000", "10:30" -> "10.30". */
/** Languages that write decimals with a comma and may group thousands with a dot ("1.000 mg", "2,5 mg"). */
export const COMMA_DECIMAL = new Set(["it", "es", "pt", "fr", "vi"]);

export function numberTokens(s: string, lang?: string): string[] {
  let t = asciiDigits(s.normalize("NFKC"));
  if (lang && COMMA_DECIMAL.has(lang)) t = t.replace(/\b(\d{1,3})((?:\.\d{3})+)(?![\d,])/g, (_, a: string, b: string) => a + b.replace(/\./g, ""));
  const found = t.match(/\d+(?:[.,:]\d+)*/g) ?? [];
  return found.map((n) => {
    let x = n.replace(/,(\d{3})(?=\D|$)/g, "$1"); // thousands
    x = x.replace(/,/g, ".").replace(/:/g, ".");
    x = x.replace(/^0+(\d)/, "$1"); // "08" -> "8"
    return x;
  });
}

/** Units that must stay exactly as written (Latin letters), because a wrong unit is a wrong dose. */
const MED_UNITS = ["mcg", "µg", "μg", "mg", "kg", "ml", "mmhg", "mmol", "iu", "lbs", "lb", "g", "l", "%", "°c", "°f", "cm", "mm"];

export function medUnits(anchor: string): string[] {
  const a = normLoose(anchor);
  const units: string[] = [];
  for (const u of MED_UNITS) {
    const esc = u.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    // A unit right after a number ("40mg", "40 mg") or standing alone as a word.
    const re = new RegExp(`(?:\\d\\s*|\\b)${esc}(?![a-z])`, "u");
    if (re.test(a) && !units.some((x) => x.includes(u))) units.push(u);
  }
  return units;
}

const EN_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];
const EN_MONTH_RE = new RegExp(`\\b(${EN_MONTHS.map((m) => `${m}|${m.slice(0, 3)}\\.?`).join("|")})\\b`, "i");

const monthCache = new Map<string, string[][]>();
/** Month names in the patient's language, long and short, from the runtime's ICU data. */
export function localMonthNames(lang: string): string[][] {
  const hit = monthCache.get(lang);
  if (hit) return hit;
  const out: string[][] = [];
  for (let m = 0; m < 12; m++) {
    const d = new Date(Date.UTC(2026, m, 15));
    const names = new Set<string>();
    for (const month of ["long", "short"] as const) {
      for (const opts of [{ month }, { month, day: "numeric" as const }]) {
        try {
          const parts = new Intl.DateTimeFormat(lang, { ...opts, timeZone: "UTC" }).formatToParts(d);
          for (const p of parts) if (p.type === "month") names.add(normLoose(p.value).replace(/\.$/, ""));
        } catch {
          /* unknown locale: fall back to English names only */
        }
      }
    }
    names.add(EN_MONTHS[m]);
    names.add(EN_MONTHS[m].slice(0, 3));
    out.push([...names].filter((n) => n.length > 0));
  }
  monthCache.set(lang, out);
  return out;
}

function hasMonth(quoteLoose: string, monthIdx: number, day: string | undefined, lang: string): boolean {
  const names = localMonthNames(lang)[monthIdx];
  if (names.some((n) => quoteLoose.includes(n))) return true;
  // Numeric dates: 12/11, 12.11, 11/12, 12-11-2026, 2026-11-12.
  const m = String(monthIdx + 1);
  if (day) {
    const pairs = [`${day}/${m}`, `${m}/${day}`, `${day}.${m}`, `${day}-${m}`, `${m}-${day}`];
    const flat = quoteLoose.replace(/\b0(\d)/g, "$1");
    if (pairs.some((p) => flat.includes(p))) return true;
  }
  return false;
}

const sameNumber = (a: string, b: string) => a === b || (Number.isFinite(Number(a)) && Number(a) === Number(b));

/** "2 pm" may be written as 14 or 14:00 in a 24-hour language; "12 am" as 0. */
export function numberAlternatives(anchor: string, n: string): string[] {
  const t = asciiDigits(anchor.normalize("NFKC")).toLowerCase();
  const esc = n.replace(".", "[.:]");
  const pm = new RegExp(`(^|\\D)${esc}\\s*(p\\.?\\s?m\\.?)(?![a-z])`).test(t);
  const am = new RegExp(`(^|\\D)${esc}\\s*(a\\.?\\s?m\\.?)(?![a-z])`).test(t);
  const [h, m] = n.split(".");
  const hour = Number(h);
  if (pm && hour >= 1 && hour < 12) return [n, String(hour + 12) + (m ? "." + m : "")];
  if (am && hour === 12) return [n, "0" + (m ? "." + m : "")];
  return [n];
}

export type AnchorKind = "date" | "number" | "name";
export function anchorKind(anchor: string): AnchorKind {
  if (EN_MONTH_RE.test(anchor)) return "date";
  if (/\d/.test(asciiDigits(anchor))) return "number";
  return "name";
}

/** Check one anchor against a quote. Returns null when it is carried, or the reason it is not. */
export function checkAnchor(anchor: string, quote: string, lang: string): string | null {
  const qLoose = normLoose(quote);
  const kind = anchorKind(anchor);
  const qNums = numberTokens(quote, lang);
  if (kind === "name") {
    // A doctor's title may be the local one ("Dott.ssa", "Dra.", "डॉ."): match the name without the English title.
    const bare = normLoose(anchor).replace(/^(dr|doctor|mr|mrs|ms|miss|prof)\.?\s+/, "");
    return qLoose.includes(normLoose(anchor)) || (bare !== normLoose(anchor) && qLoose.includes(bare))
      ? null
      : `"${anchor}" is not in the sentence (it must be kept exactly as written)`;
  }
  const nums = numberTokens(anchor);
  const missingNums = nums.filter((n) => !numberAlternatives(anchor, n).some((alt) => qNums.some((q) => sameNumber(q, alt))));
  if (kind === "date") {
    const monthIdx = EN_MONTHS.findIndex((m) => new RegExp(`\\b${m.slice(0, 3)}`, "i").test(anchor));
    const day = nums.find((n) => Number(n) >= 1 && Number(n) <= 31);
    const problems: string[] = [];
    if (missingNums.length) problems.push(`number ${missingNums.join(", ")} missing`);
    if (monthIdx >= 0 && !hasMonth(qLoose, monthIdx, day, lang)) problems.push(`month "${EN_MONTHS[monthIdx]}" missing`);
    return problems.length ? `date "${anchor}": ${problems.join("; ")}` : null;
  }
  const problems: string[] = [];
  if (missingNums.length) problems.push(`number ${missingNums.join(", ")} missing`);
  const units = medUnits(anchor).filter((u) => !medUnits(quote).includes(u));
  if (units.length) problems.push(`unit ${units.join(", ")} missing`);
  return problems.length ? `"${anchor}": ${problems.join("; ")}` : null;
}

export function verifyFact(fact: Fact, quote: string, patientText: string, lang: string): AnchorProblem[] {
  if (!quote || !quote.trim()) return [{ kind: "quote", anchor: "", reason: "no sentence in the patient version carries this fact" }];
  if (!quoteIsVerbatim(patientText, quote)) return [{ kind: "quote", anchor: "", reason: "the quoted sentence is not in the patient version word for word" }];
  const problems: AnchorProblem[] = [];
  for (const a of fact.anchors) {
    if (!a.trim()) continue;
    const r = checkAnchor(a, quote, lang);
    if (r) problems.push({ kind: "anchor", anchor: a, reason: r });
  }
  return problems;
}

/**
 * A model-written line (a re-teach sentence) may not introduce numbers that the fact does not contain.
 * Returns the invented numbers, empty when the line is safe to show.
 */
export function inventedNumbers(line: string, fact: Fact): string[] {
  // Each number in the fact, plus its 24-hour form when it is a 12-hour time ("2 pm" may be said as 14).
  const allowed = [fact.statement, ...fact.anchors].flatMap((src) => numberTokens(src).flatMap((n) => numberAlternatives(src, n)));
  return numberTokens(line).filter((n) => !allowed.some((a) => sameNumber(a, n)));
}

/** The leaflet as numbered sentences. The checker points at sentence ids; code assembles the quote, so it is verbatim by construction. */
export function splitLeaflet(paragraphs: string[]): { id: string; para: number; text: string }[] {
  const out: { id: string; para: number; text: string }[] = [];
  paragraphs.forEach((p, para) => {
    // Split after sentence-ending punctuation, but not after short abbreviations such as "Dr." or inside numbers like 2.5.
    const parts = p.split(/(?<=(?<!\b(?:Dr|Dra|Mr|Mrs|Ms|Sr|Sra|Sig|St|Prof|Dott|Dott\.ssa|Dottssa|Mme|Mlle)\b)[.!?;؟।。！？])\s+|(?<=[。！？])/u);
    for (const t of parts.map((x) => x.trim()).filter(Boolean)) out.push({ id: `s${out.length + 1}`, para, text: t });
  });
  return out;
}

const NON_LATIN = new Set(["hi", "bn", "ta", "ar", "zh"]);
const KEEP_LATIN = new Set([...MED_UNITS, "dr", "mr", "mrs", "ms", "prof"]);
const COMMON_EN = new Set(
  "minutes minute morning evening tablets tablet injection injections procedure clinic doctor dentist nurse hospital emergency cardiology orthopaedic orthopedic kidney function appointment therapy physiotherapy dressing walker before during after between weight shortness breath swelling fainting bleeding stools urine bruising headache injury surgeon review needed instead continue starting increase decrease remember missed together daily weekly monthly october november december january february august september tomorrow".split(" "),
);

/**
 * In a patient text written in a non-Latin script, Latin-letter words should only be medicine names, people's names
 * and medical units. Anything else ("Oct", "am", "min", "OPD") is English left untranslated. Returns those words.
 */
export function strayEnglish(quote: string, lang: string, allFacts: Fact[]): string[] {
  if (!NON_LATIN.has(lang)) return [];
  const allowed = new Set<string>(KEEP_LATIN);
  for (const f of allFacts) {
    for (const a of f.anchors) if (anchorKind(a) === "name") for (const w of normLoose(a).split(/[^a-z]+/)) if (w) allowed.add(w);
    // Medicine names the extractor did not list as anchors ("paracetamol"): long words from the facts that are not common English.
    for (const w of normLoose(f.statement).match(/[a-z]+/g) ?? []) if (w.length >= 6 && !COMMON_EN.has(w)) allowed.add(w);
  }
  // Temperature units ("101°F") are medical units, not stray English.
  const words = normLoose(quote).replace(/°\s*[cf]\b/g, " ").match(/[a-z]+/g) ?? [];
  return [...new Set(words.filter((w) => !allowed.has(w)))];
}
