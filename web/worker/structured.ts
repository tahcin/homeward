// Structured checks decided in code: medicine dose and frequency compared field by field, warning-sign items
// required one by one, local-script unit names, and anchor hygiene. Models only fill in the fields.

import { normLoose, numberTokens } from "./lock.ts";

// ---------------------------------------------------------------- medicine tuples

export interface MedTuple {
  drug?: string | null;
  dose_value?: number | null;
  dose_unit?: string | null;
  times_per_day?: number | null;
  every_n_hours?: number | null;
  time_of_day?: string[] | null;
  duration_days?: number | null;
  action?: string | null; // take | stop | continue | avoid | start | hold
  negated?: boolean | null;
  as_needed?: boolean | null;
  max_per_day?: number | null;
}

const num = (v: unknown): number | null => {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") {
    const t = numberTokens(v)[0];
    return t !== undefined ? Number(t) : null;
  }
  return null;
};

/** "08:00", "9 pm", "morning", "bedtime" -> morning | midday | afternoon | evening | night. */
export function timeBucket(t: string): string | null {
  const s = normLoose(t);
  if (/bedtime|night|sleep/.test(s)) return "night";
  if (/morning|breakfast|wake/.test(s)) return "morning";
  if (/noon|midday|lunch/.test(s)) return "midday";
  if (/afternoon/.test(s)) return "afternoon";
  if (/evening|dinner|supper/.test(s)) return "evening";
  const m = s.match(/(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?/);
  if (!m) return null;
  let h = Number(m[1]);
  if (m[3]?.startsWith("p") && h < 12) h += 12;
  if (m[3]?.startsWith("a") && h === 12) h = 0;
  if (h >= 5 && h < 11) return "morning";
  if (h >= 11 && h < 14) return "midday";
  if (h >= 14 && h < 17) return "afternoon";
  if (h >= 17 && h < 21) return "evening";
  return "night";
}

/** Doses per day from any of: times_per_day, every_n_hours, or the listed times of day. */
export function timesPerDay(t: MedTuple): number | null {
  const n = num(t.times_per_day);
  if (n) return n;
  const h = num(t.every_n_hours);
  if (h) return Math.round((24 / h) * 100) / 100;
  if (Array.isArray(t.time_of_day) && t.time_of_day.length) return t.time_of_day.length;
  return null;
}

/** take / continue / start count as "take"; stop / avoid / hold, or a negated take, count as "do not take". */
export function polarity(t: MedTuple): "take" | "not take" | null {
  const a = (t.action ?? "").toLowerCase();
  let p: "take" | "not take" | null = /stop|avoid|hold|pause|don/.test(a) ? "not take" : /take|continue|start|use|inject|apply|keep/.test(a) ? "take" : null;
  if (p && t.negated === true) p = p === "take" ? "not take" : "take";
  return p;
}

const UNIT_CANON: Record<string, string> = { milligram: "mg", milligrams: "mg", mg: "mg", microgram: "mcg", mcg: "mcg", gram: "g", grams: "g", g: "g", milliliter: "ml", millilitre: "ml", ml: "ml", unit: "units", units: "units", tablet: "tablet", tablets: "tablet", puff: "puff", puffs: "puff", drop: "drop", drops: "drop" };
const unitCanon = (u?: string | null) => {
  if (!u) return null;
  const k = normLoose(u);
  return UNIT_CANON[k] ?? UNIT_CANON[k.replace(/s$/, "")] ?? k;
};

/** Field-by-field comparison of what the fact says and what the patient sentence says. Empty when they agree. */
export function compareMed(fact: MedTuple, pat: MedTuple): string[] {
  const out: string[] = [];
  const fd = num(fact.dose_value);
  const pd = num(pat.dose_value);
  if (fd !== null) {
    if (pd === null) out.push("dose missing");
    else if (pd !== fd) out.push(`dose differs (fact ${fd}, patient text ${pd})`);
  }
  const fu = unitCanon(fact.dose_unit);
  const pu = unitCanon(pat.dose_unit);
  if (fu && pu && fu !== pu) out.push(`dose unit differs (fact ${fu}, patient text ${pu})`);
  const ff = timesPerDay(fact);
  const pf = timesPerDay(pat);
  if (ff !== null && !fact.as_needed) {
    if (pf === null) out.push("frequency missing");
    else if (Math.abs(pf - ff) > 0.01) out.push(`frequency differs (fact ${ff} a day, patient text ${pf} a day)`);
  }
  const fb = new Set((fact.time_of_day ?? []).map(timeBucket).filter(Boolean));
  const pb = new Set((pat.time_of_day ?? []).map(timeBucket).filter(Boolean));
  if (fb.size && pb.size && (fb.size !== pb.size || [...fb].some((b) => !pb.has(b)))) out.push(`time of day differs (fact ${[...fb].join(", ")}, patient text ${[...pb].join(", ")})`);
  const fp = polarity(fact);
  const pp = polarity(pat);
  if (fp && pp && fp !== pp) out.push(`action differs (fact: ${fp}, patient text: ${pp})`);
  if (fact.as_needed === true && pat.as_needed !== true) out.push("'only when needed' missing");
  if (fact.as_needed === false && pat.as_needed === true) out.push("patient text says 'only when needed' but the fact does not");
  const fm = num(fact.max_per_day);
  if (fm !== null && num(pat.max_per_day) !== fm) out.push(`maximum per day ${num(pat.max_per_day) === null ? "missing" : "differs"} (fact ${fm})`);
  const fdur = num(fact.duration_days);
  if (fdur !== null && num(pat.duration_days) !== fdur) out.push(`duration ${num(pat.duration_days) === null ? "missing" : "differs"} (fact ${fdur} days)`);
  return out;
}

// ---------------------------------------------------------------- warning-sign items

export interface ItemCheck {
  item: string;
  present: boolean;
  span?: string;
}

/** Every listed symptom or threshold must be present, with a quoted span that really is in the back-translation. */
export function missingItems(items: string[], checks: ItemCheck[], backTranslation: string): string[] {
  const hay = normLoose(backTranslation);
  return items.filter((item) => {
    const c = checks.find((x) => normLoose(x.item) === normLoose(item));
    if (!c || !c.present) return true;
    const span = normLoose(c.span ?? "").replace(/^["']|["']$/g, "");
    return !span || !hay.includes(span);
  });
}

// ---------------------------------------------------------------- local-script units

/** Local names for medical units, accepted where the lock expects the Latin unit (only right after a number). */
export const UNIT_ALIASES: Record<string, string[]> = {
  mg: ["毫克", "मिलीग्राम", "मि.ग्रा.", "मि.ग्रा", "মিলিগ্রাম", "মি.গ্রা.", "ملغ", "مجم", "ملجم", "மி.கி", "மில்லிகிராம்", "miligam", "miligramos", "miligramo", "milligrammi", "milligrammo", "milligrammes", "milligramme", "miligramas", "miligrama"],
  g: ["克", "ग्राम", "গ্রাম", "غرام", "جرام", "غ", "கிராம்", "gam", "gramos", "gramo", "grammi", "grammo", "grammes", "gramme", "gramas", "grama"],
  ml: ["毫升", "मिलीलीटर", "मि.ली.", "मि.ली", "মিলিলিটার", "মি.লি.", "مل", "ملل", "மி.லி", "மில்லிலிட்டர்", "mililit", "mililitros", "mililitro", "millilitri", "millilitro", "millilitres", "millilitre"],
  kg: ["公斤", "千克", "किलोग्राम", "किलो", "কেজি", "কিলোগ্রাম", "কিলো", "كغ", "كيلوغرام", "كيلو", "கிலோ", "kilôgam", "kilô", "kilos", "kilo", "chili", "chilo"],
  l: ["升", "लीटर", "লিটার", "لتر", "லிட்டர்", "lít", "litros", "litro", "litri", "litres", "litre"],
  mcg: ["微克", "माइक्रोग्राम", "মাইক্রোগ্রাম", "ميكروغرام", "மைக்ரோகிராம்", "microgam", "microgramos", "microgrammi", "microgrammes"],
};

/** True when the quote writes this unit in a local form right after a number ("40 毫克", "2,5 miligramos"). */
export function hasLocalUnit(quote: string, unit: string): boolean {
  const aliases = UNIT_ALIASES[unit.toLowerCase()] ?? [];
  const q = quote.normalize("NFC");
  return aliases.some((a) => new RegExp(`\\d\\s*${a.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`, "iu").test(q));
}

// ---------------------------------------------------------------- anchors and tidy-up

const COMMON = new Set(
  "leg legs arm arms heart chest head stomach belly back foot feet hand hands eye eyes skin wound blood urine stool stools breath shortness swelling fever pain bleeding weight pressure sugar food meals meal water alcohol salt fluids diet morning evening afternoon night bedtime daily twice once times time day days week weeks month months hour hours minute minutes doctor nurse clinic hospital emergency appointment test tests exercise walk walking rest sleep dizzy dizziness nausea vomiting cough rash redness".split(" "),
);

/**
 * Anchors must be names, numbers with units, dates, times or thresholds. Ordinary words ("leg", "heart",
 * "three times a day") are dropped; names written in lower case (most medicines) are kept.
 */
export function cleanAnchors(statement: string, anchors: string[]): string[] {
  return anchors.filter((a) => {
    if (/\d/.test(a)) return true;
    const words = normLoose(a).split(/[^a-zÀ-ɏ]+/).filter(Boolean);
    if (!words.length) return false;
    const inStatement = statement.match(new RegExp(a.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i"))?.[0] ?? a;
    const capitalised = /^\p{Lu}/u.test(inStatement) || /\b(dr|mr|mrs|ms|prof)\.?\s/i.test(inStatement);
    if (words.every((w) => COMMON.has(w) || w.length <= 3)) return false;
    if (capitalised) return true;
    // Lower-case: keep single long words that are not common (medicines such as "furosemide", "paracetamol").
    return words.length <= 2 && words[0].length >= 6 && !COMMON.has(words[0]);
  });
}

/** "furosemide (Furosemide)" -> "Furosemide": brackets only when the local name differs from the label name. */
export function tidyDuplicateBrackets(text: string): string {
  return text.replace(/([\p{L}][\p{L}\-]*) \(([\p{L}][\p{L}\-]*)\)/gu, (m, a: string, b: string) => (a.toLowerCase() === b.toLowerCase() ? b : m));
}

/** A suspicious-word finding that only complains the word is not English (in a non-English leaflet) is not a finding. */
export const notEnglishComplaint = (why: string) => /\bnot (an? )?english\b|\bnot in english\b|\btransliterat/i.test(why);
