// Fact-lock benchmark: runs the real Homeward pipeline (extract the facts from each synthetic discharge note, write
// the patient version, lock every fact) with ONE planted fault per case, plus clean controls, and records what the
// lock caught, how, and what it flagged that was not planted.
//
//   node scripts/lock-bench.mjs <base-url> [--n 24] [--concurrency 2] [--max-usd 0.40] [--dry]
//
// Writes ../ml/results/lock_bench.json and lock_bench.md. Cases come from ../data/cases.jsonl and cases_v2.jsonl.

import fs from "node:fs";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname);
const root = path.join(here, "..", "..");
const args = process.argv.slice(2);
const base = args[0];
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : dflt;
};
const N = Number(opt("n", 24));
const CONC = Number(opt("concurrency", 2));
const MAX_USD = Number(opt("max-usd", 0.4));
const DRY = args.includes("--dry");
if (!base) {
  console.error("usage: node scripts/lock-bench.mjs <base-url> [--n 24] [--concurrency 2] [--max-usd 0.40] [--dry]");
  process.exit(1);
}

const LANGS = ["hi", "bn", "ar", "zh", "it", "es", "ta", "vi"];
const FAULTS = ["dose", "frequency", "drop_negation", "negation", "date", "drop_symptom", "mistranslate", "dose_half"];
const FAULT_LABEL = {
  dose: "dose doubled",
  dose_half: "dose halved",
  frequency: "wrong frequency",
  drop_negation: "negation dropped",
  negation: "action reversed (stop becomes continue)",
  date: "wrong date",
  drop_symptom: "warning symptom dropped",
  mistranslate: "word mistranslated",
  none: "clean control",
};

// Which facts each fault can target (mirrors the Worker's targeting, so the plan only asks for possible faults).
const CAN = {
  dose: (f) => f.category.startsWith("medication") && f.critical && f.anchors.some((a) => /\d/.test(a) && /mg|ml|g\b/i.test(a)) && f.anchors.some((a) => !/\d/.test(a)),
  dose_half: (f) => CAN.dose(f),
  negation: (f) => /\b(stop|do not|don't|avoid|hold)\b/i.test(f.statement) && f.anchors.some((a) => !/\d/.test(a)),
  frequency: (f) => f.category.startsWith("medication") && /(once|twice|three times|every \d+|daily|a day|per day|bedtime|morning|evening)/i.test(f.statement),
  drop_negation: (f) => /\b(do not|don't|never|avoid|no)\b/i.test(f.statement),
  date: (f) => f.category === "follow_up" && /\d/.test(f.statement),
  drop_symptom: (f) => f.category === "warning_sign" && /,|\bor\b/.test(f.statement),
  mistranslate: (f) => f.critical,
};

function loadCases() {
  const rows = [];
  for (const fn of ["cases.jsonl", "cases_v2.jsonl"]) {
    const p = path.join(root, "data", fn);
    if (!fs.existsSync(p)) continue;
    for (const line of fs.readFileSync(p, "utf8").split("\n")) if (line.trim()) rows.push({ ...JSON.parse(line), _src: fn });
  }
  // Interleave languages so consecutive cases differ: hi, bn, ar, ... then the next case of each.
  const byLang = Object.fromEntries(LANGS.map((l) => [l, rows.filter((r) => r.language === l).sort((a, b) => a.case_id.localeCompare(b.case_id))]));
  const out = [];
  for (let k = 0; out.length < N; k++) {
    let added = false;
    for (const l of LANGS) {
      if (out.length >= N) break;
      if (byLang[l][k]) {
        out.push(byLang[l][k]);
        added = true;
      }
    }
    if (!added) break;
  }
  return out;
}

async function post(p, body) {
  const origin = new URL(base).origin;
  for (let attempt = 1; ; attempt++) {
    const r = await fetch(base + p, { method: "POST", headers: { "content-type": "application/json", origin }, body: JSON.stringify(body) });
    if (p === "/api/prepare" && r.ok) return r;
    const text = await r.text();
    let j = null;
    try {
      j = JSON.parse(text);
    } catch {
      /* not JSON */
    }
    if (r.ok && j) return j;
    if (attempt >= 3) throw new Error(`${p} ${r.status}: ${text.slice(0, 200)}`);
    await new Promise((res) => setTimeout(res, 5000 * attempt));
  }
}

function detector(reason) {
  if (/back-translation failed|no answer|reading .* failed/.test(reason)) return "checker failure";
  if (/^no sentence|word for word/.test(reason)) return "code: no sentence carries it";
  if (/English left untranslated/.test(reason)) return "code: English left in the text";
  if (/suspicious wording/.test(reason)) return "literal back-translation (suspicious word)";
  if (/second reading found/.test(reason)) return "second reading";
  if (/meaning differs/.test(reason)) return "meaning check";
  return "code: anchor check (number, unit, date or name)";
}

const usage = (steps) => ({
  tokens_in: steps.reduce((n, s) => n + (s.tokens_in ?? 0), 0),
  tokens_out: steps.reduce((n, s) => n + (s.tokens_out ?? 0), 0),
  cost_usd: steps.reduce((n, s) => n + (s.cost_usd ?? 0), 0),
  calls: steps.filter((s) => s.model !== "code").length,
});

let spent = 0;

async function runCase(c, planned) {
  const t0 = Date.now();
  const ex = await post("/api/extract", { note: c.summary_text });
  const steps = [...ex.steps];
  // Choose the planned fault if this case has a fact it can target; otherwise the next fault type that can.
  let fault = planned;
  if (fault !== "none" && !ex.facts.some(CAN[fault])) {
    const i = FAULTS.indexOf(fault);
    fault = [...FAULTS.slice(i + 1), ...FAULTS.slice(0, i)].find((k) => ex.facts.some(CAN[k])) ?? "none";
  }
  const res = await post("/api/prepare", { facts: ex.facts, topics: ex.topics, language: c.language, level: "simple", ...(fault !== "none" ? { fault } : {}) });
  const events = (await res.text())
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l))
    .filter((e) => e.type !== "ping");
  const err = events.find((e) => e.type === "error");
  if (err) throw new Error(`${c.case_id}: ${err.message} ${err.detail ?? ""}`);
  const v = events.find((e) => e.type === "done")?.version;
  if (!v) throw new Error(`${c.case_id}: no version`);
  steps.push(...events.filter((e) => e.type === "step").map((e) => e.step));
  const u = usage(steps);
  spent += u.cost_usd;

  const lockById = Object.fromEntries(v.lock.map((l) => [l.id, l]));
  const round0 = (id) => lockById[id]?.history.find((h) => h.round === 0)?.problems ?? [];
  const injected = fault !== "none" && Boolean(v.fault);
  const fid = v.fault?.fact_id;
  const finalText = v.paragraphs.join(" ");
  let planted = null;
  if (injected) {
    const probs = round0(fid);
    const caught = probs.length > 0;
    planted = {
      kind: fault,
      fact_id: fid,
      critical: ex.facts.find((f) => f.id === fid)?.critical ?? false,
      before: v.fault.before,
      after: v.fault.after,
      caught,
      detectors: [...new Set(probs.map((p) => detector(p.reason)))],
      reasons: probs.map((p) => p.reason),
      final_status: lockById[fid]?.status,
      // A missed fault is only harmless if the damaged sentence no longer appears in the final text.
      shipped: !caught && finalText.includes(v.fault.after),
    };
  }
  // Problems on facts other than the planted one: genuine writer errors or false alarms (read the reasons).
  const others = v.lock
    .filter((l) => l.id !== fid && round0(l.id).length > 0)
    .map((l) => ({ id: l.id, final_status: l.status, detectors: [...new Set(round0(l.id).map((p) => detector(p.reason)))], reasons: round0(l.id).map((p) => p.reason) }));
  return {
    case_id: c.case_id,
    source: c._src,
    language: c.language,
    condition: c.condition,
    planned_fault: planned,
    fault: injected ? fault : "none",
    fault_requested_but_not_injected: fault !== "none" && !injected,
    facts: v.lock.length,
    planted,
    other_facts_with_issues: others,
    final: { locked: v.lock.filter((l) => l.status === "locked").length, repaired: v.lock.filter((l) => l.status === "repaired").length, flagged: v.lock.filter((l) => l.status === "flagged").length },
    writer: steps.find((s) => s.step === "Write the patient version")?.model,
    fallbacks: steps.filter((s) => s.fallback_from).length,
    ...u,
    wall_s: Math.round((Date.now() - t0) / 1000),
  };
}

function pct(a, b) {
  return b ? `${Math.round((a / b) * 100)}%` : "n/a";
}

function summarize(results) {
  const planted = results.filter((r) => r.planted);
  const caught = planted.filter((r) => r.planted.caught);
  const shipped = planted.filter((r) => r.planted.shipped);
  const cleanFacts = results.reduce((n, r) => n + r.facts - (r.planted ? 1 : 0), 0);
  const otherIssues = results.reduce((n, r) => n + r.other_facts_with_issues.length, 0);
  const controls = results.filter((r) => r.fault === "none");
  const controlFacts = controls.reduce((n, r) => n + r.facts, 0);
  const controlIssues = controls.reduce((n, r) => n + r.other_facts_with_issues.length, 0);
  const finalFlagged = results.reduce((n, r) => n + r.final.flagged, 0);
  const totalFacts = results.reduce((n, r) => n + r.facts, 0);
  const group = (key) => {
    const out = {};
    for (const r of results) {
      const k = key(r);
      out[k] ??= { cases: 0, planted: 0, caught: 0, facts: 0, other_issues: 0, final_flagged: 0 };
      out[k].cases++;
      out[k].facts += r.facts;
      out[k].other_issues += r.other_facts_with_issues.length;
      out[k].final_flagged += r.final.flagged;
      if (r.planted) {
        out[k].planted++;
        if (r.planted.caught) out[k].caught++;
      }
    }
    return out;
  };
  const byDetector = {};
  for (const r of caught) for (const d of r.planted.detectors) byDetector[d] = (byDetector[d] ?? 0) + 1;
  return {
    cases: results.length,
    languages: [...new Set(results.map((r) => r.language))],
    planted_faults: planted.length,
    caught: caught.length,
    shipped_uncaught: shipped.length,
    planted_final_status: Object.fromEntries(["locked", "repaired", "flagged"].map((s) => [s, planted.filter((r) => r.planted.final_status === s).length])),
    non_planted_facts: cleanFacts,
    non_planted_facts_with_issues: otherIssues,
    control_cases: controls.length,
    control_facts: controlFacts,
    control_facts_with_issues: controlIssues,
    total_facts: totalFacts,
    final_flagged_facts: finalFlagged,
    caught_by: byDetector,
    by_fault: group((r) => r.fault),
    by_language: group((r) => r.language),
    cost_usd: results.reduce((n, r) => n + r.cost_usd, 0),
    calls: results.reduce((n, r) => n + r.calls, 0),
    tokens_in: results.reduce((n, r) => n + r.tokens_in, 0),
    tokens_out: results.reduce((n, r) => n + r.tokens_out, 0),
    median_wall_s: [...results.map((r) => r.wall_s)].sort((a, b) => a - b)[Math.floor(results.length / 2)] ?? null,
  };
}

function markdown(sum, results, failures) {
  const L = [];
  L.push("# Fact-lock benchmark", "");
  L.push(
    `Real Homeward pipeline on AMD Token Factory (Gemma 4 31B extracts and writes, DeepSeek V4 Flash checks), ${sum.cases} synthetic discharge notes in ${sum.languages.length} languages (${sum.languages.join(", ")}). One planted fault per case was put into the first draft; ${sum.control_cases} cases were clean controls. Facts were extracted by the pipeline itself, without nurse review.`,
    "",
  );
  L.push(`**Caught ${sum.caught} of ${sum.planted_faults} planted faults** (${pct(sum.caught, sum.planted_faults)}). Uncaught faults still present in the final text: ${sum.shipped_uncaught}.`, "");
  L.push(
    `**Non-planted findings:** ${sum.non_planted_facts_with_issues} of ${sum.non_planted_facts} other facts had an issue raised on the first draft (these are real writer errors or false alarms; reasons below). In the ${sum.control_cases} clean controls: ${sum.control_facts_with_issues} of ${sum.control_facts} facts. Facts still flagged for the nurse at the end: ${sum.final_flagged_facts} of ${sum.total_facts}.`,
    "",
  );
  L.push(`Planted fact final status: ${Object.entries(sum.planted_final_status).map(([k, v]) => `${k} ${v}`).join(", ")}.`, "");
  L.push(`Cost: $${sum.cost_usd.toFixed(4)} for ${sum.calls} model calls (${sum.tokens_in.toLocaleString()} tokens in, ${sum.tokens_out.toLocaleString()} out). Median wall time per case: ${sum.median_wall_s} s.`, "");
  L.push("## Caught by", "", "| Check | Faults caught |", "|---|---|");
  for (const [k, v] of Object.entries(sum.caught_by).sort((a, b) => b[1] - a[1])) L.push(`| ${k} | ${v} |`);
  L.push("", "A fault can be caught by more than one check.", "");
  L.push("## By fault type", "", "| Fault | Cases | Caught | Other facts with issues |", "|---|---|---|---|");
  for (const [k, g] of Object.entries(sum.by_fault)) L.push(`| ${FAULT_LABEL[k] ?? k} | ${g.cases} | ${g.planted ? `${g.caught} of ${g.planted}` : "n/a"} | ${g.other_issues} of ${g.facts - g.planted} |`);
  L.push("", "## By language", "", "| Language | Cases | Caught | Other facts with issues | Flagged at the end |", "|---|---|---|---|---|");
  for (const [k, g] of Object.entries(sum.by_language)) L.push(`| ${k} | ${g.cases} | ${g.planted ? `${g.caught} of ${g.planted}` : "n/a"} | ${g.other_issues} of ${g.facts - g.planted} | ${g.final_flagged} |`);
  L.push("", "## Per case", "", "| Case | Lang | Fault | Caught by | Planted fact final | Other issues |", "|---|---|---|---|---|---|");
  for (const r of results) {
    L.push(
      `| ${r.case_id} | ${r.language} | ${FAULT_LABEL[r.fault]}${r.fault_requested_but_not_injected ? " (not injected)" : ""} | ${r.planted ? (r.planted.caught ? r.planted.detectors.join("; ") : `missed${r.planted.shipped ? ", shipped" : ""}`) : ""} | ${r.planted?.final_status ?? ""} | ${r.other_facts_with_issues.length} |`,
    );
  }
  L.push("", "## Planted faults (what was changed)", "");
  for (const r of results.filter((x) => x.planted)) {
    L.push(`- ${r.case_id} (${r.language}, ${FAULT_LABEL[r.fault]}, ${r.planted.fact_id}): "${r.planted.before}" -> "${r.planted.after}". ${r.planted.caught ? `Caught: ${r.planted.reasons[0]?.slice(0, 160)}` : "Not caught."}`);
  }
  L.push("", "## Non-planted findings (to read: writer error or false alarm?)", "");
  for (const r of results) for (const o of r.other_facts_with_issues) L.push(`- ${r.case_id} ${r.language} ${o.id} (${o.final_status}): ${o.reasons[0]?.slice(0, 180)}`);
  if (failures.length) {
    L.push("", "## Cases that did not complete", "");
    for (const f of failures) L.push(`- ${f.case_id}: ${f.error.slice(0, 200)}`);
  }
  return L.join("\n") + "\n";
}

const cases = loadCases();
const plan = cases.map((c, i) => ({ c, fault: (i + 1) % 4 === 0 ? "none" : FAULTS[i - Math.floor((i + 1) / 4)] ?? FAULTS[i % FAULTS.length] }));
// Re-spread the faults over the non-control slots so each type appears about equally.
let k = 0;
for (const p of plan) if (p.fault !== "none") p.fault = FAULTS[k++ % FAULTS.length];
const todo = DRY ? plan.slice(0, 1) : plan;
console.log(`${todo.length} cases: ${todo.map((p) => `${p.c.case_id}/${p.c.language}/${p.fault}`).join(" ")}`);

const results = [];
const failures = [];
let next = 0;
async function worker() {
  while (next < todo.length) {
    const p = todo[next++];
    if (spent > MAX_USD) {
      failures.push({ case_id: p.c.case_id, error: `skipped: budget of $${MAX_USD} reached` });
      continue;
    }
    try {
      const r = await runCase(p.c, p.fault);
      results.push(r);
      console.log(
        `${r.case_id} ${r.language} ${r.fault}: ${r.planted ? (r.planted.caught ? `caught by ${r.planted.detectors.join(", ")}` : "MISSED") : "control"}; other issues ${r.other_facts_with_issues.length}; final ${JSON.stringify(r.final)}; $${r.cost_usd.toFixed(4)}; ${r.wall_s}s (spent $${spent.toFixed(4)})`,
      );
    } catch (e) {
      failures.push({ case_id: p.c.case_id, error: String(e.message ?? e) });
      console.log(`${p.c.case_id} FAILED: ${String(e.message ?? e).slice(0, 200)}`);
    }
  }
}
await Promise.all(Array.from({ length: CONC }, worker));

results.sort((a, b) => a.case_id.localeCompare(b.case_id));
const summary = summarize(results);
const outDir = path.join(root, "ml", "results");
const suffix = DRY ? "_dry" : "";
fs.writeFileSync(path.join(outDir, `lock_bench${suffix}.json`), JSON.stringify({ generated_at: new Date().toISOString(), base, summary, results, failures }, null, 2));
fs.writeFileSync(path.join(outDir, `lock_bench${suffix}.md`), markdown(summary, results, failures));
console.log(JSON.stringify({ caught: summary.caught, planted: summary.planted_faults, other_issues: summary.non_planted_facts_with_issues, of: summary.non_planted_facts, cost: summary.cost_usd.toFixed(4) }));
