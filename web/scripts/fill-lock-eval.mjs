// Copies the fact-lock benchmark totals (../ml/results/lock_bench.json) into public/eval.json -> fact_lock.
// Only numbers from the benchmark run are written; nothing is typed in by hand.
//   node scripts/fill-lock-eval.mjs

import fs from "node:fs";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname);
const bench = JSON.parse(fs.readFileSync(path.join(here, "..", "..", "ml", "results", "lock_bench.json"), "utf8"));
const evalPath = path.join(here, "..", "public", "eval.json");
const ev = JSON.parse(fs.readFileSync(evalPath, "utf8"));
const s = bench.summary;
const r = bench.results;

const firstPass = r.reduce((n, c) => n + c.final.locked, 0);
const caughtBy = Object.entries(s.caught_by)
  .sort((a, b) => b[1] - a[1])
  .map(([k, v]) => `${k} ${v}`)
  .join(", ");

ev.fact_lock = {
  description: `${s.cases} synthetic discharge notes in ${s.languages.length} languages run through the full pipeline on AMD Token Factory (Gemma 4 31B extracts and writes, DeepSeek V4 Flash checks), facts extracted without nurse review.`,
  source: "ml/results/lock_bench.json",
  generated_at: bench.generated_at,
  versions: s.cases,
  facts: s.total_facts,
  locked_first_pass: firstPass,
  repaired: r.reduce((n, c) => n + c.final.repaired, 0),
  flagged: s.final_flagged_facts,
  injected_faults: s.planted_faults,
  injected_faults_caught: s.caught,
  shipped_uncaught: s.shipped_uncaught,
  shipped_kinds: [...new Set(r.filter((c) => c.planted?.shipped).map((c) => ({ frequency: "a wrong frequency", drop_symptom: "a dropped warning symptom", dose: "a doubled dose", dose_half: "a halved dose", drop_negation: "a dropped negation", negation: "a reversed action", date: "a wrong date", mistranslate: "a mistranslated word" })[c.fault] ?? c.fault))].join(", "),
  caught_by: caughtBy,
  non_planted_facts: s.non_planted_facts,
  non_planted_issues: s.non_planted_facts_with_issues,
  control_cases: s.control_cases,
  control_facts: s.control_facts,
  control_issues: s.control_facts_with_issues,
  cost_usd: Number(s.cost_usd.toFixed(4)),
  median_wall_s: s.median_wall_s,
};
fs.writeFileSync(evalPath, JSON.stringify(ev, null, 2) + "\n");
console.log(ev.fact_lock);
