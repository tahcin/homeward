// Records the three sample patients by running the real pipeline once, so visitors can replay them without
// spending the shared Token Factory budget.
//
//   node scripts/build-samples.mjs <base-url> facts   <id...>   # extract facts  -> scripts/runs/<id>.facts.json
//   node scripts/build-samples.mjs <base-url> labels  <id...>   # add short nurse labels to those facts
//   node scripts/build-samples.mjs <base-url> version <id...>   # patient version -> public/samples/<id>.json
//   node scripts/build-samples.mjs <base-url> answers <id...>   # grade the scripted answers in scripts/answers/<id>.json
//
// Between "facts" and "version" a person reviews the facts file, as the nurse would in the app.
// Two samples put a deliberate error into the first draft (a reversed instruction, a doubled dose) so the
// recorded run shows the fact lock catching and repairing it; the app labels these runs as such.

import fs from "node:fs";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname);
const [base, stage, ...ids] = process.argv.slice(2);
if (!base || !stage) {
  console.error("usage: node scripts/build-samples.mjs <base-url> facts|version|answers <id...>");
  process.exit(1);
}

const SAMPLES = {
  "heart-failure-it": { note: "heart-failure-it.txt", language: "it", level: "simple", fault: "negation", patient: "Giuseppe Ferraro, 78", condition: "Heart failure, water pill dose doubled" },
  "anticoagulant-hi": { note: "anticoagulant-hi.txt", language: "hi", level: "very_simple", fault: "dose", patient: "Ramesh Kumar Verma, 66", condition: "Atrial fibrillation, new blood thinner" },
  "knee-replacement-es": { note: "knee-replacement-es.txt", language: "es", level: "simple", patient: "Carmen Ruiz Delgado, 69", condition: "Knee replacement, daily injections at home" },
};

const runs = path.join(here, "runs");
const out = path.join(here, "..", "public", "samples");
fs.mkdirSync(runs, { recursive: true });
fs.mkdirSync(out, { recursive: true });

async function post(p, body) {
  const origin = new URL(base).origin;
  let r = await fetch(base + p, { method: "POST", headers: { "content-type": "application/json", origin }, body: JSON.stringify(body) });
  if (p === "/api/prepare") return r;
  for (let attempt = 1; ; attempt++) {
    const text = await r.text();
    let j;
    try {
      j = JSON.parse(text);
    } catch {
      j = null;
    }
    if (r.ok && j) return j;
    if (attempt >= 3) throw new Error(`${p} ${r.status}: ${text.slice(0, 300)}`);
    r = await fetch(base + p, { method: "POST", headers: { "content-type": "application/json", origin }, body: JSON.stringify(body) });
  }
}

for (const id of ids) {
  const s = SAMPLES[id];
  if (!s) throw new Error(`unknown sample ${id}`);
  const note = fs.readFileSync(path.join(here, "notes", s.note), "utf8");
  const factsFile = path.join(runs, `${id}.facts.json`);
  const sampleFile = path.join(out, `${id}.json`);
  // A recording is staged here and published to public/samples only once it is complete and clean.
  const pendingFile = path.join(runs, `${id}.pending.json`);

  if (stage === "facts") {
    const r = await post("/api/extract", { note });
    fs.writeFileSync(factsFile, JSON.stringify(r, null, 2));
    console.log(id, r.facts.length, "facts", r.topics.length, "topics", r.steps.map((x) => `${x.model} ${x.ms}ms`).join(", "));
  }

  if (stage === "labels") {
    const ex = JSON.parse(fs.readFileSync(factsFile, "utf8"));
    const r = await post("/api/labels", { facts: ex.facts });
    ex.facts = ex.facts.map((f) => (r.labels[f.id] ? { ...f, label: r.labels[f.id] } : f));
    fs.writeFileSync(factsFile, JSON.stringify(ex, null, 2));
    console.log(id, ex.facts.map((f) => `${f.id}: ${f.label ?? "(none)"}`).join(" | "));
  }

  if (stage === "version") {
    const ex = JSON.parse(fs.readFileSync(factsFile, "utf8"));
    // The recorded run must show Gemma writing, and the deliberate error going in when one is asked for;
    // if Gemma was briefly unavailable (DeepSeek fallback) the run is recorded again.
    let events = [];
    let version = null;
    for (let attempt = 1; attempt <= 6 && !version; attempt++) {
      const res = await post("/api/prepare", { facts: ex.facts, topics: ex.topics, language: s.language, level: s.level, fault: s.fault });
      const text = await res.text();
      try {
        events = text.trim().split("\n").map((l) => JSON.parse(l));
      } catch {
        console.log(id, "attempt", attempt, "bad stream:", text.slice(0, 120));
        continue;
      }
      const err = events.find((e) => e.type === "error");
      if (err) {
        console.log(id, "attempt", attempt, "error:", err.message, err.detail ?? "");
        continue;
      }
      const v = events.find((e) => e.type === "done")?.version;
      const writer = events.find((e) => e.type === "step" && e.step.step === "Write the patient version")?.step.model;
      const checkerFailed = v?.lock.some((l) => l.history.some((h) => h.problems.some((p) => /back-translation failed|meaning check gave no answer|second reading of this critical warning sign failed/.test(p.reason))));
      const flagged = v?.lock.filter((l) => l.status === "flagged").map((l) => l.id) ?? [];
      if (!v || writer !== "Gemma-4-31B-IT" || (s.fault && !v.fault) || checkerFailed || flagged.length) {
        if (flagged.length) console.log("   flagged:", flagged.join(", "));
        for (const l of v?.lock ?? []) {
          const lastProblems = l.history.at(-1)?.problems ?? [];
          if (l.status === "flagged" || lastProblems.some((p) => /failed|no answer/.test(p.reason))) {
            console.log("   ", l.id, l.status, lastProblems.map((p) => p.reason.slice(0, 160)).join(" | "));
          }
        }
        for (const e of events) {
          if (e.type === "step" && /failed|not injected|rejected/.test(`${e.step.step} ${e.step.note ?? ""}`)) console.log("    step:", e.step.step, "-", (e.step.note ?? "").slice(0, 200));
        }
        console.log(id, "attempt", attempt, "rejected: writer", writer, "fault", Boolean(v?.fault), "checker failed", Boolean(checkerFailed));
        if (v && s.fault && !v.fault) console.log("   draft starts:", v.paragraphs.slice(0, 3).join(" / ").slice(0, 400));
        continue;
      }
      version = v;
    }
    if (!version) throw new Error(`${id}: no acceptable run after 6 attempts`);
    const steps = [...ex.steps, ...events.filter((e) => e.type === "step").map((e) => e.step)];
    const sample = {
      id,
      patient: s.patient,
      condition: s.condition,
      recorded_at: new Date().toISOString(),
      note,
      facts: ex.facts,
      topics: ex.topics,
      nurse_checks: ex.nurse_checks ?? [],
      version,
      steps,
      answers: {},
    };
    fs.writeFileSync(pendingFile, JSON.stringify(sample, null, 2));
    console.log(id, "(staged, run answers to publish)", version.lock.map((l) => `${l.id}:${l.status}`).join(" "), steps.map((x) => `${x.step} ${x.model} ${x.ms}ms`).join(" | "));
  }

  if (stage === "answers") {
    // scripts/answers/<id>.json: { "<topic id>": ["first answer", "optional retry answer"] }
    const scripted = JSON.parse(fs.readFileSync(path.join(here, "answers", `${id}.json`), "utf8"));
    const sample = JSON.parse(fs.readFileSync(fs.existsSync(pendingFile) ? pendingFile : sampleFile, "utf8"));
    sample.answers = {};
    for (const t of sample.topics) {
      const tries = scripted[t.id];
      if (!tries) continue;
      sample.answers[t.id] = [];
      let factIds = t.fact_ids;
      for (const transcript of tries) {
        const facts = sample.facts.filter((f) => factIds.includes(f.id));
        const r = await post("/api/grade", { facts, question: t.question, transcript, language: sample.version.language });
        sample.answers[t.id].push({ transcript, fact_ids: factIds, ...r });
        console.log(id, t.id, r.grades.map((g) => `${g.id}:${g.verdict}`).join(" "), r.steps.map((x) => `${x.model} ${x.ms}ms`).join(", "));
        factIds = r.grades.filter((g) => g.verdict !== "correct").map((g) => g.id);
        if (!factIds.length) break;
      }
    }
    // Refuse to publish anything incomplete: every topic needs graded answers, and no fact may be flagged.
    const missing = sample.topics.filter((t) => !sample.answers[t.id]?.length).map((t) => t.id);
    const flagged = sample.version.lock.filter((l) => l.status === "flagged").map((l) => l.id);
    if (missing.length || flagged.length) throw new Error(`${id}: not publishing (missing answers: ${missing.join(",") || "none"}; flagged: ${flagged.join(",") || "none"})`);
    fs.writeFileSync(sampleFile, JSON.stringify(sample, null, 2));
    if (fs.existsSync(pendingFile)) fs.unlinkSync(pendingFile);
    console.log(id, "published");
  }
}
