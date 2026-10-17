// Builds public/evidence-results.json for the Evidence page from the ML pipeline's own output
// (../ml/results/eval.json). Runs before every `pnpm build`, so the charts always show the latest run.
// If the ML results are not on this machine, the existing public file is kept as is.
//
// "Ours" (the Homeward grader every chart is benchmarked against) is chosen in this order:
//   1. the HOMEWARD_OURS environment variable (a system key, e.g. HOMEWARD_OURS=g12b_lora pnpm build)
//   2. eval.json's own `ours.system_key`, if the pipeline writes one
//   3. the fine-tuned system (one with a LoRA adapter) with the best accuracy on the independent test set
// Its base model is the system that loads the same model without an adapter.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../../ml/results/eval.json");
const out = resolve(here, "../public/evidence-results.json");

if (!existsSync(src)) {
  console.log(`build-evidence: ${src} not found, keeping ${out}`);
  process.exit(0);
}

const d = JSON.parse(readFileSync(src, "utf8"));
const pct = (x) => (typeof x === "number" ? Math.round(x * 1000) / 10 : null);
const r1 = (x) => (typeof x === "number" ? Math.round(x * 10) / 10 : null);

/** "/workspace/models/gemma-4-E4B-it" -> "Gemma 4 E4B"; "DeepSeek-V4-Flash" -> "DeepSeek V4 Flash". */
function modelName(path) {
  if (!path) return null;
  let b = basename(String(path)).replace(/-(it|IT|instruct|Instruct)$/, "");
  b = b.replace(/-/g, " ").replace(/^gemma/i, "Gemma");
  return b;
}
/** "homeward_run2" -> "run 2"; else "/workspace/runs/lora1/adapter" -> "run 1". */
function runName(adapter, key) {
  const k = /run[_-]?(\d+)$/i.exec(key ?? "");
  if (k) return `run ${Number(k[1])}`;
  if (!adapter) return null;
  const parts = String(adapter).split("/").filter(Boolean);
  const dir = parts[parts.length - 1] === "adapter" ? parts[parts.length - 2] : parts[parts.length - 1];
  const m = /(\d+)$/.exec(dir ?? "");
  return m ? `run ${Number(m[1])}` : dir ?? null;
}

// Parameter counts from amd-act-iii-2026/model-sizes.md, only where that file marks them verified.
// It reads any markdown table with a model-name column ("Token Factory name", "Model" or "Name"; an
// "Official release" column is also matched), "Total params" / "Active params" columns, and either a
// "Confidence" column (a row counts when it starts with "High") or a "Verified"/"Status" column.
// Cells can be verbose: the first "30.7B"-style figure is used, "(= 180B" wins when given (a sum of
// parts), and a bare count like "2,516,756,480" is converted. Dense models show one figure; MoE models
// show "284B total, 13B active"; an "about 3B" active figure (name only) is kept as "about 3B".
const sizesPath = process.env.HOMEWARD_SIZES || resolve(here, "../../../model-sizes.md");
const norm = (x) => String(x ?? "").toLowerCase().replace(/[^a-z0-9.]/g, "");
const sizeRows = [];
if (existsSync(sizesPath)) {
  const lines = readFileSync(sizesPath, "utf8").split("\n");
  let head = null;
  for (const line of lines) {
    if (!line.trim().startsWith("|")) {
      head = null;
      continue;
    }
    const cells = line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
    if (cells.every((c) => /^:?-{2,}:?$/.test(c) || c === "")) continue;
    if (!head) {
      head = cells.map((c) => c.toLowerCase().replace(/\*/g, ""));
      continue;
    }
    const col = (re) => head.findIndex((h) => re.test(h));
    const get = (re) => (col(re) >= 0 ? (cells[col(re)] ?? "").replace(/\*/g, "").trim() : "");
    const model = get(/model|name/);
    if (!model) continue;
    const conf = get(/confidence/);
    const ver = get(/verif|status|confirmed/);
    const verified = conf
      ? /^high/i.test(conf)
      : ver
        ? /\b(verified|yes|confirmed)\b|✓|✔/i.test(ver) && !/\b(unverified|not verified|no|unconfirmed)\b/i.test(ver)
        : false;
    sizeRows.push({
      model,
      release: get(/official|release|repo/),
      key: get(/^(key|system)$/),
      arch: get(/architecture|arch/),
      total: get(/total/),
      active: get(/active/),
      verified,
      source: (get(/source|url|link/).match(/https?:\/\/\S+/) ?? [null])[0],
    });
  }
}
/** First size figure in a verbose cell, in billions. */
function toB(x) {
  if (!x) return null;
  const sum = /\(=\s*([\d.]+)\s*([BMT])/i.exec(x);
  const m = sum ?? /([\d.]+)\s*([BMT])\b/i.exec(x);
  if (m) {
    const v = parseFloat(m[1]);
    const u = m[2].toUpperCase();
    return u === "T" ? v * 1000 : u === "M" ? v / 1000 : v;
  }
  const n = /(\d{1,3}(?:,\d{3}){2,})/.exec(x);
  return n ? Number(n[1].replace(/,/g, "")) / 1e9 : null;
}
const fmtB = (b) => (b === null ? null : b >= 100 ? `${Math.round(b)}B` : b >= 10 ? `${Number(b.toFixed(1))}B` : `${Number(b.toFixed(2)).toString().replace(/\.?0+$/, "")}B`);
function sizeFor(key, name) {
  const n = norm(name);
  const exact = (r) => norm(r.model) === n || norm(r.release.split(" ")[0]) === n || norm(r.model).replace(/it$/, "") === n;
  const loose = (r) => n.length > 3 && (norm(r.model).startsWith(n) || n.startsWith(norm(r.model)) || norm(r.release).includes(n));
  const row = sizeRows.find((r) => r.key && norm(r.key) === norm(key)) ?? sizeRows.find(exact) ?? sizeRows.find(loose);
  const none = { verified: false, total: null, active: null, total_b: null, active_b: null, display: "size not published" };
  if (!row || !row.verified) return none;
  const total_b = toB(row.total);
  const dense = /^dense/i.test(row.arch) || /\(dense/i.test(row.active);
  const effective = /effective/i.test(row.active) ? toB(row.active) : null;
  const about = /^about/i.test(row.active);
  // Encoder-decoder models give two active figures; quote the one used while generating.
  const decode = /([\d.]+)\s*B per token in decode/i.exec(row.active);
  const active_b = dense ? total_b : decode ? parseFloat(decode[1]) : toB(row.active);
  if (total_b === null) return none;
  const total = fmtB(total_b);
  const active = active_b === null ? null : `${about ? "about " : ""}${fmtB(active_b)}`;
  let display;
  if (effective) display = `${total} (${fmtB(effective)} effective)`;
  else if (dense || !active || active_b === total_b) display = total;
  else display = `${total} total, ${active} active`;
  return { verified: true, total, active, total_b, active_b: dense ? total_b : active_b, display, source: row.source };
}

const indep = d.test_sets?.independent?.systems ?? {};
const adapterSystems = Object.entries(indep).filter(([, s]) => s.meta?.adapter);
const best = adapterSystems.sort((a, b) => (b[1].metrics?.accuracy ?? 0) - (a[1].metrics?.accuracy ?? 0))[0]?.[0];
const oursKey = process.env.HOMEWARD_OURS || d.ours?.system_key || best || "e4b_lora";
const oursMeta = indep[oursKey]?.meta ?? {};
const baseKey =
  Object.entries(indep).find(([k, s]) => k !== oursKey && !s.meta?.adapter && s.meta?.model && s.meta.model === oursMeta.model)?.[0] ?? null;
const oursModel = modelName(oursMeta.model) ?? "Gemma 4 E4B";
// The teacher whose labels Homeward was distilled from: HOMEWARD_TEACHER, eval.json's teacher.system_key,
// or the Gemma 4 31B system.
const teacherKey =
  process.env.HOMEWARD_TEACHER ||
  d.teacher?.system_key ||
  Object.entries(indep).find(([, s]) => !s.meta?.adapter && /gemma.?4.?31b/i.test(String(s.meta?.model ?? "")))?.[0] ||
  null;
const oursRun = runName(oursMeta.adapter, oursKey);
const costs = d.cost?.per_system ?? {};

function labelFor(key, s) {
  const m = s.meta ?? {};
  if (key === oursKey) return "Homeward (fine-tuned)";
  const name = modelName(m.model) ?? key;
  if (key === baseKey) return `${name}, base`;
  if (key === teacherKey) return `${name}, teacher${s.meta?.device ? "" : " (cloud)"}`;
  return s.meta?.device ? name : `${name} (cloud)`;
}
function roleFor(key, s) {
  if (key === oursKey) return "ours";
  if (key === baseKey) return "base";
  if (key === teacherKey) return "teacher";
  return s.meta?.device ? "onsite" : "cloud";
}
function runsOn(s) {
  return s.meta?.device ? s.meta.device.replace("Radeon Pro", "Radeon PRO") : "AMD GPUs via AMD Token Factory";
}

function testSet(t) {
  if (!t) return null;
  // Only the selected Homeward grader is shown; intermediate fine-tuning runs are left out.
  const systems = Object.entries(t.systems || {})
    .filter(([id, s]) => id === oursKey || !s.meta?.adapter)
    .map(([id, s]) => {
    const m = s.metrics || {};
    const ci = (k) => (Array.isArray(m[k]) ? m[k].map(pct) : null);
    const pk = Object.keys(m).find((k) => k.startsWith("paired_vs_"));
    const pv = pk ? m[pk] : null;
    return {
      id,
      role: roleFor(id, s),
      label: labelFor(id, s),
      model: modelName(s.meta?.model),
      run: id === oursKey ? oursRun : null,
      size: sizeFor(id, modelName(s.meta?.model)),
      runs_on: runsOn(s),
      accuracy: pct(m.accuracy),
      accuracy_ci: ci("accuracy_95ci"),
      macro_f1: m.macro_f1 ?? null,
      false_reassurance: pct(m.false_reassurance_rate),
      false_reassurance_ci: ci("false_reassurance_rate_95ci"),
      false_reassurance_n: m.false_reassurance_n ?? null,
      critical_false_reassurance_n: m.false_reassurance_critical_n ?? null,
      over_flag: pct(m.over_flag_rate),
      json_valid: pct(m.json_valid_rate),
      ci_method: m.method ?? null,
      paired_vs_base: pv ? { base: pk.slice("paired_vs_".length), fixes: pv.this_right_ref_wrong, breaks: pv.this_wrong_ref_right, p: pv.mcnemar_exact_p } : null,
      latency_bs1_p50_s: r1(s.meta?.bs1?.p50_s ?? null),
      usd_per_discharge: costs[id]?.usd_per_discharge ?? null,
    };
  });
  // Ours first, then its base, then the rest in file order.
  const rank = { ours: 0, base: 1, teacher: 2, onsite: 3, cloud: 4 };
  systems.sort((a, b) => rank[a.role] - rank[b.role]);
  return {
    answers: t.test_set?.answers ?? null,
    facts: t.test_set?.facts ?? null,
    cases: t.test_set?.cases ?? null,
    languages: t.test_set?.languages ?? [],
    labels: t.test_set?.labels ?? null,
    systems,
  };
}

const asr = d.asr || {};
const wer = {};
for (const [sys, bySrc] of Object.entries(asr.asr || {})) {
  wer[sys] = {};
  for (const [srcName, byLang] of Object.entries(bySrc)) {
    wer[sys][srcName] = {};
    for (const [lang, v] of Object.entries(byLang)) {
      wer[sys][srcName][lang] = {
        clips: v.clips,
        words: v.words,
        word_edits: v.word_edits,
        wer: pct(v.wer),
        wer_normalised: pct(v.normalised?.wer),
        cer: pct(v.cer),
        cer_normalised: pct(v.normalised?.cer),
      };
    }
  }
}

const loop = asr.onprem_loop || {};
const onprem = d.cost?.onprem_w7900d || {};
const capacity = Object.entries(d.capacity || {})
  .map(([bs, c]) => ({ batch: Number(bs), per_hour: Math.round(c.answers_per_s * 3600), peak_vram_gib: r1(c.peak_vram_gib) }))
  .sort((a, b) => a.batch - b.batch);

const result = {
  schema: "homeward.evidence.v2",
  source: "ml/results/eval.json",
  generated_at: d.generated_at ?? null,
  sizes_file: existsSync(sizesPath) ? "model-sizes.md" : null,
  whisper_size: sizeFor("whisper", "whisper-large-v3-turbo"),
  teacher_key: teacherKey,
  ours: {
    system_key: oursKey,
    label: "Homeward (fine-tuned)",
    model: oursModel,
    run: oursRun,
    description: oursMeta.adapter ? `${oursModel} fine-tuned with LoRA${oursRun ? `, ${oursRun}` : ""}` : oursModel,
    base_key: baseKey,
    lora: d.training_args ? { rank: d.training_args.rank, alpha: d.training_args.alpha, epochs: d.training_args.epochs } : null,
    asr_system: "e4b",
    asr_label: `${oursModel} speech recognition`,
  },
  grader: {
    independent: testSet(d.test_sets?.independent),
    in_distribution: testSet(d.test_sets?.in_distribution),
  },
  asr: {
    clips: asr.clips ?? null,
    // The independent transcripts were written by Claude, not by people; say so wherever the note travels.
    tts_note: asr.tts_note ? asr.tts_note.replace(/hand-written independent test transcripts/g, "independent test transcripts written by a different model family (Claude)").replace(/hand-written/g, "Claude-written") : null,
    normalisation: asr.normalisation ?? null,
    wer,
    latency_s: asr.asr_latency_s ?? null,
    grading_from_transcripts: asr.grading_from_transcripts ?? null,
  },
  onprem_loop: {
    clips: loop.clips ?? null,
    p50_s: r1(loop.loop_p50_s),
    p95_s: r1(loop.loop_p95_s),
    asr_p50_s: r1(loop.asr_p50_s),
    grade_p50_s: r1(loop.grade_p50_s),
    peak_vram_gib: r1(loop.peak_vram_gib),
  },
  capacity: {
    by_batch: capacity,
    discharges_per_hour_per_card: onprem.discharges_per_hour_per_card ? Math.round(onprem.discharges_per_hour_per_card) : null,
    grading_only_discharges_per_hour_per_card: onprem.grading_only_discharges_per_hour_per_card ? Math.round(onprem.grading_only_discharges_per_hour_per_card) : null,
    answers_per_discharge: d.cost?.answers_per_discharge_assumption ?? null,
    assumptions: onprem.assumptions ?? null,
  },
  training: d.training
    ? {
        device: d.training.device?.replace("Radeon Pro", "Radeon PRO"),
        examples: d.training.train_examples,
        minutes: r1(d.training.train_wall_s / 60),
        peak_vram_gib: r1(d.training.peak_vram_allocated_gib),
      }
    : null,
};

writeFileSync(out, JSON.stringify(result, null, 1) + "\n");
console.log(`build-evidence: wrote ${out} from ${d.generated_at}; ours = ${oursKey} (${result.ours.description}), base = ${baseKey}`);
