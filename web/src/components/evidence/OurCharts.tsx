import { useState } from "react";
import { ChartCard, COLORS, HChart, Legend, type Group } from "./HChart.tsx";
import { LANG, type Results, type SystemResult } from "./types.ts";

// Every chart here puts the Homeward grader first, in the accent colour; comparators are muted.
const colorOf = (s: SystemResult) => (s.role === "ours" ? COLORS.gemma : COLORS.muted);
const f1 = (v: number) => `${v.toFixed(1)}%`;
const pts = (d: number) => `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)} points`;
const times = (n: number) => (n === 1 ? "once" : n === 2 ? "twice" : `${n} times`);
const pfmt = (p: number) => (p < 0.001 ? p.toExponential(1) : p.toFixed(p < 0.01 ? 3 : 2));

export function useSystems(r: Results) {
  const t = r.grader.independent;
  const sys = t?.systems ?? [];
  const ours = sys.find((s) => s.role === "ours") ?? null;
  const base = sys.find((s) => s.role === "base") ?? null;
  const others = sys.filter((s) => s.role !== "ours");
  return { t, sys, ours, base, others };
}

const SOURCE_OURS = (
  <>
    Source: Homeward evaluation run, <code>ml/results/eval.json</code>. On-site systems ran on an AMD Radeon PRO W7900D; cloud systems on AMD GPUs through AMD Token Factory.
  </>
);

function sysGroup(s: SystemResult, r: Results, item: Group["items"][number]): Group {
  return {
    label: s.label,
    sub: s.role === "ours" ? r.ours.description : s.runs_on ?? undefined,
    strong: s.role === "ours",
    items: [item],
  };
}

const LEGEND_OURS = (r: Results, shape: "bar" | "dot" = "bar") => (
  <Legend
    items={[
      { label: `${r.ours.label}: ${r.ours.description}`, color: COLORS.gemma, shape },
      { label: "Comparators", color: COLORS.muted, shape },
    ]}
  />
);

export function GradingChart({ r }: { r: Results }) {
  const { t, sys, ours, base } = useSystems(r);
  if (!t || !ours || ours.accuracy === null) return null;
  const valid = sys.filter((s) => s.accuracy !== null);
  const lo = Math.min(...valid.map((s) => s.accuracy_ci?.[0] ?? s.accuracy ?? 100));
  const d0 = Math.max(0, Math.floor((lo - 5) / 10) * 10);
  const groups = valid.map((s) =>
    sysGroup(s, r, {
      series: "Fact accuracy",
      value: s.accuracy,
      lo: s.accuracy_ci?.[0],
      hi: s.accuracy_ci?.[1],
      color: colorOf(s),
      display: f1(s.accuracy!),
      detail: [
        ...(s.accuracy_ci ? [`95% interval ${s.accuracy_ci[0]}–${s.accuracy_ci[1]}%`] : []),
        ...(s.role !== "ours" && s.accuracy !== null ? [`Homeward ${pts(ours.accuracy! - s.accuracy)} against this`] : []),
      ],
    }),
  );
  const ahead = valid.filter((s) => s.role !== "ours" && s.accuracy! > ours.accuracy!);
  const behind = valid.filter((s) => s.role !== "ours" && s.role !== "base" && s.accuracy! < ours.accuracy!);
  const parts: string[] = [];
  if (base?.accuracy != null) parts.push(`${pts(ours.accuracy - base.accuracy)} over base ${base.model ?? "model"}`);
  behind.forEach((s) => parts.push(`${pts(ours.accuracy! - s.accuracy!)} over ${s.label.replace(" (cloud)", "")}`));
  const title = `${ours.label} grades ${f1(ours.accuracy)} of facts correctly, ${parts.join(" and ")}${
    ahead.length ? `; ${ahead.map((s) => `${s.label} is still ahead at ${f1(s.accuracy!)}`).join(", ")}` : ""
  }`;
  const pv = ours.paired_vs_base;
  return (
    <ChartCard
      id="grade"
      title={title}
      sub={
        <>
          Share of {t.facts} facts in {t.answers} independent teach-back answers ({t.cases} cases, {t.languages.length} languages), written by a different model family (Claude) and never seen by the data generator or in training, where the grader's verdict matched the designed label. Dots are measured accuracy; whiskers are 95% intervals.
          {pv && base ? ` On the same facts, fine-tuning fixed ${pv.fixes} that the base model got wrong and broke ${pv.breaks} it got right (exact McNemar p = ${pfmt(pv.p)}).` : ""}
        </>
      }
      caveats={[
        `Small test set: ${t.answers} answers from ${t.cases} cases, so the intervals are wide. They come from a ${ours.ci_method ?? "case-level bootstrap"}, which resamples whole cases.`,
        "The answers are synthetic: written by a different model family (Claude) to read like patients, never seen by the data generator or in training, and from conditions held out of training. Real patients will be messier.",
        "The axis starts above zero to show the intervals; read differences from the numbers, not the distance from the left edge.",
      ]}
      source={SOURCE_OURS}
      table={{
        head: ["System", "Runs on", "Fact accuracy", "95% interval", "Homeward vs this", "Macro-F1"],
        numeric: [false, false, true, true, true, true],
        rows: valid.map((s) => [
          s.label,
          s.runs_on ?? "",
          f1(s.accuracy!),
          s.accuracy_ci ? `${s.accuracy_ci[0]}–${s.accuracy_ci[1]}%` : "",
          s.role === "ours" ? "" : pts(ours.accuracy! - s.accuracy!),
          s.macro_f1 !== null ? s.macro_f1.toFixed(3) : "",
        ]),
      }}
    >
      <HChart groups={groups} domain={[d0, 100]} mark="dot" format={(v) => `${v}%`} axisLabel="Facts graded correctly (%)" ariaLabel="Grading accuracy with 95% intervals, Homeward first" labelWidth={260} />
      {LEGEND_OURS(r, "dot")}
    </ChartCard>
  );
}

export function ReassuranceChart({ r }: { r: Results }) {
  const { sys, ours, base } = useSystems(r);
  const list = sys.filter((s) => s.critical_false_reassurance_n);
  if (!ours?.critical_false_reassurance_n || !list.length) return null;
  const [on, N] = ours.critical_false_reassurance_n;
  const max = Math.max(...list.map((s) => s.critical_false_reassurance_n![0]));
  const best = [...list].sort((a, b) => a.critical_false_reassurance_n![0] - b.critical_false_reassurance_n![0])[0];
  let vsBase = "";
  if (base?.critical_false_reassurance_n) {
    const bn = base.critical_false_reassurance_n[0];
    vsBase = on < bn ? `, fewer than base ${base.model} (${bn})` : on === bn ? `, the same as base ${base.model}` : `, so it is not yet better than base ${base.model} (${bn})`;
  }
  const title = `${ours.label} called a misunderstood critical fact "understood" ${times(on)} in ${N}${vsBase}${
    best.role !== "ours" ? `; ${best.label} did so ${times(best.critical_false_reassurance_n![0])}` : ""
  }`;
  const groups = list.map((s) => {
    const [n, d] = s.critical_false_reassurance_n!;
    return sysGroup(s, r, {
      series: "Critical false reassurance",
      value: n,
      color: colorOf(s),
      display: `${n} of ${d}`,
      detail: [`${((n / d) * 100).toFixed(1)}% of critical facts the patient got wrong or left out`],
    });
  });
  return (
    <ChartCard
      id="fr"
      title={title}
      sub={`Count of critical facts the patient got wrong or left out (${N} in the test set) that the grader still marked correct. This is the unsafe error, so lower is better.`}
      caveats={[
        `These are small counts out of ${N}; a difference of one or two facts is within noise. Fine-tuning raised accuracy but has not yet reduced this count, which is why the hosted demo grades with Gemma 4 31B and the next training run targets it.`,
        "Homeward never relies on the grader alone: every critical fact needs a passing grade, the patient gets one re-teach, and the nurse confirms before discharge.",
      ]}
      source={SOURCE_OURS}
      table={{
        head: ["System", "Critical facts falsely passed", "All facts falsely passed", "False reassurance rate", "95% interval"],
        numeric: [false, true, true, true, true],
        rows: list.map((s) => [
          s.label,
          `${s.critical_false_reassurance_n![0]} of ${s.critical_false_reassurance_n![1]}`,
          s.false_reassurance_n ? `${s.false_reassurance_n[0]} of ${s.false_reassurance_n[1]}` : "",
          s.false_reassurance !== null ? f1(s.false_reassurance) : "",
          s.false_reassurance_ci ? `${s.false_reassurance_ci[0]}–${s.false_reassurance_ci[1]}%` : "",
        ]),
      }}
    >
      <HChart groups={groups} domain={[0, Math.max(10, Math.ceil((max + 1) / 2) * 2)]} mark="bar" format={(v) => `${v}`} axisLabel={`Critical facts falsely marked understood (out of ${N})`} ariaLabel="Critical false reassurance count, Homeward first" labelWidth={260} ticks={5} />
      {LEGEND_OURS(r)}
    </ChartCard>
  );
}

export function OverflagChart({ r }: { r: Results }) {
  const { t, sys, ours, base } = useSystems(r);
  const list = sys.filter((s) => s.over_flag !== null);
  if (!t || !ours || ours.over_flag === null || !list.length) return null;
  const lower = list.filter((s) => s.role !== "ours" && s.over_flag! > ours.over_flag!);
  const better = list.filter((s) => s.role !== "ours" && s.over_flag! < ours.over_flag!);
  const title = `${ours.label} sends ${f1(ours.over_flag)} of correctly understood facts back for re-teaching${
    base?.over_flag != null ? `, down from ${f1(base.over_flag)} for base ${base.model}` : ""
  }${lower.filter((s) => s.role !== "base").length ? ` and below ${lower.filter((s) => s.role !== "base").map((s) => `${s.label.replace(" (cloud)", "")} (${f1(s.over_flag!)})`).join(", ")}` : ""}${
    (() => {
      const level = list.filter((s) => s.role !== "ours" && s.role !== "base" && s.over_flag === ours.over_flag);
      return level.length ? `, level with ${level.map((s) => s.label.replace(" (cloud)", "")).join(" and ")}` : "";
    })()
  }${
    better.length ? `; ${better.map((s) => `${s.label} is lower at ${f1(s.over_flag!)}`).join(", ")}` : ""
  }`;
  const max = Math.max(...list.map((s) => s.over_flag!));
  const groups = list.map((s) =>
    sysGroup(s, r, { series: "Over-flag rate", value: s.over_flag, color: colorOf(s), display: f1(s.over_flag!), detail: ["Facts the patient got right that the grader did not pass"] }),
  );
  return (
    <ChartCard
      id="overflag"
      title={title}
      sub="Share of facts the patient explained correctly that the grader still did not pass. These are false alarms: not unsafe, but each one costs the nurse a re-teach. Lower is better."
      caveats={["Same small, synthetic test set as above. A grader can trade false alarms for false reassurance, so read this chart together with the one above."]}
      source={SOURCE_OURS}
      table={{
        head: ["System", "Over-flag rate", "JSON replies valid"],
        numeric: [false, true, true],
        rows: list.map((s) => [s.label, f1(s.over_flag!), s.json_valid !== null ? f1(s.json_valid) : ""]),
      }}
    >
      <HChart groups={groups} domain={[0, Math.max(20, Math.ceil(max / 5) * 5)]} mark="bar" format={(v) => `${v}%`} axisLabel="Correct facts not passed (%), lower is better" ariaLabel="Over-flag rate, Homeward first" labelWidth={260} ticks={4} />
      {LEGEND_OURS(r)}
    </ChartCard>
  );
}

export function AsrChart({ r }: { r: Results }) {
  const [src, setSrc] = useState<"human" | "tts">("human");
  const w = r.asr.wer;
  const oursSys = r.ours.asr_system;
  if (!w[oursSys] || !w.whisper) return null;
  const langs = Object.keys(w[oursSys][src] ?? {}).sort((a, b) => (a === "ALL" ? -1 : b === "ALL" ? 1 : (LANG[a] ?? a).localeCompare(LANG[b] ?? b)));
  // Chinese is written without spaces, so word error rate is meaningless there; use character error rate.
  const val = (sys: string, lang: string) => {
    const c = w[sys]?.[src]?.[lang];
    if (!c) return null;
    return lang === "zh" ? c.cer_normalised ?? c.cer : c.wer_normalised ?? c.wer;
  };
  const oursName = `Homeward on-site: ${r.ours.asr_label}`;
  const groups: Group[] = langs.map((l) => {
    const c = w[oursSys][src][l];
    return {
      label: (LANG[l] ?? l) + (l === "zh" ? " (characters)" : ""),
      sub: `${c.clips} clips, ${c.words} words`,
      strong: l === "ALL",
      items: [
        { series: oursName, value: val(oursSys, l), color: COLORS.gemma, display: val(oursSys, l) !== null ? f1(val(oursSys, l)!) : undefined },
        { series: "Whisper large-v3-turbo", value: val("whisper", l), color: COLORS.muted, display: val("whisper", l) !== null ? f1(val("whisper", l)!) : undefined },
      ],
    };
  });
  const allE = val(oursSys, "ALL");
  const allW = val("whisper", "ALL");
  const bnE = val(oursSys, "bn");
  const bnW = val("whisper", "bn");
  const by = (pick: (e: number, wh: number) => boolean) =>
    langs.filter((l) => l !== "ALL" && val(oursSys, l) !== null && val("whisper", l) !== null && pick(val(oursSys, l)!, val("whisper", l)!)).map((l) => LANG[l] ?? l);
  const list = (xs: string[]) => (xs.length <= 1 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);
  const q = (v: number | null) => (v !== null ? f1(v) : "?");
  const title =
    src === "human"
      ? `On the owner's own voice, Homeward's on-site speech recognition made ${q(allE)} word errors against ${q(allW)} for Whisper large-v3-turbo, and the gap is widest in Bengali (${q(bnE)} against ${q(bnW)})`
      : (() => {
          const g = by((e, wh) => wh - e >= 5);
          const wn = by((e, wh) => e - wh >= 5);
          return `On synthetic voices Homeward's speech recognition and Whisper are close overall (${q(allE)} against ${q(allW)})${g.length ? `; Homeward is clearly better in ${list(g)}` : ""}${wn.length ? `, Whisper in ${list(wn)}` : ""}`;
        })();
  const max = Math.max(...groups.flatMap((g) => g.items.map((i) => i.value ?? 0)));
  const clips = r.asr.clips?.[src];
  return (
    <ChartCard
      id="asr"
      title={title}
      sub={
        <>
          Word error rate (lower is better) when each model transcribes the same spoken teach-back answers, both on one AMD Radeon PRO W7900D. Homeward uses {r.ours.model} for
          speech too, so one model copy does both jobs. {src === "human" ? `${clips} clips recorded by the project owner in English, Hindi and Bengali.` : `${clips} clips of macOS text-to-speech voices reading the independent answers (written by Claude) in 11 languages.`}
        </>
      }
      caveats={[
        src === "human"
          ? "Very small sample: one speaker, 5 clips per language, about 70 to 90 words each. Treat it as a direction, not a benchmark."
          : "Synthetic voices are clean studio speech, which flatters both models compared with real patients. Tagalog has no macOS voice, so it is not here.",
        "Both models were told the language. Text is lower-cased with punctuation removed; Chinese Traditional characters are mapped to Simplified and Arabic letters normalised. Writing a drug name in another script still counts as an error.",
        "Chinese is scored by characters (CER) because it has no spaces between words.",
      ]}
      source={
        <>
          Source: Homeward speech test, <code>ml/results/asr.json</code>. Models: google/gemma-4-E4B-it (adapter off for transcription) and openai/whisper-large-v3-turbo, greedy decoding.
        </>
      }
      table={{
        head: ["Language", "Clips", "Words", oursName, "Whisper large-v3-turbo"],
        numeric: [false, true, true, true, true],
        rows: langs.map((l) => [
          (LANG[l] ?? l) + (l === "zh" ? " (CER)" : ""),
          w[oursSys][src][l].clips,
          w[oursSys][src][l].words,
          val(oursSys, l) !== null ? f1(val(oursSys, l)!) : "",
          val("whisper", l) !== null ? f1(val("whisper", l)!) : "",
        ]),
      }}
    >
      <div className="seg ev-seg" role="group" aria-label="Which recordings">
        <button aria-pressed={src === "human"} onClick={() => setSrc("human")}>
          Owner's real voice
        </button>
        <button aria-pressed={src === "tts"} onClick={() => setSrc("tts")}>
          Synthetic voices
        </button>
      </div>
      <HChart groups={groups} domain={[0, Math.max(50, Math.ceil(max / 10) * 10)]} mark="bar" format={(v) => `${Number.isInteger(v) ? v : v.toFixed(1)}%`} axisLabel="Word error rate (%), lower is better" ariaLabel="Speech recognition word error rate by language, Homeward first" labelWidth={170} />
      <Legend
        items={[
          { label: oursName, color: COLORS.gemma },
          { label: "Whisper large-v3-turbo", color: COLORS.muted },
        ]}
      />
    </ChartCard>
  );
}

export function EndToEndChart({ r }: { r: Results }) {
  const g = r.asr.grading_from_transcripts;
  const oursSys = r.ours.asr_system;
  if (!g || !g[`${oursSys}/human`] || !g["whisper/human"]) return null;
  const rows = [
    { key: oursSys, label: `Homeward on-site: ${r.ours.model} speech + Homeward grader`, color: COLORS.gemma, strong: true },
    { key: "whisper", label: "Whisper large-v3-turbo + Homeward grader", color: COLORS.muted, strong: false },
    { key: "ref", label: "Exact transcript + Homeward grader (ceiling)", color: COLORS.context, strong: false },
  ];
  const panel = (src: "human" | "tts") =>
    rows
      .filter((x) => g[`${x.key}/${src}`])
      .map((x) => {
        const c = g[`${x.key}/${src}`];
        const v = (c.correct_vs_gold / c.facts) * 100;
        return {
          label: x.label,
          strong: x.strong,
          items: [
            {
              series: src === "human" ? "Owner's real voice" : "Synthetic voices",
              value: Math.round(v * 10) / 10,
              color: x.color,
              display: `${c.correct_vs_gold} of ${c.facts}`,
              detail: [`${v.toFixed(1)}% of facts graded correctly`, `Critical facts falsely passed: ${c.critical_false_reassurance} of ${c.critical_bad_facts}`],
            },
          ],
        } as Group;
      });
  const h = (k: string) => g[`${k}/human`];
  const t = (k: string) => g[`${k}/tts`];
  const oh = h(oursSys);
  const wh = h("whisper");
  const title = `End to end on the owner's real voice, Homeward's on-site stack graded ${oh.correct_vs_gold} of ${oh.facts} facts correctly against ${wh.correct_vs_gold} of ${wh.facts} when Whisper did the listening${
    t(oursSys) && t("whisper") ? `; on synthetic voices it was ${t(oursSys).correct_vs_gold} against ${t("whisper").correct_vs_gold} of ${t(oursSys).facts}, with ${t(oursSys).critical_false_reassurance} critical false passes against ${t("whisper").critical_false_reassurance}` : ""
  }`;
  return (
    <ChartCard
      id="e2e"
      title={title}
      sub="The whole teach-back check: speech is transcribed, then the same Homeward grader grades every fact from that transcript. Bars are facts graded correctly against the designed labels; the exact-transcript row shows what the grader does with perfect hearing."
      caveats={[
        `Small samples: ${oh.answers} answers and ${oh.facts} facts on the owner's voice, ${t(oursSys)?.answers ?? "?"} answers and ${t(oursSys)?.facts ?? "?"} facts on synthetic voices. A two-fact gap on 24 is a direction, not proof.`,
        "Graded by the fine-tuned Gemma 4 E4B from the speech test run. On synthetic voices the exact transcript is not a strict ceiling: it had more critical false passes than the transcribed versions, because recognition errors sometimes push the grader towards flagging.",
      ]}
      source={
        <>
          Source: Homeward speech test, <code>ml/results/asr.json</code> (grading from transcripts), one AMD Radeon PRO W7900D.
        </>
      }
      table={{
        head: ["Pipeline", "Real voice: facts correct", "Synthetic: facts correct", "Synthetic: critical false passes"],
        numeric: [false, true, true, true],
        rows: rows
          .filter((x) => h(x.key) || t(x.key))
          .map((x) => [
            x.label,
            h(x.key) ? `${h(x.key).correct_vs_gold} of ${h(x.key).facts}` : "",
            t(x.key) ? `${t(x.key).correct_vs_gold} of ${t(x.key).facts}` : "",
            t(x.key) ? `${t(x.key).critical_false_reassurance} of ${t(x.key).critical_bad_facts}` : "",
          ]),
      }}
    >
      <div className="ev-panels">
        <div className="ev-panel">
          <h4>Owner's real voice</h4>
          <p>
            {oh.answers} spoken answers, {oh.facts} facts
          </p>
          <HChart groups={panel("human")} domain={[0, 100]} mark="bar" format={(v) => `${v}%`} axisLabel="Facts graded correctly (%)" ariaLabel="End-to-end facts graded correctly, owner's voice" labelWidth={240} ticks={4} />
        </div>
        {t(oursSys) && (
          <div className="ev-panel">
            <h4>Synthetic voices</h4>
            <p>
              {t(oursSys).answers} spoken answers, {t(oursSys).facts} facts, 11 languages
            </p>
            <HChart groups={panel("tts")} domain={[0, 100]} mark="bar" format={(v) => `${v}%`} axisLabel="Facts graded correctly (%)" ariaLabel="End-to-end facts graded correctly, synthetic voices" labelWidth={240} ticks={4} />
          </div>
        )}
      </div>
      <Legend
        items={[
          { label: "Homeward on-site stack", color: COLORS.gemma },
          { label: "Whisper for speech", color: COLORS.muted },
          { label: "Exact transcript (reference)", color: COLORS.context },
        ]}
      />
    </ChartCard>
  );
}

export function DeploymentCard({ r }: { r: Results }) {
  const { sys } = useSystems(r);
  const cap = r.capacity;
  const loop = r.onprem_loop;
  const cloud = sys.filter((s) => s.role === "cloud" || s.role === "teacher");
  const top = cap.by_batch[cap.by_batch.length - 1];
  if (!cap.by_batch.length) return null;
  const groups: Group[] = cap.by_batch.map((b) => ({
    label: `Batch of ${b.batch}`,
    items: [{ series: "Teach-backs graded per hour", value: b.per_hour, color: COLORS.gemma, display: b.per_hour.toLocaleString(), detail: b.peak_vram_gib ? [`Peak memory ${b.peak_vram_gib} GiB of 48 GiB`] : undefined }],
  }));
  const max = Math.max(...cap.by_batch.map((b) => b.per_hour));
  const money = (v: number) => `$${v < 0.01 ? v.toFixed(5) : v.toFixed(3)}`;
  return (
    <ChartCard
      id="deploy"
      title={`One AMD Radeon PRO W7900D grades about ${top.per_hour.toLocaleString()} teach-back answers an hour with the Homeward grader, enough for ${cap.discharges_per_hour_per_card ?? "?"} discharges an hour including speech recognition, with no patient data leaving the hospital`}
      sub="Throughput of the Homeward grader on one card at different batch sizes, measured on the in-distribution test answers. Below it, the on-site stack next to the cloud graders we compared."
      caveats={[
        `Capacity assumes ${cap.answers_per_discharge ?? 6} graded answers per discharge, grading batched at ${top.batch} and speech recognition one clip at a time, on one card with no other load.`,
        "Cloud prices are AMD Token Factory list prices for the tokens our test answers used. Cloud latency in our runs included queueing under a shared hackathon rate limit, so it is not a fair speed comparison and is not shown.",
        "Token Factory did not accept audio input when we tested it, so the cloud graders still need a separate speech service.",
        ...(loop.clips ? [`The ${loop.clips}-clip timing of one spoken answer (batch size 1) was measured with an earlier fine-tuned adapter on the same Gemma 4 E4B base.`] : []),
      ]}
      source={
        <>
          Source: Homeward evaluation run, <code>ml/results/eval.json</code> (capacity, cost and on-prem loop).
        </>
      }
      table={{
        head: ["Batch size", "Teach-backs per hour", "Peak memory"],
        numeric: [true, true, true],
        rows: cap.by_batch.map((b) => [b.batch, b.per_hour.toLocaleString(), b.peak_vram_gib ? `${b.peak_vram_gib} GiB` : ""]),
      }}
    >
      <HChart groups={groups} domain={[0, Math.ceil(max / 2000) * 2000]} mark="bar" format={(v) => (v >= 1000 ? `${v / 1000}K` : `${v}`)} axisLabel="Teach-back answers graded per hour, one card" ariaLabel="Homeward grader throughput by batch size" labelWidth={120} ticks={5} />
      <div className="ev-compare">
        <div className="ev-cmp ours">
          <h4>{r.ours.label}, on site</h4>
          <ul>
            {cap.discharges_per_hour_per_card != null && <li>{cap.discharges_per_hour_per_card} discharges per hour per card, speech plus grading</li>}
            {cap.grading_only_discharges_per_hour_per_card != null && <li>{cap.grading_only_discharges_per_hour_per_card.toLocaleString()} per hour for grading alone</li>}
            {loop.p50_s != null && (
              <li>
                One spoken answer, heard and graded: {loop.p50_s} s median, {loop.p95_s} s at the 95th percentile ({loop.clips} clips)
              </li>
            )}
            {loop.peak_vram_gib != null && <li>Peak memory {loop.peak_vram_gib} GiB of the card's 48 GiB, one model copy for speech and grading</li>}
            <li>No per-call fee and no patient audio or text leaves the hospital</li>
          </ul>
        </div>
        {cloud.map((s) => (
          <div className="ev-cmp" key={s.id}>
            <h4>{s.label}</h4>
            <ul>
              {s.usd_per_discharge != null && <li>{money(s.usd_per_discharge)} per discharge at list price</li>}
              {s.accuracy != null && <li>{f1(s.accuracy)} fact accuracy in our test</li>}
              <li>Runs in AMD's cloud only; patient text leaves the hospital</li>
              <li>No speech recognition on the same service</li>
            </ul>
          </div>
        ))}
      </div>
    </ChartCard>
  );
}

/** Size label for a row: only sizes marked verified in model-sizes.md are shown. */
const sizeText = (s: SystemResult) => s.size?.display ?? "size not published";
function ratio(s: SystemResult, ours: SystemResult) {
  const a = s.size?.total_b;
  const b = ours.size?.total_b;
  if (!a || !b) return null;
  return a / b;
}
const xFmt = (x: number) => (x >= 10 ? `${Math.round(x)}×` : `${x.toFixed(1)}×`);
const shortSize = (s: SystemResult) => (s.size?.verified ? s.size.total ?? s.size.display : null);
const list = (xs: string[]) => (xs.length <= 1 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);

function sizedGroups(rows: SystemResult[]) {
  // Sizes only, no "times our size" multiplier: for mixture-of-experts models total against total overstates the gap.
  const sub = (s: SystemResult) => sizeText(s);
  const acc: Group[] = rows.map((s) => ({
    label: s.label,
    sub: sub(s),
    strong: s.role === "ours",
    items: [
      {
        series: "Fact accuracy",
        value: s.accuracy,
        lo: s.accuracy_ci?.[0],
        hi: s.accuracy_ci?.[1],
        color: colorOf(s),
        display: f1(s.accuracy!),
        detail: [...(s.accuracy_ci ? [`95% interval ${s.accuracy_ci[0]}–${s.accuracy_ci[1]}%`] : []), sizeText(s), s.runs_on ?? ""].filter(Boolean),
      },
    ],
  }));
  const crit: Group[] = rows
    .filter((s) => s.critical_false_reassurance_n)
    .map((s) => ({
      label: s.label,
      sub: sub(s),
      strong: s.role === "ours",
      items: [
        {
          series: "Critical misses",
          value: s.critical_false_reassurance_n![0],
          color: colorOf(s),
          display: `${s.critical_false_reassurance_n![0]} of ${s.critical_false_reassurance_n![1]}`,
          detail: ["Critical facts the patient got wrong or left out, graded correct", sizeText(s)],
        },
      ],
    }));
  return { acc, crit };
}

function TwoPanels({ rows, N, id }: { rows: SystemResult[]; N: number | undefined; id: string }) {
  const { acc, crit } = sizedGroups(rows);
  const lo = Math.min(...rows.map((s) => s.accuracy_ci?.[0] ?? s.accuracy ?? 100));
  const d0 = Math.max(0, Math.floor((lo - 5) / 10) * 10);
  const cmax = Math.max(0, ...rows.map((s) => s.critical_false_reassurance_n?.[0] ?? 0));
  return (
    <div className="ev-panels">
      <div className="ev-panel">
        <h4>Facts graded correctly</h4>
        <p>Higher is better; whiskers are 95% intervals</p>
        <HChart groups={acc} domain={[d0, 100]} mark="dot" format={(v) => `${v}%`} axisLabel="Facts graded correctly (%)" ariaLabel={`${id}: accuracy by model, Homeward first`} labelWidth={210} ticks={4} />
      </div>
      <div className="ev-panel">
        <h4>Critical misses</h4>
        <p>Misunderstood critical facts graded correct, out of {N}; lower is better</p>
        <HChart groups={crit} domain={[0, Math.max(8, Math.ceil((cmax + 1) / 2) * 2)]} mark="bar" format={(v) => `${v}`} axisLabel={`Critical misses (out of ${N})`} ariaLabel={`${id}: critical misses by model, Homeward first`} labelWidth={210} ticks={4} />
      </div>
    </div>
  );
}

const sizeTable = (rows: SystemResult[]) => ({
  head: ["System", "Parameters", "Fact accuracy", "95% interval", "Critical misses", "Runs on"],
  numeric: [false, false, true, true, true, false],
  rows: rows.map((s) => {
    return [
      s.label,
      sizeText(s),
      s.accuracy !== null ? f1(s.accuracy) : "",
      s.accuracy_ci ? `${s.accuracy_ci[0]}–${s.accuracy_ci[1]}%` : "",
      s.critical_false_reassurance_n ? `${s.critical_false_reassurance_n[0]} of ${s.critical_false_reassurance_n[1]}` : "",
      s.runs_on ?? "",
    ];
  }),
});

/** Chart 1: Homeward against every model it did not learn from. The teacher only appears if Homeward matches or beats it. */
export function SizeChart({ r }: { r: Results }) {
  const { t, sys, ours, base } = useSystems(r);
  if (!t || !ours || ours.accuracy === null) return null;
  const teacher = sys.find((s) => s.role === "teacher");
  const includeTeacher = teacher?.accuracy != null && ours.accuracy >= teacher.accuracy;
  const rest = sys
    .filter((s) => s.role !== "ours" && s.accuracy !== null && (s.role !== "teacher" || includeTeacher))
    .sort((a, b) => (a.role === "teacher" ? -1 : b.role === "teacher" ? 1 : 0) || (b.size?.total_b ?? -1) - (a.size?.total_b ?? -1) || (b.accuracy ?? 0) - (a.accuracy ?? 0));
  const rows = [ours, ...rest];
  const name = (s: SystemResult) => s.label.replace(" (cloud)", "");
  const others = rest.filter((s) => s.role !== "base");
  const oc = ours.critical_false_reassurance_n?.[0];
  const N = ours.critical_false_reassurance_n?.[1];
  // "Beats" only where both numbers favour Homeward: higher accuracy and no more critical misses.
  // Higher accuracy alone is said as "higher accuracy than", with the critical misses next to it.
  const higher = others.filter((s) => s.accuracy! > 0 && s.accuracy! < ours.accuracy!);
  const crit = (s: SystemResult) => s.critical_false_reassurance_n?.[0];
  const beaten = higher.filter((s) => oc !== undefined && crit(s) !== undefined && oc <= crit(s)!);
  const accOnly = higher.filter((s) => !beaten.includes(s));
  const ahead = others.filter((s) => s.accuracy! >= ours.accuracy! && s.role !== "teacher");
  const largest = [...beaten].sort((a, b) => (b.size?.total_b ?? 0) - (a.size?.total_b ?? 0))[0];
  const oSize = shortSize(ours);
  let title: string;
  if (largest) {
    const lSize = shortSize(largest);
    title = `Our fine-tuned ${oSize ?? ""} Gemma beats ${lSize ? `a ${lSize} model, ${name(largest)},` : name(largest)} at checking discharge understanding: ${f1(ours.accuracy)} against ${f1(largest.accuracy!)}`.replace("  ", " ");
    const lc = crit(largest);
    if (oc !== undefined && lc !== undefined) title += oc < lc ? `, with fewer critical misses (${oc} against ${lc})` : `, with the same number of critical misses (${oc})`;
    const alsoBeat = beaten.filter((s) => s !== largest);
    if (alsoBeat.length) title += `; it also beats ${list(alsoBeat.map(name))}`;
  } else {
    title = `${ours.label} grades ${f1(ours.accuracy)} of facts correctly`;
  }
  if (accOnly.length) {
    title += `; it has higher accuracy than ${list(accOnly.map(name))}, ${accOnly.length > 1 ? "which let" : "which lets"} through fewer critical misses (${list(accOnly.map((s) => `${crit(s) ?? "?"}`))} against ${oc})`;
  }
  if (ahead.length) title += `; ${list(ahead.map((s) => `${name(s)} (${f1(s.accuracy!)})`))} ${ahead.length > 1 ? "score" : "scores"} higher`;
  if (includeTeacher && teacher) title += `, and it matches or beats its teacher ${teacher.model ?? name(teacher)}`;
  const pv = ours.paired_vs_base;
  return (
    <ChartCard
      id="size"
      title={title}
      sub={
        <>
          Every model here is one Homeward did not learn from. Each graded the same {t.facts} facts in {t.answers} independent teach-back answers ({t.cases} cases,{" "}
          {t.languages.length} languages), written by a different model family (Claude) and never seen by the data generator or in training. Every other model was given
          the same grading prompt with no examples (zero-shot); Homeward is fine-tuned for this task and output format. Each row gives the model's size.
          {pv && base ? ` Against its own base model, Homeward fixed ${pv.fixes} facts and broke ${pv.breaks} (exact McNemar p = ${pfmt(pv.p)}).` : ""}
          {teacher && !includeTeacher ? ` Its teacher, ${teacher.model ?? name(teacher)}, is compared separately below.` : ""}
        </>
      }
      caveats={[
        `Small test set: ${t.answers} answers from ${t.cases} cases, so the 95% intervals (${ours.ci_method ?? "case-level bootstrap"}) are wide; overlapping intervals are not a clear win.`,
        `Critical misses are small counts out of ${N ?? "?"}; one or two either way is within noise. Homeward never relies on the grader alone: the patient gets a re-teach and the nurse confirms.`,
        "The answers are synthetic. Sizes come from the makers' model cards (amd-act-iii-2026/model-sizes.md) and are shown only where verified. Mixture-of-experts models list total and active parameters.",
        "The accuracy axis starts above zero to show the intervals.",
      ]}
      source={SOURCE_OURS}
      table={sizeTable(rows)}
    >
      <TwoPanels rows={rows} N={N} id="Homeward vs other models" />
      {LEGEND_OURS(r, "dot")}
    </ChartCard>
  );
}

const fractionWords = (x: number) => {
  const n = Math.round(x);
  const w: Record<number, string> = { 2: "half", 3: "a third", 4: "a quarter", 5: "a fifth", 6: "a sixth", 8: "an eighth", 10: "a tenth" };
  return w[n] ? `about ${w[n]} of the size` : `${xFmt(x)} smaller`;
};

/** Chart 2: how much of its teacher Homeward keeps. */
export function TeacherChart({ r }: { r: Results }) {
  const { t, sys, ours } = useSystems(r);
  const teacher = sys.find((s) => s.role === "teacher");
  if (!t || !ours || !teacher || ours.accuracy === null || teacher.accuracy === null) return null;
  const keep = Math.round((ours.accuracy / teacher.accuracy) * 100);
  const x = ratio(teacher, ours);
  const N = ours.critical_false_reassurance_n?.[1];
  const title = `Distilled from ${teacher.model ?? teacher.label}, Homeward keeps ${keep}% of its accuracy${x ? ` at ${fractionWords(x)}` : ""}, on one workstation GPU in the hospital`;
  const oc = ours.critical_false_reassurance_n?.[0];
  const tc = teacher.critical_false_reassurance_n?.[0];
  return (
    <ChartCard
      id="teacher"
      title={title}
      sub={
        <>
          Homeward learned from labels its teacher helped produce, so the teacher is the bar to measure against, not a competitor. Accuracy {f1(ours.accuracy)} against{" "}
          {f1(teacher.accuracy)} ({keep}% kept); critical misses {oc} against {tc} of {N}; size {sizeText(ours)} against {sizeText(teacher)}.
        </>
      }
      caveats={[
        `Same small, synthetic test set as above (${t.answers} answers), so a ${Math.abs(ours.accuracy - teacher.accuracy).toFixed(1)}-point gap has wide uncertainty; see the intervals.`,
        "Accuracy kept is Homeward's accuracy divided by the teacher's. Size uses the makers' published parameter counts.",
      ]}
      source={SOURCE_OURS}
      table={sizeTable([ours, teacher])}
    >
      <TwoPanels rows={[ours, teacher]} N={N} id="Homeward vs its teacher" />
      <div className="ev-compare two">
        <div className="ev-cmp ours">
          <h4>{ours.label}</h4>
          <ul>
            <li>{sizeText(ours)}</li>
            <li>Runs on one AMD Radeon PRO W7900D inside the hospital</li>
            <li>Patient audio and text never leave the building</li>
          </ul>
        </div>
        <div className="ev-cmp">
          <h4>{teacher.label}</h4>
          <ul>
            <li>{sizeText(teacher)}</li>
            <li>Served in AMD's cloud through AMD Token Factory</li>
            <li>Patient text has to leave the hospital</li>
          </ul>
        </div>
      </div>
    </ChartCard>
  );
}
