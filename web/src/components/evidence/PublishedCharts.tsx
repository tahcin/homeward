import { ChartCard, COLORS, HChart, Legend, type Group } from "./HChart.tsx";
import { LANG, type BenchPanel, type Benchmarks, type Results } from "./types.ts";

const famColor = { gemma4: COLORS.gemma, gemma3: COLORS.context, other: COLORS.compare } as const;
const fmt = (v: number) => (Number.isInteger(v) ? `${v}` : v.toFixed(1));

function SourceLinks({ ids, b, lead = "Source" }: { ids: string[]; b: Benchmarks; lead?: string }) {
  const uniq = [...new Set(ids)].filter((id) => b.sources[id]);
  return (
    <>
      {lead}
      {lead === "Source" && uniq.length > 1 ? "s" : ""}:{" "}
      {uniq.map((id, i) => (
        <span key={id}>
          {i > 0 && "; "}
          <a href={b.sources[id].url} target="_blank" rel="noreferrer">
            {b.sources[id].title}
          </a>{" "}
          ({b.sources[id].publisher}, accessed {b.sources[id].accessed})
        </span>
      ))}
    </>
  );
}

function Panel({ p }: { p: BenchPanel }) {
  // Rows stay in the file's order (smallest model to largest), so each size class sits together.
  const rows = p.rows;
  const max = Math.max(...rows.map((r) => r.value));
  const groups: Group[] = rows.map((r) => ({
    label: r.model,
    strong: r.family === "gemma4",
    items: [{ series: p.benchmark, value: r.value, color: famColor[r.family], display: fmt(r.value), detail: [r.params ?? "", ...(r.note ? [r.note] : [])].filter(Boolean) }],
  }));
  const domainMax = max <= 100 && p.metric.includes("%") ? 100 : Math.ceil(max / 10) * 10;
  return (
    <div className={p.wide ? "ev-panel wide" : "ev-panel"}>
      <h4>{p.benchmark}</h4>
      <p>
        {p.setting}. {p.metric}, {p.higher_is_better ? "higher is better" : "lower is better"}.
      </p>
      <HChart groups={groups} domain={[0, domainMax]} mark="bar" format={fmt} axisLabel={p.metric} ariaLabel={`${p.benchmark}: published scores`} labelWidth={190} ticks={p.wide ? 5 : 4} />
      {p.note && <p>{p.note}</p>}
    </div>
  );
}

export function PublishedLlmChart({ b }: { b: Benchmarks }) {
  const panels = b.llm.panels.filter((p) => p.rows.length > 0);
  if (!panels.length) return null;
  const fams = new Set(panels.flatMap((p) => p.rows.map((r) => r.family)));
  return (
    <ChartCard
      id="pub"
      title={b.llm.title}
      sub={b.llm.note}
      caveats={[
        "These numbers are published by the model makers, not measured by us. Each panel only compares scores reported for the same benchmark under the same setting; models without a matching published number are left out rather than mixed in.",
        "Makers run their own evaluation harnesses, so small gaps (a point or two) are not meaningful.",
      ]}
      source={<SourceLinks ids={panels.flatMap((p) => p.rows.map((r) => r.source))} b={b} />}
      table={{
        head: ["Benchmark", "Setting", "Model", "Score", "Source"],
        numeric: [false, false, false, true, false],
        rows: panels.flatMap((p) => p.rows.map((r) => [p.benchmark, p.setting, r.model, fmt(r.value), b.sources[r.source]?.publisher ?? r.source])),
      }}
    >
      <Legend
        items={[
          { label: "Gemma 4", color: COLORS.gemma },
          ...(fams.has("gemma3") ? [{ label: "Gemma 3 (previous generation)", color: COLORS.context }] : []),
          ...(fams.has("other") ? [{ label: "Other model makers", color: COLORS.compare }] : []),
        ]}
      />
      <div className="ev-panels">
        {panels.map((p) => (
          <Panel key={p.id} p={p} />
        ))}
      </div>
    </ChartCard>
  );
}

const wf = (v: number | null | undefined) => (v === null || v === undefined ? "" : `${fmt(v)}%`);

export function FleursPublishedChart({ b }: { b: Benchmarks }) {
  const F = b.whisper_fleurs;
  const rows = F.rows.filter((x) => x.gemma4_e4b !== null && x.whisper_large_v2 !== null);
  if (!rows.length) return null;
  const sorted = [...rows].sort((a, c) => (LANG[a.lang] ?? a.lang).localeCompare(LANG[c.lang] ?? c.lang));
  const groups: Group[] = sorted.map((x) => ({
    label: (LANG[x.lang] ?? x.lang) + (x.metric === "CER" ? " (characters)" : ""),
    items: [
      { series: "Gemma 4 E4B (Google)", value: x.gemma4_e4b, color: COLORS.gemma, display: wf(x.gemma4_e4b), detail: x.note ? [x.note] : undefined },
      { series: "Whisper large-v2 (OpenAI)", value: x.whisper_large_v2, color: COLORS.compare, display: wf(x.whisper_large_v2), detail: x.whisper_large_v3_chart !== null ? [`Whisper large-v3 reads about ${fmt(x.whisper_large_v3_chart)}% on OpenAI's chart`] : undefined },
    ],
  }));
  const max = Math.max(...groups.flatMap((g) => g.items.map((i) => i.value ?? 0)));
  return (
    <ChartCard
      id="fleurs-pub"
      title={F.title}
      sub="Word error rate on the FLEURS speech benchmark for the languages both makers report, as published by Google and OpenAI. Lower is better."
      caveats={[
        "Published by the model makers, not measured by us. Each maker scored with its own text normaliser, so gaps of a point or less are not meaningful.",
        "Whisper large-v2 is the latest Whisper with an exact per-language table. OpenAI shows large-v3 only as a chart; those readings are in the table below and in each tooltip. " + F.v3_note,
        "Model sizes differ: Gemma 4 E4B is a general model with 4.5B effective parameters; Whisper large-v2 is a 1.55B speech-only model.",
      ]}
      source={<SourceLinks ids={[F.sources.gemma4_e4b, F.sources.whisper_large_v2, F.sources.whisper_large_v3_chart]} b={b} />}
      table={{
        head: ["Language", "Gemma 4 E4B", "Whisper large-v2", "Whisper large-v3 (chart reading)"],
        numeric: [false, true, true, true],
        rows: sorted.map((x) => [(LANG[x.lang] ?? x.lang) + (x.metric === "CER" ? " (CER)" : ""), wf(x.gemma4_e4b), wf(x.whisper_large_v2), wf(x.whisper_large_v3_chart)]),
      }}
    >
      <HChart groups={groups} domain={[0, Math.max(25, Math.ceil(max / 5) * 5)]} mark="bar" format={(v) => `${fmt(v)}%`} axisLabel="FLEURS word error rate (%), lower is better" ariaLabel="Published FLEURS error rates, Gemma 4 E4B and Whisper large-v2" labelWidth={170} />
      <Legend
        items={[
          { label: "Gemma 4 E4B, published by Google", color: COLORS.gemma },
          { label: "Whisper large-v2, published by OpenAI", color: COLORS.compare },
        ]}
      />
    </ChartCard>
  );
}

export function FleursChart({ b, r }: { b: Benchmarks; r: Results }) {
  const F = b.whisper_fleurs;
  const w = r.asr.wer;
  const ours = (sys: string, lang: string) => {
    const c = w[sys]?.tts?.[lang];
    if (!c) return null;
    return lang === "zh" ? c.cer_normalised ?? c.cer : c.wer_normalised ?? c.wer;
  };
  // Only languages we measured; Tagalog has no synthetic voice.
  const rows = F.rows.filter((x) => ours("whisper", x.lang) !== null || ours("e4b", x.lang) !== null);
  if (!rows.length) return null;
  const sorted = [...rows].sort((a, c) => (LANG[a.lang] ?? a.lang).localeCompare(LANG[c.lang] ?? c.lang));
  const groups: Group[] = sorted.map((x) => {
    const zh = x.metric === "CER";
    const pub = (v: number | null) => (v === null ? "not published" : `${fmt(v)}%`);
    const items: Group["items"] = [
      {
        series: "Whisper",
        value: ours("whisper", x.lang),
        ref: x.whisper_large_v2,
        color: COLORS.compare,
        display: wf(ours("whisper", x.lang)),
        detail: [`Measured by us (large-v3-turbo), synthetic voices`, `Published on FLEURS (large-v2): ${pub(x.whisper_large_v2)}`, ...(x.note ? [x.note] : [])],
      },
      {
        series: "Gemma 4 E4B",
        value: ours("e4b", x.lang),
        ref: x.gemma4_e4b,
        color: COLORS.gemma,
        display: wf(ours("e4b", x.lang)),
        detail: [`Measured by us, synthetic voices`, `Published on FLEURS by Google: ${pub(x.gemma4_e4b)}`],
      },
    ].filter((i) => i.value !== null);
    return { label: (LANG[x.lang] ?? x.lang) + (zh ? " (characters)" : ""), items };
  });
  const max = Math.max(...groups.flatMap((g) => g.items.map((i) => Math.max(i.value ?? 0, i.ref ?? 0))));
  const both = sorted.filter((x) => x.whisper_large_v2 !== null && ours("whisper", x.lang) !== null);
  const higher = both.filter((x) => ours("whisper", x.lang)! > x.whisper_large_v2!).length;
  const title = `Whisper's error rate on our medical teach-back clips is higher than its published FLEURS score in ${higher} of ${both.length} languages, so published benchmarks understate how hard this task is`;
  return (
    <ChartCard
      id="fleurs"
      title={title}
      sub="Each row joins the maker's published FLEURS score (hollow ring) to what we measured on our synthetic-voice clips of medical teach-back answers (filled dot). Google publishes Gemma 4 FLEURS scores for 8 of these languages, so some rows have no ring. Lower is better."
      caveats={[
        "The settings differ, so read this as context, not a head-to-head: FLEURS is read Wikipedia sentences, each maker scored with its own normaliser; ours is medical answers full of drug names and doses, spoken by text-to-speech voices and scored with our normalisation.",
        "We ran whisper-large-v3-turbo; the published ring is large-v2, the latest Whisper with an exact per-language table. " + F.note,
        "Our sample is 10 clips per language (about 200 to 350 words). Chinese is scored by characters throughout.",
      ]}
      source={
        <>
          <SourceLinks ids={[F.sources.whisper_large_v2, F.sources.gemma4_e4b]} b={b} lead="Published" />. Ours: Homeward speech test, <code>ml/results/asr.json</code>.
        </>
      }
      table={{
        head: ["Language", "Whisper large-v2, published", "Whisper turbo, ours", "Gemma 4 E4B, published", "Gemma 4 E4B, ours"],
        numeric: [false, true, true, true, true],
        rows: sorted.map((x) => [(LANG[x.lang] ?? x.lang) + (x.metric === "CER" ? " (CER)" : ""), wf(x.whisper_large_v2), wf(ours("whisper", x.lang)), wf(x.gemma4_e4b), wf(ours("e4b", x.lang))]),
      }}
    >
      <Legend
        items={[
          { label: "Published on FLEURS", color: "var(--muted)", shape: "ring" },
          { label: "Measured by us", color: "var(--muted)", shape: "dot" },
          { label: "Whisper (published large-v2, measured large-v3-turbo)", color: COLORS.compare, shape: "dot" },
          { label: "Gemma 4 E4B", color: COLORS.gemma, shape: "dot" },
        ]}
      />
      <HChart groups={groups} domain={[0, Math.max(50, Math.ceil(max / 10) * 10)]} mark="dot" format={(v) => `${v}%`} axisLabel="Word error rate (%), lower is better" ariaLabel="Published FLEURS error rates next to our measured error rates" labelWidth={170} />
    </ChartCard>
  );
}
