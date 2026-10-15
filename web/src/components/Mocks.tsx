// Small live renderings of real product moments for the landing page. Every string comes from a recorded sample
// run (public/samples), so nothing shown here is invented.

import { useEffect, useState } from "react";
import strings from "../i18n.json";
import { loadSample, type Sample } from "../lib/session.ts";
import { Icon } from "./Icon.tsx";

export const VERDICT: Record<string, { label: string; cls: string }> = {
  correct: { label: "Understood", cls: "ok" },
  partial: { label: "Partly", cls: "warn" },
  incorrect: { label: "Wrong", cls: "bad" },
  missing: { label: "Not mentioned", cls: "bad" },
};

/** Loads a recorded sample. `failed` turns true if it can't be fetched, so callers can fall back. */
export function useSample(id: string): Sample | null {
  return useSampleState(id).sample;
}

export function useSampleState(id: string): { sample: Sample | null; failed: boolean } {
  const [state, setState] = useState<{ sample: Sample | null; failed: boolean }>({ sample: null, failed: false });
  useEffect(() => {
    let live = true;
    loadSample(id)
      .then((sample) => live && setState({ sample, failed: false }))
      .catch(() => live && setState({ sample: null, failed: true }));
    return () => {
      live = false;
    };
  }, [id]);
  return state;
}

export const firstSentence = (t: string) => (t.match(/^[^.!?।。]+[.!?।。]?/u)?.[0] ?? t).trim();

/** Splits a fact label like "Apixaban 5 mg: twice a day" into the medicine and the instruction. */
export function splitLabel(label: string): { name: string; rule: string } {
  const i = label.indexOf(":");
  return i > 0 ? { name: label.slice(0, i).trim(), rule: label.slice(i + 1).trim() } : { name: label, rule: "" };
}

/** Every verdict a fact got across the tries on its topic, in order. */
export function verdictsFor(s: Sample, topicId: string, factId: string): string[] {
  return (s.answers[topicId] ?? []).map((a) => a.grades.find((g) => g.id === factId)?.verdict).filter(Boolean) as string[];
}

function Verdicts({ list, critical }: { list: string[]; critical: boolean }) {
  return (
    <span className="lpm-verdicts">
      {list.map((v, i) => (
        <span key={i} className="lpm-verdicts">
          {i > 0 && <Icon name="right" size={12} />}
          <span className={`tag ${v === "correct" ? "ok" : critical && i === list.length - 1 ? "bad" : "warn"}`}>{VERDICT[v]?.label ?? v}</span>
        </span>
      ))}
    </span>
  );
}

/** Step 1: one fact as the nurse sees it, with the words and numbers that must survive translation. */
export function FactRowMock({ s }: { s: Sample }) {
  const f = s.facts.find((x) => x.category === "medication_change" && x.critical) ?? s.facts[0];
  return (
    <div className="lpm-card" aria-hidden>
      <div className="lpm-row">
        <span className="tag crit">Critical</span>
        <span className="lpm-small">Medicine change</span>
      </div>
      <p className="lpm-text">{f.statement}</p>
      <div className="lpm-chips">
        {f.anchors.slice(0, 4).map((a) => (
          <span key={a} className="lpm-chip">
            <Icon name="lock" size={11} />
            {a}
          </span>
        ))}
      </div>
    </div>
  );
}

/** Step 2: the patient's question screen, in the patient's language. */
export function QuestionMock({ s }: { s: Sample }) {
  const t = (strings as Record<string, Record<string, string>>)[s.version.language] ?? strings.en;
  const topic = s.topics[0];
  return (
    <div className="lpm-card lpm-tablet" aria-hidden lang={s.version.language}>
      <p className="lpm-small">{t.questionOf.replace("{n}", "1").replace("{total}", String(s.topics.length))}</p>
      <p className="lpm-q">{s.version.questions[topic.id] || topic.question}</p>
      <span className="lpm-mic">
        <Icon name="mic" size={18} />
        {t.answerVoice}
      </span>
    </div>
  );
}

/** Step 3: the first topic's critical facts on the nurse's board, with the re-teach history. */
export function BoardMock({ s }: { s: Sample }) {
  const topic = s.topics[0];
  const rows = topic.fact_ids
    .map((id) => s.facts.find((f) => f.id === id)!)
    .filter((f) => f && f.critical)
    .slice(0, 3)
    .map((f) => ({ f, v: verdictsFor(s, topic.id, f.id) }));
  const allCritical = s.facts.filter((f) => f.critical);
  const ready = allCritical.every((f) => {
    const t = s.topics.find((x) => x.fact_ids.includes(f.id));
    return t && verdictsFor(s, t.id, f.id).at(-1) === "correct";
  });
  return (
    <div className="lpm-card lpm-board" aria-hidden>
      {rows.map(({ f, v }) => (
        <div key={f.id} className="lpm-board-row">
          <span className="lpm-label">{splitLabel(f.label ?? f.statement).name.split(";")[0]}</span>
          <Verdicts list={v} critical={f.critical} />
        </div>
      ))}
      <p className={`lpm-ready ${ready ? "ok" : "bad"}`}>
        <Icon name={ready ? "check" : "alert"} size={15} />
        {ready ? "Every critical fact understood" : "Re-teach needed"}
      </p>
    </div>
  );
}
