// Hero vignette: the caregiver's spoken answer in Hindi, its English rendering, and the nurse's board.
// Strings are read at runtime from the recorded Hindi sample, so a re-recorded sample shows up here as-is.
// If the sample can't be fetched, the strings from the recording made on 2026-10-15 are shown instead.

import { Icon } from "../Icon.tsx";
import { VERDICT, firstSentence, splitLabel, useSampleState, verdictsFor } from "../Mocks.tsx";
import type { Sample } from "../../lib/session.ts";

interface Row {
  id: string;
  name: string;
  rule: string;
  heard: string;
  critical: boolean;
  verdicts: string[];
}

interface Moment {
  said: string;
  saidEn: string;
  rows: Row[];
}

const FALLBACK: Moment = {
  said: "पापा Apixaban सुबह एक बार लेंगे, और Metoprolol दिन में दो बार, सुबह 9 बजे और रात 9 बजे।",
  saidEn: "Papa Apixaban morning one time will take, and Metoprolol day in two times, morning 9 o'clock and night 9 o'clock.",
  rows: [
    { id: "f1", name: "Apixaban 5 mg", rule: "twice a day, 9 am and 9 pm", heard: "Apixaban once in the morning", critical: true, verdicts: ["incorrect", "partial"] },
    { id: "f2", name: "Metoprolol 25 mg", rule: "once a day, morning", heard: "Metoprolol twice a day at 9 am and 9 pm", critical: false, verdicts: ["incorrect", "incorrect"] },
    { id: "f3", name: "Aspirin 75 mg", rule: "stop", heard: "Aspirin is stopped", critical: true, verdicts: ["correct"] },
  ],
};

/** The first answer on the medicines topic, with up to three facts: a critical miss, another miss, and one understood. */
function fromSample(s: Sample): Moment | null {
  const topic = s.topics[0];
  const first = s.answers[topic?.id]?.[0];
  if (!first) return null;
  const rows: Row[] = first.grades.map((g) => {
    const f = s.facts.find((x) => x.id === g.id);
    const { name, rule } = splitLabel(f?.label ?? f?.statement ?? g.id);
    return { id: g.id, name, rule, heard: g.heard, critical: !!f?.critical, verdicts: verdictsFor(s, topic.id, g.id) };
  });
  const missedCritical = rows.filter((r) => r.critical && r.verdicts[0] !== "correct");
  const missedOther = rows.filter((r) => !r.critical && r.verdicts[0] !== "correct");
  const understood = rows.filter((r) => r.verdicts.at(-1) === "correct");
  const pick = [...missedCritical.slice(0, 1), ...missedOther.slice(0, 1), ...understood.filter((r) => r.critical).slice(0, 1)];
  if (!pick.length) return null;
  return { said: firstSentence(first.transcript), saidEn: firstSentence(first.transcript_en), rows: pick };
}

function BoardLine({ r, delay }: { r: Row; delay: number }) {
  const last = r.verdicts.at(-1) ?? "";
  const ok = last === "correct";
  const tone = ok ? "ok" : r.critical ? "bad" : "warn";
  return (
    <li className={`lpv-line ${tone}`} style={{ animationDelay: `${delay}ms` }}>
      <div className="lpv-line-top">
        <span className="lpv-med">{r.name}</span>
        <span className="lpv-tags">
          {r.critical && !ok && <span className="tag crit">Critical</span>}
          {r.verdicts.map((v, i) => (
            <span key={i} className="lpv-tags">
              {i > 0 && <Icon name="right" size={11} />}
              <span className={`tag ${v === "correct" ? "ok" : i === r.verdicts.length - 1 && r.critical ? "bad" : "warn"}`}>{VERDICT[v]?.label ?? v}</span>
            </span>
          ))}
        </span>
      </div>
      {ok ? (
        <p className="lpv-rule">{r.rule}</p>
      ) : (
        <dl className="lpv-compare">
          <dt>Should be</dt>
          <dd>{r.rule}</dd>
          <dt>Heard</dt>
          <dd>{r.heard}</dd>
        </dl>
      )}
    </li>
  );
}

export function HeroVignette() {
  const { sample, failed } = useSampleState("anticoagulant-hi");
  const m = sample ? fromSample(sample) ?? FALLBACK : failed ? FALLBACK : null;
  if (!m) return <div className="lpv lpv-wait" aria-hidden />;
  const critical = m.rows.find((r) => r.critical && r.verdicts.at(-1) !== "correct");
  return (
    <figure className="lpv" aria-label="A recorded teach-back in Hindi. The son explains his father's medicines, and the nurse's board shows which ones he got wrong.">
      <div className="lpv-said">
        <p className="lpv-who">His son explains the medicines, in Hindi</p>
        <p className="lpv-quote" lang="hi">
          {m.said}
        </p>
        <p className="lpv-en">
          <span>Word for word:</span> {m.saidEn}
        </p>
      </div>
      <div className="lpv-board">
        <p className="lpv-board-head">
          <span>Nurse's board</span>
          {critical && <span className="lpv-board-flag">Re-teach needed: {critical.name.split(" ")[0]}</span>}
        </p>
        <ul>
          {m.rows.map((r, i) => (
            <BoardLine key={r.id} r={r} delay={450 + i * 160} />
          ))}
        </ul>
      </div>
    </figure>
  );
}
