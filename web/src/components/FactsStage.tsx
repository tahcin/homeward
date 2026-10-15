import { useState } from "react";
import { CATEGORIES, type Category, type Fact, type Topic } from "../../shared/types.ts";
import { loadSample, signature, type Session, type Stage } from "../lib/session.ts";
import type { Update } from "../pages/Flow.tsx";
import { Icon } from "./Icon.tsx";

const CATEGORY_LABEL: Record<Category, string> = {
  medication: "Medicine",
  medication_change: "Medicine change",
  warning_sign: "Warning signs",
  follow_up: "Follow-up",
  activity: "Activity",
  diet: "Diet",
  self_monitoring: "Self-check",
  wound_care: "Wound care",
  device: "Device",
};
export const categoryLabel = (c: Category) => CATEGORY_LABEL[c] ?? c;

export function fmtDate(iso: string) {
  try {
    return new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
  } catch {
    return iso;
  }
}

function AnchorChips({ fact, onChange }: { fact: Fact; onChange: (anchors: string[]) => void }) {
  const [draft, setDraft] = useState("");
  const add = () => {
    const v = draft.trim();
    if (v && !fact.anchors.includes(v)) onChange([...fact.anchors, v]);
    setDraft("");
  };
  return (
    <div className="chips" aria-label="Anchors: words and numbers that must survive translation">
      <span className="tiny faint">Must survive translation:</span>
      {fact.anchors.map((a) => (
        <span className="chip" key={a} title={fact.statement.toLowerCase().includes(a.toLowerCase()) ? undefined : "Not in the sentence above"}>
          {a}
          <button aria-label={`Remove ${a}`} onClick={() => onChange(fact.anchors.filter((x) => x !== a))}>
            <Icon name="x" size={16} />
          </button>
        </span>
      ))}
      <input
        className="chip-input"
        value={draft}
        placeholder="Add, then Enter"
        aria-label="Add an anchor"
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            add();
          }
        }}
        onBlur={add}
      />
    </div>
  );
}

export function FactsStage({ s, update, goStage }: { s: Session; update: Update; goStage: (st: Stage) => void }) {
  const [checked, setChecked] = useState(s.factsConfirmed);
  const [busy, setBusy] = useState(false);
  const replay = s.source === "sample" && s.sample && signature(s.facts, s.topics) === s.sample.signature;

  /** Any edit un-confirms the facts and drops what was built on them. */
  const edit = (facts: Fact[], topics: Topic[]) => {
    if (replay && !window.confirm("This sample was recorded. If you change its facts, the patient version will be written live (about one to four minutes). Continue?")) return;
    setChecked(false);
    update({ facts, topics, factsConfirmed: false, version: undefined, versionReplayed: false, attempts: [], acknowledged: [] });
  };
  const setFact = (id: string, patch: Partial<Fact>) => edit(s.facts.map((f) => (f.id === id ? { ...f, ...patch } : f)), s.topics);
  const removeFact = (id: string) =>
    window.confirm("Delete this fact? It will not be checked with the patient.") &&
    edit(
      s.facts.filter((f) => f.id !== id),
      s.topics.map((t) => ({ ...t, fact_ids: t.fact_ids.filter((x) => x !== id) })).filter((t) => t.fact_ids.length > 0),
    );
  const addFact = (topicId: string) => {
    let n = s.facts.length + 1;
    while (s.facts.some((f) => f.id === `f${n}`)) n++;
    const f: Fact = { id: `f${n}`, category: "medication", critical: false, statement: "", anchors: [] };
    edit([...s.facts, f], s.topics.map((t) => (t.id === topicId ? { ...t, fact_ids: [...t.fact_ids, f.id] } : t)));
  };
  const addTopic = () => {
    let n = s.topics.length + 1;
    while (s.topics.some((t) => t.id === `t${n}`)) n++;
    let m = s.facts.length + 1;
    while (s.facts.some((f) => f.id === `f${m}`)) m++;
    const f: Fact = { id: `f${m}`, category: "activity", critical: false, statement: "", anchors: [] };
    edit([...s.facts, f], [...s.topics, { id: `t${n}`, question: "Tell me what you need to remember about this when you get home.", fact_ids: [f.id] }]);
  };

  const empty = s.facts.some((f) => !f.statement.trim()) || s.topics.some((t) => !t.question.trim());
  const criticalCount = s.facts.filter((f) => f.critical).length;

  async function confirm() {
    setBusy(true);
    const now = Date.now();
    let patch: Partial<Session> = { factsConfirmed: true, stage: "version", times: { ...s.times, factsConfirmed: now } };
    if (replay && s.sample) {
      try {
        const sample = await loadSample(s.sample.id);
        patch = {
          ...patch,
          version: sample.version,
          versionReplayed: true,
          language: sample.version.language,
          level: sample.version.level,
          steps: [...s.steps.filter((x) => x.step === "Find the facts"), ...sample.steps.filter((x) => x.step !== "Find the facts").map((x) => ({ ...x, replayed: true }))],
        };
      } catch {
        /* fall through to a live run */
      }
    }
    update(patch);
    setBusy(false);
    window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
  }

  return (
    <div>
      <div className="stage-head">
        <div>
          <h2>Facts to check</h2>
          <p>
            These are the facts the patient must understand. Check each one against the note, fix anything wrong, and mark the ones where a misunderstanding could
            cause harm as critical. Nothing goes to the patient until you confirm.
          </p>
        </div>
        {replay && s.sample && (
          <span className="replay">
            <Icon name="play" size={14} />
            Replayed from a recorded run on {fmtDate(s.sample.recorded_at)}
          </span>
        )}
      </div>

      <div className="split wide-left">
        <div className="stack">
          {s.nurse_checks.length > 0 && (
            <div className="note warn">
              <Icon name="alert" />
              <div>
                <strong>Worth checking in the note</strong>
                <ul className="clean" style={{ marginTop: 4 }}>
                  {s.nurse_checks.map((c, i) => (
                    <li key={i}>{c}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {s.topics.map((t, ti) => (
            <section className="card topic" key={t.id} aria-label={`Question ${ti + 1}`}>
              <div className="topic-head">
                <label className="field">
                  <span className="small muted" style={{ fontWeight: 500 }}>
                    Teach-back question {ti + 1}
                  </span>
                  <textarea
                    className="input"
                    rows={2}
                    value={t.question}
                    onChange={(e) => edit(s.facts, s.topics.map((x) => (x.id === t.id ? { ...x, question: e.target.value } : x)))}
                  />
                </label>
                <p className="tiny faint" style={{ marginTop: 6 }}>
                  Keep it open: name the topic, not the answer (no doses, times or numbers).
                </p>
              </div>
              {t.fact_ids.map((fid) => {
                const f = s.facts.find((x) => x.id === fid);
                if (!f) return null;
                return (
                  <div className="fact" key={f.id}>
                    <div className="fact-top">
                      <span className="fact-id">{f.id}</span>
                      <select className="sel" value={f.category} aria-label="Category" onChange={(e) => setFact(f.id, { category: e.target.value as Category })}>
                        {CATEGORIES.map((c) => (
                          <option key={c} value={c}>
                            {CATEGORY_LABEL[c]}
                          </option>
                        ))}
                      </select>
                      <label className="switch">
                        <input type="checkbox" checked={f.critical} onChange={(e) => setFact(f.id, { critical: e.target.checked })} />
                        <span>Critical</span>
                      </label>
                      <button className="btn quiet small" style={{ marginLeft: "auto" }} onClick={() => removeFact(f.id)} aria-label={`Delete fact ${f.id}`}>
                        <Icon name="trash" size={15} />
                        Delete
                      </button>
                    </div>
                    <textarea
                      className="input"
                      rows={2}
                      value={f.statement}
                      aria-label={`Fact ${f.id}`}
                      placeholder="One plain sentence the patient could follow"
                      onChange={(e) => setFact(f.id, { statement: e.target.value, label: undefined })}
                    />
                    <AnchorChips fact={f} onChange={(anchors) => setFact(f.id, { anchors })} />
                  </div>
                );
              })}
              <div className="fact">
                <button className="btn quiet small" style={{ justifySelf: "start" }} onClick={() => addFact(t.id)}>
                  <Icon name="plus" size={15} />
                  Add a fact to this question
                </button>
              </div>
            </section>
          ))}
          <button className="btn small" onClick={addTopic}>
            <Icon name="plus" size={15} />
            Add a question
          </button>
        </div>

        <aside className="sticky">
          <details className="card pad" open={typeof window !== "undefined" && window.innerWidth > 900}>
            <summary>Original discharge note</summary>
            <div className="notebox">{s.note}</div>
          </details>
        </aside>
      </div>

      <div className="confirm-bar">
        <label className="check">
          <input type="checkbox" checked={checked} onChange={(e) => setChecked(e.target.checked)} disabled={empty} />
          <span>
            I have checked these {s.facts.length} facts against the discharge note ({criticalCount} critical).
          </span>
        </label>
        <div className="btn-row">
          <button className="btn quiet" onClick={() => goStage("note")}>
            <Icon name="left" />
            Note
          </button>
          <button className="btn primary" disabled={!checked || empty || busy || s.facts.length === 0} onClick={confirm}>
            Confirm facts
            <Icon name="right" />
          </button>
        </div>
      </div>
      {empty && <p className="small" style={{ color: "var(--bad)" }}>Fill in or delete the empty fact or question before confirming.</p>}
    </div>
  );
}
