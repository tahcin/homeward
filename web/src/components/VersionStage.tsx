import { useRef, useState } from "react";
import { LANGUAGES, LEVELS, langByCode, type LockResult, type PrepareEvent, type Step } from "../../shared/types.ts";
import { useConfig } from "../App.tsx";
import { ApiError, preparePatientVersion } from "../lib/api.ts";
import type { Session, Stage } from "../lib/session.ts";
import type { Update } from "../pages/Flow.tsx";
import { fmtDate } from "./FactsStage.tsx";
import { Icon } from "./Icon.tsx";
import { ErrorNote } from "./NoteStage.tsx";

type Line = { key: number; text: string; when?: string; live?: boolean };

export function LockStatus({ l }: { l: LockResult }) {
  if (l.status === "locked")
    return (
      <span className="tag ok status-ic">
        <Icon name="lock" size={13} />
        Locked
      </span>
    );
  if (l.status === "repaired")
    return (
      <span className="tag warn status-ic">
        <Icon name="wrench" size={13} />
        Caught and repaired
      </span>
    );
  return (
    <span className="tag warn status-ic">
      <Icon name="alert" size={13} />
      Needs your review
    </span>
  );
}

const secs = (ms: number) => `${(ms / 1000).toFixed(1)} s`;

export function VersionStage({ s, update, goStage }: { s: Session; update: Update; goStage: (st: Stage) => void }) {
  const config = useConfig();
  const [lang, setLang] = useState(s.language);
  const [level, setLevel] = useState(s.level);
  const [fault, setFault] = useState<"" | "dose" | "negation">("");
  const [lines, setLines] = useState<Line[]>([]);
  const [busy, setBusy] = useState(false);
  const [picking, setPicking] = useState(false); // choosing a new language while the current version stays available
  const [error, setError] = useState<ApiError | string | null>(null);
  const n = useRef(0);
  const v = s.version;

  async function generate() {
    setBusy(true);
    setError(null);
    setLines([]);
    const steps: Step[] = [];
    const t0 = Date.now();
    const push = (text: string, live = true, when?: string) =>
      setLines((cur) => [...cur.map((l) => ({ ...l, live: false })), { key: n.current++, text, live, when }]);
    try {
      const version = await preparePatientVersion({ facts: s.facts, topics: s.topics, language: lang, level, fault: fault || undefined }, (e: PrepareEvent) => {
        if (e.type === "progress") push(e.message);
        if (e.type === "step") {
          steps.push(e.step);
          push(`${e.step.step}: ${e.step.model}${e.step.fallback_from ? ` (fallback from ${e.step.fallback_from})` : ""}`, false, secs(e.step.ms));
        }
      });
      push(`Done in ${secs(Date.now() - t0)}`, false);
      setPicking(false);
      update((cur) => ({
        version,
        versionReplayed: false,
        language: lang,
        level,
        acknowledged: [],
        attempts: [],
        steps: [...cur.steps.filter((x) => x.step === "Find the facts" || x.step === "Read the document (OCR)"), ...steps],
      }));
    } catch (e) {
      setError(e instanceof ApiError ? e : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  if (!v || picking) {
    return (
      <div className="stack-lg">
        <div className="stage-head">
          <div>
            <h2>Patient version</h2>
            <p>Choose the patient's language and reading level. Gemma 4 31B writes the instructions, then an independent model and plain code check every fact.</p>
          </div>
        </div>
        <section className="stack">
          <h3>Language</h3>
          <div className="langs" role="group" aria-label="Language">
            {LANGUAGES.map((l) => (
              <button key={l.code} className="lang-opt" aria-pressed={lang === l.code} onClick={() => setLang(l.code)} disabled={busy}>
                <b lang={l.code} dir={l.rtl ? "rtl" : undefined}>
                  {l.native}
                </b>
                <span>{l.name}</span>
              </button>
            ))}
          </div>
        </section>
        <section className="stack">
          <h3>Reading level</h3>
          <div className="seg" role="group" aria-label="Reading level">
            {LEVELS.map((x) => (
              <button key={x.id} aria-pressed={level === x.id} onClick={() => setLevel(x.id)} disabled={busy} title={x.hint}>
                {x.label}
              </button>
            ))}
          </div>
          <p className="small muted">{LEVELS.find((x) => x.id === level)?.hint}.</p>
        </section>
        <details>
          <summary>Test the safety check</summary>
          <div className="card pad stack" style={{ marginTop: 10 }}>
            <p className="small muted">
              Put a deliberate error into the first draft and watch the fact lock catch it. A doubled dose is caught by the code check; a reversed instruction keeps every
              number and name, so only the meaning check can catch it.
            </p>
            <div className="seg" role="group" aria-label="Deliberate error">
              {(
                [
                  ["", "No error"],
                  ["dose", "Double a dose"],
                  ["negation", "Reverse an instruction"],
                ] as const
              ).map(([id, label]) => (
                <button key={id} aria-pressed={fault === id} onClick={() => setFault(id)} disabled={busy}>
                  {label}
                </button>
              ))}
            </div>
          </div>
        </details>
        <div className="btn-row">
          {v && picking ? (
            <button className="btn quiet" onClick={() => setPicking(false)} disabled={busy}>
              <Icon name="left" />
              Cancel, keep the {langByCode(v.language).name} version
            </button>
          ) : (
            <button className="btn quiet" onClick={() => goStage("facts")} disabled={busy}>
              <Icon name="left" />
              Facts
            </button>
          )}
          <button className="btn primary" onClick={generate} disabled={busy || config?.live === false}>
            {busy && <Icon name="loader" className="spin" />}
            {busy ? "Writing and checking" : `Write in ${langByCode(lang).name}`}
          </button>
        </div>
        {lines.length > 0 && (
          <ol className="progress card pad" aria-live="polite">
            {lines.map((l) => (
              <li key={l.key}>
                <span className={`dot${l.live && busy ? " live" : ""}`} />
                <span>{l.text}</span>
                {l.when && <span className="when">{l.when}</span>}
              </li>
            ))}
          </ol>
        )}
        {busy && <p className="small muted">Writing, checking and back-translating takes about one to four minutes on the shared service; each step appears above as it finishes.</p>}
        {config?.live === false && (
          <ErrorNote error={new ApiError("budget", "Live generation is closed for today, so a new patient version cannot be written now. The sample patients still work end to end.")} />
        )}
        <ErrorNote error={error} />
      </div>
    );
  }

  const L = langByCode(v.language);
  const counts = { locked: 0, repaired: 0, flagged: 0 };
  v.lock.forEach((l) => counts[l.status]++);
  const unacked = v.lock.filter((l) => l.status === "flagged" && !s.acknowledged.includes(l.id));
  const faultFact = v.fault ? s.facts.find((f) => f.id === v.fault!.fact_id) : undefined;
  const faultLock = v.fault ? v.lock.find((l) => l.id === v.fault!.fact_id) : undefined;

  return (
    <div className="stack-lg">
      <div className="stage-head">
        <div>
          <h2>Patient version in {L.name}</h2>
          <p>
            {counts.locked + counts.repaired} of {v.lock.length} facts verified
            {counts.repaired ? `, ${counts.repaired} after a repair` : ""}
            {counts.flagged ? `. ${counts.flagged} need${counts.flagged === 1 ? "s" : ""} your review before the patient sees it.` : "."}
          </p>
        </div>
        <div className="btn-row">
          {s.versionReplayed && s.sample && (
            <span className="replay">
              <Icon name="play" size={14} />
              Replayed from a recorded run on {fmtDate(s.sample.recorded_at)}
            </span>
          )}
          <button
            className="btn small"
            onClick={() => {
              setLang(v.language);
              setLevel(v.level);
              setPicking(true);
            }}
          >
            Change language or level
          </button>
        </div>
      </div>

      {v.fault && (
        <div className={`note ${faultLock?.status === "flagged" ? "bad" : "ok"}`}>
          <Icon name={faultLock?.status === "flagged" ? "alert" : "check"} />
          <div style={{ display: "grid", gap: 4 }}>
            <strong>
              {v.fault.kind === "dose" ? "A doubled dose was put into the first draft on purpose." : "A reversed instruction was put into the first draft on purpose."}{" "}
              {faultLock?.status === "flagged" ? "The lock caught it; the repair did not pass, so it is flagged for you." : "The lock caught it and the sentence was rewritten."}
            </strong>
            <span lang={L.code} dir={L.rtl ? "rtl" : undefined}>
              Injected: “{v.fault.after}”
            </span>
            {faultFact && <span>Fact: {faultFact.statement}</span>}
          </div>
        </div>
      )}

      <div className="split">
        <section className="card pad leaflet" aria-label={`Patient version in ${L.name}`}>
          <h3 lang={L.code} dir={L.rtl ? "rtl" : undefined}>
            {v.title}
          </h3>
          <div lang={L.code} dir={L.rtl ? "rtl" : undefined}>
            {v.paragraphs.map((p, i) => (
              <p key={i}>{p}</p>
            ))}
          </div>
        </section>
        <section className="card pad leaflet" aria-label="English back-translation">
          <h3>Back-translation for you</h3>
          {v.back_translation.length ? (
            v.back_translation.map((p, i) => (
              <p key={i} className="muted">
                {p}
              </p>
            ))
          ) : (
            <p className="muted">The back-translation is not available for this run. Each fact's own sentence is back-translated below.</p>
          )}
          <p className="tiny faint" style={{ marginTop: 14 }}>
            Translated back by DeepSeek V4 Flash, a different model from the writer.
          </p>
        </section>
      </div>

      <section className="card" aria-labelledby="lock-h">
        <div className="pad" style={{ paddingBottom: 8 }}>
          <h3 id="lock-h">Fact lock</h3>
          <p className="small muted" style={{ marginTop: 4 }}>
            For each fact, the checker points at the sentences that carry it, and code confirms they contain every medicine name, number and date;
            then those sentences are translated back word for word (a wrong or nonsensical word fails the fact) and compared with the fact for meaning (a reversed instruction, a different frequency or a missing threshold fails). Medicine names
            stay in Latin letters so they match the box label.
          </p>
        </div>
        {v.lock.map((l) => {
          const f = s.facts.find((x) => x.id === l.id);
          return (
            <div className="lock-row" key={l.id}>
              <div className="fact-top">
                <LockStatus l={l} />
                {f?.critical && <span className="tag crit">Critical</span>}
                <span className="fact-id">{l.id}</span>
              </div>
              <p className="statement">{f?.statement}</p>
              {l.quote ? (
                <p className="quote" lang={L.code} dir={L.rtl ? "rtl" : undefined}>
                  “{l.quote}”
                </p>
              ) : (
                <p className="quote muted">No sentence carries this fact.</p>
              )}
              {l.quote_en && <p className="back">Back-translated: {l.quote_en}</p>}
              {l.history.map((h) => (
                <p className="history" key={h.round}>
                  {h.round === 0 ? "First draft" : `After repair ${h.round}`}: {h.problems.map((p) => p.reason).join("; ")}
                </p>
              ))}
              {l.status === "flagged" && (
                <label className="check small">
                  <input
                    type="checkbox"
                    checked={s.acknowledged.includes(l.id)}
                    onChange={(e) =>
                      update((cur) => ({ acknowledged: e.target.checked ? [...cur.acknowledged, l.id] : cur.acknowledged.filter((x) => x !== l.id) }))
                    }
                  />
                  <span>I have read this sentence and will explain this fact to the patient myself.</span>
                </label>
              )}
            </div>
          );
        })}
      </section>

      <div className="confirm-bar">
        <span className="small muted">
          {unacked.length ? `Review ${unacked.length} flagged ${unacked.length === 1 ? "fact" : "facts"} to continue.` : "Next, the patient reads or listens in their language, then explains it back."}
        </span>
        <div className="btn-row">
          <button className="btn quiet" onClick={() => goStage("facts")}>
            <Icon name="left" />
            Facts
          </button>
          <button
            className="btn primary"
            disabled={unacked.length > 0}
            onClick={() => {
              update({ teach: undefined });
              goStage("teach");
            }}
          >
            Give the tablet to the patient
            <Icon name="right" />
          </button>
        </div>
      </div>
    </div>
  );
}
