import { useEffect, useState } from "react";
import type { Update } from "../pages/Flow.tsx";
import { langByCode, type Fact, type Verdict } from "../../shared/types.ts";
import { clips, finalGrades, fmtDuration, type AuditStep, type Session, type Stage } from "../lib/session.ts";
import type { Attempt, FactGrade } from "../../shared/types.ts";
import { categoryLabel } from "./FactsStage.tsx";
import { Icon } from "./Icon.tsx";
import { LockStatus } from "./VersionStage.tsx";

const VERDICT: Record<Verdict, { label: string; cls: string }> = {
  correct: { label: "Understood", cls: "ok" },
  partial: { label: "Partly", cls: "warn" },
  incorrect: { label: "Wrong", cls: "bad" },
  missing: { label: "Not mentioned", cls: "bad" },
};

/** On-site grader replay (format in ONPREM_REPLAY.md): the fine-tuned E4B's verdicts for a sample's recorded answers. */
interface OnpremEntry {
  attempt: number;
  transcript: string;
  fact_ids: string[];
  grades?: FactGrade[];
  parse_error?: boolean;
  latency_ms?: number;
  tokens_in?: number;
  tokens_out?: number;
  tokens_per_s?: number;
  peak_vram_gib?: number;
}
interface Onprem {
  sample_id: string;
  recorded_at: string;
  model: { name: string; base?: string; adapter?: string };
  hardware: { gpu: string; rocm?: string; runtime?: string };
  answers: Record<string, OnpremEntry[]>;
}

function useOnprem(id: string | undefined): Onprem | null {
  const [d, setD] = useState<Onprem | null>(null);
  useEffect(() => {
    if (!id) return;
    fetch(`/samples/${id}.onprem.json`)
      // Unknown paths fall back to the app's HTML, so only a JSON response counts.
      .then((r) => (r.ok && (r.headers.get("content-type") ?? "").includes("json") ? r.json() : null))
      .then((j) => setD(j && j.answers ? (j as Onprem) : null))
      .catch(() => setD(null));
  }, [id]);
  return d;
}

/** Swap in the on-site grades for each recorded attempt whose transcript and facts match exactly. */
function withOnprem(attempts: Attempt[], o: Onprem): { attempts: Attempt[]; used: OnpremEntry[]; unmatched: number } {
  const used: OnpremEntry[] = [];
  let unmatched = 0;
  const seen: Record<string, number> = {};
  const out = attempts.map((a) => {
    const i = (seen[a.topic_id] = (seen[a.topic_id] ?? -1) + 1);
    const e = o.answers[a.topic_id]?.[i];
    const ids = a.grades.map((g) => g.id).join();
    if (!e || e.parse_error || !e.grades || e.transcript.trim() !== a.transcript.trim() || e.fact_ids.join() !== ids) {
      unmatched++;
      return a;
    }
    used.push(e);
    return { ...a, grades: a.grades.map((g) => ({ ...g, ...(e.grades!.find((x) => x.id === g.id) ?? {}), reteach_patient: undefined })) };
  });
  return { attempts: out, used, unmatched };
}

/** Older recordings said "AMD Instinct"; the Token Factory manual only says AMD GPUs, so that is what we show. */
const hostLabel = (h: string) => (/Instinct/.test(h) ? "AMD GPUs via AMD Token Factory" : h);

export function Audit({ steps }: { steps: AuditStep[] }) {
  const total = steps.reduce((n, x) => n + x.ms, 0);
  const tin = steps.reduce((n, x) => n + (x.tokens_in ?? 0), 0);
  const tout = steps.reduce((n, x) => n + (x.tokens_out ?? 0), 0);
  const cost = steps.reduce((n, x) => n + (x.cost_usd ?? 0), 0);
  return (
    <div className="table-scroll">
      <table className="audit">
        <thead>
          <tr>
            <th>Step</th>
            <th>Model</th>
            <th>Where it ran</th>
            <th className="num">Time</th>
            <th className="num">Tokens in / out</th>
          </tr>
        </thead>
        <tbody>
          {steps.map((x, i) => (
            <tr key={i}>
              <td>
                {x.step}
                {x.replayed && (
                  <span className="tag" style={{ marginLeft: 6 }}>
                    recorded
                  </span>
                )}
                {x.note && <div className="tiny faint">{x.note}</div>}
              </td>
              <td>
                {x.model}
                {x.fallback_from && <div className="tiny" style={{ color: "var(--warn)" }}>fallback: {x.fallback_from} did not answer</div>}
              </td>
              <td>{hostLabel(x.host)}</td>
              <td className="num">{(x.ms / 1000).toFixed(1)} s</td>
              <td className="num">{x.tokens_in != null ? `${x.tokens_in.toLocaleString()} / ${(x.tokens_out ?? 0).toLocaleString()}` : ""}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <td colSpan={3}>
              <b>Total</b>
              {cost > 0 && (
                <span className="muted small">
                  {" "}
                  · ${cost.toFixed(4)} at the Token Factory list prices returned by its API (the dashboard shows higher point rates; the event quota is free)
                </span>
              )}
            </td>
            <td className="num">
              <b>{(total / 1000).toFixed(1)} s</b>
            </td>
            <td className="num">
              <b>
                {tin.toLocaleString()} / {tout.toLocaleString()}
              </b>
            </td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

function summaryText(s: Session, ready: boolean, rows: { fact: Fact; verdicts: Verdict[]; heard: string; reteach: string }[]): string {
  const L = langByCode(s.language);
  const lines = [
    `Discharge teach-back (Homeward), ${new Date().toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}. Language: ${L.name}.`,
    `Teach-back result: ${ready ? "Understanding confirmed: every critical fact explained back correctly." : "Re-teach needed."}`,
    "Homeward supports patient education. It does not make discharge decisions; the clinician does.",
    "",
  ];
  for (const r of rows) {
    const v = r.verdicts.length ? r.verdicts.map((x) => VERDICT[x].label).join(" -> ") : "Not checked";
    lines.push(`- ${r.fact.critical ? "[critical] " : ""}${r.fact.statement}`);
    lines.push(`  Teach-back: ${v}${r.heard ? `. Patient said: ${r.heard}` : ""}`);
    if (r.reteach && r.verdicts.at(-1) !== "correct") lines.push(`  Re-teach: ${r.reteach}`);
  }
  lines.push("", "Patient version written by Gemma 4 31B and checked by code plus an independent model; teach-back graded per fact. Synthetic demo data.");
  return lines.join("\n");
}

export function Board({ s, update, goStage, reset }: { s: Session; update: Update; goStage: (st: Stage) => void; reset: () => void }) {
  // The note-to-decision clock stops the first time the nurse sees this board.
  useEffect(() => {
    if (!s.times.decided) update((cur) => ({ times: { ...cur.times, decided: Date.now() } }));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const [copied, setCopied] = useState(false);
  const L = langByCode(s.language);
  const onprem = useOnprem(s.source === "sample" ? s.sample?.id : undefined);
  const [onsite, setOnsite] = useState(false);
  const swap = onprem && onsite ? withOnprem(s.attempts, onprem) : null;
  const fg = finalGrades(swap ? { ...s, attempts: swap.attempts } : s);
  const rows = s.facts.map((fact) => {
    const g = fg.get(fact.id);
    return {
      fact,
      verdicts: g ? g.history.map((h) => h.verdict) : ([] as Verdict[]),
      heard: g ? g.grade.heard : "",
      reteach: g ? g.grade.reteach : "",
      attempts: g ? g.attempts : [],
    };
  });
  const critical = rows.filter((r) => r.fact.critical);
  const criticalMisses = critical.filter((r) => r.verdicts.at(-1) !== "correct");
  const ready = criticalMisses.length === 0 && critical.length > 0;
  const otherMisses = rows.filter((r) => !r.fact.critical && r.verdicts.length > 0 && r.verdicts.at(-1) !== "correct");
  const understood = rows.filter((r) => r.verdicts.at(-1) === "correct").length;
  const decided = s.times.decided ?? Date.now();
  // In a replayed sample the clocks only measure how fast someone clicked through, so show the recorded model time instead.
  const replay = s.source === "sample" && Boolean(s.versionReplayed);
  const modelMs = s.steps.reduce((n, x) => n + x.ms, 0);

  return (
    <div className="stack-lg">
      <div className="stage-head">
        <div>
          <h2>Teach-back results</h2>
          <p>
            Every fact, the patient's own words, and the verdict after the re-teach. "Understanding confirmed" needs every critical fact explained back correctly; the model cannot waive one. Homeward supports patient education. It does not make discharge decisions; the clinician does.
          </p>
        </div>
        <div className="btn-row no-print">
          <button
            className="btn small"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(summaryText(s, ready, rows));
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
              } catch {
                /* clipboard blocked */
              }
            }}
          >
            <Icon name={copied ? "check" : "copy"} size={15} />
            {copied ? "Copied" : "Copy for the chart"}
          </button>
          <button className="btn small" onClick={() => window.print()}>
            <Icon name="print" size={15} />
            Print
          </button>
        </div>
      </div>

      {onprem && (
        <div className="card pad no-print" style={{ display: "grid", gap: 6 }}>
          <label className="switch" style={{ fontSize: "0.98rem", color: "var(--ink)" }}>
            <input type="checkbox" checked={onsite} onChange={(e) => setOnsite(e.target.checked)} />
            <span>Grade with the on-site model (recorded on the {onprem.hardware.gpu})</span>
          </label>
          <span className="small muted">
            {onsite
              ? `Showing ${onprem.model.name}'s verdicts for the same recorded answers${swap && swap.unmatched ? `; ${swap.unmatched} answer(s) had no matching on-site grade and keep the hosted grade` : ""}.`
              : "Shows how our fine-tuned Gemma 4 E4B, running on one workstation GPU, graded these same answers."}
          </span>
        </div>
      )}

      <div className={`verdict ${ready ? "ok" : "bad"}`} role="status">
        <Icon name={ready ? "check" : "alert"} size={28} />
        <div>
          <h2>{ready ? "Understanding confirmed" : "Re-teach needed"}</h2>
          {ready ? (
            <p style={{ marginTop: 6 }}>All {critical.length} critical facts were explained back correctly.</p>
          ) : (
            <ul>
              {criticalMisses.map((r) => (
                <li key={r.fact.id} title={r.fact.statement}>
                  <b>{r.fact.label ?? r.fact.statement}</b> ({r.verdicts.length ? VERDICT[r.verdicts.at(-1)!].label.toLowerCase() : "not checked"})
                </li>
              ))}
            </ul>
          )}
          {otherMisses.length > 0 && (
            <p style={{ marginTop: 10, color: "var(--ink)" }}>
              Also go over: {otherMisses.map((r) => r.fact.label ?? r.fact.statement).join("; ")}.
            </p>
          )}
        </div>
      </div>

      <div className="metrics">
        {replay ? (
          <div className="card pad metric">
            <b>{fmtDuration(modelMs)}</b>
            <span>Model time in the recorded run (this replay's clock is not a measurement)</span>
          </div>
        ) : (
          <div className="card pad metric">
            <b>{fmtDuration(decided - s.times.started)}</b>
            <span>From discharge note to these results</span>
          </div>
        )}
        <div className="card pad metric">
          <b>{replay ? "not measured" : s.times.factsShown && s.times.factsConfirmed ? fmtDuration(s.times.factsConfirmed - s.times.factsShown) : "n/a"}</b>
          <span>{replay ? "Time on the facts screen (not measured in a replay)" : "Time on the facts screen"}</span>
        </div>
        <div className="card pad metric">
          <b>
            {understood} of {rows.length}
          </b>
          <span>Facts understood after re-teach</span>
        </div>
        <div className="card pad metric">
          <b>
            {critical.length - criticalMisses.length} of {critical.length}
          </b>
          <span>Critical facts understood</span>
        </div>
      </div>

      {s.topics.map((t, ti) => (
        <section className="card" key={t.id} aria-label={`Question ${ti + 1}`}>
          <div className="pad" style={{ paddingBottom: 10, borderBottom: "1px solid var(--line)" }}>
            <h3>
              {ti + 1}. {t.question}
            </h3>
            {s.version?.questions[t.id] && (
              <p className="small muted" lang={L.code} dir={L.rtl ? "rtl" : undefined} style={{ marginTop: 4 }}>
                {s.version.questions[t.id]}
              </p>
            )}
          </div>
          {t.fact_ids.map((fid) => {
            const r = rows.find((x) => x.fact.id === fid);
            if (!r) return null;
            const last = r.verdicts.at(-1);
            const lock = s.version?.lock.find((l) => l.id === fid);
            return (
              <div className={`board-row${r.fact.critical && last !== "correct" ? " critical-miss" : ""}`} key={fid}>
                <div className="fact-top">
                  {r.fact.critical && <span className="tag crit">Critical</span>}
                  <span className="tag">{categoryLabel(r.fact.category)}</span>
                  {lock && <LockStatus l={lock} />}
                  <span className="attempts" style={{ marginLeft: "auto" }}>
                    {r.verdicts.length === 0 && <span className="tag">Not checked</span>}
                    {r.verdicts.map((x, i) => (
                      <span key={i} className="attempts">
                        {i > 0 && <Icon name="right" size={13} className="faint" />}
                        <span className={`tag ${x === "correct" ? "ok" : r.fact.critical ? "bad" : "warn"}`}>{VERDICT[x].label}</span>
                      </span>
                    ))}
                  </span>
                </div>
                {r.fact.label ? (
                  <p>
                    <b>{r.fact.label}</b>
                    <span className="muted small" style={{ display: "block" }}>
                      {r.fact.statement}
                    </span>
                  </p>
                ) : (
                  <p>{r.fact.statement}</p>
                )}
                {r.heard && (
                  <p className="said">
                    Heard: <b>{r.heard}</b>
                  </p>
                )}
                {r.reteach && last !== "correct" && <p className="said">Re-teach: {r.reteach}</p>}
                {r.attempts.length > 0 && (
                  <details>
                    <summary>Patient's own words</summary>
                    <div className="stack">
                      {r.attempts.map((a, i) => {
                        const clip = clips.get(`${a.topic_id}:${i}`);
                        return (
                          <div key={i} className="small">
                            <p className="faint tiny">
                              {r.attempts.length > 1 ? (i === 0 ? "First answer" : "After re-teach") : "Answer"} · {a.input === "voice" ? "spoken" : a.input === "sample" ? "sample answer" : "typed"}
                              {a.replayed ? " · recorded run" : ""}
                            </p>
                            <p lang={L.code} dir={L.rtl ? "rtl" : undefined}>
                              “{a.transcript}”
                            </p>
                            {a.transcript_en && <p className="muted">Patient said (English rendering): {a.transcript_en}</p>}
                            {clip && <audio controls src={clip} style={{ marginTop: 6, maxWidth: "100%" }} />}
                          </div>
                        );
                      })}
                    </div>
                  </details>
                )}
              </div>
            );
          })}
        </section>
      ))}

      <section className="card pad stack" aria-labelledby="audit-h">
        <h3 id="audit-h">Audit trail</h3>
        <p className="small muted">
          Every model call in this check, the model that actually answered and where it ran. Patient text is not stored on the server, and Homeward does not upload the
          recorded audio. Speech recognition in this hosted demo is done by your browser, which may send the audio to its maker (for example Google, in Chrome).
        </p>
        <Audit steps={s.steps} />
        {swap && onprem && swap.used.length > 0 && (
          <div className="table-scroll" style={{ marginTop: 12 }}>
            <p className="small" style={{ marginBottom: 6 }}>
              <b>On-site grading</b>: {onprem.model.name} on the {onprem.hardware.gpu}
              {onprem.hardware.runtime ? `, ${onprem.hardware.runtime}` : ""}, recorded {new Date(onprem.recorded_at).toLocaleDateString("en-GB")}.
            </p>
            <table className="audit">
              <thead>
                <tr>
                  <th>Answer</th>
                  <th className="num">Latency</th>
                  <th className="num">Tokens in / out</th>
                  <th className="num">Tokens per second</th>
                  <th className="num">Peak VRAM</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(onprem.answers).flatMap(([tid, list]) =>
                  list.map((e) => (
                    <tr key={`${tid}-${e.attempt}`}>
                      <td>
                        Question {s.topics.findIndex((t) => t.id === tid) + 1}, {e.attempt === 0 ? "first answer" : "after re-teach"}
                      </td>
                      <td className="num">{e.latency_ms != null ? `${(e.latency_ms / 1000).toFixed(1)} s` : ""}</td>
                      <td className="num">{e.tokens_in != null ? `${e.tokens_in} / ${e.tokens_out ?? ""}` : ""}</td>
                      <td className="num">{e.tokens_per_s != null ? e.tokens_per_s.toFixed(1) : ""}</td>
                      <td className="num">{e.peak_vram_gib != null ? `${e.peak_vram_gib.toFixed(1)} GiB` : ""}</td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <div className="btn-row no-print">
        <button
          className="btn"
          onClick={() => {
            update({ teach: undefined });
            goStage("teach");
          }}
        >
          <Icon name="redo" size={16} />
          Give the tablet back to the patient
        </button>
        <button
          className="btn quiet"
          onClick={() => {
            if (!window.confirm("Start a new discharge check? This check will be cleared from this tab.")) return;
            reset();
            goStage("note");
          }}
        >
          Start a new discharge check
        </button>
      </div>
    </div>
  );
}
