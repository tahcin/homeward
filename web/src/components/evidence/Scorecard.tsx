import { useSystems } from "./OurCharts.tsx";
import type { Results, SystemResult } from "./types.ts";

const f1 = (v: number) => `${v.toFixed(1)}%`;
const signed = (d: number) => `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)}`;

/** The headline: the Homeward grader, then one tile per comparator saying how it compares. */
export function Scorecard({ r }: { r: Results }) {
  const { ours, others } = useSystems(r);
  if (!ours) return null;
  const tr = r.training;
  const crit = (s: SystemResult) => s.critical_false_reassurance_n?.[0];

  const tile = (s: SystemResult) => {
    const d = ours.accuracy !== null && s.accuracy !== null ? ours.accuracy - s.accuracy : null;
    const oc = crit(ours);
    const sc = crit(s);
    const lines: string[] = [];
    if (s.role === "base" && ours.paired_vs_base) lines.push(`Fixes ${ours.paired_vs_base.fixes} facts it got wrong, breaks ${ours.paired_vs_base.breaks}`);
    if (ours.over_flag !== null && s.over_flag !== null) lines.push(`False alarms ${f1(s.over_flag)} → ${f1(ours.over_flag)} for Homeward`);
    if (oc !== undefined && sc !== undefined)
      lines.push(`Critical false passes: Homeward ${oc}, this ${sc}${oc > sc ? " (Homeward not yet better)" : oc < sc ? " (Homeward better)" : " (level)"}`);
    if (s.role === "cloud") lines.push("Cloud only; Homeward runs on site");
    if (s.json_valid !== null && s.json_valid < 100) lines.push(`Valid structured replies ${f1(s.json_valid)}, Homeward ${f1(ours.json_valid ?? 0)}`);
    return (
      <div className="card" key={s.id}>
        <div className="v-who">vs {s.label}</div>
        <div className="v-big">
          {d !== null ? `${signed(d)} points` : "Pending"}
          <small>accuracy</small>
        </div>
        <ul>
          {lines.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </div>
    );
  };

  const asrO = r.asr.wer[r.ours.asr_system]?.human?.ALL;
  const asrW = r.asr.wer.whisper?.human?.ALL;
  const bnO = r.asr.wer[r.ours.asr_system]?.human?.bn;
  const bnW = r.asr.wer.whisper?.human?.bn;
  const g = r.asr.grading_from_transcripts;
  const eo = g?.[`${r.ours.asr_system}/human`];
  const ew = g?.["whisper/human"];

  return (
    <>
      <div className="card ev-ours">
        <div className="o-top">
          <span className="o-name">{r.ours.label}</span>
          <span className="o-desc">
            {r.ours.description}
            {tr ? `, trained on ${tr.examples} examples in ${tr.minutes} minutes on one ${tr.device}` : ""}
          </span>
        </div>
        <div className="o-stats">
          {ours.accuracy !== null && (
            <div className="o-stat">
              <b>{f1(ours.accuracy)}</b>
              <span>facts graded correctly{ours.accuracy_ci ? ` (95% interval ${ours.accuracy_ci[0]}–${ours.accuracy_ci[1]}%)` : ""}</span>
            </div>
          )}
          {ours.over_flag !== null && (
            <div className="o-stat">
              <b>{f1(ours.over_flag)}</b>
              <span>false alarms on facts the patient got right</span>
            </div>
          )}
          {ours.critical_false_reassurance_n && (
            <div className="o-stat">
              <b>
                {ours.critical_false_reassurance_n[0]} of {ours.critical_false_reassurance_n[1]}
              </b>
              <span>critical facts falsely passed</span>
            </div>
          )}
          {r.capacity.discharges_per_hour_per_card != null && (
            <div className="o-stat">
              <b>{r.capacity.discharges_per_hour_per_card}</b>
              <span>discharges an hour on one W7900D, speech included</span>
            </div>
          )}
        </div>
      </div>
      <div className="ev-vs">
        {others.map(tile)}
        {asrO?.wer != null && asrW?.wer != null && (
          <div className="card">
            <div className="v-who">Speech: vs Whisper large-v3-turbo</div>
            <div className="v-big">
              {f1(asrO.wer)}
              <small>vs {f1(asrW.wer)} word errors</small>
            </div>
            <ul>
              <li>On the owner's real voice, {asrO.clips} clips</li>
              {bnO?.wer != null && bnW?.wer != null && <li>Bengali {f1(bnO.wer)} vs {f1(bnW.wer)}</li>}
              {eo && ew && (
                <li>
                  Facts graded correctly end to end: {eo.correct_vs_gold} of {eo.facts} vs {ew.correct_vs_gold} of {ew.facts}
                </li>
              )}
            </ul>
          </div>
        )}
      </div>
    </>
  );
}
