import { useEffect, useState } from "react";
import { Link } from "../App.tsx";
import { AsrChart, DeploymentCard, EndToEndChart, OverflagChart, SizeChart, TeacherChart } from "../components/evidence/OurCharts.tsx";
import { Scorecard } from "../components/evidence/Scorecard.tsx";
import type { Benchmarks, Results } from "../components/evidence/types.ts";
import "../components/evidence/evidence.css";

// Only the sources this page still cites: the Gemma 4 card and report, Whisper turbo, and AMD's specification pages.
const SOURCE_IDS = ["gemma4_card", "gemma4_report", "whisper_turbo_card", "amd_w7900d", "amd_w7900", "amd_mi300x"];

function useJson<T>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    fetch(url, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setData)
      .catch(() => setFailed(true));
  }, [url]);
  return { data, failed };
}

/** The W7900D note quotes our own memory figures, so it is written from the results file rather than kept as text. */
function hwNote(h: Benchmarks["hardware"][number], r: Results | null): string | undefined {
  if (!/W7900D/.test(h.name) || !r) return h.note;
  const train = r.training?.peak_vram_gib;
  const loop = r.onprem_loop?.peak_vram_gib;
  if (train == null || loop == null) return h.note;
  return `The card we fine-tuned and served Gemma 4 E4B on. Training peaked at ${train} GiB and the full speech-plus-grading loop at ${loop} GiB, so one workstation card holds it with room to spare.`;
}

export function Evidence() {
  const res = useJson<Results>("/evidence-results.json");
  const bench = useJson<Benchmarks>("/benchmarks.json");
  const r = res.data;
  const b = bench.data;

  return (
    <div className="wrap ev">
      <section>
        <h1 style={{ maxWidth: "22ch" }}>How the Homeward grader compares</h1>
        <p className="lead">
          Homeward's on-site grader decides whether a patient really understood each discharge fact. It is our own fine-tuned Gemma model, built to run on one AMD
          workstation GPU inside the hospital (the hosted demo on this site grades with Gemma 4 31B instead). Every chart below puts it first against the base model it came
          from, the cloud models we could have used instead, and Whisper for speech.
        </p>
        {res.failed && <p className="note warn" style={{ marginTop: 16 }}>Our results file could not be loaded.</p>}
        {!r && !res.failed && <p className="muted" style={{ marginTop: 16 }}>Loading results.</p>}
        {r && <Scorecard r={r} />}
        {r?.generated_at && <p className="tiny faint" style={{ marginTop: 10 }}>Results generated {r.generated_at} from the latest evaluation run. Synthetic test data only.</p>}
      </section>

      {r && (
        <section aria-labelledby="grade-h">
          <h2 id="grade-h">Grading: Homeward against other models</h2>
          <p className="sec-lead">
            Tested on {r.grader.independent?.answers ?? "?"} independent teach-back answers in {r.grader.independent?.languages.length ?? "?"} languages, written by a different model family (Claude) and never seen by the data generator or in training. Three numbers matter:
            how often the verdict is right, how often a misunderstood critical fact is passed (the unsafe error), and how often a correct answer is sent back (the nurse's extra work).
          </p>
          <div className="ev-grid">
            <SizeChart r={r} />
            <TeacherChart r={r} />
            <OverflagChart r={r} />
          </div>
        </section>
      )}

      {r && (
        <section aria-labelledby="speech-h">
          <h2 id="speech-h">Speech and the full on-site check</h2>
          <p className="sec-lead">Patients answer out loud. Homeward transcribes with the same Gemma model that grades, so the whole check runs on one card.</p>
          <div className="ev-grid">
            <AsrChart r={r} />
            <EndToEndChart r={r} />
          </div>
        </section>
      )}

      {r && (
        <section aria-labelledby="deploy-h">
          <h2 id="deploy-h">Running it in the hospital</h2>
          <p className="sec-lead">What one AMD Radeon PRO W7900D handles, next to the cloud graders that would need patient data to leave the building.</p>
          <DeploymentCard r={r} />
        </section>
      )}

      {b && (
        <section aria-labelledby="why-h">
          <h2 id="why-h">Why we built on Gemma 4</h2>
          <p className="sec-lead" style={{ marginBottom: 0 }}>
            We started from Gemma 4 E4B because one model handles both jobs: Google reports 76.6% on multilingual MMMLU for E4B (
            <a href={b.sources.gemma4_card?.url} target="_blank" rel="noreferrer">
              model card
            </a>
            ) and a 7.5% average error rate across 12 languages on FLEURS speech recognition, counting words (characters for Chinese, Japanese and Korean) (
            <a href={b.sources.gemma4_report?.url} target="_blank" rel="noreferrer">
              Gemma 4 technical report, Table 7
            </a>
            ), small enough to fine-tune and serve on one workstation card.
          </p>
        </section>
      )}

      {b && b.hardware.length > 0 && (
        <section aria-labelledby="hw-h">
          <h2 id="hw-h">The AMD hardware</h2>
          <p className="sec-lead">Why these models fit: the memory on one card decides what can run on a hospital's own machine.</p>
          <div className="ev-hw">
            {b.hardware.map((h) => (
              <div className="card" key={h.name}>
                <h3>{h.name}</h3>
                <ul>
                  {h.facts.map((f) => (
                    <li key={f.label}>
                      {f.label}: {f.value}
                    </li>
                  ))}
                </ul>
                {hwNote(h, r) && <p className="tiny muted" style={{ marginTop: 8 }}>{hwNote(h, r)}</p>}
                {b.sources[h.source] && (
                  <p className="tiny faint" style={{ marginTop: 8 }}>
                    Source:{" "}
                    <a href={b.sources[h.source].url} target="_blank" rel="noreferrer">
                      {b.sources[h.source].title}
                    </a>
                  </p>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {b && (
        <section aria-labelledby="src-h">
          <h2 id="src-h">Sources</h2>
          <p className="sec-lead">Published figures were checked on these pages on the dates shown.</p>
          <ol className="ev-sources">
            {Object.entries(b.sources)
              .filter(([id]) => SOURCE_IDS.includes(id))
              .map(([id, s]) => (
              <li key={id}>
                <a href={s.url} target="_blank" rel="noreferrer">
                  {s.title}
                </a>
                , {s.publisher}. Accessed {s.accessed}.
              </li>
            ))}
          </ol>
          <p className="muted" style={{ marginTop: 18 }}>
            How the pipeline works is on <Link to="/how">How it works</Link>; you can also <Link to="/check">try a discharge check</Link>.
          </p>
        </section>
      )}
    </div>
  );
}
