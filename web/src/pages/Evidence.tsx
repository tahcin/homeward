import { useEffect, useState } from "react";
import { Link } from "../App.tsx";
import { AsrChart, DeploymentCard, EndToEndChart, GradingChart, OverflagChart, ReassuranceChart } from "../components/evidence/OurCharts.tsx";
import { Scorecard } from "../components/evidence/Scorecard.tsx";
import { FleursChart, FleursPublishedChart, PublishedLlmChart } from "../components/evidence/PublishedCharts.tsx";
import type { Benchmarks, Results } from "../components/evidence/types.ts";
import "../components/evidence/evidence.css";

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
          Homeward's grader decides whether a patient really understood each discharge fact. It is our own fine-tuned Gemma model, running on one AMD workstation GPU
          inside the hospital, and every chart below puts it first against the base model it came from, the cloud models we could have used instead, and Whisper for speech.
        </p>
        {res.failed && <p className="note warn" style={{ marginTop: 16 }}>Our results file could not be loaded.</p>}
        {!r && !res.failed && <p className="muted" style={{ marginTop: 16 }}>Loading results.</p>}
        {r && <Scorecard r={r} />}
        {r?.generated_at && <p className="tiny faint" style={{ marginTop: 10 }}>Results generated {r.generated_at} from the latest evaluation run. Synthetic test data only.</p>}
      </section>

      {r && (
        <section aria-labelledby="grade-h">
          <h2 id="grade-h">Grading: Homeward against each comparator</h2>
          <p className="sec-lead">
            Tested on {r.grader.independent?.answers ?? "?"} independent teach-back answers in {r.grader.independent?.languages.length ?? "?"} languages, written by a different model family (Claude) and never seen by the data generator or in training. Three numbers matter:
            how often the verdict is right, how often a misunderstood critical fact is passed (the unsafe error), and how often a correct answer is sent back (the nurse's extra work).
          </p>
          <div className="ev-grid">
            <GradingChart r={r} />
            <ReassuranceChart r={r} />
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

      <section aria-labelledby="pub-h">
        <h2 id="pub-h">Why we built on Gemma 4</h2>
        <p className="sec-lead">
          Context for the choice of base model. These are official numbers from model cards and papers, not our measurements, and two models only sit side by side where they
          were scored on the same benchmark in the same way.
        </p>
        {bench.failed && <p className="note warn">The published benchmark file could not be loaded.</p>}
        {b && (
          <div className="ev-grid">
            <PublishedLlmChart b={b} />
            <FleursPublishedChart b={b} />
            {r && <FleursChart b={b} r={r} />}
          </div>
        )}
      </section>

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
                {h.note && <p className="tiny muted" style={{ marginTop: 8 }}>{h.note}</p>}
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
          <p className="sec-lead">Published figures were checked on these pages on the dates shown. {b.note}</p>
          <ol className="ev-sources">
            {Object.entries(b.sources).map(([id, s]) => (
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
