import { useEffect, useState } from "react";
import { Link } from "../App.tsx";

type Kind = "ai" | "code" | "person";
interface Node {
  id: string;
  title: string;
  sub: string[];
  kind: Kind;
}

const NODES: Node[] = [
  { id: "note", title: "Discharge note", sub: ["Pasted, or a PDF or photo", "read by MinerU 2.5"], kind: "person" },
  { id: "facts", title: "Find the facts", sub: ["Gemma 4 31B lists facts,", "anchors and questions"], kind: "ai" },
  { id: "confirm", title: "Nurse confirms", sub: ["Edits, marks critical;", "nothing passes unconfirmed"], kind: "person" },
  { id: "write", title: "Patient version", sub: ["Gemma 4 31B writes it", "in the patient's language"], kind: "ai" },
  { id: "lock", title: "Fact lock", sub: ["Sentences + anchors: code", "Meaning: DeepSeek V4"], kind: "code" },
  { id: "speak", title: "Teach-back", sub: ["Patient explains in", "their own words"], kind: "person" },
  { id: "grade", title: "Grade each fact", sub: ["Gemma 4 31B, or our", "fine-tuned Gemma 4 E4B"], kind: "ai" },
  { id: "reteach", title: "Re-teach once", sub: ["Locked sentence again,", "then one more try"], kind: "code" },
  { id: "board", title: "Results for the nurse", sub: ["Confirmed only when every", "critical fact is understood"], kind: "code" },
];

const EDGES: [string, string][] = [
  ["note", "facts"],
  ["facts", "confirm"],
  ["confirm", "write"],
  ["write", "lock"],
  ["lock", "speak"],
  ["speak", "grade"],
  ["grade", "reteach"],
  ["reteach", "board"],
];

function Diagram({ vertical }: { vertical: boolean }) {
  const W = vertical ? 340 : 960;
  const bw = vertical ? 236 : 162;
  const bh = vertical ? 76 : 84;
  // Desktop: a snake, row one left to right, row two right to left. Mobile: one column.
  const pos: Record<string, { x: number; y: number }> = {};
  NODES.forEach((n, i) => {
    if (vertical) pos[n.id] = { x: 16, y: 16 + i * (bh + 26) };
    else if (i < 5) pos[n.id] = { x: 16 + i * (bw + 30), y: 56 };
    else pos[n.id] = { x: 16 + (4 - (i - 5)) * (bw + 30), y: 240 };
  });
  const H = vertical ? 16 + NODES.length * (bh + 26) : 380;
  const mid = (id: string) => ({ cx: pos[id].x + bw / 2, cy: pos[id].y + bh / 2 });

  function edge(a: string, b: string, i: number) {
    const A = pos[a];
    const B = pos[b];
    let d: string;
    let tip: { x: number; y: number; dir: "r" | "l" | "d" };
    if (vertical || (a === "lock" && b === "speak")) {
      const x = A.x + bw / 2;
      d = `M${x} ${A.y + bh} L${x} ${B.y - 6}`;
      tip = { x, y: B.y, dir: "d" };
    } else if (B.x > A.x) {
      const y = A.y + bh / 2;
      d = `M${A.x + bw} ${y} L${B.x - 6} ${y}`;
      tip = { x: B.x, y, dir: "r" };
    } else {
      const y = A.y + bh / 2;
      d = `M${A.x} ${y} L${B.x + bw + 6} ${y}`;
      tip = { x: B.x + bw, y, dir: "l" };
    }
    return (
      <g key={i}>
        <path className="edge" d={d} />
        <Arrow {...tip} />
      </g>
    );
  }

  // Loops: lock -> write (repair), reteach -> speak (one more try).
  const loops = vertical
    ? [
        { from: "lock", to: "write", label: "repair, at most twice" },
        { from: "reteach", to: "speak", label: "one retry" },
      ].map((l, i) => {
        const a = mid(l.from);
        const b = mid(l.to);
        const x = 16 + bw;
        const r = 40;
        return (
          <g key={`l${i}`}>
            <path className="edge back" d={`M${x} ${a.cy} C${x + r} ${a.cy} ${x + r} ${b.cy} ${x + 6} ${b.cy}`} />
            <Arrow x={x} y={b.cy} dir="l" back />
            <text x={x + r - 2} y={(a.cy + b.cy) / 2 + 4} className="sub" textAnchor="start" style={{ fill: "var(--warn)" }}>
              {l.label.split(", ")[0]}
            </text>
          </g>
        );
      })
    : [
        (() => {
          const a = mid("lock");
          const b = mid("write");
          const y = pos.lock.y;
          return (
            <g key="l0">
              <path className="edge back" d={`M${a.cx} ${y} C${a.cx} ${y - 40} ${b.cx} ${y - 40} ${b.cx} ${y - 6}`} />
              <Arrow x={b.cx} y={y} dir="d" back />
              <text x={(a.cx + b.cx) / 2} y={y - 36} className="sub" textAnchor="middle" style={{ fill: "var(--warn)" }}>
                fails a check: repair, at most twice, then flag the nurse
              </text>
            </g>
          );
        })(),
        (() => {
          const a = mid("reteach");
          const b = mid("speak");
          const y = pos.reteach.y + bh;
          return (
            <g key="l1">
              <path className="edge back" d={`M${a.cx} ${y} C${a.cx} ${y + 44} ${b.cx} ${y + 44} ${b.cx} ${y + 6}`} />
              <Arrow x={b.cx} y={y} dir="u" back />
              <text x={(a.cx + b.cx) / 2} y={y + 50} className="sub" textAnchor="middle" style={{ fill: "var(--warn)" }}>
                missed a fact: one more try
              </text>
            </g>
          );
        })(),
      ];

  return (
    <svg className="diagram" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Homeward pipeline from discharge note to results for the nurse">
      {EDGES.map(([a, b], i) => edge(a, b, i))}
      {loops}
      {NODES.map((n, i) => (
        <g key={n.id}>
          <rect className={`box ${n.kind}`} x={pos[n.id].x} y={pos[n.id].y} width={bw} height={bh} rx={9} />
          <text x={pos[n.id].x + 12} y={pos[n.id].y + (vertical ? 24 : 26)} fontSize={13.5} fontWeight={700}>
            {i + 1}. {n.title}
          </text>
          {n.sub.map((line, j) => (
            <text key={j} x={pos[n.id].x + 12} y={pos[n.id].y + (vertical ? 44 : 48) + j * 15} className="sub">
              {line}
            </text>
          ))}
        </g>
      ))}
    </svg>
  );
}

function Arrow({ x, y, dir, back }: { x: number; y: number; dir: "r" | "l" | "d" | "u"; back?: boolean }) {
  const s = 6;
  const pts =
    dir === "r"
      ? `${x},${y} ${x - s},${y - s / 1.4} ${x - s},${y + s / 1.4}`
      : dir === "l"
        ? `${x},${y} ${x + s},${y - s / 1.4} ${x + s},${y + s / 1.4}`
        : dir === "d"
          ? `${x},${y} ${x - s / 1.4},${y - s} ${x + s / 1.4},${y - s}`
          : `${x},${y} ${x - s / 1.4},${y + s} ${x + s / 1.4},${y + s}`;
  return <polygon className={`arrow${back ? " back" : ""}`} points={pts} />;
}

type EvalModel = { id: string; label: string; runs_on: string; ci95?: Record<string, [number, number]>; [k: string]: unknown };
interface EvalFile {
  status: string;
  updated: string | null;
  grader: {
    test_set: { description: string; cases: number | null; answers: number | null; facts_graded: number | null; languages: number | null };
    metrics: { id: string; label: string; unit: string; hint?: string }[];
    models: EvalModel[];
  };
  training: { gpu: string; method: string; base_model: string; examples: number | null; epochs: number | null; minutes: number | null };
  fact_lock: Record<string, string | number | null>;
}

const num = (v: unknown) => typeof v === "number" && Number.isFinite(v);
function cell(m: EvalModel, id: string, unit: string) {
  const v = m[id];
  if (!num(v)) return <span className="faint">not measured</span>;
  const ci = m.ci95?.[id];
  const main = unit === "ms" ? `${(v as number).toLocaleString()} ms` : `${v}${unit}`;
  return (
    <>
      {main}
      {ci && <span className="faint tiny" style={{ display: "block" }}>95% CI {ci[0]}–{ci[1]}</span>}
    </>
  );
}

function Evaluation() {
  const [e, setE] = useState<EvalFile | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    fetch("/eval.json", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setE)
      .catch(() => setFailed(true));
  }, []);
  if (failed) return <p className="muted">The evaluation file could not be loaded.</p>;
  if (!e) return <p className="muted">Loading the evaluation.</p>;
  if (e.status !== "final") return <p className="note">The grader evaluation is still running; results will appear here when it finishes.</p>;
  const ts = e.grader.test_set;
  const fl = e.fact_lock;
  const tr = e.training;
  const hasLock = num(fl.versions) && num(fl.facts);
  return (
    <div className="stack-lg">
      <div className="card pad stack">
        <h3>Teach-back grader</h3>
        <p className="small muted">
          {ts.answers} teach-back answers ({ts.facts_graded} facts graded) for {ts.cases} held-out cases in {ts.languages} languages. The answers were written by Claude, a
          different model family from the one that generated the training data, and none of them were seen in training.
        </p>
        <div className="table-scroll">
          <table className="audit">
            <thead>
              <tr>
                <th>Model</th>
                <th>Runs on</th>
                {e.grader.metrics.map((m) => (
                  <th key={m.id} className="num" title={m.hint}>
                    {m.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {e.grader.models.map((m) => (
                <tr key={m.id}>
                  <td>{m.label}</td>
                  <td>{m.runs_on}</td>
                  {e.grader.metrics.map((x) => (
                    <td key={x.id} className="num">
                      {cell(m, x.id, x.unit)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small">
          What this means for safety: fine-tuning raised verdict accuracy, but not safety. On the 108 critical facts the patients got wrong or left out, the fine-tuned
          model missed 7 and the base model 5; the confidence intervals overlap, so neither is better on this test. Gemma 4 31B missed 1, which is why this hosted demo
          grades with Gemma 4 31B. The small model is the on-site option, for hospitals where data cannot leave the building; either way, the nurse reviews every result on
          the board.
        </p>
        <p className="tiny faint">
          {e.grader.metrics
            .filter((m) => m.hint)
            .map((m) => `${m.label}: ${m.hint}.`)
            .join(" ")}
        </p>
      </div>
      <div className="grid2">
        {num(tr.examples) && (
          <div className="card pad stack">
            <h3>Fine-tuning</h3>
            <ul className="clean small">
              <li>
                {tr.method} on {tr.base_model}, trained on one {tr.gpu}
              </li>
              <li>{tr.examples} training examples</li>
              {num(tr.epochs) && <li>{tr.epochs} epochs</li>}
              {num(tr.minutes) && <li>{tr.minutes} minutes of training</li>}
            </ul>
          </div>
        )}
        {hasLock && (
          <div className="card pad stack">
            <h3>Fact lock benchmark</h3>
            <p className="small muted">
              {String(fl.description ?? "")} {num(fl.control_cases) ? `${Number(fl.versions) - Number(fl.control_cases)} cases had one fault planted in the first draft; ${fl.control_cases} were clean controls.` : ""}
            </p>
            <ul className="clean small">
              {num(fl.injected_faults) && (
                <li>
                  <b>
                    Caught {fl.injected_faults_caught} of {fl.injected_faults} planted faults
                  </b>
                  {fl.caught_by ? ` (${String(fl.caught_by)})` : ""}
                </li>
              )}
              {num(fl.shipped_uncaught) && Number(fl.shipped_uncaught) > 0 && (
                <li>
                  {Number(fl.shipped_uncaught) === 1 ? "1 planted fault was" : `${fl.shipped_uncaught} planted faults were`} not caught and reached the final text
                  {fl.shipped_kinds ? ` (${String(fl.shipped_kinds)})` : ""}. The nurse's review of the patient version is still needed.
                </li>
              )}
              {num(fl.non_planted_facts) && (
                <li>
                  Issues raised on {fl.non_planted_issues} of {fl.non_planted_facts} facts with nothing planted (real writer errors or false alarms)
                  {num(fl.control_facts) ? `; ${fl.control_issues} of ${fl.control_facts} in the clean controls` : ""}
                </li>
              )}
              <li>
                {fl.versions} patient versions, {fl.facts} facts; at the end {fl.locked_first_pass} locked on the first pass, {fl.repaired} repaired, {fl.flagged} flagged to
                the nurse
              </li>
              {num(fl.cost_usd) && <li>Token Factory usage for the whole run: ${Number(fl.cost_usd).toFixed(2)}</li>}
            </ul>
          </div>
        )}
      </div>
      {e.updated && <p className="tiny faint">Updated {e.updated}.</p>}
    </div>
  );
}

const WHERE: [string, string, string, string][] = [
  ["Read a PDF or photo of the note", "MinerU 2.5 Pro", "AMD GPUs", "AMD Token Factory"],
  ["Find the facts and teach-back questions", "Gemma 4 31B", "AMD GPUs", "AMD Token Factory"],
  ["Write the patient version, repair failed facts", "Gemma 4 31B", "AMD GPUs", "AMD Token Factory"],
  ["Fact lock: find each fact's sentences, literal back-translation with suspicious words, compare meaning", "DeepSeek V4 Flash, an independent checker", "AMD GPUs", "AMD Token Factory"],
  ["Fact lock: anchors and stray English in every sentence; the confirmation rule", "Homeward code, no model", "Cloudflare Worker", "Cloudflare"],
  ["Grade the teach-back, write re-teach lines", "Gemma 4 31B (on-site build: our fine-tuned Gemma 4 E4B, not connected to this demo)", "AMD GPUs (on site: AMD Radeon PRO W7900D)", "AMD Token Factory (on site: self-hosted)"],
  ["Transcribe the spoken answer", "Browser speech recognition (on site: Gemma 4 E4B)", "Patient's device (on site: AMD Radeon PRO W7900D)", "Browser (on site: self-hosted)"],
  ["Second literal reading in the recorded Italian and Spanish samples only", "Qwen3.8-Flash-Next", "AMD GPUs", "AMD Token Factory"],
];

export function How() {
  const [vertical, setVertical] = useState(() => window.innerWidth < 760);
  useEffect(() => {
    const on = () => setVertical(window.innerWidth < 760);
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);

  return (
    <div className="wrap how" style={{ paddingTop: 40 }}>
      <section>
        <h1 style={{ maxWidth: "18ch" }}>How Homeward works</h1>
        <p className="note" style={{ marginTop: 14, maxWidth: "70ch" }}>
          Intended purpose: Homeward supports patient education. It does not make discharge decisions; the clinician does.
        </p>
        <p className="lead" style={{ marginTop: 14, fontSize: "1.1rem" }}>
          Language models write, check and grade; plain code makes the decisions that matter. A fact only reaches the patient if its sentence is really in the text,
          carries every medicine name, number and date, and means the same thing when translated back.
        </p>
      </section>

      <section aria-labelledby="pipe-h">
        <h2 id="pipe-h">The pipeline</h2>
        <div className="card pad">
          <Diagram vertical={vertical} />
          <div className="legend">
            <span>
              <i className="ai" />
              Model step
            </span>
            <span>
              <i className="code" />
              Decided by code
            </span>
            <span>
              <i className="person" />
              Person
            </span>
          </div>
        </div>
      </section>

      <section aria-labelledby="models-h">
        <h2 id="models-h">What runs where</h2>
        <p className="lead" style={{ marginBottom: 12 }}>
          Every language-model call on this site runs on AMD GPUs through AMD Token Factory (the Token Factory manual does not name the GPU model); speech recognition is your browser's. The results board lists each call of a check with the model that answered and
          how long it took.
        </p>
        <div className="card table-scroll">
          <table className="audit">
            <thead>
              <tr>
                <th>Step</th>
                <th>Model</th>
                <th>Hardware</th>
                <th>Provider</th>
              </tr>
            </thead>
            <tbody>
              {WHERE.map((r) => (
                <tr key={r[0]}>
                  <td>{r[0]}</td>
                  <td>{r[1]}</td>
                  <td>{r[2]}</td>
                  <td>{r[3]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="grid2" style={{ marginTop: 14 }}>
          <div className="card pad stack">
            <h3>Hosted demo (this site)</h3>
            <p className="small muted">
              Gemma 4 31B finds the facts, writes the patient version and grades the teach-back. DeepSeek V4 Flash only answers for Gemma if Gemma fails, and the audit trail
              marks it as a fallback. Spoken answers are transcribed by your browser, because Token Factory does not yet serve audio input. When the Italian and Spanish samples were recorded, Qwen3.8-Flash-Next did the second literal reading; it was then dropped because it misreads Devanagari vowel signs, and DeepSeek V4 Flash does that pass now.
            </p>
          </div>
          <div className="card pad stack">
            <h3>Hospital deployment</h3>
            <p className="small muted">
              Real discharge notes cannot go to a public API (Token Factory's own terms forbid health data). In the hospital, Gemma 4 E4B, fine-tuned with LoRA on one AMD
              Radeon PRO W7900D (48 GB), listens to the answer and grades it on site, where the data already lives.
            </p>
          </div>
        </div>
      </section>

      <section aria-labelledby="safety-h">
        <h2 id="safety-h">Safety design</h2>
        <div className="grid2">
          <div className="card pad stack">
            <h3>It only restates the clinician</h3>
            <p className="small muted">
              No diagnosis, no treatment advice. Every sentence comes from a fact the nurse confirmed. A re-teach line that introduces a number the fact does not contain is
              dropped by code, and the patient sees the locked sentence instead.
            </p>
          </div>
          <div className="card pad stack">
            <h3>Wrong translations are caught, not trusted</h3>
            <p className="small muted">
              The checker points at the sentences that carry each fact, and code confirms they contain every anchor. Digits in Devanagari, Bengali or Arabic script are
              normalised; month names are matched in the patient's language; 2 pm may be written 14:00. Every sentence is translated back word for word by an independent model that is told to keep errors, never guess what was meant, and name any wrong or nonsensical word; one such word fails the fact. That model also compares the meaning, which catches a reversed instruction that keeps every number, and critical warning signs get a second reading. Why it matters: physicians flagged safety concerns, mostly omissions, in 18 of 100 reviews of GPT-4 patient-friendly discharge summaries (
              <a href="https://pubmed.ncbi.nlm.nih.gov/38466307/" target="_blank" rel="noreferrer">
                Zaretsky et al., JAMA Network Open, 2024
              </a>
              ).
            </p>
          </div>
          <div className="card pad stack">
            <h3>Uncertainty goes to a person</h3>
            <p className="small muted">
              A fact that still fails after two repairs is flagged and cannot reach the patient until the nurse reads it. The grader is told to pick the less reassuring
              verdict when unsure. The patient confirms the transcript of what they said before it is graded.
            </p>
          </div>
          <div className="card pad stack">
            <h3>Critical facts cannot be waived</h3>
            <p className="small muted">
              "Understanding confirmed" is computed by code: every critical fact must be explained back correctly. The nurse sees the patient's own words, an English rendering and
              the audio, and makes the final call.
            </p>
          </div>
          <div className="card pad stack">
            <h3>Patients are told it is AI</h3>
            <p className="small muted">
              The patient's first screen says, in their language, that the tablet uses AI to read the instructions with them and check what they understood, and that the
              nurse reviews everything and makes every decision (EU AI Act, article 50; Italian Law 132/2025, article 7).
            </p>
          </div>
          <div className="card pad stack">
            <h3>Nothing is stored</h3>
            <p className="small muted">
              The server keeps no patient text. It stores only a salted hash of your IP address with a timestamp, for rate limiting, and the day's spend. Homeward does not upload
              recorded audio; in this hosted demo, your browser's own speech recognition may send it to the browser's maker.
            </p>
          </div>
          <div className="card pad stack">
            <h3>Synthetic data only</h3>
            <p className="small muted">
              This hosted demo is for synthetic or fully de-identified notes. The sample patients are fictional, and their runs were recorded once through the same pipeline.
            </p>
          </div>
        </div>
      </section>

      <section aria-labelledby="eval-h">
        <h2 id="eval-h">Evaluation</h2>
        <Evaluation />
        <p className="muted" style={{ marginTop: 14 }}>
          Charts with confidence intervals, and published benchmarks for the same class of models, are on the <Link to="/evidence">Evidence</Link> page.
        </p>
      </section>

      <section>
        <p className="muted">
          <Link to="/check">Try a discharge check</Link> or open a sample patient from the <Link to="/">home page</Link>.
        </p>
      </section>
    </div>
  );
}
