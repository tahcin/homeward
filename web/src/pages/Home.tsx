import { useState } from "react";
import { Link, useRouter } from "../App.tsx";
import { Icon } from "../components/Icon.tsx";
import { HeroBackdrop } from "../components/landing/HeroBackdrop.tsx";
import { HeroVignette } from "../components/landing/Vignette.tsx";
import { BoardMock, FactRowMock, QuestionMock, useSample } from "../components/Mocks.tsx";
import { loadSample, newSession, saveSession, sessionFromSample } from "../lib/session.ts";
import "../landing.css";

export const SAMPLE_CARDS = [
  {
    id: "heart-failure-it",
    language: "Italian",
    native: "Italiano",
    patient: "Giuseppe, 78",
    condition: "Heart failure. His water pill dose doubled and a painkiller must stop.",
    shows: "The lock catches a reversed instruction",
  },
  {
    id: "anticoagulant-hi",
    language: "Hindi",
    native: "हिन्दी",
    patient: "Ramesh, 66, with his son",
    condition: "New blood thinner after atrial fibrillation. His aspirin stops.",
    shows: "A doubled dose and a medicine mix-up, both caught",
  },
  {
    id: "knee-replacement-es",
    language: "Spanish",
    native: "Español",
    patient: "Carmen, 69",
    condition: "Knee replacement. Daily injections at home for two weeks.",
    shows: "A clean run, one fact re-taught",
  },
];

// Every figure below is quoted from research.md (checked 2026-10-15), with the population and measure it applies to.
type Src = { label: string; href: string };
const S = {
  trivedi: { label: "Trivedi et al., JAMA Intern Med 2023", href: "https://pubmed.ncbi.nlm.nih.gov/36939674/" },
  hoek: { label: "Hoek et al., Ann Emerg Med 2020", href: "https://pubmed.ncbi.nlm.nih.gov/31439363/" },
  ziaeian: { label: "Ziaeian et al., J Gen Intern Med 2012", href: "https://pubmed.ncbi.nlm.nih.gov/22798200/" },
  forster03: { label: "Forster et al., Ann Intern Med 2003", href: "https://pubmed.ncbi.nlm.nih.gov/12558354/" },
  forster04: { label: "Forster et al., CMAJ 2004", href: "https://pubmed.ncbi.nlm.nih.gov/14757670/" },
  hcup: { label: "AHRQ HCUP Statistical Brief 304", href: "https://hcup-us.ahrq.gov/reports/statbriefs/sb304-readmissions-2016-2020.jsp" },
  auerbach: { label: "Auerbach et al., JAMA Intern Med 2016", href: "https://pubmed.ncbi.nlm.nih.gov/26954564/" },
  cms: { label: "CMS data via Healthcare Dive, Oct 2026", href: "https://www.healthcaredive.com/news/hospital-readmission-reduction-program-penalties-rise-2027-cms/832473/" },
  oh: { label: "Oh et al., Patient Educ Couns 2023", href: "https://pubmed.ncbi.nlm.nih.gov/36411152/" },
  engel: { label: "Engel et al., Ann Emerg Med 2009", href: "https://pubmed.ncbi.nlm.nih.gov/18619710/" },
  khoong: { label: "Khoong et al., JAMA Intern Med 2019", href: "https://pmc.ncbi.nlm.nih.gov/articles/PMC6450297/" },
  emteba: { label: "Mahajan et al., Int J Emerg Med 2020", href: "https://pmc.ncbi.nlm.nih.gov/articles/PMC7513274/" },
} satisfies Record<string, Src>;

const Sources = ({ list }: { list: Src[] }) => (
  <p className="lp-src">
    {list.map((s, i) => (
      <span key={s.href}>
        {i > 0 && <span aria-hidden> · </span>}
        <a href={s.href} target="_blank" rel="noreferrer">
          {s.label}
        </a>
      </span>
    ))}
  </p>
);

const CHAIN: { fig: string; title: string; text: string; src: Src[] }[] = [
  {
    fig: "47%",
    title: "They forget and misunderstand",
    text: "Across 51 studies, emergency patients correctly recalled only about half (47%) of the discharge instructions they were told. Among older inpatients, 81% either received a medication error at discharge or didn't understand at least one change to their medicines.",
    src: [S.hoek, S.ziaeian],
  },
  {
    fig: "1 in 5",
    title: "So they get hurt at home",
    text: "About 1 in 5 medical patients had an adverse event in the weeks after discharge, and two thirds of those involved medicines. In a Canadian replication, half were preventable or could have been made less severe.",
    src: [S.forster03, S.forster04],
  },
  {
    fig: "3.85 million",
    title: "So they come back",
    text: "US hospitals saw 3.85 million 30-day readmissions in 2020, costing about $16,300 each. About a quarter were judged potentially preventable, and Medicare is cutting payments to 80% of evaluated hospitals for excess readmissions in fiscal 2027.",
    src: [S.hcup, S.auerbach, S.cms],
  },
  {
    fig: "1 of 33",
    title: "The fix exists, but rarely happens",
    text: "For heart failure, teach-back at discharge cut the odds of readmission by more than half in a meta-analysis. Yet when observers followed 33 patients from 7 am to discharge, only 1 was asked to explain the plan back.",
    src: [S.oh, S.trivedi],
  },
];

const BARRIERS: { title: string; problem: string; src: Src; answer: string }[] = [
  {
    title: "Translation can change the meaning",
    problem: "When emergency discharge instructions were machine-translated, 8% of the Chinese sentences carried potential for clinically significant harm.",
    src: S.khoong,
    answer: "Every medicine, dose and date in the patient's version is checked in code, and an independent model checks the meaning. Anything that fails is repaired or flagged to the nurse.",
  },
  {
    title: "Teach-back needs time and a shared language",
    problem: "In a Dutch emergency department, teach-back took 1:39 minutes and cut the share of patients leaving with a gap from 49% to 11.9%. It still needs someone free to do it, in the patient's language.",
    src: S.emteba,
    answer: "Homeward does the asking, listening and grading, so the nurse only reviews the result.",
  },
  {
    title: "“Any questions?” finds nothing",
    problem: "78% of emergency patients had a gap in understanding at least one part of their discharge, and when they had one, they noticed it only 1 time in 5.",
    src: S.engel,
    answer: "Questions are open and never give the dose or time away, so a gap shows up even when the patient feels sure.",
  },
];

export function Home() {
  const { go } = useRouter();
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState("");
  const italian = useSample("heart-failure-it");

  async function openSample(id: string) {
    setLoading(id);
    setError("");
    try {
      const s = await loadSample(id);
      saveSession(sessionFromSample(s));
      go("/check");
    } catch {
      setError("That sample could not be loaded. Please try again.");
    } finally {
      setLoading(null);
    }
  }

  function ownNote() {
    saveSession(newSession());
    go("/check");
  }

  return (
    <div className="lp">
      <section className="lp-hero-band" aria-labelledby="hero-h">
        <HeroBackdrop />
        <div className="wrap lp-hero">
          <h1 id="hero-h">Patients go home misunderstanding their instructions, and too many come back because of it.</h1>
          <p className="lp-lede">
            Homeward lets every patient explain their instructions back in their own language before they leave, and shows the nurse exactly what to go over again.
          </p>
          <div className="lp-cta">
            <a className="btn primary lp-btn" href="#samples">
              Try a sample patient
              <Icon name="right" />
            </a>
            <button className="btn lp-btn" onClick={ownNote}>
              Use your own discharge note
            </button>
          </div>
        </div>
      </section>

      <section className="lp-why" aria-labelledby="why-h">
        <div className="wrap">
          <div className="lp-head">
            <h2 id="why-h">Why it matters</h2>
          </div>
          <ol className="lp-chain">
            {CHAIN.map((c) => (
              <li key={c.title}>
                <p className="lp-fig" aria-hidden>
                  {c.fig}
                </p>
                <div className="lp-chain-body">
                  <h3>{c.title}</h3>
                  <p>{c.text}</p>
                  <Sources list={c.src} />
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="wrap lp-sec" aria-labelledby="way-h">
        <div className="lp-head">
          <h2 id="way-h">What gets in the way, and what Homeward does</h2>
        </div>
        <div className="lp-barriers">
          {BARRIERS.map((b) => (
            <article className="lp-barrier" key={b.title}>
              <div className="lp-barrier-problem">
                <h3>{b.title}</h3>
                <p>{b.problem}</p>
                <Sources list={[b.src]} />
              </div>
              <p className="lp-barrier-answer">
                <Icon name="check" size={18} />
                <span>{b.answer}</span>
              </p>
            </article>
          ))}
        </div>
      </section>

      <section className="wrap lp-sec" aria-labelledby="bed-h">
        <div className="lp-head">
          <h2 id="bed-h">At the bedside</h2>
          <p>Shown with Giuseppe, who speaks Italian.</p>
        </div>
        <ol className="lp-steps">
          <li>
            <span className="lp-step-n" aria-hidden>
              1
            </span>
            <h3>The nurse confirms the facts</h3>
            <p>Medicines, warning signs and appointments are pulled from the discharge note. Nothing goes further until the nurse confirms them.</p>
            {italian && <FactRowMock s={italian} />}
          </li>
          <li>
            <span className="lp-step-n" aria-hidden>
              2
            </span>
            <h3>The patient explains it back</h3>
            <p>They read or listen to their instructions, answer each question by voice, and check the transcript before it is graded.</p>
            {italian && <QuestionMock s={italian} />}
          </li>
          <li>
            <span className="lp-step-n" aria-hidden>
              3
            </span>
            <h3>The nurse sees what to go over</h3>
            <p>Anything missed is explained again in simpler words and asked once more. The board shows “Understanding confirmed” only when every critical fact is understood.</p>
            {italian && <BoardMock s={italian} />}
          </li>
        </ol>
      </section>

      <section className="wrap lp-sec" aria-labelledby="moment-h">
        <div className="lp-moment">
          <div className="lp-head lp-moment-text">
            <h2 id="moment-h">One answer, checked fact by fact</h2>
            <p>Ramesh, 66, is going home on a new blood thinner. The patient is fictional and the answer was scripted by us; the grades come from the real model.</p>
            <button className="lp-more lp-linkbtn" onClick={() => openSample("anticoagulant-hi")} disabled={loading !== null}>
              {loading === "anticoagulant-hi" ? "Opening" : "Open Ramesh's Hindi sample"}
              <Icon name="right" size={16} />
            </button>
          </div>
          <HeroVignette />
        </div>
      </section>

      <section id="samples" className="wrap lp-sec" aria-labelledby="samples-h">
        <div className="lp-head">
          <h2 id="samples-h">Try a sample patient</h2>
          <p>Each opens a recorded run, so it works instantly. Answer the questions yourself and you are graded live.</p>
        </div>
        <div className="lp-samples">
          {SAMPLE_CARDS.map((c) => (
            <button key={c.id} className="lp-sample" onClick={() => openSample(c.id)} disabled={loading !== null} aria-describedby={`${c.id}-d`}>
              <span className="lp-sample-lang">
                {c.language}
                <span lang={c.id.slice(-2)}>{c.native}</span>
              </span>
              <span className="lp-sample-who">{c.patient}</span>
              <span className="lp-sample-what" id={`${c.id}-d`}>
                {c.condition}
              </span>
              <span className="lp-sample-shows">
                <span>{loading === c.id ? "Opening" : c.shows}</span>
                <Icon name="right" size={16} />
              </span>
            </button>
          ))}
        </div>
        {error && (
          <p className="note bad" role="alert" style={{ marginTop: 12 }}>
            {error}
          </p>
        )}
      </section>

      <section className="wrap lp-sec" aria-labelledby="hosp-h">
        <div className="lp-hosp">
          <div>
            <h2 id="hosp-h">Built for hospitals</h2>
            <p>
              In a hospital, a fine-tuned Gemma 4 E4B listens and grades on one AMD Radeon PRO W7900D, on site, so patient data stays inside the hospital.
            </p>
            <Link to="/how" className="lp-more">
              How it works
              <Icon name="right" size={16} />
            </Link>
          </div>
          <p className="lp-measured">
            On the author's own recorded Hindi answer, transcription and grading took <b>14.7 seconds</b> on one card, using <b>15.7 GB</b> of its 48 GB.
          </p>
        </div>
      </section>
    </div>
  );
}
