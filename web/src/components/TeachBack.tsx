import { useEffect, useRef, useState } from "react";
import { langByCode, type Attempt, type FactGrade } from "../../shared/types.ts";
import { useConfig, useHideChrome } from "../App.tsx";
import strings from "../i18n.json";
import { ApiError, gradeAnswer, transcribe } from "../lib/api.ts";
import { clips, loadSample, type Sample, type Session, type Stage } from "../lib/session.ts";
import { blobToBase64, listen, recognitionSupported, speak, stopSpeaking, voiceFor, voicesReady, type Listener } from "../lib/speech.ts";
import type { Update } from "../pages/Flow.tsx";
import { Bidi } from "./Bidi.tsx";
import { Icon } from "./Icon.tsx";
import { ErrorNote } from "./NoteStage.tsx";

type Phase = "ask" | "listening" | "playing" | "transcribing" | "confirm" | "typing" | "grading" | "result" | "done";
type Strings = (typeof strings)["en"];

export function TeachBack({ s, update, goStage }: { s: Session; update: Update; goStage: (st: Stage) => void }) {
  const hide = useHideChrome();
  const config = useConfig();
  const v = s.version!;
  const L = langByCode(v.language);
  const t: Strings = { ...strings.en, ...((strings as Record<string, Partial<Strings>>)[L.code] ?? {}) };
  const dir = L.rtl ? "rtl" : undefined;

  const [idx, setIdx] = useState(s.teach?.idx ?? -1); // -1: reading the instructions
  const [attempt, setAttempt] = useState(s.teach?.attempt ?? 0);
  const [phase, setPhase] = useState<Phase>("ask");
  const [text, setText] = useState("");
  const [input, setInput] = useState<"voice" | "typed" | "sample">("voice");
  // On a reload in the middle of a retry, the first try's grades come back from the session.
  const [last, setLast] = useState<Attempt | null>(() => {
    const t0 = s.teach;
    if (!t0 || t0.attempt !== 1 || t0.idx < 0) return null;
    return s.attempts.find((a) => a.topic_id === s.topics[t0.idx]?.id) ?? null;
  });
  const [reread, setReread] = useState(false);
  useEffect(() => {
    update({ teach: { idx, attempt } });
  }, [idx, attempt]); // eslint-disable-line react-hooks/exhaustive-deps
  const [error, setError] = useState<ApiError | string | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [hasVoice, setHasVoice] = useState(true);
  const [sample, setSample] = useState<Sample | null>(null);
  const listener = useRef<Listener | null>(null);
  const [recStart, setRecStart] = useState<number | null>(null);
  const [gradeStart, setGradeStart] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (recStart === null && gradeStart === null) return;
    const t = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(t);
  }, [recStart, gradeStart]);
  const gradeSecs = gradeStart === null ? 0 : Math.max(0, Math.floor((now - gradeStart) / 1000));
  const recSecs = recStart === null ? 0 : Math.max(0, Math.floor((now - recStart) / 1000));
  const recClock = `${Math.floor(recSecs / 60)}:${String(recSecs % 60).padStart(2, "0")}`;

  const serverAsr = config?.asr === "onprem" || config?.asr === "mimo";
  const canSpeak = serverAsr || recognitionSupported();

  useEffect(() => {
    hide(true);
    voicesReady().then(() => setHasVoice(Boolean(voiceFor(L.bcp47))));
    if (s.source === "sample" && s.versionReplayed && s.sample) loadSample(s.sample.id).then(setSample).catch(() => {});
    return () => {
      hide(false);
      stopSpeaking();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const topic = idx >= 0 ? s.topics[idx] : null;
  const askIds = topic ? (attempt === 0 ? topic.fact_ids : (last?.grades ?? []).filter((g) => g.verdict !== "correct").map((g) => g.id)) : [];
  const question = topic ? v.questions[topic.id] || topic.question : "";

  function say(txt: string) {
    if (speaking) {
      stopSpeaking();
      setSpeaking(false);
      return;
    }
    const ok = speak(txt, L.bcp47, () => setSpeaking(false));
    setSpeaking(ok);
    if (!ok) setHasVoice(false);
  }

  async function startListening() {
    setError(null);
    stopSpeaking();
    setSpeaking(false);
    setText("");
    try {
      listener.current = await listen(L.bcp47, setText, !serverAsr, (m) => setError(m));
      setRecStart(Date.now());
      setNow(Date.now());
      setPhase("listening");
      setInput("voice");
    } catch {
      setPhase("typing");
    }
  }

  /** A recorded sample answer appears word by word, as dictation would, then goes to the same confirmation step. */
  const playTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  function playSample(transcript: string) {
    stopSpeaking();
    setSpeaking(false);
    setError(null);
    setInput("sample");
    setPhase("playing");
    const words = transcript.split(/(\s+)/);
    let i = 0;
    setText("");
    if (playTimer.current) clearInterval(playTimer.current);
    playTimer.current = setInterval(() => {
      i += 2;
      setText(words.slice(0, i).join(""));
      if (i >= words.length) {
        clearInterval(playTimer.current!);
        playTimer.current = null;
        setText(transcript);
        setPhase("confirm");
      }
    }, 90);
  }
  useEffect(() => () => {
    if (playTimer.current) clearInterval(playTimer.current);
  }, []);

  async function stopListening() {
    const l = listener.current;
    listener.current = null;
    setRecStart(null);
    if (!l) return;
    const { text: heard, wav } = await l.stop();
    if (wav && topic) {
      const key = `${topic.id}:${attempt}`;
      const old = clips.get(key);
      if (old) URL.revokeObjectURL(old);
      clips.set(key, URL.createObjectURL(wav));
    }
    let finalText = heard;
    if (serverAsr && wav) {
      setPhase("transcribing");
      try {
        const out = await transcribe(await blobToBase64(wav), L.code);
        finalText = out.text;
        update((cur) => ({ steps: [...cur.steps, ...out.steps] }));
      } catch (e) {
        setError(e instanceof ApiError ? e : "Transcription failed.");
        setPhase("typing");
        return;
      }
    }
    setText(finalText || text);
    setPhase(finalText || text ? "confirm" : "ask");
  }

  async function check() {
    if (!topic || !text.trim()) return;
    setPhase("grading");
    setGradeStart(Date.now());
    setNow(Date.now());
    setError(null);
    const facts = s.facts.filter((f) => askIds.includes(f.id));
    const recorded = sample?.answers?.[topic.id]?.[attempt];
    // Running a question again replaces its audit rows instead of adding duplicates.
    const scope = `${topic.id}:${attempt}`;
    const dropScope = (steps: Session["steps"]) =>
      steps.filter((x) => !(x.scope === scope || (attempt === 0 && x.scope?.startsWith(`${topic.id}:`))));
    try {
      let grades: FactGrade[];
      let transcript_en: string;
      let replayed = false;
      if (recorded && recorded.transcript.trim() === text.trim() && recorded.fact_ids.join() === askIds.join()) {
        grades = recorded.grades;
        transcript_en = recorded.transcript_en;
        replayed = true;
        update((cur) => ({ steps: [...dropScope(cur.steps), ...recorded.steps.map((x) => ({ ...x, replayed: true, scope }))] }));
      } else {
        const out = await gradeAnswer(facts, topic.question, text.trim(), L.code);
        grades = out.grades;
        transcript_en = out.transcript_en;
        update((cur) => ({ steps: [...dropScope(cur.steps), ...out.steps.map((x) => ({ ...x, scope }))] }));
      }
      const a: Attempt = { topic_id: topic.id, transcript: text.trim(), transcript_en, input, grades, replayed };
      // A first try replaces any earlier run of this question; a retry replaces only an earlier retry.
      update((cur) => {
        const same = cur.attempts.filter((x) => x.topic_id === topic.id);
        const keep = cur.attempts.filter((x) => x.topic_id !== topic.id || (attempt === 1 && same.indexOf(x) === 0));
        return { attempts: [...keep, a] };
      });
      setLast(a);
      setGradeStart(null);
      setPhase("result");
    } catch (e) {
      setError(e instanceof ApiError ? e : "Something went wrong.");
      setGradeStart(null);
      setPhase(input === "typed" ? "typing" : "confirm");
    }
  }

  function next() {
    stopSpeaking();
    setSpeaking(false);
    setText("");
    setLast(null);
    setAttempt(0);
    setError(null);
    if (idx + 1 >= s.topics.length) setPhase("done");
    else {
      setIdx(idx + 1);
      setPhase("ask");
    }
    window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
  }

  function retry() {
    stopSpeaking();
    setSpeaking(false);
    setText("");
    setAttempt(1);
    setPhase("ask");
    window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
  }

  const listenBtn = (txt: string) =>
    hasVoice ? (
      <button className="btn" onClick={() => say(txt)}>
        <Icon name={speaking ? "stop" : "speaker"} />
        {speaking ? t.stop : t.listen}
      </button>
    ) : null;

  const missed = last ? last.grades.filter((g) => g.verdict !== "correct") : [];
  const reteachText = missed
    .map((g) => {
      const lock = v.lock.find((l) => l.id === g.id);
      return [g.reteach_patient, lock?.quote].filter(Boolean).join(" ");
    })
    .join(" ");
  const recorded = topic ? sample?.answers?.[topic.id]?.[attempt] : undefined;
  const total = s.topics.length;

  return (
    <div className="patient" lang={L.code}>
      <div className="patient-top">
        <span className="muted small" lang="en">
          {L.native}
        </span>
        <button
          className="btn quiet small"
          lang="en"
          onClick={() => {
            listener.current?.stop();
            goStage(s.attempts.length ? "board" : "version");
          }}
        >
          Return to nurse
        </button>
      </div>
      <div className="patient-body stack-lg" dir={dir}>
        {reread && (
          <div className="stack-lg fade-in">
            <h1>
              <Bidi text={v.title || t.yourInstructions} rtl={L.rtl} />
            </h1>
            <div className="btn-row">{listenBtn([v.title, ...v.paragraphs].join(". "))}</div>
            <div className="leaflet">
              {v.paragraphs.map((p, i) => (
                <p key={i}>
                  <Bidi text={p} rtl={L.rtl} />
                </p>
              ))}
            </div>
            <button
              className="btn primary"
              onClick={() => {
                stopSpeaking();
                setSpeaking(false);
                setReread(false);
                window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
              }}
            >
              <Icon name={dir ? "right" : "left"} />
              {t.questionOf.replace("{n}", String(idx + 1)).replace("{total}", String(s.topics.length))}
            </button>
          </div>
        )}

        {!reread && idx >= 0 && phase !== "done" && (
          <div className="progress-bar" aria-hidden>
            <span style={{ width: `${(idx / total) * 100}%` }} />
          </div>
        )}

        {idx === -1 && (
          <>
            <h1>
              <Bidi text={v.title || t.yourInstructions} rtl={L.rtl} />
            </h1>
            <div className="btn-row">
              {hasVoice && (
                <button className="btn" onClick={() => say([v.title, ...v.paragraphs].join(". "))}>
                  <Icon name={speaking ? "stop" : "speaker"} />
                  {speaking ? t.stop : t.listen}
                </button>
              )}
            </div>
            {!hasVoice && <p className="small muted">{t.noVoice}</p>}
            <div className="leaflet">
              {v.paragraphs.map((p, i) => (
                <p key={i}>
                  <Bidi text={p} rtl={L.rtl} />
                </p>
              ))}
            </div>
            <button
              className="btn primary"
              onClick={() => {
                stopSpeaking();
                setSpeaking(false);
                setIdx(0);
                setPhase("ask");
                window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
              }}
            >
              {t.startQuestions}
              <Icon name={dir ? "left" : "right"} />
            </button>
          </>
        )}

        {!reread && topic && phase !== "done" && (
          <>
            <p className="muted">{t.questionOf.replace("{n}", String(idx + 1)).replace("{total}", String(total))}</p>
            {phase !== "result" && (
              <>
                <h1 className="fade-in" key={`${topic.id}-${attempt}`}>
                  <Bidi text={question} rtl={L.rtl} />
                </h1>
                <div className="btn-row">
                  {listenBtn(question)}
                  <button className="btn quiet" onClick={() => setReread(true)}>
                    {t.readAgain}
                  </button>
                </div>
              </>
            )}

            {(phase === "ask" || phase === "listening") && canSpeak && (
              <div className="stack">
                <button
                  className={`btn mic${phase === "listening" ? " on" : " primary"}`}
                  onClick={phase === "listening" ? stopListening : startListening}
                  aria-pressed={phase === "listening"}
                >
                  <Icon name={phase === "listening" ? "stop" : "mic"} size={40} />
                  <span>{phase === "listening" ? t.listening : t.answerVoice}</span>
                  {phase === "listening" && (
                    <span className="rec" aria-hidden>
                      <span className="rec-dot" />
                      {recClock}
                    </span>
                  )}
                </button>
                {phase === "listening" && text && (
                  <p className="heard" aria-live="polite">
                    {text}
                  </p>
                )}
                {phase === "ask" && (
                  <div className="btn-row">
                    <button className="btn quiet" onClick={() => setPhase("typing")}>
                      <Icon name="keyboard" />
                      {t.typeInstead}
                    </button>
                    {recorded && (
                      <button className="btn quiet" onClick={() => playSample(recorded.transcript)}>
                        <Icon name="play" />
                        {t.sampleAnswer}
                      </button>
                    )}
                  </div>
                )}

              </div>
            )}

            {(phase === "ask" || phase === "typing") && recorded && (
              <p className="demo-hint" lang="en" dir="ltr">
                Demo: tap “{t.sampleAnswer}” for the recorded answer, or answer yourself.
              </p>
            )}
            {(phase === "ask" || phase === "typing") && config?.live === false && (
              <p className="demo-hint" lang="en" dir="ltr">
                Live grading is closed for today; recorded sample answers still work.
              </p>
            )}

            {phase === "ask" && !canSpeak && (
              <p className="note patient-note">
                <Icon name="info" />
                <span>{t.micUnavailable}</span>
              </p>
            )}

            {phase === "playing" && (
              <div className="stack fade-in">
                <div className="btn mic on" aria-hidden style={{ cursor: "default" }}>
                  <Icon name="mic" size={40} />
                  <span>{t.listening.split(/[.।。]/)[0]}</span>
                </div>
                <p className="heard" aria-live="polite">
                  {text}
                </p>
              </div>
            )}

            {phase === "transcribing" && (
              <p className="big muted">
                <Icon name="loader" className="spin" /> {t.checking}
              </p>
            )}

            {phase === "confirm" && (
              <div className="stack fade-in">
                <h2>{t.confirmHeard}</h2>
                <p className="heard">
                  <Bidi text={text} rtl={L.rtl} />
                </p>
                <div className="btn-row">
                  {listenBtn(text)}
                </div>
                <div className="btn-row">
                  <button className="btn primary" onClick={check}>
                    <Icon name="check" />
                    {t.yes}
                  </button>
                  {canSpeak && (
                    <button className="btn" onClick={startListening}>
                      <Icon name="redo" />
                      {t.sayAgain}
                    </button>
                  )}
                  <button
                    className="btn quiet"
                    onClick={() => {
                      setInput("typed");
                      setPhase("typing");
                    }}
                  >
                    <Icon name="keyboard" />
                    {t.typeInstead}
                  </button>
                </div>
              </div>
            )}

            {(phase === "typing" || (phase === "ask" && !canSpeak)) && (
              <div className="stack fade-in">
                <label className="field">
                  <span>{t.orType}</span>
                  <textarea
                    className="input answer"
                    value={text}
                    onChange={(e) => {
                      setText(e.target.value);
                      setInput("typed");
                    }}
                    rows={4}
                    autoFocus
                  />
                </label>
                <div className="btn-row">
                  <button className="btn primary" onClick={check} disabled={!text.trim()}>
                    {t.check}
                  </button>
                  {canSpeak && (
                    <button className="btn quiet" onClick={startListening}>
                      <Icon name="mic" />
                      {t.answerVoice}
                    </button>
                  )}
                  {recorded && !text && (
                    <button className="btn quiet" onClick={() => playSample(recorded.transcript)}>
                      <Icon name="play" />
                      {t.sampleAnswer}
                    </button>
                  )}
                </div>
              </div>
            )}

            {phase === "grading" && (
              <p className="big muted" aria-live="polite">
                <Icon name="loader" className="spin" /> {t.checking}{" "}
                <span className="rec" style={{ fontSize: 18 }} aria-hidden>
                  {Math.floor(gradeSecs / 60)}:{String(gradeSecs % 60).padStart(2, "0")}
                </span>
              </p>
            )}

            {phase === "result" && last && (
              <div className="stack-lg fade-in">
                {missed.length === 0 ? (
                  <>
                    <h1>{t.goodAll}</h1>
                    <div className="btn-row">{listenBtn(t.goodAll)}</div>
                  </>
                ) : (
                  <>
                    <h1>{t.againIntro}</h1>
                    <div className="stack">
                      {missed.map((g) => {
                        const lock = v.lock.find((l) => l.id === g.id);
                        return (
                          <div className="reteach big" key={g.id}>
                            {g.reteach_patient && (
                              <p>
                                <Bidi text={g.reteach_patient} rtl={L.rtl} />
                              </p>
                            )}
                            {lock?.quote && (
                              <p style={{ marginTop: g.reteach_patient ? 8 : 0 }}>
                                <Bidi text={lock.quote} rtl={L.rtl} />
                              </p>
                            )}
                          </div>
                        );
                      })}
                    </div>
                    <div className="btn-row">{listenBtn(`${t.againIntro} ${reteachText}`)}</div>
                  </>
                )}
                <div className="btn-row">
                  {missed.length > 0 && attempt === 0 ? (
                    <>
                      <button className="btn primary" onClick={retry}>
                        <Icon name="mic" />
                        {t.tryAgain}
                      </button>
                      {sample && topic && !sample.answers?.[topic.id]?.[1] && (
                        <button className="btn quiet" lang="en" dir="ltr" onClick={next}>
                          {idx + 1 >= total ? "Demo: skip to the nurse" : "Demo: skip to the next question"}
                        </button>
                      )}
                    </>
                  ) : (
                    <button className="btn primary" onClick={next}>
                      {idx + 1 >= total ? t.finish : t.next}
                      <Icon name={dir ? "left" : "right"} />
                    </button>
                  )}
                </div>
              </div>
            )}
          </>
        )}

        {phase === "done" && (
          <div className="stack-lg fade-in">
            <h1>{t.finished}</h1>
            <div className="btn-row">{listenBtn(t.finished)}</div>
            <div className="btn-row">
              <button
                className="btn primary"
                lang="en"
                dir="ltr"
                onClick={() => {
                  stopSpeaking();
                  update((cur) => ({ times: { ...cur.times, decided: Date.now() } }));
                  goStage("board");
                }}
              >
                Return to nurse
                <Icon name="right" />
              </button>
            </div>
          </div>
        )}

        <ErrorNote error={error} />
      </div>
    </div>
  );
}
