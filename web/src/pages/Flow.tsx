import { useCallback, useEffect, useRef, useState } from "react";
import { Board } from "../components/Board.tsx";
import { FactsStage } from "../components/FactsStage.tsx";
import { NoteStage } from "../components/NoteStage.tsx";
import { TeachBack } from "../components/TeachBack.tsx";
import { VersionStage } from "../components/VersionStage.tsx";
import { loadSession, newSession, saveSession, STAGES, type Session, type Stage } from "../lib/session.ts";

export type Update = (patch: Partial<Session> | ((s: Session) => Partial<Session>)) => void;

function reachable(s: Session, stage: Stage): boolean {
  switch (stage) {
    case "note":
      return true;
    case "facts":
      return s.facts.length > 0;
    case "version":
      return s.factsConfirmed;
    case "teach":
      return Boolean(s.version) && s.version!.lock.every((l) => l.status !== "flagged" || s.acknowledged.includes(l.id));
    case "board":
      return s.attempts.length > 0;
  }
}

export function Flow() {
  const [s, setS] = useState<Session>(() => loadSession() ?? newSession());

  useEffect(() => saveSession(s), [s]);

  // On narrow screens the step strip scrolls sideways; keep the current step in view (horizontally only).
  const strip = useRef<HTMLElement>(null);
  useEffect(() => {
    const nav = strip.current;
    const cur = nav?.querySelector<HTMLElement>('[aria-current="step"]');
    if (nav && cur) nav.scrollLeft = cur.offsetLeft - nav.clientWidth / 2 + cur.clientWidth / 2;
  }, [s.stage]);

  const update: Update = useCallback((patch) => {
    setS((cur) => ({ ...cur, ...(typeof patch === "function" ? patch(cur) : patch) }));
  }, []);

  const goStage = (stage: Stage) => {
    update({ stage });
    window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
  };

  if (s.stage === "teach" && s.version) return <TeachBack s={s} update={update} goStage={goStage} />;

  const idx = STAGES.findIndex((x) => x.id === s.stage);
  return (
    <div className="wrap">
      <nav className="steps" aria-label="Progress" ref={strip}>
        {STAGES.map((st, i) => (
          <button
            key={st.id}
            aria-current={st.id === s.stage ? "step" : undefined}
            className={i < idx ? "done" : undefined}
            disabled={!reachable(s, st.id)}
            onClick={() => goStage(st.id)}
          >
            <span className="num">{i + 1}</span>
            {st.label}
          </button>
        ))}
      </nav>
      {s.stage === "note" && <NoteStage s={s} update={update} goStage={goStage} reset={() => setS(newSession())} />}
      {s.stage === "facts" && <FactsStage s={s} update={update} goStage={goStage} />}
      {s.stage === "version" && <VersionStage s={s} update={update} goStage={goStage} />}
      {s.stage === "board" && <Board s={s} update={update} goStage={goStage} reset={() => setS(newSession())} />}
    </div>
  );
}
