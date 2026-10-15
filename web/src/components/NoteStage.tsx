import { useRef, useState } from "react";
import { useConfig, useRouter } from "../App.tsx";
import { ApiError, extractFacts, ocrFile } from "../lib/api.ts";
import { loadSample, sessionFromSample, type Session, type Stage } from "../lib/session.ts";
import type { Update } from "../pages/Flow.tsx";
import { Icon } from "./Icon.tsx";

export function ErrorNote({ error }: { error: ApiError | string | null }) {
  const { go } = useRouter();
  if (!error) return null;
  const msg = typeof error === "string" ? error : error.message;
  const budget = typeof error !== "string" && (error.code === "budget" || error.code === "ip" || error.code === "off");
  return (
    <div className="note bad fade-in" role="alert">
      <Icon name="alert" />
      <span>
        {msg}{" "}
        {budget && (
          <a
            href="/#samples"
            onClick={(e) => {
              e.preventDefault();
              go("/");
              setTimeout(() => document.getElementById("samples")?.scrollIntoView(), 50);
            }}
          >
            Open a sample patient
          </a>
        )}
      </span>
    </div>
  );
}

export function NoteStage({ s, update, goStage, reset }: { s: Session; update: Update; goStage: (st: Stage) => void; reset: () => void }) {
  const config = useConfig();
  const [busy, setBusy] = useState<"" | "ocr" | "extract">("");
  const [error, setError] = useState<ApiError | string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const live = config?.live !== false;

  async function onFile(file: File) {
    setError(null);
    if (file.size > 5 * 1024 * 1024) return setError("That file is larger than 5 MB. Try a smaller scan, or paste the text.");
    if (!/^(application\/pdf|image\/(png|jpeg|webp))$/.test(file.type)) return setError("Upload a PDF, PNG, JPEG or WebP file.");
    setBusy("ocr");
    try {
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(String(r.result));
        r.onerror = () => reject(new Error("read"));
        r.readAsDataURL(file);
      });
      const out = await ocrFile(dataUrl);
      update((cur) => ({ note: out.text, source: "own", sample: undefined, steps: [...cur.steps, ...out.steps] }));
    } catch (e) {
      setError(e instanceof ApiError ? e : "The file could not be read.");
    } finally {
      setBusy("");
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function findFacts() {
    setError(null);
    setBusy("extract");
    try {
      const out = await extractFacts(s.note);
      update((cur) => ({
        facts: out.facts,
        topics: out.topics,
        nurse_checks: out.nurse_checks,
        factsConfirmed: false,
        version: undefined,
        attempts: [],
        acknowledged: [],
        steps: [...cur.steps.filter((x) => x.step === "Read the document (OCR)"), ...out.steps],
        times: { started: cur.times.started, factsShown: Date.now() },
        stage: "facts",
      }));
      window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
    } catch (e) {
      setError(e instanceof ApiError ? e : "Something went wrong.");
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="stack-lg">
      <div className="stage-head">
        <div>
          <h2>Discharge note</h2>
          <p>Paste the discharge instructions as written, or upload a PDF or photo of them. Homeward finds the facts the patient must understand.</p>
        </div>
        {s.note && (
          <button className="btn quiet small" onClick={reset}>
            Start over
          </button>
        )}
      </div>

      {!s.note.trim() && (
        <div className="btn-row small">
          <span className="muted">No note to hand? Open a sample patient:</span>
          {[
            ["heart-failure-it", "Giuseppe · Italian"],
            ["anticoagulant-hi", "Ramesh · Hindi"],
            ["knee-replacement-es", "Carmen · Spanish"],
          ].map(([id, label]) => (
            <button
              key={id}
              className="btn small"
              onClick={async () => {
                try {
                  const sm = await loadSample(id);
                  update({ ...sessionFromSample(sm), version: undefined, versionReplayed: false, teach: undefined });
                } catch {
                  setError("That sample could not be loaded. Please try again.");
                }
              }}
            >
              {label}
            </button>
          ))}
        </div>
      )}

      <div className="note warn">
        <Icon name="alert" />
        <span>
          Use synthetic or fully de-identified notes only. This hosted demo sends text to AMD Token Factory, whose terms do not allow real health data.
        </span>
      </div>

      <label className="field">
        <span>Discharge note</span>
        <textarea
          className="input mono"
          rows={16}
          value={s.note}
          placeholder={"DISCHARGE INSTRUCTIONS\nMedications: ...\nReturn precautions: ...\nFollow-up: ..."}
          onChange={(e) => update({ note: e.target.value, source: "own", sample: undefined })}
          disabled={busy !== ""}
        />
      </label>

      <div className="btn-row">
        <button className="btn primary" onClick={findFacts} disabled={busy !== "" || s.note.trim().length < 40 || !live}>
          {busy === "extract" ? <Icon name="loader" className="spin" /> : null}
          {busy === "extract" ? "Finding the facts" : "Find the facts"}
        </button>
        <input ref={fileRef} type="file" accept="application/pdf,image/png,image/jpeg,image/webp" hidden onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
        <button className="btn" onClick={() => fileRef.current?.click()} disabled={busy !== "" || !live}>
          {busy === "ocr" ? <Icon name="loader" className="spin" /> : <Icon name="upload" />}
          {busy === "ocr" ? "Reading the file" : "Upload PDF or photo"}
        </button>
        {busy === "extract" && <span className="muted small">Gemma 4 31B is reading the note on AMD GPUs through AMD Token Factory. This usually takes about a minute.</span>}
        {busy === "ocr" && <span className="muted small">MinerU 2.5 is reading the document on AMD GPUs through AMD Token Factory.</span>}
      </div>
      {!live && <ErrorNote error={new ApiError("budget", "Live generation is closed for today. The sample patients still work end to end.")} />}
      <ErrorNote error={error} />
      {s.facts.length > 0 && busy === "" && (
        <p className="small muted">
          Facts from the last run are kept.{" "}
          <button className="btn quiet small" onClick={() => goStage("facts")}>
            Go to the facts
          </button>
        </p>
      )}
    </div>
  );
}
