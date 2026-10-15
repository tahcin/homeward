// Read aloud with the browser's own voices, and capture spoken answers.
// Recognition uses the browser's speech service when there is no self-hosted transcriber; the patient can
// always correct the text, or type instead.

export function voiceFor(bcp47: string): SpeechSynthesisVoice | null {
  if (typeof speechSynthesis === "undefined") return null;
  const voices = speechSynthesis.getVoices();
  const lang = bcp47.toLowerCase();
  const base = lang.split("-")[0];
  const alt = base === "fil" ? ["fil", "tl"] : [base];
  return (
    voices.find((v) => v.lang.toLowerCase().replace("_", "-") === lang) ??
    voices.find((v) => alt.includes(v.lang.toLowerCase().split(/[-_]/)[0])) ??
    null
  );
}

/** Voices load asynchronously in Chrome; resolve once they are available (or after a short wait). */
export function voicesReady(): Promise<void> {
  if (typeof speechSynthesis === "undefined") return Promise.resolve();
  if (speechSynthesis.getVoices().length) return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => resolve();
    speechSynthesis.addEventListener("voiceschanged", done, { once: true });
    setTimeout(done, 1500);
  });
}

export function speak(text: string, bcp47: string, onEnd?: () => void): boolean {
  const voice = voiceFor(bcp47);
  if (!voice) return false;
  speechSynthesis.cancel();
  // Long texts are split by sentence; Chrome cuts off single utterances after about 15 seconds.
  const parts = text.match(/[^.!?。！？।\n]+[.!?。！？।]?/g)?.map((p) => p.trim()).filter(Boolean) ?? [text];
  parts.forEach((p, i) => {
    const u = new SpeechSynthesisUtterance(p);
    u.voice = voice;
    u.lang = voice.lang;
    u.rate = 0.92;
    if (i === parts.length - 1 && onEnd) u.onend = onEnd;
    speechSynthesis.speak(u);
  });
  return true;
}

export function stopSpeaking() {
  if (typeof speechSynthesis !== "undefined") speechSynthesis.cancel();
}

// ---------------------------------------------------------------- recognition

type Rec = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((e: any) => void) | null;
  onerror: ((e: any) => void) | null;
  onend: (() => void) | null;
};

export const recognitionSupported = () => typeof window !== "undefined" && Boolean((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition);

export interface Listener {
  stop(): Promise<{ text: string; wav: Blob | null }>;
}

/**
 * Starts listening. The microphone is always recorded as 16 kHz mono WAV (kept in this browser only, so the
 * nurse can replay the answer, and sent to a server transcriber when one is configured). When browserText is
 * true, the browser's speech recognition also runs and onText receives the running transcript.
 */
export async function listen(bcp47: string, onText: (t: string) => void, browserText: boolean, onError: (msg: string) => void): Promise<Listener> {
  let finalText = "";
  let rec: Rec | null = null;
  const Ctor = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  if (Ctor && browserText) {
    rec = new Ctor() as Rec;
    rec.lang = bcp47;
    rec.continuous = true;
    rec.interimResults = true;
    rec.onresult = (e: any) => {
      let interim = "";
      finalText = "";
      for (let i = 0; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) finalText += r[0].transcript + " ";
        else interim += r[0].transcript;
      }
      onText((finalText + interim).trim());
    };
    rec.onerror = (e: any) => {
      if (e.error === "no-speech" || e.error === "aborted") return;
      onError(e.error === "not-allowed" ? "The microphone is blocked. Allow it in the browser, or type the answer." : "Speech recognition stopped. You can type the answer instead.");
    };
    rec.start();
  }

  let stopWav: (() => Promise<Blob>) | null = null;
  try {
    stopWav = await recordPcm(onError);
  } catch {
    stopWav = null;
  }

  return {
    async stop() {
      let wav: Blob | null = null;
      if (stopWav) wav = await stopWav();
      if (rec) {
        await new Promise<void>((resolve) => {
          rec!.onend = () => resolve();
          rec!.stop();
          setTimeout(resolve, 1200);
        });
      }
      return { text: finalText.trim(), wav };
    },
  };
}

/** Records the microphone and returns a function that stops and yields 16 kHz mono 16-bit WAV. */
async function recordPcm(onError: (m: string) => void): Promise<() => Promise<Blob>> {
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  } catch {
    onError("The microphone is blocked. Allow it in the browser, or type the answer.");
    throw new Error("mic");
  }
  const ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const chunks: Float32Array[] = [];
  proc.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(proc);
  proc.connect(ctx.destination);
  return async () => {
    proc.disconnect();
    src.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    const rate = ctx.sampleRate;
    await ctx.close();
    const len = chunks.reduce((n, c) => n + c.length, 0);
    const all = new Float32Array(len);
    let o = 0;
    for (const c of chunks) {
      all.set(c, o);
      o += c.length;
    }
    return encodeWav(await resample(all, rate, 16000), 16000);
  };
}

export async function resample(data: Float32Array, from: number, to: number): Promise<Float32Array> {
  if (from === to || data.length === 0) return data;
  const frames = Math.ceil((data.length * to) / from);
  const off = new OfflineAudioContext(1, frames, to);
  const buf = off.createBuffer(1, data.length, from);
  buf.copyToChannel(new Float32Array(data), 0);
  const s = off.createBufferSource();
  s.buffer = buf;
  s.connect(off.destination);
  s.start();
  const out = await off.startRendering();
  return out.getChannelData(0);
}

export function encodeWav(samples: Float32Array, rate: number): Blob {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const w = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF");
  v.setUint32(4, 36 + samples.length * 2, true);
  w(8, "WAVE");
  w(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  w(36, "data");
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const x = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, x < 0 ? x * 0x8000 : x * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

export async function blobToBase64(b: Blob): Promise<string> {
  const bytes = new Uint8Array(await b.arrayBuffer());
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
