import { test } from "node:test";
import assert from "node:assert/strict";
import { checkAnchor, inventedNumbers, numberTokens, quoteIsVerbatim, splitLeaflet, strayEnglish, verifyFact } from "./lock.ts";
import type { Fact } from "../shared/types.ts";

const fact: Fact = {
  id: "f1",
  category: "medication_change",
  critical: true,
  statement: "Take furosemide 40 mg twice a day, up from 20 mg once a day.",
  anchors: ["furosemide", "40 mg", "20 mg"],
};

test("number tokens normalise decimal commas, times and other scripts", () => {
  assert.deepEqual(numberTokens("2,5 mg alle 10:30"), ["2.5", "10.30"]);
  assert.deepEqual(numberTokens("१० मिलीग्राम"), ["10"]);
  assert.deepEqual(numberTokens("৪০ mg"), ["40"]);
  assert.deepEqual(numberTokens("1,000 ml"), ["1000"]);
  assert.deepEqual(numberTokens("٥ mg"), ["5"]);
});

test("verbatim quote check tolerates spacing and curly quotes but not edits", () => {
  const text = "Prenda la  furosemide 40 mg due volte al giorno.\nPesi ogni mattina.";
  assert.ok(quoteIsVerbatim(text, "Prenda la furosemide 40 mg due volte al giorno."));
  assert.ok(quoteIsVerbatim(text, "“Prenda la furosemide 40 mg due volte al giorno”"));
  assert.ok(!quoteIsVerbatim(text, "Prenda la furosemide 20 mg due volte al giorno."));
  assert.ok(!quoteIsVerbatim(text, "Prenda la ... al giorno."));
});

test("Italian sentence carries all anchors", () => {
  const q = "Prenda la Furosemide 40 mg due volte al giorno (prima prendeva 20 mg una volta al giorno).";
  assert.deepEqual(verifyFact(fact, q, q, "it"), []);
});

test("a changed dose fails the lock", () => {
  const q = "Prenda la furosemide 20 mg due volte al giorno.";
  const p = verifyFact(fact, q, q, "it");
  assert.equal(p.length, 1);
  assert.equal(p[0].anchor, "40 mg");
});

test("a translated unit fails the lock", () => {
  assert.ok(checkAnchor("40 mg", "फ्यूरोसेमाइड 40 मिलीग्राम दिन में दो बार", "hi"));
  assert.equal(checkAnchor("40 mg", "furosemide 40 mg दिन में दो बार", "hi"), null);
});

test("dates match on localised month names or numeric dates", () => {
  assert.equal(checkAnchor("November 12", "Visita il 12 novembre alle 10:30", "it"), null);
  assert.equal(checkAnchor("November 12", "12 नवंबर को डॉक्टर से मिलें", "hi"), null);
  assert.equal(checkAnchor("Nov 12, 2026", "Cita el 12/11/2026", "es"), null);
  assert.ok(checkAnchor("November 12", "Visita il 13 novembre", "it"));
  assert.ok(checkAnchor("November 12", "Visita il 12 dicembre", "it"));
  assert.equal(checkAnchor("November 12", "11月12日复诊", "zh"), null);
});

test("re-teach lines may not invent numbers", () => {
  assert.deepEqual(inventedNumbers("Prenda 40 mg due volte.", fact), []);
  assert.deepEqual(inventedNumbers("Prenda 80 mg.", fact), ["80"]);
});

test("12-hour times match their 24-hour forms", () => {
  assert.equal(checkAnchor("2 pm", "alle 14:00", "it"), null);
  assert.equal(checkAnchor("8 am", "alle 8:00 di mattina", "it"), null);
  assert.equal(checkAnchor("9 pm", "रात 9 बजे", "hi"), null);
  assert.ok(checkAnchor("2 pm", "alle 2:30", "it"));
  assert.equal(checkAnchor("3 days", "in 3 giorni", "it"), null);
  assert.equal(checkAnchor("1.5 L/day", "1,5 L al giorno", "it"), null);
  assert.ok(checkAnchor("1.5 L/day", "1,5 litri al giorno", "it"));
});

test("re-teach lines may use the 24-hour form of a fact's time", () => {
  assert.deepEqual(inventedNumbers("Prenda 40 mg alle 8:00 e alle 14:00.", fact2), []);
  assert.deepEqual(inventedNumbers("Prenda 40 mg alle 9:00.", fact2), ["9.00"]);
});
const fact2: Fact = { ...fact, statement: "Take furosemide 40 mg twice a day at 8 am and 2 pm.", anchors: ["furosemide", "40 mg", "8 am", "2 pm"] };

test("leaflets split into sentences across scripts, not inside numbers or titles", () => {
  const s = splitLeaflet(["Prenda 2.5 mg al giorno. Vada da Dr. Elena Bianchi il 12 novembre.", "नई दवा Apixaban 5 mg लें। इसे सुबह 9:00 को लें।", "每天服用 5 mg。按时复诊。"]);
  assert.deepEqual(s.map((x) => x.text), ["Prenda 2.5 mg al giorno.", "Vada da Dr. Elena Bianchi il 12 novembre.", "नई दवा Apixaban 5 mg लें।", "इसे सुबह 9:00 को लें।", "每天服用 5 mg。", "按时复诊。"]);
  assert.deepEqual(s.map((x) => x.para), [0, 0, 1, 1, 2, 2]);
});

test("English left in a non-Latin patient text is caught", () => {
  const facts: Fact[] = [{ id: "f1", category: "follow_up", critical: false, statement: "See Dr. Anjali Mehta on 9 Nov 2026 at 11:00.", anchors: ["Dr. Anjali Mehta", "9 Nov 2026", "11:00"] }];
  assert.deepEqual(strayEnglish("9 Nov 2026 को 11:00 बजे Dr. Anjali Mehta से मिलें।", "hi", facts), ["nov"]);
  assert.deepEqual(strayEnglish("9 नवंबर 2026 को 11:00 बजे Dr. Anjali Mehta से मिलें।", "hi", facts), []);
  assert.deepEqual(strayEnglish("Apixaban 5 mg सुबह 9 am लें।", "hi", [{ ...facts[0], anchors: ["Apixaban", "5 mg"] }]), ["am"]);
  assert.deepEqual(strayEnglish("Visita il 9 Nov", "it", facts), []);
});

test("medicine names from the facts are not mistaken for stray English", () => {
  const facts: Fact[] = [{ id: "f5", category: "medication", critical: true, statement: "Avoid ibuprofen and diclofenac for pain, but paracetamol is okay for 10 min.", anchors: ["ibuprofen", "diclofenac"] }];
  assert.deepEqual(strayEnglish("दर्द के लिए ibuprofen और diclofenac न लें। paracetamol ले सकते हैं।", "hi", facts), []);
  assert.deepEqual(strayEnglish("10 min से ज्यादा", "hi", facts), ["min"]);
});

test("Latin month and unit words are rejected in Arabic and Hindi text", () => {
  const facts: Fact[] = [
    { id: "f1", category: "follow_up", critical: false, statement: "See Dr. Luis Herrera on 12 Nov 2026.", anchors: ["Dr. Luis Herrera", "12 Nov 2026"] },
    { id: "f2", category: "activity", critical: false, statement: "Walk 10-15 min twice a day.", anchors: ["10-15 min"] },
  ];
  assert.deepEqual(strayEnglish("راجع Dr. Luis Herrera في 12 Nov 2026.", "ar", facts), ["nov"]);
  assert.deepEqual(strayEnglish("راجع Dr. Luis Herrera في 12 نوفمبر 2026.", "ar", facts), []);
  assert.deepEqual(strayEnglish("امشِ 10-15 min مرتين يومياً.", "ar", facts), ["min"]);
  assert.deepEqual(strayEnglish("नाक से 10 min से ज़्यादा खून बहे", "hi", facts), ["min"]);
  assert.equal(checkAnchor("12 Nov 2026", "راجع الطبيب في 12 نوفمبر 2026", "ar"), null);
});

test("temperature units are not stray English", () => {
  const facts: Fact[] = [{ id: "f1", category: "warning_sign", critical: true, statement: "Go to the ER if fever is above 101°F.", anchors: ["101°F"] }];
  assert.deepEqual(strayEnglish("बुखार 101°F से ज़्यादा हो तो अस्पताल जाएं।", "hi", facts), []);
  assert.deepEqual(strayEnglish("بخار أعلى من 38 °C", "ar", facts), []);
});

test("local drug names with the box-label name in brackets, local titles and decimal commas pass", () => {
  assert.equal(checkAnchor("paracetamol", "Prenda paracetamolo (paracetamol) 500 mg fino a 3 volte al giorno.", "it"), null);
  assert.equal(checkAnchor("Furosemide", "Prenda furosemide (Furosemide) 40 mg due volte al giorno.", "it"), null);
  assert.equal(checkAnchor("Dr. Elena Bianchi", "Vada dalla Dott.ssa Elena Bianchi il 12 novembre.", "it"), null);
  assert.equal(checkAnchor("Dr. Luis Herrera", "Consulte a la Dra. Luis Herrera el 10 de noviembre.", "es"), null);
  assert.ok(checkAnchor("Dr. Elena Bianchi", "Vada dalla Dott.ssa Maria Rossi il 12 novembre.", "it"));
  assert.equal(checkAnchor("2.5 mg", "Prenda Bisoprolol 2,5 mg al giorno.", "it"), null);
  assert.equal(checkAnchor("1.5 L", "Beva al massimo 1,5 L al giorno.", "it"), null);
  assert.equal(checkAnchor("1000 mg", "Tome 1.000 mg al día.", "es"), null);
  assert.ok(checkAnchor("2.5 mg", "Prenda Bisoprolol 25 mg al giorno.", "it"));
});

test("sentences are not split after local titles", () => {
  const s = splitLeaflet(["Vada dal Dott. Rossi il 12 novembre. Porti i referti."]);
  assert.deepEqual(s.map((x) => x.text), ["Vada dal Dott. Rossi il 12 novembre.", "Porti i referti."]);
});
