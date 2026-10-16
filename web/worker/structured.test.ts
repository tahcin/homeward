import { test } from "node:test";
import assert from "node:assert/strict";
import { cleanAnchors, compareMed, hasLocalUnit, missingItems, notEnglishComplaint, timesPerDay, tidyDuplicateBrackets } from "./structured.ts";

test("frequency forms are equivalent: twice a day, every 12 hours, 9 am and 9 pm", () => {
  assert.equal(timesPerDay({ times_per_day: 2 }), 2);
  assert.equal(timesPerDay({ every_n_hours: 12 }), 2);
  assert.equal(timesPerDay({ time_of_day: ["9 am", "9 pm"] }), 2);
  const fact = { drug: "Apixaban", dose_value: 5, dose_unit: "mg", times_per_day: 2, time_of_day: ["9 am", "9 pm"], action: "take" };
  assert.deepEqual(compareMed(fact, { drug: "Apixaban", dose_value: 5, dose_unit: "mg", every_n_hours: 12, time_of_day: ["morning 9:00", "night 9:00"], action: "take" }), []);
});

test("a wrong frequency, time of day, action or dose fails with the field named", () => {
  const fact = { drug: "Metoprolol", dose_value: 25, dose_unit: "mg", times_per_day: 1, time_of_day: ["morning"], action: "take" };
  assert.match(compareMed(fact, { ...fact, times_per_day: 2, time_of_day: ["morning", "night"] }).join(), /frequency differs/);
  assert.match(compareMed(fact, { ...fact, time_of_day: ["evening"] }).join(), /time of day differs/);
  assert.match(compareMed(fact, { ...fact, action: "take", negated: true }).join(), /action differs/);
  assert.match(compareMed(fact, { ...fact, dose_value: 50 }).join(), /dose differs/);
  assert.match(compareMed({ drug: "aspirin", action: "stop" }, { drug: "aspirin", action: "continue" }).join(), /action differs/);
  assert.match(compareMed({ drug: "oxycodone", as_needed: true, max_per_day: 4 }, { drug: "oxycodone", as_needed: true }).join(), /maximum per day missing/);
  assert.deepEqual(compareMed({ drug: "aspirin", action: "stop" }, { drug: "aspirin", action: "take", negated: true }), []);
});

test("warning-sign items are all required, with a real quoted span", () => {
  const items = ["black stools", "vomiting blood", "severe headache", "fall with a head injury"];
  const bt = "On having black or bloody stool call 112. On vomiting blood or severe headache call 112.";
  const checks = [
    { item: "black stools", present: true, span: "black or bloody stool" },
    { item: "vomiting blood", present: true, span: "vomiting blood" },
    { item: "severe headache", present: true, span: "severe headache" },
    { item: "fall with a head injury", present: true, span: "falls and hits the head" },
  ];
  assert.deepEqual(missingItems(items, checks, bt), ["fall with a head injury"]);
});

test("local-script unit names count as the unit, only right after a number", () => {
  assert.ok(hasLocalUnit("每天服用 40 毫克", "mg"));
  assert.ok(hasLocalUnit("40 मिलीग्राम दिन में दो बार", "mg"));
  assert.ok(hasLocalUnit("Tome 2,5 miligramos", "mg"));
  assert.ok(hasLocalUnit("salt under 3 克 a day", "g"));
  assert.ok(!hasLocalUnit("عمل كامل", "ml"));
  assert.ok(hasLocalUnit("اشرب 500 مل", "ml"));
});

test("anchors keep names and numbers, drop ordinary words", () => {
  const st = "Elevate your leg above your heart three times a day and take Cephalexin 500 mg; call Dr. Patel if furosemide makes you dizzy.";
  assert.deepEqual(cleanAnchors(st, ["leg", "heart", "three times a day", "Cephalexin", "500 mg", "Dr. Patel", "furosemide"]), ["Cephalexin", "500 mg", "Dr. Patel", "furosemide"]);
});

test("brackets are dropped when the local and label names match", () => {
  assert.equal(tidyDuplicateBrackets("Prenda furosemide (Furosemide) 40 mg e paracetamolo (paracetamol)."), "Prenda Furosemide 40 mg e paracetamolo (paracetamol).");
});

test("'not English' complaints are not findings", () => {
  assert.ok(notEnglishComplaint("The Hindi word 'पानी' means 'water', but it is transliterated as 'pani' which is not English."));
  assert.ok(!notEnglishComplaint("लत means addiction; nonsensical here."));
});
