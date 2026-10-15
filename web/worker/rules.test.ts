import { test } from "node:test";
import assert from "node:assert/strict";
import { highRiskGroup } from "./rules.ts";

test("high-risk medicines are recognised in medicine facts only", () => {
  assert.equal(highRiskGroup("medication", "Take new medicine Metoprolol succinate 25 mg once daily in the morning."), "beta-blocker");
  assert.equal(highRiskGroup("medication_change", "Stop taking aspirin 75 mg."), "antiplatelet");
  assert.equal(highRiskGroup("medication", "Inject insulin glargine 20 units at bedtime."), "insulin or other glucose-lowering medicine");
  assert.equal(highRiskGroup("medication", "Take Omeprazole 20 mg daily before breakfast."), null);
  assert.equal(highRiskGroup("warning_sign", "Call if you take too much furosemide and feel dizzy."), null);
  assert.equal(highRiskGroup("medication", "Take Paracetamol 1 g every 6 h."), null);
});
