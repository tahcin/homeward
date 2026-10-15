// Clinical safety rules applied in code, not left to the model.

/**
 * Medicines whose misunderstanding commonly causes serious harm. A medicine fact naming one of these is marked
 * critical by default when facts are extracted; the nurse can still untick it.
 */
const HIGH_RISK: { group: string; names: string[] }[] = [
  { group: "anticoagulant", names: ["warfarin", "acenocoumarol", "apixaban", "rivaroxaban", "dabigatran", "edoxaban", "enoxaparin", "dalteparin", "tinzaparin", "heparin", "fondaparinux"] },
  { group: "antiplatelet", names: ["aspirin", "acetylsalicylic", "clopidogrel", "ticagrelor", "prasugrel", "dipyridamole"] },
  { group: "beta-blocker", names: ["metoprolol", "bisoprolol", "carvedilol", "atenolol", "propranolol", "nebivolol", "labetalol", "sotalol", "nadolol"] },
  { group: "insulin or other glucose-lowering medicine", names: ["insulin", "glargine", "detemir", "degludec", "lispro", "aspart", "glulisine", "glipizide", "gliclazide", "glimepiride", "glibenclamide", "glyburide", "repaglinide"] },
  { group: "opioid", names: ["morphine", "oxycodone", "hydromorphone", "fentanyl", "tramadol", "codeine", "tapentadol", "methadone", "buprenorphine", "hydrocodone", "oxymorphone"] },
  { group: "diuretic", names: ["furosemide", "torsemide", "torasemide", "bumetanide", "hydrochlorothiazide", "chlorthalidone", "indapamide", "metolazone", "spironolactone", "eplerenone"] },
  { group: "digoxin", names: ["digoxin"] },
  { group: "lithium", names: ["lithium"] },
  { group: "methotrexate", names: ["methotrexate"] },
  { group: "potassium", names: ["potassium chloride", "potassium supplement", "potassium"] },
  { group: "anticonvulsant", names: ["levetiracetam", "phenytoin", "valproate", "valproic", "carbamazepine", "oxcarbazepine", "lamotrigine", "lacosamide", "topiramate", "phenobarbital", "clobazam", "zonisamide", "brivaracetam"] },
];

/** The high-risk group a medicine fact belongs to, or null. Only medicine facts count. */
export function highRiskGroup(category: string, statement: string): string | null {
  if (!category.startsWith("medication")) return null;
  const s = statement.toLowerCase();
  for (const g of HIGH_RISK) if (g.names.some((n) => new RegExp(`\\b${n}`).test(s))) return g.group;
  return null;
}

export const HIGH_RISK_HINT = HIGH_RISK.map((g) => g.group).join(", ");
