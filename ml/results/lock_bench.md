# Fact-lock benchmark

Real Homeward pipeline on AMD Token Factory (Gemma 4 31B extracts and writes, DeepSeek V4 Flash checks), 24 synthetic discharge notes in 8 languages (it, es, hi, bn, ar, zh, ta, vi). One planted fault per case was put into the first draft; 7 cases were clean controls. Facts were extracted by the pipeline itself, without nurse review.

**Caught 16 of 17 planted faults** (94%). Uncaught faults still present in the final text: 1.

**Non-planted findings:** 68 of 234 other facts had an issue raised on the first draft (these are real writer errors or false alarms; reasons below). In the 7 clean controls: 29 of 69 facts. Facts still flagged for the nurse at the end: 24 of 251.

Planted fact final status: locked 1, repaired 14, flagged 2.

Cost: $0.0543 for 535 model calls (333,593 tokens in, 131,722 out). Median wall time per case: 281 s.

## Non-planted findings by source

| Source | Facts |
|---|---|
| writer left an English word untranslated | 34 |
| meaning check disagreed | 9 |
| suspicious word (read these: several are checker false alarms) | 7 |
| field check disagreed (dose, frequency, timing, action) | 6 |
| checker failed to answer (retried, not repaired) | 5 |
| writer translated a medical unit (policy keeps mg, g in Latin letters) | 5 |
| warning-sign item missing | 2 |
| number or date missing | 1 |

A fact can have findings from more than one source. Writer and extractor findings are real deviations from the rules; the suspicious-word list includes checker false alarms (listed at the end).

## Caught by

| Check | Faults caught |
|---|---|
| code: anchor check (number, unit, date or name) | 6 |
| code: field check (dose, frequency, timing, action) | 5 |
| code: English left in the text | 3 |
| meaning check | 2 |
| literal back-translation (suspicious word) | 1 |

A fault can be caught by more than one check.

## By fault type

| Fault | Cases | Caught | Other facts with issues |
|---|---|---|---|
| wrong date | 3 | 3 of 3 | 2 of 27 |
| wrong frequency | 5 | 5 of 5 | 15 of 46 |
| negation dropped | 2 | 2 of 2 | 4 of 20 |
| clean control | 7 | n/a | 29 of 69 |
| warning symptom dropped | 2 | 1 of 2 | 4 of 18 |
| dose doubled | 2 | 2 of 2 | 4 of 21 |
| action reversed (stop becomes continue) | 1 | 1 of 1 | 4 of 9 |
| dose halved | 1 | 1 of 1 | 3 of 13 |
| word mistranslated | 1 | 1 of 1 | 3 of 11 |

## By language

| Language | Cases | Caught | Other facts with issues | Flagged at the end |
|---|---|---|---|---|
| it | 3 | 3 of 3 | 3 of 31 | 1 |
| es | 3 | 3 of 3 | 3 of 29 | 3 |
| hi | 3 | 2 of 2 | 8 of 26 | 4 |
| bn | 3 | 2 of 3 | 8 of 25 | 2 |
| ar | 3 | 3 of 3 | 9 of 33 | 4 |
| zh | 3 | n/a | 19 of 30 | 2 |
| ta | 3 | 3 of 3 | 12 of 32 | 4 |
| vi | 3 | n/a | 6 of 28 | 4 |

## Per case

| Case | Lang | Fault | Caught by | Planted fact final | Other issues |
|---|---|---|---|---|---|
| c001 | it | wrong date | code: anchor check (number, unit, date or name) | repaired | 0 |
| c002 | es | wrong date | code: anchor check (number, unit, date or name) | repaired | 1 |
| c003 | hi | wrong frequency | code: English left in the text | repaired | 3 |
| c004 | bn | wrong frequency | code: field check (dose, frequency, timing, action) | repaired | 2 |
| c006 | ar | negation dropped | code: English left in the text | repaired | 3 |
| c007 | zh | clean control |  |  | 9 |
| c010 | ta | warning symptom dropped | code: English left in the text | flagged | 1 |
| c012 | vi | clean control |  |  | 2 |
| c013 | it | wrong frequency | code: field check (dose, frequency, timing, action) | repaired | 0 |
| c014 | es | negation dropped | meaning check | repaired | 1 |
| c015 | hi | clean control (not injected) |  |  | 4 |
| c016 | bn | wrong frequency | code: field check (dose, frequency, timing, action) | repaired | 3 |
| c018 | ar | dose doubled | code: anchor check (number, unit, date or name) | flagged | 3 |
| c019 | zh | clean control |  |  | 2 |
| c022 | ta | action reversed (stop becomes continue) | code: field check (dose, frequency, timing, action); meaning check | repaired | 4 |
| c024 | vi | clean control |  |  | 3 |
| c025 | it | dose halved | code: anchor check (number, unit, date or name) | repaired | 3 |
| c026 | es | dose doubled | code: anchor check (number, unit, date or name) | repaired | 1 |
| c027 | hi | wrong date | code: anchor check (number, unit, date or name) | repaired | 1 |
| c028 | bn | warning symptom dropped | missed, shipped | locked | 3 |
| c030 | ar | word mistranslated | literal back-translation (suspicious word) | repaired | 3 |
| c031 | zh | clean control |  |  | 8 |
| c034 | ta | wrong frequency | code: field check (dose, frequency, timing, action) | repaired | 7 |
| c036 | vi | clean control |  |  | 1 |

## Planted faults (what was changed)

- c001 (it, wrong date, f6): "Si rechi alla clinica di cardiologia il 10 novembre 2026." -> "Si rechi alla clinica di cardiologia il 11 novembre 2026.". Caught: date "Nov 10, 2026": number 10 missing
- c002 (es, wrong date, f4): "Tenga una cita de seguimiento con cardiología el 16/11/26." -> "Tenga una cita de seguimiento con cardiología el 17/11/26.". Caught: "11/16/26": number 16 missing
- c003 (hi, wrong frequency, f1): "Azithromycin 250 mg को अगले 3 दिनों तक रोज़ाना मुँह से लें।" -> "Azithromycin 250 mg को अगले 3 दिनों तक हर 12 घंटे में मुँह से लें।". Caught: English left untranslated: "h"; write it in Hindi
- c004 (bn, wrong frequency, f1): "প্রথম 5 দিন প্রতিদিন 40 mg নিন।" -> "প্রথম 5 দিন প্রতি 12 ঘণ্টা অন্তর 40 mg নিন।". Caught: field check: frequency differs (fact 1 a day, patient text 2 a day)
- c006 (ar, negation dropped, f10): "تجنب القيادة إذا كان سكر الدم أقل من 70 mg/dL." -> "قد إذا كان سكر الدم أقل من 70 mg/dL.". Caught: English left untranslated: "dl"; write it in Arabic
- c010 (ta, warning symptom dropped, f8): "ஓய்வெடுத்தும் சரியாகாத 5 min க்கும் மேலாக நீடிக்கும் மார்பு வலி, வியர்த்தல், மூச்சுத் திணறல் அல்லது மயக்கம் ஏற்பட்டால் அவசர சிகிச்சை பிரிவுக்குத் திரும்புங்கள்." -> "ஓய்வெடுத்தும் சரியாகாத 5 min க்கும் மேலாக நீடிக்கும் மார்பு வலி, வியர்த்தல், மூச்சுத் திணறல் ஏற்பட்டால் அவசர சிகிச்சை பிரிவுக்குத் திரும்புங்கள்.". Caught: English left untranslated: "min"; write it in Tamil
- c013 (it, wrong frequency, f1): "Prenda cefalessina (Cephalexin) 500 mg per bocca quattro volte al giorno per 10 giorni." -> "Prenda cefalessina (Cephalexin) 500 mg per bocca due volte al giorno per 10 giorni.". Caught: field check: frequency differs (fact 4 a day, patient text 2 a day)
- c014 (es, negation dropped, f6): "Evite levantar objetos pesados de más de 10 lb durante 1 semana." -> "Levante objetos pesados de más de 10 lb durante 1 semana.". Caught: meaning differs (missing detail)
- c016 (bn, wrong frequency, f1): "Prednisone 40 mg প্রতিদিন মুখে খাবেন 5 দিন ধরে।" -> "Prednisone 40 mg প্রতি ১২ ঘণ্টা অন্তর মুখে খাবেন 5 দিন ধরে।". Caught: field check: frequency differs (fact 1 a day, patient text 2 a day)
- c018 (ar, dose doubled, f1): "ابدأ في تناول levetiracetam بجرعة 500 mg مرتين يومياً في الصباح والمساء." -> "ابدأ في تناول levetiracetam بجرعة 1000 mg مرتين يومياً في الصباح والمساء.". Caught: "500 mg": number 500 missing
- c022 (ta, action reversed (stop becomes continue), f3): "Lisinopril 10 mg மருந்தை எடுத்துக் கொள்வதை நிறுத்துங்கள்." -> "Lisinopril 10 mg மருந்தை எடுத்துக் கொள்வதை தொடருங்கள்.". Caught: field check: frequency missing
- c025 (it, dose halved, f1): "La dose di Furosemide aumenta da 20 mg a 40 mg due volte al giorno." -> "La dose di Furosemide aumenta da 10 mg a 40 mg due volte al giorno.". Caught: "20 mg": number 20 missing
- c026 (es, dose doubled, f1): "Empiece a tomar apixaban 5mg dos veces al día." -> "Empiece a tomar apixaban 10mg dos veces al día.". Caught: "5mg": number 5 missing
- c027 (hi, wrong date, f6): "फेफड़ों की दोबारा जांच के लिए 11/10/2026 को क्लिनिक वापस आएं।" -> "फेफड़ों की दोबारा जांच के लिए 12/10/2026 को क्लिनिक वापस आएं।". Caught: "11/10/2026": number 11 missing
- c028 (bn, warning symptom dropped, f4): "রক্তবমি বা কাশির সাথে রক্ত আসলে জরুরি বিভাগে যান।" -> "কাশির সাথে রক্ত আসলে জরুরি বিভাগে যান।". Not caught.
- c030 (ar, word mistranslated, f1): "خذ insulin glargine بمقدار 24 وحدة تحت الجلد كل ليلة عند النوم." -> "خذ insulin glargine بمقدار 24 وحدة تحت الجند كل ليلة عند النوم.". Caught: suspicious wording "الجند": It means 'the soldier' but the intended word is 'الجلد' (the skin).
- c034 (ta, wrong frequency, f1): "Aspirin 81 mg மருந்தினை தினமும் வாயின் வழியாக எடுத்துக்கொள்ளத் தொடங்குங்கள்." -> "Aspirin 81 mg மருந்தினை ஒரு நாளைக்கு இரண்டு முறை வாயின் வழியாக எடுத்துக்கொள்ளத் தொடங்குங்கள்.". Caught: field check: frequency differs (fact 1 a day, patient text 2 a day)

## Non-planted findings (to read: writer error or false alarm?)

- c002 es f8 (flagged): missing: rapid irregular heartbeat with shortness of breath
- c003 hi f2 (repaired): the meaning check gave no answer for this fact
- c003 hi f3 (flagged): English left untranslated: "clots", "h"; write it in Hindi
- c003 hi f7 (repaired): meaning differs (timing): The patient sentence says '2 week after 11/17/2026' which implies meeting 2 weeks after that date, but the fact says follow up in 2 weeks on 11/17/2026, m
- c004 bn f2 (repaired): English left untranslated: "puffs"; write it in Bengali
- c004 bn f3 (flagged): English left untranslated: "puff"; write it in Bengali
- c006 ar f2 (flagged): English left untranslated: "dl"; write it in Arabic
- c006 ar f6 (repaired): "2": number 2 missing
- c006 ar f7 (repaired): the second reading of this critical warning sign failed
- c007 zh f1 (repaired): English left untranslated: "h"; write it in Mandarin Chinese
- c007 zh f2 (locked): the back-translation failed, so the meaning could not be checked
- c007 zh f3 (flagged): "3 g": unit g missing
- c007 zh f4 (repaired): English left untranslated: "days", "h"; write it in Mandarin Chinese
- c007 zh f5 (locked): the back-translation failed, so the meaning could not be checked
- c007 zh f6 (repaired): English left untranslated: "weeks"; write it in Mandarin Chinese
- c007 zh f7 (locked): the back-translation failed, so the meaning could not be checked
- c007 zh f8 (repaired): English left untranslated: "weeks"; write it in Mandarin Chinese
- c007 zh f9 (repaired): English left untranslated: "h"; write it in Mandarin Chinese
- c010 ta f9 (repaired): English left untranslated: "cardiology"; write it in Tamil
- c012 vi f2 (flagged): "3 g": unit g missing
- c012 vi f9 (flagged): suspicious wording "đỏ mới": It literally means 'red new' but likely intended 'new redness'; as written it is nonsensical.
- c014 es f2 (flagged): field check: time of day differs (fact morning, evening, patient text morning, night)
- c015 hi f3 (flagged): "3 g": unit g missing
- c015 hi f5 (flagged): missing: worsening flank pain
- c015 hi f9 (flagged): meaning differs (missing detail): The patient sentence omits 'or strenuous exercise' from the instruction.
- c015 hi f11 (repaired): English left untranslated: "h"; write it in Hindi
- c016 bn f2 (repaired): English left untranslated: "puffs"; write it in Bengali
- c016 bn f3 (flagged): English left untranslated: "puff"; write it in Bengali
- c016 bn f8 (repaired): English left untranslated: "puffs"; write it in Bengali
- c018 ar f3 (flagged): "3 g": unit g missing
- c018 ar f4 (repaired): English left untranslated: "min"; write it in Arabic
- c018 ar f10 (repaired): English left untranslated: "week"; write it in Arabic
- c019 zh f4 (flagged): suspicious wording "停止": The sentence says 'stop' but the context of a discharge leaflet likely means 'continue' or 'take'; 'stop' is nonsensical for a medication instruction.
- c019 zh f8 (repaired): suspicious wording "出现": 'Appear' is used for '出现' but in English 'appear' means 'seem' or 'become visible'; the correct medical term is 'experience' or 'have'.
- c022 ta f6 (repaired): English left untranslated: "am"; write it in Tamil
- c022 ta f7 (repaired): English left untranslated: "pm"; write it in Tamil
- c022 ta f8 (repaired): meaning differs (missing detail): The patient sentence says 'avoid heavy things to lift' but does not specify the 10 lb threshold.
- c022 ta f9 (repaired): suspicious wording "நோயாளிகள்": It means 'patients' but in context of avoiding crowded places, it likely means 'sick people' or 'infected persons'; as written it is nonsensical to 
- c024 vi f4 (flagged): "3 g": unit g missing
- c024 vi f7 (flagged): suspicious wording "nói ngọng": 'nói ngọng' means 'lisp' or 'slurred speech' — but 'ngọng' specifically is a lisp, not general slurring; however it is commonly used for slurred spe
- c024 vi f9 (repaired): suspicious wording "Thận": It means 'kidney' but likely intended 'nephrology' department; however, as written it is a body organ, not a department name.
- c025 it f6 (flagged): meaning differs (dose): Patient says one tablet per day, but fact says take one tablet as needed, not daily.
- c025 it f8 (repaired): meaning differs (missing detail): Patient sentence omits the instruction to weigh every morning.
- c025 it f12 (repaired): meaning differs (action): Patient sentence is nonsense and does not convey the fact.
- c026 es f5 (flagged): suspicious wording "al": 'al día' means 'per day' but literally 'to the day' - I translated as 'to day' which is wrong English but literal.
- c027 hi f3 (repaired): field check: maximum per day differs (fact 3000)
- c028 bn f2 (repaired): field check: time of day differs (fact morning, evening, patient text morning, night)
- c028 bn f5 (repaired): English left untranslated: "weeks"; write it in Bengali
- c028 bn f6 (repaired): English left untranslated: "weeks"; write it in Bengali
- c030 ar f3 (flagged): field check: duration missing (fact 9 days)
- c030 ar f9 (repaired): English left untranslated: "rao"; write it in Arabic
- c030 ar f11 (repaired): English left untranslated: "rao"; write it in Arabic
- c031 zh f1 (repaired): English left untranslated: "h"; write it in Mandarin Chinese
- c031 zh f3 (repaired): English left untranslated: "h"; write it in Mandarin Chinese
- c031 zh f4 (repaired): English left untranslated: "h", "days"; write it in Mandarin Chinese
- c031 zh f6 (repaired): English left untranslated: "week"; write it in Mandarin Chinese
- c031 zh f8 (repaired): English left untranslated: "weeks"; write it in Mandarin Chinese
- c031 zh f9 (repaired): English left untranslated: "f"; write it in Mandarin Chinese
- c031 zh f10 (repaired): English left untranslated: "weeks"; write it in Mandarin Chinese
- c031 zh f11 (repaired): English left untranslated: "weeks"; write it in Mandarin Chinese
- c034 ta f4 (flagged): field check: frequency differs (fact 2 a day, patient text 1 a day)
- c034 ta f6 (flagged): field check: dose missing
- c034 ta f7 (repaired): English left untranslated: "min"; write it in Tamil
- c034 ta f8 (repaired): English left untranslated: "min"; write it in Tamil
- c034 ta f10 (flagged): English left untranslated: "cardiology"; write it in Tamil
- c034 ta f12 (repaired): English left untranslated: "hrs", "week"; write it in Tamil
- c034 ta f13 (repaired): meaning differs (missing detail): Patient sentence omits 'avoid grapefruit'.
- c036 vi f1 (repaired): meaning differs (missing detail)
