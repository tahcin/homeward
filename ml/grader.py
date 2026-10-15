"""Shared helpers for the teach-back grader: build the prompt from prompts/grader.json
(used verbatim), turn designed labels into the gold JSON, and parse model replies.

Used by verify.py, build_sft.py, train_lora.py and eval.py. No third-party imports, so
it runs both locally and on the GPU notebook.
"""

import json
import re
from pathlib import Path

VERDICTS = ["correct", "partial", "incorrect", "missing"]

_HERE = Path(__file__).resolve().parent
_PROMPT_CANDIDATES = [_HERE.parent / "prompts" / "grader.json", _HERE / "grader.json"]


def load_prompt(path: str | None = None) -> dict:
    for p in ([Path(path)] if path else _PROMPT_CANDIDATES):
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError("grader.json not found")


def fact_line(f: dict) -> str:
    crit = ", critical" if f.get("critical") else ""
    return f'- {f["id"]} [{f["category"]}{crit}]: {f["statement"]}'


def user_prompt(row: dict, prompt: dict) -> str:
    return prompt["user"].format(
        facts="\n".join(fact_line(f) for f in row["facts"]),
        question=row["question"],
        language_name=row["language_name"],
        transcript=row["transcript"],
    )


def messages(row: dict, prompt: dict) -> list[dict]:
    return [{"role": "system", "content": prompt["system"]},
            {"role": "user", "content": user_prompt(row, prompt)}]


def gold(row: dict) -> dict:
    """Designed labels as the grader's JSON reply (drops the internal 'error' tag)."""
    out = []
    for lab in row["labels"]:
        v = lab["verdict"]
        out.append({"id": lab["id"], "verdict": v,
                    "heard": "" if v == "missing" else (lab.get("heard") or ""),
                    "reteach": "" if v == "correct" else (lab.get("reteach") or "")})
    return {"facts": out}


def gold_text(row: dict) -> str:
    return json.dumps(gold(row), ensure_ascii=False)


def row_key(row: dict) -> str:
    return f'{row["case_id"]}/{row["topic_id"]}/{row["n"]}'


def parse_reply(text: str):
    """Return the parsed JSON object or None. Tolerates code fences, thinking blocks and stray text."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"<\|channel\>.*?<channel\|>", "", t, flags=re.S)
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    t = re.sub(r"<turn\|>|<eos>|<end_of_turn>", "", t).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        s, e = t.find("{"), t.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except json.JSONDecodeError:
                return None
    return None


def verdicts_from(obj, fact_ids: list[str]) -> tuple[list[str | None], bool]:
    """Map a parsed reply to one verdict per expected fact id.

    Returns (verdicts, strictly_valid). strictly_valid means: a dict with a 'facts' list
    holding exactly the expected ids in order, each with a known verdict. Facts the reply
    does not cover come back as None (scored as a wrong answer, never as correct).
    """
    if not isinstance(obj, dict) or not isinstance(obj.get("facts"), list):
        return [None] * len(fact_ids), False
    got = {}
    for f in obj["facts"]:
        if isinstance(f, dict) and isinstance(f.get("id"), str):
            v = str(f.get("verdict", "")).strip().lower()
            got.setdefault(f["id"], v if v in VERDICTS else None)
    ids = [f.get("id") for f in obj["facts"] if isinstance(f, dict)]
    out = [got.get(i) for i in fact_ids]
    strict = ids == fact_ids and all(v is not None for v in out)
    return out, strict
