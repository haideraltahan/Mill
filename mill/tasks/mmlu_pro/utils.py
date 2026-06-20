"""MMLU-Pro — prompt and chain-of-thought answer-extraction metric.

The model is asked to reason step by step and end with a line of the form
``Answer: $LETTER``; the gold letter is regex-extracted from the response and
compared to the reference, mirroring the lighteval / official MMLU-Pro setup.
"""
from __future__ import annotations

import re
from string import ascii_uppercase

from mill.api.instance import OutputType
from mill.api.metrics import register_metric
from mill.api.task import Doc


def mmlu_pro_prompt(row: dict) -> Doc:
    """Build a CoT multiple-choice prompt for one MMLU-Pro row (up to 10 options)."""
    options = row["options"]
    letters = ascii_uppercase[: len(options)]
    choices_block = "\n".join(f"{letter}. {opt}" for letter, opt in zip(letters, options))
    query = (
        "Answer the following multiple choice question. The last line of your "
        "response should be of the form 'Answer: $LETTER' (without quotes) where "
        f"$LETTER is one of {''.join(letters)}. Think step by step before answering."
        f"\n\n{row['question'].strip()}\n\n{choices_block}\n\nAnswer:"
    )

    # Gold letter: prefer the dataset's letter, fall back to the index.
    gold = row.get("answer")
    if not gold and row.get("answer_index") is not None:
        gold = ascii_uppercase[int(row["answer_index"])]

    return Doc(
        query=query,
        choices=list(letters),
        target_index=str(gold).strip().upper(),
        metadata={"category": row.get("category", ""), "answer_index": row.get("answer_index")},
    )


# ── Answer extraction ───────────────────────────────────────────────────────
# Tried in order; first match wins. Falls back to the last standalone A–J letter.
_PATTERNS = [
    re.compile(r"answer is \(?([A-J])\)?", re.IGNORECASE),
    re.compile(r"answer\s*:\s*\(?([A-J])\)?", re.IGNORECASE),
    re.compile(r"\b([A-J])\b(?!.*\b[A-J]\b)", re.DOTALL),
]


def extract_answer_letter(text: str) -> str | None:
    """Best-effort extraction of the A–J answer letter from a model response."""
    for pattern in _PATTERNS:
        match = pattern.search(text or "")
        if match:
            return match.group(1).upper()
    return None


@register_metric("mmlu_pro_acc", higher_is_better=True, output_type=OutputType.GENERATIVE)
def mmlu_pro_acc(doc: Doc, response: str) -> float:
    """1.0 if the extracted answer letter matches the gold letter, else 0.0."""
    gold = doc.target_index
    if not isinstance(gold, str):
        gold = doc.choices[gold] if (doc.choices and isinstance(gold, int)) else ""
    pred = extract_answer_letter(response)
    return float(pred is not None and pred == str(gold).strip().upper())
