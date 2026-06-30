"""MMMU-Pro answer parsing — faithful port of the official grader.

Mirrors ``parse_multi_choice_response`` / ``get_multi_choice_info`` from the
official evaluation script so Mill's score matches the reported MMMU-Pro
multiple-choice accuracy:
https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/evaluate.py

The one intentional deviation: when nothing can be parsed the official guesses
with the *global* ``random`` module (non-reproducible). We take a seeded
``random.Random`` instead so scores are deterministic while preserving the
official's ~1/N expected contribution for genuinely unparseable responses.
"""
from __future__ import annotations

import random
from string import ascii_uppercase


def get_multi_choice_info(options: list[str]) -> tuple[dict[str, str], list[str]]:
    """Return ``(index2ans, all_choices)`` for an option list, lettered from A."""
    index2ans: dict[str, str] = {}
    all_choices: list[str] = []
    for i, option in enumerate(options):
        letter = ascii_uppercase[i]
        index2ans[letter] = option
        all_choices.append(letter)
    return index2ans, all_choices


def parse_multi_choice_response(
    response: str,
    all_choices: list[str],
    index2ans: dict[str, str],
    rng: random.Random | None = None,
) -> str:
    """Parse the predicted choice letter from a model response.

    Faithful port of the official MMMU-Pro grader. ``rng`` (a seeded
    ``random.Random``) is used only for the random-guess fallback.
    """
    rng = rng or random.Random()

    # Primary path: the text after the LAST "Answer:" should name exactly one
    # option letter.
    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:"):].strip()
        matching_options = [choice for choice in all_choices if choice in answer_str]
        if len(matching_options) == 1:
            return matching_options[0]

    if isinstance(response, str):
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
        response = " " + response + " "  # pad to avoid partial matches
    else:
        response = ""

    index_ans = True
    ans_with_brack = False
    candidates: list[str] = []
    for choice in all_choices:               # e.g. (A) (B) (C)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if not candidates:
        for choice in all_choices:           # e.g. "A " "B " "C "
            if f"{choice} " in response:
                candidates.append(choice)
    if not candidates:
        for choice in all_choices:           # e.g. "A." "B." "C."
            if f"{choice}." in response:
                candidates.append(choice)
    # Long answers with no letter: try to match the option text itself.
    if not candidates and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # matched on content, not the letter

    if not candidates:                       # nothing parsed -> random guess
        return rng.choice(all_choices)
    if len(candidates) == 1:
        return candidates[0]

    # Multiple candidates: keep the one that appears last in the response.
    start_indexes: list[int] = []
    if index_ans:
        if ans_with_brack:
            start_indexes = [response.rfind(f"({can})") for can in candidates]
        else:
            start_indexes = [response.rfind(f" {can} ") for can in candidates]
    else:
        start_indexes = [response.lower().rfind(index2ans[can].lower()) for can in candidates]
    return candidates[start_indexes.index(max(start_indexes))]
