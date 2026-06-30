"""General-purpose helpers shared across Mill tasks."""
from __future__ import annotations

import random
import re
from string import ascii_uppercase
from typing import Sequence

from mill.constants import DEFAULT_SEED

# ── Reproducibility ─────────────────────────────────────────────────────────────
# A single eval-wide seed drives every source of randomness: per-sample option
# shuffles, few-shot sampling, and random-guess fallbacks. `evaluate_task` calls
# `set_global_seed` once at the start of each task; prompt/metric functions — which
# only receive a row or a Doc, never the seed — read it back via `sample_rng`.

_GLOBAL_SEED = DEFAULT_SEED


def set_global_seed(seed: int) -> None:
    """Set the eval-wide seed and seed the stdlib / numpy / torch global RNGs.

    Called once per task before any data is built, so anything downstream
    (shuffles, model sampling) is reproducible for a given ``seed``.
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_global_seed() -> int:
    return _GLOBAL_SEED


def sample_rng(*parts: object) -> random.Random:
    """A ``random.Random`` derived from the global seed plus stable identifiers.

    Seeding from ``(seed, *parts)`` rather than the shared global stream makes a
    draw depend only on the sample's own identity — so the same sample yields the
    same result regardless of iteration order, parallelism, or ``--limit``. Pass
    a stable per-sample id (e.g. a dataset ``__key__``) so each sample is
    independent yet reproducible.
    """
    key = "|".join(str(p) for p in (_GLOBAL_SEED, *parts))
    return random.Random(key)

# "the answer is X" / "final answer: (X)" style markers. The choice letter is
# captured *case-sensitively* (uppercase only) and must stand alone (the
# look-around assertions), so lowercase letters inside ordinary words — "is",
# "not", "the" — and uppercase letters mid-word — the "D" in "Develop" — can
# never be mistaken for the answer. This is the strongest, least-trickable cue.
_ANSWER_MARKER_RE = re.compile(
    r"(?i:(?:final|correct|best)\s+answers?|answers?)"   # answer | final/correct/best answer
    r"\s*(?i:is|are|=|:|->|=>)?\s*"                      # optional connective (is / : / = ...)
    r"[\(\[\{*\"'`]*\s*"                                 # optional bracket / quote / bold opener
    r"(?<![A-Za-z])([A-Z])(?![A-Za-z])"                 # a standalone uppercase choice letter
)


def parse_multi_choice_response(
    response: str,
    options: Sequence[str],
    rng: random.Random | None = None,
) -> str:
    """Extract the chosen option letter from a free-form model response.

    Tries several cues, from most to least trustworthy, and always returns one
    of the valid letters (A, B, C, ... assigned to ``options`` in order):

      1. An explicit "Answer: X" / "the final answer is (X)" marker.
      2. A bracketed or delimited letter — "(X)", "X ", "X.".
      3. The verbatim text of an option (e.g. "cat"), for letterless answers.
      4. A deterministic random guess, as a last resort.

    Built not to be tricked by letters that merely appear inside reasoning or
    option text: the marker letter must stand alone, the structured passes look
    for delimited letters only, and when several candidates tie the one
    appearing *last* (a model's final word) wins.

    ``rng`` seeds only the random-guess fallback; it defaults to a generator
    keyed by the global seed and ``response`` so identical responses always score
    identically and the fallback honours ``--seed``.
    """
    letters = list(ascii_uppercase[: len(options)])
    if not letters:
        raise ValueError("parse_multi_choice_response needs at least one option")
    index2ans = dict(zip(letters, options))
    valid = set(letters)
    rng = rng or sample_rng("parse_multi_choice", response)
    response = response if isinstance(response, str) else ""

    # 1. Explicit answer marker — the last one wins (models often restate).
    markers = [m.group(1) for m in _ANSWER_MARKER_RE.finditer(response) if m.group(1) in valid]
    if markers:
        return markers[-1]

    # 2. Structured / standalone letters, then option text. Punctuation is
    #    trimmed from the ends and the text padded so a letter at a boundary
    #    still matches as "(X)" / "X " / "X.".
    text = response
    for ch in [",", ".", "!", "?", ";", ":", "'", '"']:
        text = text.strip(ch)
    text = f" {text} "

    matched_text = False
    candidates = [c for c in letters if f"({c})" in text]
    bracketed = bool(candidates)
    if not candidates:
        candidates = [c for c in letters if f"{c} " in text]
    if not candidates:
        candidates = [c for c in letters if f"{c}." in text]
    if not candidates and len(text.split()) > 5:
        candidates = [c for c in letters if index2ans[c].lower() in text.lower()]
        matched_text = bool(candidates)

    if not candidates:                       # 4. nothing parsed -> random guess
        return rng.choice(letters)
    if len(candidates) == 1:
        return candidates[0]

    # Several candidates: keep whichever is mentioned last.
    if matched_text:
        return max(candidates, key=lambda c: text.lower().rfind(index2ans[c].lower()))
    token = (lambda c: f"({c})") if bracketed else (lambda c: f" {c} ")
    return max(candidates, key=lambda c: text.rfind(token(c)))


def clip_mcq_doc(
    question: str,
    options: Sequence[str],
    answer_index: int,
    visuals: list,
    *,
    task_name: str = "",
    metadata: dict | None = None,
):
    """Render a multiple-choice question as a CLIP zero-shot-retrieval ``Doc``.

    Each option becomes a candidate caption ``f"{question} {option}"`` (the
    unibench VQA convention): a ``ZeroShotClassifier`` scores the image against
    every caption by image-text similarity and the argmax index is graded
    against ``answer_index``. Pair this with a task whose ``task_type`` is
    ``ZERO_SHOT_CLASSIFICATION`` and ``zeroshot_templates=["{c}"]`` (identity, so
    each caption is embedded verbatim instead of being wrapped in a
    "a photo of a {c}." template).

    The caption carries the whole question, so strip image-placeholder markers
    (e.g. "<image 1>") from ``question`` before calling. CLIP scores a single
    image (``visuals[0]``) and truncates each caption to its context length
    (77 tokens for most CLIP variants).
    """
    from mill.api.task import Doc

    captions = [f"{question} {option}".strip() for option in options]
    return Doc(
        query="",
        choices=captions,
        target_index=answer_index,
        visuals=list(visuals),
        metadata=dict(metadata or {}),
        task_name=task_name,
    )
