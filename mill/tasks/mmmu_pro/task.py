"""MMMU-Pro (standard, 10 options) — multimodal multiple-choice with CoT.

The ``MMMU/MMMU_Pro`` ``standard (10 options)`` config: college-level multimodal
questions carrying up to 7 images and 10 answer options (A–J). A vision-language
model sees the image(s) and the lettered options, reasons step by step, and ends
with ``Answer: $LETTER``; the letter is extracted from the generation and graded.

Mirrors the official MMMU-Pro *standard* chain-of-thought setup:
https://github.com/MMMU-Benchmark/MMMU/tree/main/mmmu-pro
"""
from __future__ import annotations

import ast
import re
from string import ascii_uppercase

from mill.api.instance import OutputType
from mill.api.metrics import get_metric, register_metric
from mill.api.task import Doc, MillBenchmarkConfig, MillTaskConfig
from mill.api.taxonomy import TaskType
from mill.tasks.mmmu_pro.utils import get_multi_choice_info, parse_multi_choice_response
from mill.utils import clip_mcq_doc, sample_rng

# Official CoT instruction for the standard setting (prompts.yaml -> cot.standard).
# It refers to the "preceding" question, so it is appended after the options.
_COT_INSTRUCTION = (
    "Answer the preceding multiple choice question. The last line of your response "
    "should be of the following format: 'Answer: $LETTER' (without quotes) where "
    "LETTER is one of options. Think step by step before answering."
)

# "<image 3>" style markers embedded in the question text.
_IMAGE_TOKEN_RE = re.compile(r"<image\s+(\d+)>")


def _as_list(value) -> list:
    """Coerce a field that may be a stringified Python list (or None) to a list.

    MMMU-Pro stores both ``options`` and ``img_type`` as ``"[...]"`` strings.
    """
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
    return list(value)


def mmmu_pro_prompt(row: dict) -> Doc:
    """Build a multimodal CoT MCQ prompt for one MMMU-Pro 'standard' row."""
    options = _as_list(row["options"])
    letters = ascii_uppercase[: len(options)]
    choices_block = "\n".join(f"{letter}. {opt}" for letter, opt in zip(letters, options))

    # Gather the images the question cites, in citation order, and collapse each
    # "<image N>" marker to a generic "<image>" token (official behaviour).
    question = row["question"]
    image_order = [int(n) for n in _IMAGE_TOKEN_RE.findall(question)]
    question = _IMAGE_TOKEN_RE.sub("<image>", question)
    visuals = [row[f"image_{i}"] for i in image_order if row.get(f"image_{i}") is not None]
    if not visuals:  # no inline references — fall back to any attached images
        visuals = [row[f"image_{i}"] for i in range(1, 8) if row.get(f"image_{i}") is not None]

    query = f"{question}\n{choices_block}\n{_COT_INSTRUCTION}"
    return Doc(
        query=query,
        choices=list(letters),
        target_index=str(row["answer"]).strip().upper(),  # gold letter, e.g. "B"
        visuals=visuals,
        # `options` is kept so the metric can match answers by option *text*.
        metadata={
            "subject": row.get("subject", ""),
            "id": row.get("id", ""),
            "options": options,
            "topic_difficulty": row.get("topic_difficulty", ""),
            "img_type": _as_list(row.get("img_type")),
        },
        task_name="mmmu_pro",
    )


@register_metric("mmmu_pro_acc", higher_is_better=True, output_type=OutputType.GENERATIVE)
def mmmu_pro_acc(doc: Doc, response: str) -> float:
    """1.0 if the parsed choice letter matches the gold letter (official grader)."""
    options = doc.metadata.get("options") or list(doc.choices or [])
    index2ans, all_choices = get_multi_choice_info(options)
    # Seed the random-guess fallback per-sample so scores stay reproducible and
    # honour --seed (sample_rng folds in the eval-wide global seed).
    rng = sample_rng("mmmu_pro", doc.metadata.get("id") or doc.query)
    pred = parse_multi_choice_response(response or "", all_choices, index2ans, rng)
    return float(pred == str(doc.target_index).strip().upper())


mmmu_pro_task = MillTaskConfig(
    name="mmmu_pro",
    version=1,
    hf_repo="MMMU/MMMU_Pro",
    hf_subset="standard (10 options)",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    prompt_function=mmmu_pro_prompt,
    task_type=TaskType.MULTIPLE_CHOICE,
    output_type=OutputType.GENERATIVE,        # graded by CoT generation + answer-letter extraction
    input_modalities=["image", "text"],       # requires a vision-language model
    generation_size=1024,                     # allow for long CoT generations
    n_shots=0,
    metrics=[get_metric("mmmu_pro_acc")],
    description=(
        "MMMU-Pro (standard, 10 options): college-level multimodal multiple-choice "
        "questions with up to 7 images and 10 options each. A vision-language model "
        "reasons step by step and answers with a letter, which is extracted and graded."
    ),
    categories=["vision", "multiple-choice", "reasoning", "academic"],
    capabilities=["visual reasoning", "chain-of-thought reasoning", "domain knowledge"],
    paper_url="https://arxiv.org/abs/2409.02813",
    approx_num_samples={"test": 1730},
)

# ── CLIP zero-shot-retrieval variant (for CLIP-style models) ──────────────────
# A ZeroShotClassifier can't generate a CoT answer, so it scores the image
# against each option rendered as "question + option" (unibench VQA convention)
# and the best-matching option is graded. CLIP sees a single image and truncates
# the caption to its 77-token context, so this is a weak read on MMMU-Pro — it
# exists for CLIP-family coverage/comparability, not as a strong solver.


def mmmu_pro_clip_prompt(row: dict) -> Doc:
    """CLIP zero-shot-retrieval rendering of one MMMU-Pro 'standard' row."""
    options = _as_list(row["options"])

    # Gather cited images in citation order (CLIP scores the first one) and strip
    # the "<image N>" markers — they are noise to CLIP's text encoder.
    raw_question = row["question"]
    image_order = [int(n) for n in _IMAGE_TOKEN_RE.findall(raw_question)]
    visuals = [row[f"image_{i}"] for i in image_order if row.get(f"image_{i}") is not None]
    if not visuals:  # no inline references — fall back to any attached images
        visuals = [row[f"image_{i}"] for i in range(1, 8) if row.get(f"image_{i}") is not None]
    # Replace "<image N>" with natural language so the caption stays on-distribution
    # for CLIP's text encoder (CLIP sees the image itself, not the marker).
    question = re.sub(r"\s+", " ", _IMAGE_TOKEN_RE.sub("the image", raw_question)).strip()

    answer = str(row["answer"]).strip().upper()
    letters = ascii_uppercase[: len(options)]
    answer_index = letters.index(answer) if answer in letters else 0

    return clip_mcq_doc(
        question=question,
        options=options,
        answer_index=answer_index,
        visuals=visuals,
        task_name="mmmu_pro_clip",
        metadata={
            "subject": row.get("subject", ""),
            "id": row.get("id", ""),
            "options": options,
        },
    )


mmmu_pro_clip_task = MillTaskConfig(
    name="mmmu_pro_clip",
    version=1,
    hf_repo="MMMU/MMMU_Pro",
    hf_subset="standard (10 options)",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    prompt_function=mmmu_pro_clip_prompt,
    task_type=TaskType.ZERO_SHOT_CLASSIFICATION,
    zeroshot_templates=["{c}"],               # identity — caption embedded verbatim
    input_modalities=["image", "text"],       # requires an image-text model (CLIP)
    n_shots=0,
    metrics=[get_metric("acc")],
    description=(
        "MMMU-Pro (standard, 10 options) rendered for CLIP-style models: each option "
        "is scored as 'question + option' against the image by image-text similarity "
        "(unibench convention) and the best-matching option is graded. CLIP sees only "
        "the first image and truncates long questions to its 77-token context."
    ),
    categories=["vision", "multiple-choice", "zero-shot"],
    capabilities=["image-text alignment", "visual recognition"],
    paper_url="https://arxiv.org/abs/2409.02813",
    approx_num_samples={"test": 1730},
)

TASKS_TABLE = [mmmu_pro_task, mmmu_pro_clip_task]

mmmu_pro_benchmark = MillBenchmarkConfig(
    name="mmmu_pro",
    task_names=["mmmu_pro", "mmmu_pro_clip"],
    metric_names=["mmmu_pro_acc"],
    weighted_aggregate=False,
    pick_variant_by_model=True,
    description=(
        "MMMU-Pro (standard, 10 options): robust multi-discipline multimodal "
        "understanding and reasoning, scored by chain-of-thought generation with "
        "answer-letter extraction."
    ),
    categories=["vision", "multiple-choice", "reasoning", "academic", "multitask"],
    capabilities=["visual reasoning", "chain-of-thought reasoning", "domain knowledge"],
    paper_url="https://arxiv.org/abs/2409.02813",
)

BENCHMARKS_TABLE = [mmmu_pro_benchmark]
