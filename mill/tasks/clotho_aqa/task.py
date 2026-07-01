"""Clotho-AQA — crowdsourced audio question answering (single-word answers).

Each sample pairs a 15-30 s audio clip with a question whose answer is a single
word (often yes/no). An audio-language model listens to the clip and answers in
one word; the answer is graded by exact match after lowercasing and stripping
punctuation.

Mirrors the lmms-eval ``clotho_aqa_test`` setup (``clotho_aqa_test_filtered``
split, single-word prompt, case/punctuation-insensitive exact match):
https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/clotho_aqa

The dataset's other rendering (``clotho_asqa_test_v2``) is scored by a GPT-4o
judge and is intentionally not ported here — exact match is reproducible without
an external judge.
"""
from __future__ import annotations

import string

from mill.api.instance import OutputType
from mill.api.metrics import get_metric, register_metric
from mill.api.task import Doc, MillBenchmarkConfig, MillTaskConfig
from mill.api.taxonomy import TaskType

# Official lmms-eval prompt: the bare question followed by this instruction
# (pre_prompt is empty). Audio is prepended to the prompt by the context builder.
_POST_PROMPT = "Answer the question using a single word only."

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, and collapse whitespace (lmms-eval exact_match)."""
    text = text.lower().translate(_PUNCT_TABLE)
    return " ".join(text.split())


def clotho_aqa_prompt(row: dict) -> Doc:
    """Build an audio QA prompt for one Clotho-AQA row."""
    question = (row.get("question") or "").strip()
    return Doc(
        query=f"{question} {_POST_PROMPT}",
        target_index=str(row.get("answer", "")),  # gold single-word answer
        audios=[row["audio"]],                     # HF datasets audio dict
        metadata={"question": question},
        task_name="clotho_aqa",
    )


@register_metric("clotho_aqa_exact_match", higher_is_better=True, output_type=OutputType.GENERATIVE)
def clotho_aqa_exact_match(doc: Doc, response: str) -> float:
    """1.0 if the normalized generation equals the normalized gold answer.

    Faithful to lmms-eval's ``exact_match`` with ``ignore_case`` and
    ``ignore_punctuation`` enabled.
    """
    gold = doc.target_index if isinstance(doc.target_index, str) else ""
    return float(_normalize(response or "") == _normalize(gold))


clotho_aqa_task = MillTaskConfig(
    name="clotho_aqa",
    version=1,
    hf_repo="lmms-lab/ClothoAQA",
    hf_subset="clotho_aqa",
    hf_avail_splits=["clotho_aqa_test_filtered", "clotho_aqa_val_filtered"],
    evaluation_splits=["clotho_aqa_test_filtered"],
    prompt_function=clotho_aqa_prompt,
    task_type=TaskType.GENERATIVE_QA,
    output_type=OutputType.GENERATIVE,
    input_modalities=["audio", "text"],   # requires an audio-language model
    generation_size=8,                    # official max_new_tokens=8 (single word)
    n_shots=0,
    metrics=[get_metric("clotho_aqa_exact_match")],
    description=(
        "Clotho-AQA: crowdsourced audio question answering. Each 15-30 s audio clip "
        "is paired with a question whose answer is a single word (often yes/no); an "
        "audio-language model answers in one word, graded by case/punctuation-"
        "insensitive exact match."
    ),
    categories=["audio", "question-answering"],
    capabilities=["audio understanding", "acoustic reasoning"],
    paper_url="https://arxiv.org/abs/2204.09634",
    approx_num_samples={"clotho_aqa_test_filtered": 1442, "clotho_aqa_val_filtered": 1048},
)

# ── CLAP zero-shot-retrieval variant (for CLAP-style audio-text encoders) ─────
# A ZeroShotClassifier can't answer a question, so it scores the audio against
# each candidate answer rendered as "question + answer" (the clip_mcq_doc /
# unibench VQA convention) and the best-matching option is graded. The
# `clotho_aqa_test_filtered` split is entirely yes/no, so the option set is
# {yes, no}. Like `mmmu_pro_clip`, this is a deliberately weak read — a
# contrastive audio encoder has no notion of the question or of negation — and
# exists for CLAP-family coverage, not as a strong solver (chance ≈ 55.9%,
# the yes-majority rate).

_CLAP_OPTIONS = ["yes", "no"]


def clotho_aqa_clap_prompt(row: dict) -> Doc:
    """CLAP zero-shot-retrieval rendering of one (yes/no) Clotho-AQA row."""
    question = (row.get("question") or "").strip()
    gold = str(row.get("answer", "")).strip().lower()
    answer_index = _CLAP_OPTIONS.index(gold) if gold in _CLAP_OPTIONS else 0
    captions = [f"{question} {opt}".strip() for opt in _CLAP_OPTIONS]
    return Doc(
        query="",
        choices=captions,                 # candidate captions, embedded verbatim
        target_index=answer_index,
        audios=[row["audio"]],
        metadata={"question": question, "options": _CLAP_OPTIONS},
        task_name="clotho_aqa_clap",
    )


clotho_aqa_clap_task = MillTaskConfig(
    name="clotho_aqa_clap",
    version=1,
    hf_repo="lmms-lab/ClothoAQA",
    hf_subset="clotho_aqa",
    hf_avail_splits=["clotho_aqa_test_filtered", "clotho_aqa_val_filtered"],
    evaluation_splits=["clotho_aqa_test_filtered"],
    prompt_function=clotho_aqa_clap_prompt,
    task_type=TaskType.ZERO_SHOT_CLASSIFICATION,
    zeroshot_templates=["{c}"],           # identity — caption embedded verbatim
    input_modalities=["audio"],           # requires an audio-text encoder (CLAP)
    n_shots=0,
    metrics=[get_metric("acc")],
    description=(
        "Clotho-AQA (yes/no) rendered for CLAP-style audio-text encoders: the audio "
        "is scored against each answer as 'question + answer' by audio-text similarity "
        "and the best-matching option is graded. A deliberately weak read for "
        "contrastive-audio coverage, not a strong solver."
    ),
    categories=["audio", "question-answering", "zero-shot"],
    capabilities=["audio-text alignment", "audio understanding"],
    paper_url="https://arxiv.org/abs/2204.09634",
    approx_num_samples={"clotho_aqa_test_filtered": 1442, "clotho_aqa_val_filtered": 1048},
)

TASKS_TABLE = [clotho_aqa_task, clotho_aqa_clap_task]

# One benchmark, two renderings: a generative audio-LM runs the free-form QA task,
# a CLAP-style audio-text encoder runs the zero-shot variant (picked by capability).
clotho_aqa_benchmark = MillBenchmarkConfig(
    name="clotho_aqa",
    task_names=["clotho_aqa", "clotho_aqa_clap"],
    metric_names=["clotho_aqa_exact_match", "acc"],
    weighted_aggregate=False,
    pick_variant_by_model=True,
    description=(
        "Clotho-AQA: crowdsourced audio question answering with single-word answers. "
        "Exact match for generative audio-LMs; a yes/no CLAP zero-shot rendering for "
        "contrastive audio-text encoders."
    ),
    categories=["audio", "question-answering"],
    capabilities=["audio understanding", "acoustic reasoning"],
    paper_url="https://arxiv.org/abs/2204.09634",
)

BENCHMARKS_TABLE = [clotho_aqa_benchmark]
