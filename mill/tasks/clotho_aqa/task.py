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

TASKS_TABLE = [clotho_aqa_task]

clotho_aqa_benchmark = MillBenchmarkConfig(
    name="clotho_aqa",
    task_names=["clotho_aqa"],
    metric_names=["clotho_aqa_exact_match"],
    weighted_aggregate=False,
    description=(
        "Clotho-AQA: crowdsourced audio question answering with single-word answers, "
        "scored by case/punctuation-insensitive exact match."
    ),
    categories=["audio", "question-answering"],
    capabilities=["audio understanding", "acoustic reasoning"],
    paper_url="https://arxiv.org/abs/2204.09634",
)

BENCHMARKS_TABLE = [clotho_aqa_benchmark]
