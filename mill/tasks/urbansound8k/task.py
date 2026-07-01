"""UrbanSound8K — urban sound classification (10 classes).

Two renderings of the same 8,732-clip set, auto-selected per model:

- ``urbansound8k`` — **zero-shot** for CLAP-style audio-text encoders: each clip is
  scored against the 10 class names by audio-text similarity (the standard CLAP
  zero-shot setup, prompt ``"This is a sound of {c}."``).
- ``urbansound8k_generative`` — a generative rendering for audio-language models:
  the model hears the clip and the 10 candidate category names and answers with the
  category name, which is matched back to a class.

Dataset: ``danavery/urbansound8K`` (Salamon et al., 2014 —
https://urbansounddataset.weebly.com/urbansound8k.html). UrbanSound8K ships as 10
folds for supervised cross-validation, but zero-shot needs no training split, so
all 8,732 clips are scored (matching how CLAP-family papers report US8K zero-shot).
"""
from __future__ import annotations

from mill.api.instance import OutputType
from mill.api.metrics import get_metric, register_metric
from mill.api.task import Doc, MillBenchmarkConfig, MillTaskConfig
from mill.api.taxonomy import TaskType

# The 10 UrbanSound8K categories in ``classID`` order (0-9), verbatim from the
# dataset's ``class`` field. ``choices[classID]`` is therefore the gold class.
US8K_CATEGORIES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]

# Human-readable class names (underscores -> spaces) used in prompts and captions.
US8K_CLASSNAMES = [c.replace("_", " ") for c in US8K_CATEGORIES]


def _norm(text) -> str:
    """Lowercase, underscores->spaces, collapse whitespace."""
    return " ".join(str(text).lower().replace("_", " ").split())


# ── Zero-shot variant (for CLAP-style audio-text encoders) ────────────────────


def urbansound8k_prompt(row: dict) -> Doc:
    """Zero-shot rendering: score the audio against the 10 class names."""
    return Doc(
        query="",
        choices=US8K_CLASSNAMES,
        target_index=int(row["classID"]),
        audios=[row["audio"]],
        task_name="urbansound8k",
    )


urbansound8k_task = MillTaskConfig(
    name="urbansound8k",
    version=1,
    hf_repo="danavery/urbansound8K",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    prompt_function=urbansound8k_prompt,
    task_type=TaskType.ZERO_SHOT_CLASSIFICATION,
    zeroshot_templates=["This is a sound of {c}."],   # standard CLAP zero-shot prompt
    input_modalities=["audio"],                       # requires an audio-text encoder (CLAP)
    n_shots=0,
    metrics=[get_metric("acc")],
    description=(
        "UrbanSound8K zero-shot urban sound classification: 10 classes, 8,732 clips. "
        "Scored by CLAP-style audio-text similarity with the 'This is a sound of {c}.' "
        "prompt."
    ),
    categories=["audio", "audio-classification", "zero-shot"],
    capabilities=["audio-text alignment", "audio recognition"],
    paper_url="https://urbansounddataset.weebly.com/urbansound8k.html",
    approx_num_samples={"train": 8732},
)

# ── Generative variant (for audio-language models) ────────────────────────────

_GEN_INSTRUCTION = (
    "Listen to the audio and identify the sound. Choose exactly one category from "
    "the list and answer with the category name only.\nCategories: "
    + ", ".join(US8K_CLASSNAMES)
    + "."
)


def urbansound8k_generative_prompt(row: dict) -> Doc:
    """Generative rendering: the model answers with a category name."""
    gold = US8K_CLASSNAMES[int(row["classID"])]
    return Doc(
        query=_GEN_INSTRUCTION,
        target_index=gold,              # gold class name (string)
        audios=[row["audio"]],
        metadata={"category": gold},
        task_name="urbansound8k_generative",
    )


@register_metric("urbansound8k_gen_acc", higher_is_better=True, output_type=OutputType.GENERATIVE)
def urbansound8k_gen_acc(doc: Doc, response: str) -> float:
    """1.0 if the response names the gold class.

    Exact normalized match first; otherwise the longest class name that appears
    as a substring of the response wins (so a verbose answer still resolves to a
    class). Unmatched answers score 0.
    """
    gold = _norm(doc.target_index)
    resp = _norm(response)
    if not resp:
        return 0.0
    if resp == gold:
        return 1.0
    hits = [c for c in (_norm(x) for x in US8K_CLASSNAMES) if c and c in resp]
    if hits:
        return float(max(hits, key=len) == gold)
    return 0.0


urbansound8k_generative_task = MillTaskConfig(
    name="urbansound8k_generative",
    version=1,
    hf_repo="danavery/urbansound8K",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    prompt_function=urbansound8k_generative_prompt,
    task_type=TaskType.GENERATIVE_QA,
    output_type=OutputType.GENERATIVE,
    input_modalities=["audio", "text"],   # requires an audio-language model
    generation_size=32,
    n_shots=0,
    metrics=[get_metric("urbansound8k_gen_acc")],
    description=(
        "UrbanSound8K generative urban sound classification: an audio-language "
        "model hears the clip and answers with one of 10 category names, matched "
        "back to a class."
    ),
    categories=["audio", "audio-classification", "question-answering"],
    capabilities=["audio understanding", "audio recognition"],
    paper_url="https://urbansounddataset.weebly.com/urbansound8k.html",
    approx_num_samples={"train": 8732},
)

TASKS_TABLE = [urbansound8k_task, urbansound8k_generative_task]

# One benchmark, two renderings: a CLAP-style encoder runs the zero-shot task,
# an audio-language model runs the generative task (picked by capability).
urbansound8k_benchmark = MillBenchmarkConfig(
    name="urbansound8k",
    task_names=["urbansound8k", "urbansound8k_generative"],
    metric_names=["acc", "urbansound8k_gen_acc"],
    weighted_aggregate=False,
    pick_variant_by_model=True,
    description=(
        "UrbanSound8K urban sound classification top-1 accuracy: zero-shot "
        "audio-text similarity for CLAP-style encoders, generative category naming "
        "for audio-language models."
    ),
    categories=["audio", "audio-classification"],
    capabilities=["audio recognition"],
    paper_url="https://urbansounddataset.weebly.com/urbansound8k.html",
)

BENCHMARKS_TABLE = [urbansound8k_benchmark]
