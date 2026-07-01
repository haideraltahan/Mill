"""ESC-50 — environmental sound classification (50 classes).

Two renderings of the same 2,000-clip test set, auto-selected per model:

- ``esc50`` — **zero-shot** for CLAP-style audio-text encoders: the audio is
  scored against the 50 class names by audio-text similarity (the standard CLAP
  zero-shot setup, prompt ``"This is a sound of {c}."``).
- ``esc50_generative`` — a generative rendering for audio-language models: the
  model hears the clip and the 50 candidate category names and answers with the
  category name, which is matched back to a class.

Dataset: ``ashraq/esc50`` (Piczak, 2015 — https://github.com/karolpiczak/ESC-50).
ESC-50 has no train/test split for zero-shot use (the encoder isn't trained), so
all 2,000 clips are scored.
"""
from __future__ import annotations

from mill.api.instance import OutputType
from mill.api.metrics import get_metric, register_metric
from mill.api.task import Doc, MillBenchmarkConfig, MillTaskConfig
from mill.api.taxonomy import TaskType

# The 50 ESC-50 categories in ``target`` order (0-49), verbatim from the dataset's
# ``category`` field. ``choices[target]`` is therefore the gold class.
ESC50_CATEGORIES = [
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",
    "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops",
    "wind", "pouring_water", "toilet_flush", "thunderstorm", "crying_baby", "sneezing",
    "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing_teeth",
    "snoring", "drinking_sipping", "door_wood_knock", "mouse_click", "keyboard_typing",
    "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm",
    "clock_tick", "glass_breaking", "helicopter", "chainsaw", "siren", "car_horn",
    "engine", "train", "church_bells", "airplane", "fireworks", "hand_saw",
]

# Human-readable class names (underscores -> spaces) used in prompts and captions.
ESC50_CLASSNAMES = [c.replace("_", " ") for c in ESC50_CATEGORIES]


def _norm(text) -> str:
    """Lowercase, underscores->spaces, collapse whitespace."""
    return " ".join(str(text).lower().replace("_", " ").split())


# ── Zero-shot variant (for CLAP-style audio-text encoders) ────────────────────


def esc50_prompt(row: dict) -> Doc:
    """Zero-shot rendering: score the audio against the 50 class names."""
    return Doc(
        query="",
        choices=ESC50_CLASSNAMES,
        target_index=int(row["target"]),
        audios=[row["audio"]],
        task_name="esc50",
    )


esc50_task = MillTaskConfig(
    name="esc50",
    version=1,
    hf_repo="ashraq/esc50",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    prompt_function=esc50_prompt,
    task_type=TaskType.ZERO_SHOT_CLASSIFICATION,
    zeroshot_templates=["This is a sound of {c}."],   # standard CLAP ESC-50 prompt
    input_modalities=["audio"],                       # requires an audio-text encoder (CLAP)
    n_shots=0,
    metrics=[get_metric("acc")],
    description=(
        "ESC-50 zero-shot environmental sound classification: 50 classes, 2,000 clips. "
        "Scored by CLAP-style audio-text similarity with the 'This is a sound of {c}.' "
        "prompt."
    ),
    categories=["audio", "audio-classification", "zero-shot"],
    capabilities=["audio-text alignment", "audio recognition"],
    paper_url="https://github.com/karolpiczak/ESC-50",
    approx_num_samples={"train": 2000},
)

# ── Generative variant (for audio-language models) ────────────────────────────

_GEN_INSTRUCTION = (
    "Listen to the audio and identify the sound. Choose exactly one category from "
    "the list and answer with the category name only.\nCategories: "
    + ", ".join(ESC50_CLASSNAMES)
    + "."
)


def esc50_generative_prompt(row: dict) -> Doc:
    """Generative rendering: the model answers with a category name."""
    gold = ESC50_CLASSNAMES[int(row["target"])]
    return Doc(
        query=_GEN_INSTRUCTION,
        target_index=gold,              # gold class name (string)
        audios=[row["audio"]],
        metadata={"category": gold},
        task_name="esc50_generative",
    )


@register_metric("esc50_gen_acc", higher_is_better=True, output_type=OutputType.GENERATIVE)
def esc50_gen_acc(doc: Doc, response: str) -> float:
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
    hits = [c for c in (_norm(x) for x in ESC50_CLASSNAMES) if c and c in resp]
    if hits:
        return float(max(hits, key=len) == gold)
    return 0.0


esc50_generative_task = MillTaskConfig(
    name="esc50_generative",
    version=1,
    hf_repo="ashraq/esc50",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    prompt_function=esc50_generative_prompt,
    task_type=TaskType.GENERATIVE_QA,
    output_type=OutputType.GENERATIVE,
    input_modalities=["audio", "text"],   # requires an audio-language model
    generation_size=32,
    n_shots=0,
    metrics=[get_metric("esc50_gen_acc")],
    description=(
        "ESC-50 generative environmental sound classification: an audio-language "
        "model hears the clip and answers with one of 50 category names, matched "
        "back to a class."
    ),
    categories=["audio", "audio-classification", "question-answering"],
    capabilities=["audio understanding", "audio recognition"],
    paper_url="https://github.com/karolpiczak/ESC-50",
    approx_num_samples={"train": 2000},
)

TASKS_TABLE = [esc50_task, esc50_generative_task]

# One benchmark, two renderings: a CLAP-style encoder runs the zero-shot task,
# an audio-language model runs the generative task (picked by capability).
esc50_benchmark = MillBenchmarkConfig(
    name="esc50",
    task_names=["esc50", "esc50_generative"],
    metric_names=["acc", "esc50_gen_acc"],
    weighted_aggregate=False,
    pick_variant_by_model=True,
    description=(
        "ESC-50 environmental sound classification top-1 accuracy: zero-shot "
        "audio-text similarity for CLAP-style encoders, generative category naming "
        "for audio-language models."
    ),
    categories=["audio", "audio-classification"],
    capabilities=["audio recognition"],
    paper_url="https://github.com/karolpiczak/ESC-50",
)

BENCHMARKS_TABLE = [esc50_benchmark]
