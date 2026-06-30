"""ImageNet-1k zero-shot image classification.

Uses the WebDataset export at ``haideraltahan/wds_imagenet1k`` (clip_benchmark
format): each sample carries an image (column ``0.webp``) and an integer class
label (column ``cls``) in ``[0, 1000)``. A zero-shot classifier (e.g. CLIP)
scores the image against the 1000 class names, ensembling the prompt templates
per class.

Class names and templates are copied verbatim from the dataset's
``classnames.txt`` (1000 classes) and ``zeroshot_classification_templates.txt``
(80 OpenAI CLIP templates); see ``classnames.py``.

Mirrors the two-rendering protocol of the CIFAR-10 task: one benchmark, run as
CLIP-style zero-shot classification for image-text models and as a generative
lettered-MCQ for vision-language models.
"""
from string import ascii_uppercase

from mill.api.instance import OutputType
from mill.api.metrics import get_metric, register_metric
from mill.api.task import Doc, MillBenchmarkConfig, MillTaskConfig
from mill.api.taxonomy import TaskType
from mill.tasks.imagenet.classnames import IMAGENET_CLASSNAMES, IMAGENET_TEMPLATES
from mill.utils import parse_multi_choice_response, sample_rng


def imagenet_prompt(row: dict) -> Doc:
    return Doc(
        query="",
        choices=IMAGENET_CLASSNAMES,
        target_index=int(row["cls"]),
        visuals=[row["0.webp"]],
        task_name="imagenet",
    )


imagenet_task = MillTaskConfig(
    name="imagenet",
    version=1,
    hf_repo="haideraltahan/wds_imagenet1k",
    hf_builder="webdataset",
    hf_data_files={"test": "hf://datasets/haideraltahan/wds_imagenet1k/test/*.tar"},
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    prompt_function=imagenet_prompt,
    task_type=TaskType.ZERO_SHOT_CLASSIFICATION,
    zeroshot_templates=IMAGENET_TEMPLATES,
    n_shots=0,
    metrics=[get_metric("acc")],
    description=(
        "ImageNet-1k zero-shot image classification: 1000 classes, 50K validation "
        "images. Scored by CLIP-style image-text similarity with prompt-template "
        "ensembling."
    ),
    categories=["vision", "image-classification", "zero-shot"],
    capabilities=["visual recognition", "image-text alignment"],
    paper_url="https://www.image-net.org/",
    approx_num_samples={"test": 50000},
)

# ── Generative multiple-choice variant (for vision-language models) ───────────
# A VLM can't be shown all 1000 classes as lettered options, so each image gets a
# 10-way question: the true class plus 9 random distractor classes, shuffled and
# lettered A–J (mirroring the cifar10_mcq rendering). The model answers with a
# letter, which is parsed from its generation and graded. Both the distractor
# draw and the shuffle are keyed by the sample's id, so the question is identical
# under a fixed --seed and independent of iteration order.

_MCQ_NUM_OPTIONS = 10
_LETTERS = ascii_uppercase[:_MCQ_NUM_OPTIONS]  # A..J


def imagenet_mcq_prompt(row: dict) -> Doc:
    cls = int(row["cls"])
    sample_id = row.get("__key__", cls)
    rng = sample_rng("imagenet_mcq", sample_id)

    distractors = rng.sample(
        [i for i in range(len(IMAGENET_CLASSNAMES)) if i != cls],
        _MCQ_NUM_OPTIONS - 1,
    )
    option_indices = distractors + [cls]
    rng.shuffle(option_indices)
    options = [IMAGENET_CLASSNAMES[i] for i in option_indices]
    gold_letter = _LETTERS[option_indices.index(cls)]

    options_block = "\n".join(f"{letter}) {name}" for letter, name in zip(_LETTERS, options))
    query = (
        "Which of these categories best describes the image?\n"
        f"{options_block}\n"
        "Answer with the letter of the correct option."
    )
    return Doc(
        query=query,
        choices=list(_LETTERS),
        target_index=gold_letter,
        visuals=[row["0.webp"]],
        metadata={"options": options},   # the sampled, shuffled order, so the metric can grade
        task_name="imagenet_mcq",
    )


@register_metric("imagenet_mcq_acc", higher_is_better=True, output_type=OutputType.GENERATIVE)
def imagenet_mcq_acc(doc: Doc, response: str) -> float:
    """1.0 if the letter parsed from the response matches the gold letter."""
    options = doc.metadata.get("options") or []
    if not options:
        return 0.0
    pred = parse_multi_choice_response(response, options)
    return float(pred == str(doc.target_index).strip().upper())


imagenet_mcq_task = MillTaskConfig(
    name="imagenet_mcq",
    version=1,
    hf_repo="haideraltahan/wds_imagenet1k",
    hf_builder="webdataset",
    hf_data_files={"test": "hf://datasets/haideraltahan/wds_imagenet1k/test/*.tar"},
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    prompt_function=imagenet_mcq_prompt,
    task_type=TaskType.MULTIPLE_CHOICE,
    output_type=OutputType.GENERATIVE,        # graded by parsing the answer letter
    input_modalities=["image", "text"],       # requires a vision-language model
    generation_size=32,
    n_shots=0,
    metrics=[get_metric("imagenet_mcq_acc")],
    description=(
        "ImageNet-1k generative multiple-choice classification: a vision-language "
        "model sees the image and 10 lettered class options (the true class plus 9 "
        "random distractors) and answers with a letter."
    ),
    categories=["vision", "image-classification", "multiple-choice"],
    capabilities=["visual recognition", "instruction following"],
    paper_url="https://www.image-net.org/",
    approx_num_samples={"test": 50000},
)

TASKS_TABLE = [imagenet_task, imagenet_mcq_task]

# One benchmark, two renderings: the pipeline runs the zero-shot task for CLIP-style
# models and the generative-MCQ task for vision-language models, picking by capability.
imagenet_benchmark = MillBenchmarkConfig(
    name="imagenet",
    task_names=["imagenet", "imagenet_mcq"],
    metric_names=["acc"],
    pick_variant_by_model=True,
    description="ImageNet-1k top-1 accuracy (zero-shot for CLIP, generative MCQ for VLMs).",
    categories=["vision", "image-classification"],
    capabilities=["visual recognition"],
    paper_url="https://www.image-net.org/",
)

BENCHMARKS_TABLE = [imagenet_benchmark]
