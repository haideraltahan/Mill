"""CIFAR-10 zero-shot image classification.

Uses the WebDataset export at ``haideraltahan/wds_cifar10`` (clip_benchmark
format): each sample carries a 32x32 image (column ``0.webp``) and an integer
class label (column ``cls``). A zero-shot classifier (e.g. CLIP) scores the
image against the 10 class names, ensembling the prompt templates per class.

Class names and templates are copied verbatim from the dataset's
``classnames.txt`` and ``zeroshot_classification_templates.txt``.
"""
from string import ascii_uppercase

from mill.api.instance import OutputType
from mill.api.metrics import get_metric, register_metric
from mill.api.task import Doc, MillBenchmarkConfig, MillTaskConfig
from mill.api.taxonomy import TaskType
from mill.utils import parse_multi_choice_response, sample_rng

CIFAR10_CLASSNAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR10_TEMPLATES = [
    "a photo of a {c}.",
    "a blurry photo of a {c}.",
    "a black and white photo of a {c}.",
    "a low contrast photo of a {c}.",
    "a high contrast photo of a {c}.",
    "a bad photo of a {c}.",
    "a good photo of a {c}.",
    "a photo of a small {c}.",
    "a photo of a big {c}.",
    "a photo of the {c}.",
    "a blurry photo of the {c}.",
    "a black and white photo of the {c}.",
    "a low contrast photo of the {c}.",
    "a high contrast photo of the {c}.",
    "a bad photo of the {c}.",
    "a good photo of the {c}.",
    "a photo of the small {c}.",
    "a photo of the big {c}.",
]


def cifar10_prompt(row: dict) -> Doc:
    return Doc(
        query="",
        choices=CIFAR10_CLASSNAMES,
        target_index=int(row["cls"]),
        visuals=[row["0.webp"]],
        task_name="cifar10",
    )


cifar10_task = MillTaskConfig(
    name="cifar10",
    version=1,
    hf_repo="haideraltahan/wds_cifar10",
    hf_builder="webdataset",
    hf_data_files={"test": "hf://datasets/haideraltahan/wds_cifar10/test/*.tar"},
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    prompt_function=cifar10_prompt,
    task_type=TaskType.ZERO_SHOT_CLASSIFICATION,
    zeroshot_templates=CIFAR10_TEMPLATES,
    n_shots=0,
    metrics=[get_metric("acc")],
    description=(
        "CIFAR-10 zero-shot image classification: 10 classes, 10K test images. "
        "Scored by CLIP-style image-text similarity with prompt-template ensembling."
    ),
    categories=["vision", "image-classification", "zero-shot"],
    capabilities=["visual recognition", "image-text alignment"],
    paper_url="https://www.cs.toronto.edu/~kriz/cifar.html",
    approx_num_samples={"test": 10000},
)

# ── Generative multiple-choice variant (for vision-language models) ───────────
# Instead of CLIP-style similarity, a VLM is shown the image and the 10 classes
# as lettered options and asked to answer with a letter; the letter is parsed
# from its generation and graded (mirrors the mmlu_pro setup, with an image).

_LETTERS = ascii_uppercase[:10]  # A..J


def cifar10_mcq_prompt(row: dict) -> Doc:
    cls = int(row["cls"])
    # Shuffle the class->letter assignment per image so the gold answer isn't
    # pinned to a fixed letter (e.g. "cat" always being "D"), which a model could
    # exploit. The permutation is keyed by the sample's dataset id, so it is
    # reproducible under a fixed --seed and independent of iteration order.
    sample_id = row.get("__key__", cls)
    order = list(range(len(CIFAR10_CLASSNAMES)))
    sample_rng("cifar10_mcq", sample_id).shuffle(order)
    options = [CIFAR10_CLASSNAMES[i] for i in order]
    gold_letter = _LETTERS[order.index(cls)]

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
        metadata={"options": options},   # the shuffled order, so the metric can grade
        task_name="cifar10_mcq",
    )


@register_metric("cifar10_mcq_acc", higher_is_better=True, output_type=OutputType.GENERATIVE)
def cifar10_mcq_acc(doc: Doc, response: str) -> float:
    """1.0 if the letter parsed from the response matches the gold letter."""
    options = doc.metadata.get("options") or CIFAR10_CLASSNAMES
    pred = parse_multi_choice_response(response, options)
    return float(pred == str(doc.target_index).strip().upper())


cifar10_mcq_task = MillTaskConfig(
    name="cifar10_mcq",
    version=1,
    hf_repo="haideraltahan/wds_cifar10",
    hf_builder="webdataset",
    hf_data_files={"test": "hf://datasets/haideraltahan/wds_cifar10/test/*.tar"},
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    prompt_function=cifar10_mcq_prompt,
    task_type=TaskType.MULTIPLE_CHOICE,
    output_type=OutputType.GENERATIVE,        # graded by parsing the answer letter
    input_modalities=["image", "text"],       # requires a vision-language model
    generation_size=32,
    n_shots=0,
    metrics=[get_metric("cifar10_mcq_acc")],
    description=(
        "CIFAR-10 generative multiple-choice classification: a vision-language model "
        "sees the image and 10 lettered class options and answers with a letter."
    ),
    categories=["vision", "image-classification", "multiple-choice"],
    capabilities=["visual recognition", "instruction following"],
    paper_url="https://www.cs.toronto.edu/~kriz/cifar.html",
    approx_num_samples={"test": 10000},
)

TASKS_TABLE = [cifar10_task, cifar10_mcq_task]

# One benchmark, two renderings: the pipeline runs the zero-shot task for CLIP-style
# models and the generative-MCQ task for vision-language models, picking by capability.
cifar10_benchmark = MillBenchmarkConfig(
    name="cifar10",
    task_names=["cifar10", "cifar10_mcq"],
    metric_names=["acc"],
    pick_variant_by_model=True,
    description="CIFAR-10 top-1 accuracy (zero-shot for CLIP, generative MCQ for VLMs).",
    categories=["vision", "image-classification"],
    capabilities=["visual recognition"],
    paper_url="https://www.cs.toronto.edu/~kriz/cifar.html",
)

BENCHMARKS_TABLE = [cifar10_benchmark]
