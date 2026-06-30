"""MMLU-Pro — TIGER-Lab/MMLU-Pro, generative chain-of-thought evaluation.

A harder successor to MMLU: ~12K questions across 14 categories with up to 10
answer options each. The model reasons step by step and ends with
``Answer: $LETTER``; the answer letter is extracted from the generation and
scored, mirroring lighteval's `mmlu_pro` task.
"""
from mill.api.instance import OutputType
from mill.api.metrics import get_metric
from mill.api.task import MillBenchmarkConfig, MillTaskConfig
from mill.api.taxonomy import TaskType
from mill.tasks.mmlu_pro.utils import mmlu_pro_prompt  # noqa: F401  registers `mmlu_pro_acc`

mmlu_pro_task = MillTaskConfig(
    name="mmlu_pro",
    version=1,
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
    hf_avail_splits=["validation", "test"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    prompt_function=mmlu_pro_prompt,
    task_type=TaskType.MULTIPLE_CHOICE,
    output_type=OutputType.GENERATIVE,  # graded by CoT generation + answer-letter extraction
    generation_size=1024,  # allow for long CoT generations
    n_shots=0,  # zero-shot CoT (the prompt already instructs step-by-step reasoning)
    metrics=[get_metric("mmlu_pro_acc")],
    description=(
        "MMLU-Pro: ~12K harder multiple-choice questions across 14 categories with up "
        "to 10 options each. Scored by chain-of-thought generation; the answer letter "
        "is extracted from the response."
    ),
    categories=["knowledge", "multiple-choice", "reasoning", "academic"],
    capabilities=["chain-of-thought reasoning", "factual recall", "domain knowledge"],
    paper_url="https://arxiv.org/abs/2406.01574",
    approx_num_samples={"validation": 70, "test": 12032},
)

TASKS_TABLE = [mmlu_pro_task]

mmlu_pro_benchmark = MillBenchmarkConfig(
    name="mmlu_pro",
    task_names=["mmlu_pro"],
    metric_names=["mmlu_pro_acc"],
    weighted_aggregate=False,
    description=(
        "MMLU-Pro: harder, 10-option successor to MMLU evaluated with chain-of-thought "
        "generation and answer-letter extraction."
    ),
    categories=["knowledge", "multiple-choice", "reasoning", "academic", "multitask"],
    capabilities=["chain-of-thought reasoning", "factual recall", "domain knowledge"],
    paper_url="https://arxiv.org/abs/2406.01574",
)

BENCHMARKS_TABLE = [mmlu_pro_benchmark]
