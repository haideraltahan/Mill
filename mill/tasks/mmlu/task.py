"""MMLU — Massive Multitask Language Understanding (57 subjects, 4-choice MCQ).

Uses log-likelihood scoring: the model scores each choice token and picks the
highest log-prob answer.
"""
from mill.api.instance import OutputType
from mill.api.metrics import get_metric
from mill.api.task import MillBenchmarkConfig, MillTaskConfig
from mill.tasks.mmlu.utils import mmlu_prompt

# All 57 MMLU subjects
_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy",
    "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]


def _make_mmlu_task(subject: str) -> MillTaskConfig:
    display = subject.replace("_", " ").title()
    _prompt = lambda row, s=subject: mmlu_prompt(row, subject=s)
    return MillTaskConfig(
        name=f"mmlu_{subject}",
        version=1,
        hf_repo="cais/mmlu",
        hf_subset=subject,
        hf_avail_splits=["dev", "validation", "test"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        prompt_function=_prompt,
        output_type=OutputType.LOGPROBS,
        n_shots=5,
        metrics=[get_metric("acc")],
        description=f"MMLU — {display}: 4-choice multiple-choice questions testing academic knowledge.",
        categories=["knowledge", "multiple-choice", "academic"],
        capabilities=["factual recall", "domain knowledge", "reading comprehension"],
        paper_url="https://arxiv.org/abs/2009.03300",
        approx_num_samples={"dev": 5, "validation": 11, "test": 100},
    )


TASKS_TABLE = [_make_mmlu_task(s) for s in _SUBJECTS]

mmlu_benchmark = MillBenchmarkConfig(
    name="mmlu",
    task_names=[f"mmlu_{s}" for s in _SUBJECTS],
    metric_names=["acc"],
    weighted_aggregate=False,   # unweighted mean of 57 subject accuracies
    description=(
        "MMLU: 57-subject multiple-choice benchmark. Runs each subject task "
        "independently and reports the unweighted mean accuracy across subjects."
    ),
    categories=["knowledge", "multiple-choice", "academic", "multitask"],
    capabilities=["factual recall", "domain knowledge", "reading comprehension", "multitask generalization"],
    paper_url="https://arxiv.org/abs/2009.03300",
)

BENCHMARKS_TABLE = [mmlu_benchmark]
