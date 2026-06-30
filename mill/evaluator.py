"""Core evaluation loop: build requests → dispatch to model → compute metrics."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from mill.api.instance import OutputType
from mill.api.model import ensure_supported
from mill.api.taxonomy import GENERATIVE_TASK_TYPES, TaskType
from mill.constants import DEFAULT_SEED
from mill.utils import set_global_seed

if TYPE_CHECKING:
    from mill.api.model import MillModel
    from mill.api.task import MillTask
    from mill.output import OutputHandler

logger = logging.getLogger(__name__)


def evaluate_task(
    model: "MillModel",
    task: "MillTask",
    output_handler: "OutputHandler",
    limit: int | None = None,
    benchmark: str = "",
    seed: int = DEFAULT_SEED,
) -> dict:
    """Evaluate one task and write per-sample results to the output handler.

    ``seed`` makes the run reproducible: it seeds the stdlib/numpy/torch RNGs and
    every per-sample draw (option shuffles, few-shot sampling, random-guess
    fallbacks). Set once here, before any data is built, so it applies uniformly
    across all benchmarks.

    Returns a dict of aggregated metric scores.
    """
    set_global_seed(seed)
    task.seed = seed
    model_name = model.model_name
    task_name = task.name
    n_shot = task.config.n_shots

    if output_handler.is_completed(model_name, task_name, n_shot):
        logger.info(f"Cached {task_name} — loading samples for {model_name} (n_shot={n_shot}), no recompute")
        metric_names = [m.name for m in task.config.metrics]
        return output_handler.aggregate(
            model_name, task_name, n_shot, metric_names,
            benchmark=benchmark, task_type=task.config.task_type.value,
        )

    # Fail fast (before downloading data) if the model can't serve this task.
    task_type = task.config.task_type
    ensure_supported(model, task_type, task.config.output_type, task.config.input_modalities)

    logger.info(f"Evaluating {task_name} | model={model_name} | n_shot={n_shot}")
    task.download(limit=limit)
    instances = task.build_all_requests()
    if not instances:
        logger.warning(f"No instances for {task_name}")
        return {}

    # ── Dispatch by task type ─────────────────────────────────────────────────
    if task_type in GENERATIVE_TASK_TYPES:
        _run_generative(model, instances)
    elif task_type == TaskType.ZERO_SHOT_CLASSIFICATION:
        for inst, resp in zip(instances, model.zero_shot_classify(instances)):
            inst.resps = [resp]
    elif task_type == TaskType.SUPERVISED_CLASSIFICATION:
        for inst, resp in zip(instances, model.classify(instances)):
            inst.resps = [resp]

    # ── Group instances by doc (LOGPROBS tasks produce N instances per doc) ───
    doc_instances: dict[int, list] = defaultdict(list)
    for inst in instances:
        doc_id = inst.metadata.get("doc_id", inst.idx)
        doc_instances[doc_id].append(inst)

    # ── Compute per-sample metrics and accumulate ─────────────────────────────
    all_results: list[dict] = []
    for doc_id, insts in doc_instances.items():
        doc = insts[0].doc
        responses = [r for inst in insts for r in inst.resps]
        metrics = task.process_results(doc, responses)
        sample_row = {
            **doc.metadata,
            "model": model_name,
            "task": task_name,
            "n_shot": n_shot,
            "doc_id": doc_id,
            "split": insts[0].metadata.get("split", ""),
            "prediction": responses[0] if responses else "",
            "gold": doc.target_index,
            **metrics,
        }
        output_handler.add_sample(**sample_row)
        all_results.append(metrics)

    output_handler.flush(model_name, task_name, n_shot)

    metric_names = [m.name for m in task.config.metrics]
    aggregated = output_handler.aggregate(
        model_name, task_name, n_shot, metric_names,
        benchmark=benchmark, task_type=task_type.value,
    )
    logger.info(f"  {task_name}: {aggregated}")
    return aggregated


def _run_generative(model: "MillModel", instances: list) -> None:
    """Dispatch generative-family instances by their OutputType scoring method."""
    by_type: dict[OutputType, list] = defaultdict(list)
    for inst in instances:
        by_type[inst.request_type].append(inst)

    if OutputType.GENERATIVE in by_type:
        responses = model.generate_until(by_type[OutputType.GENERATIVE])
        for inst, resp in zip(by_type[OutputType.GENERATIVE], responses):
            inst.resps = [resp]

    if OutputType.LOGPROBS in by_type:
        responses = model.loglikelihood(by_type[OutputType.LOGPROBS])
        for inst, resp in zip(by_type[OutputType.LOGPROBS], responses):
            inst.resps = [resp]

    if OutputType.PERPLEXITY in by_type:
        responses = model.loglikelihood_rolling(by_type[OutputType.PERPLEXITY])
        for inst, resp in zip(by_type[OutputType.PERPLEXITY], responses):
            inst.resps = [resp]
