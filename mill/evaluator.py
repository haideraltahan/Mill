"""Core evaluation loop: build requests → dispatch to model → compute metrics."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from mill.api.instance import OutputType

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
) -> dict:
    """Evaluate one task and write per-sample results to the output handler.

    Returns a dict of aggregated metric scores.
    """
    model_name = model.model_name
    task_name = task.name
    n_shot = task.config.n_shots

    if output_handler.is_completed(model_name, task_name, n_shot):
        logger.info(f"Cached {task_name} — loading samples for {model_name} (n_shot={n_shot}), no recompute")
        metric_names = [m.name for m in task.config.metrics]
        return output_handler.aggregate(model_name, task_name, n_shot, metric_names, benchmark=benchmark)

    logger.info(f"Evaluating {task_name} | model={model_name} | n_shot={n_shot}")
    task.download(limit=limit)
    instances = task.build_all_requests()
    if not instances:
        logger.warning(f"No instances for {task_name}")
        return {}

    # ── Dispatch by output type ───────────────────────────────────────────────
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
    aggregated = output_handler.aggregate(model_name, task_name, n_shot, metric_names, benchmark=benchmark)
    logger.info(f"  {task_name}: {aggregated}")
    return aggregated
