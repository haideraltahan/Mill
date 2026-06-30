"""Top-level evaluation orchestrator.

Pipeline.run() coordinates:
  1. Resolve requested names to benchmarks and/or leaf tasks
  2. Filter out already-completed jobs (output caching)
  3. Load model
  4. Evaluate each pending leaf task
  5. Aggregate benchmark scores and display results
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mill.api.registry import (
    get_benchmark_config,
    get_task_config,
    list_benchmarks,
    load_tasks_from_path,
)
from mill.api.task import ConfigurableTask, MillBenchmarkConfig, MillTaskConfig
from mill.constants import DEFAULT_SEED, OUTPUT_DIR
from mill.evaluator import evaluate_task
from mill.output import OutputHandler

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        model: Any,  # MillModel instance or config dict
        tasks: list[str],
        output_dir: str | Path = OUTPUT_DIR,
        limit: int | None = None,
        task_paths: list[str] | None = None,
        seed: int = DEFAULT_SEED,
    ):
        self.output_handler = OutputHandler(output_dir=output_dir)
        self.limit = limit
        self.seed = seed

        # ── Discover tasks and benchmarks ─────────────────────────────────────
        if task_paths:
            for p in task_paths:
                load_tasks_from_path(p)

        # Resolve each requested name to a benchmark or a leaf task.
        # Benchmarks take priority when both registries share the same name.
        self._benchmark_configs: list[MillBenchmarkConfig] = []
        self._task_configs: list[MillTaskConfig] = []   # explicit leaf tasks only
        self._requested_names: list[str] = list(tasks)

        for name in tasks:
            try:
                self._benchmark_configs.append(get_benchmark_config(name))
                continue
            except KeyError:
                pass
            try:
                self._task_configs.append(get_task_config(name))
            except KeyError as e:
                logger.error(str(e))

        # ── Resolve model (load lazily — only if there is pending work) ────────
        if isinstance(model, dict):
            from mill.models.loader import model_name_from_config
            self._model = None
            self._model_config: dict | None = model
            self._model_abbr = model_name_from_config(model)
        else:
            self._model = model
            self._model_config = None
            self._model_abbr = model.model_name

    def run(self) -> dict[str, dict]:
        # ── Collect all leaf tasks needed (from benchmarks + explicit tasks) ──
        leaf_configs: list[MillTaskConfig] = list(self._task_configs)
        leaf_names: set[str] = {c.name for c in leaf_configs}
        # Map each leaf task name to its parent benchmark name (empty = standalone)
        task_to_benchmark: dict[str, str] = {}

        # Effective task list per benchmark: all subtasks, or — for
        # variant-selection benchmarks — the single task matching the model.
        bench_tasks: dict[str, list[str]] = {
            bc.name: self._effective_tasks(bc) for bc in self._benchmark_configs
        }

        for bc in self._benchmark_configs:
            for task_name in bench_tasks[bc.name]:
                task_to_benchmark[task_name] = bc.name
                if task_name not in leaf_names:
                    try:
                        leaf_configs.append(get_task_config(task_name))
                        leaf_names.add(task_name)
                    except KeyError as e:
                        logger.error(str(e))

        # ── Run pending leaf tasks ────────────────────────────────────────────
        pending = [
            c for c in leaf_configs
            if not self.output_handler.is_completed(self._model_abbr, c.name, c.n_shots)
        ]
        done = [c for c in leaf_configs if c not in pending]

        results: dict[str, dict] = {}

        for config in done:
            metric_names = [m.name for m in config.metrics]
            results[config.name] = self.output_handler.aggregate(
                self._model_abbr, config.name, config.n_shots, metric_names,
                benchmark=task_to_benchmark.get(config.name, ""),
                task_type=config.task_type.value,
            )

        if pending:
            if self._model is None:
                from mill.models.loader import load_model_from_config
                logger.info("Loading model %s for %d pending task(s)", self._model_abbr, len(pending))
                self._model = load_model_from_config(self._model_config)
            logger.info(
                "Running %d/%d tasks (model=%s)",
                len(pending), len(leaf_configs), self._model_abbr,
            )
            for config in pending:
                task = ConfigurableTask(config, seed=self.seed)
                results[config.name] = evaluate_task(
                    model=self._model,
                    task=task,
                    output_handler=self.output_handler,
                    limit=self.limit,
                    benchmark=task_to_benchmark.get(config.name, ""),
                    seed=self.seed,
                )
        else:
            logger.info("All leaf tasks already completed — loaded results from cache (no model load).")

        if self._model is not None:
            self._model.cleanup()

        # ── Aggregate benchmark scores from leaf task results ─────────────────
        for bc in self._benchmark_configs:
            effective = bench_tasks[bc.name]
            if not effective:
                logger.warning("Benchmark '%s': no task matches the model's capabilities", bc.name)
                results[bc.name] = {}
                continue
            n_shots = self._benchmark_n_shots(bc)
            if len(effective) == 1:
                # Single task (or selected variant): pass through directly
                results[bc.name] = results.get(effective[0], {})
            else:
                results[bc.name] = self.output_handler.aggregate_group(
                    self._model_abbr,
                    bc.name,
                    effective,
                    n_shots,
                    bc.metric_names,
                    weighted=bc.weighted_aggregate,
                    benchmark=bc.name,
                    task_type=get_task_config(effective[0]).task_type.value,
                )

        # Only display names that were explicitly requested
        requested = set(self._requested_names)
        self._display({k: v for k, v in results.items() if k in requested})
        return results

    def _benchmark_n_shots(self, bc: MillBenchmarkConfig) -> int:
        """Infer n_shots from the benchmark's tasks (all tasks share the same value)."""
        for name in bc.task_names:
            try:
                return get_task_config(name).n_shots
            except KeyError:
                continue
        return 0

    def _effective_tasks(self, bc: MillBenchmarkConfig) -> list[str]:
        """Tasks to actually run for a benchmark.

        Normal benchmarks run all ``task_names``. Variant-selection benchmarks
        run only the first task whose ``task_type`` the model supports.
        """
        if not bc.pick_variant_by_model:
            return list(bc.task_names)
        model_task_types = self._model_task_types()
        for name in bc.task_names:
            try:
                if get_task_config(name).task_type in model_task_types:
                    return [name]
            except KeyError as e:
                logger.error(str(e))
        return []

    def _model_task_types(self) -> frozenset:
        """Task types the model supports — from the instance, or its config class."""
        if self._model is not None:
            return self._model.supported_task_types
        from mill.api.model import class_supported_task_types
        from mill.models.loader import resolve_model_class
        return class_supported_task_types(resolve_model_class(self._model_config))

    def _display(self, results: dict[str, dict]) -> None:
        if not results:
            return
        print(f"\n{'Task':<35} {'Score':>10}")
        print("-" * 47)
        for task_name, metrics in results.items():
            main = next(
                (v for k, v in metrics.items()
                 if not k.endswith("_stderr") and isinstance(v, (int, float)) and k not in ("n_shot",)),
                None,
            )
            if main is not None:
                print(f"{task_name:<35} {main:>10.4f}")
        print()
