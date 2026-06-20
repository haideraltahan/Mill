"""Feather-based output caching layer.

Adapted from unibench/unibench/output.py.

Core guarantee: never recompute a (model, task, n_shot) tuple that already
has results in the cache directory.

Cache layout::

    {output_dir}/
      outputs/
        {model_abbr}/
          {task_name}_{n_shot}shot.f    # Per-sample results
      aggregate.csv                     # Long-format summary, one row per
                                        # (model, task, n_shot, metric):
                                        #   model,benchmark,task,n_shot,
                                        #   metric,performance,stderr
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import pandas as pd
from oslo_concurrency import lockutils

from mill.constants import DEFAULT_ROUND_VALUES, LOCK_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


def _stderr_of_mean(values: "pd.Series") -> float:
    """Standard error of the mean (std / sqrt(n)).

    Returns 0.0 for fewer than two values, where the spread is undefined.
    Works for any metric since it operates on the per-sample (or per-subtask)
    scores directly.
    """
    n = len(values)
    if n < 2:
        return 0.0
    std = float(values.std(ddof=1))
    if math.isnan(std):
        return 0.0
    return std / math.sqrt(n)


class OutputHandler:
    def __init__(
        self,
        output_dir: str | Path = OUTPUT_DIR,
        round_values: int = DEFAULT_ROUND_VALUES,
    ):
        self.output_dir = Path(output_dir)
        self.outputs_dir = self.output_dir / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.round_values = round_values
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        lockutils.set_defaults(lock_path=str(LOCK_DIR))
        self._local: list[dict] = []
        self._aggregate: pd.DataFrame = self._load_aggregate()

    # ── Completion check ──────────────────────────────────────────────────────

    def is_completed(self, model: str, task: str, n_shot: int) -> bool:
        """Return True if this (model, task, n_shot) was already evaluated.

        The per-sample ``.f`` file is the source of truth: samples are written in
        a single flush at the end of a run, so its presence means inference is
        done and the aggregate can be rebuilt from it without recomputing. Falls
        back to an existing ``aggregate.csv`` row when the samples file is gone.
        """
        if self._sample_path(model, task, n_shot).exists():
            return True
        self._aggregate = self._load_aggregate()
        if self._aggregate.empty or "model" not in self._aggregate.columns:
            return False
        mask = (
            (self._aggregate["model"] == model)
            & (self._aggregate["task"] == task)
            & (self._aggregate["n_shot"] == n_shot)
        )
        return bool(mask.any())

    def filter_pending(
        self, model: str, tasks: list[str], n_shot: int
    ) -> list[str]:
        """Return only tasks that have not been computed yet."""
        return [t for t in tasks if not self.is_completed(model, t, n_shot)]

    # ── Sample accumulation ───────────────────────────────────────────────────

    def add_sample(self, **kwargs) -> None:
        """Accumulate a per-sample result dict in memory."""
        import torch
        row = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().squeeze().tolist()
            row[k] = v
        self._local.append(row)

    def flush(self, model: str, task: str, n_shot: int) -> None:
        """Write accumulated samples to disk (append to existing feather file)."""
        if not self._local:
            return
        path = self._sample_path(model, task, n_shot)
        path.parent.mkdir(parents=True, exist_ok=True)
        local_df = pd.DataFrame(self._local)
        lock_name = f"sample__{model.replace('/', '__')}__{task}__{n_shot}shot"
        with lockutils.lock(lock_name, external=True, fair=True):
            if path.exists():
                existing = pd.read_feather(path)
                combined = pd.concat([existing, local_df], ignore_index=True)
            else:
                combined = local_df
            combined.round(self.round_values).reset_index(drop=True).to_feather(path)
        self._local = []

    # ── Aggregation ───────────────────────────────────────────────────────────

    def aggregate(self, model: str, task: str, n_shot: int, metric_names: list[str], benchmark: str = "") -> dict:
        """Compute mean performance per metric over all samples.

        Writes one long-format row per metric to ``aggregate.csv`` — columns
        ``model, benchmark, task, n_shot, metric, performance, stderr`` — so the
        table generalises across benchmarks regardless of each metric's name.

        Returns a ``{metric_name: mean, f"{metric_name}_stderr": stderr, ...}``
        dict for in-memory display.
        """
        path = self._sample_path(model, task, n_shot)
        if not path.exists():
            logger.warning(f"No samples file for {model}/{task}/{n_shot}shot")
            return {}

        df = pd.read_feather(path)
        scores: dict = {"model": model, "benchmark": benchmark, "task": task, "n_shot": n_shot}
        for m in metric_names:
            if m not in df.columns:
                continue
            performance = round(float(df[m].mean()), self.round_values)
            stderr = round(_stderr_of_mean(df[m]), self.round_values)
            scores[m] = performance
            scores[f"{m}_stderr"] = stderr
            self._upsert_aggregate({
                "model": model,
                "benchmark": benchmark,
                "task": task,
                "n_shot": n_shot,
                "metric": m,
                "performance": performance,
                "stderr": stderr,
            })
        return scores

    def aggregate_group(
        self,
        model: str,
        group_name: str,
        subtask_names: list[str],
        n_shot: int,
        metric_names: list[str],
        weighted: bool = True,
        benchmark: str = "",
    ) -> dict:
        """Average per-subtask aggregate scores and store under group_name.

        ``weighted=False``: simple mean over subtasks (each counts equally).
        ``weighted=True``: mean weighted by each subtask's sample count.
        """
        agg = self._load_aggregate()
        per_metric: dict[str, list[float]] = {m: [] for m in metric_names}
        sample_counts: list[int] = []

        for task in subtask_names:
            if agg.empty:
                continue
            mask = (
                (agg["model"] == model)
                & (agg["task"] == task)
                & (agg["n_shot"] == n_shot)
            )
            rows = agg[mask]
            if rows.empty or "metric" not in rows.columns:
                continue
            found_any = False
            for m in metric_names:
                mrow = rows[rows["metric"] == m]
                if not mrow.empty:
                    per_metric[m].append(float(mrow["performance"].iloc[0]))
                    found_any = True
            if found_any and weighted:
                path = self._sample_path(model, task, n_shot)
                sample_counts.append(len(pd.read_feather(path)) if path.exists() else 1)

        if not any(per_metric.values()):
            logger.warning("No subtask results found for group '%s'", group_name)
            return {}

        bench = benchmark or group_name
        scores: dict = {"model": model, "benchmark": bench, "task": group_name, "n_shot": n_shot}
        for m, vals in per_metric.items():
            if not vals:
                continue
            if weighted and sample_counts:
                w = sample_counts[: len(vals)]
                performance = round(sum(v * c for v, c in zip(vals, w)) / sum(w), self.round_values)
            else:
                performance = round(sum(vals) / len(vals), self.round_values)
            # Spread of the per-subtask scores around the group mean.
            stderr = round(_stderr_of_mean(pd.Series(vals)), self.round_values)
            scores[m] = performance
            scores[f"{m}_stderr"] = stderr
            self._upsert_aggregate({
                "model": model,
                "benchmark": bench,
                "task": group_name,
                "n_shot": n_shot,
                "metric": m,
                "performance": performance,
                "stderr": stderr,
            })

        logger.info("  %s (group): %s", group_name, scores)
        return scores

    @lockutils.synchronized(name="aggregate", external=True, fair=True)
    def _upsert_aggregate(self, row: dict) -> None:
        new_df = pd.DataFrame([row])
        p = self._aggregate_path
        agg = pd.read_csv(p) if p.exists() else pd.DataFrame()
        if not agg.empty and "metric" in agg.columns:
            mask = (
                (agg["model"] == row["model"])
                & (agg["task"] == row["task"])
                & (agg["n_shot"] == row["n_shot"])
                & (agg["metric"] == row["metric"])
            )
            agg = agg[~mask]
        agg = pd.concat([agg, new_df], ignore_index=True)
        agg.to_csv(self._aggregate_path, index=False)
        self._aggregate = agg

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_samples(self, model: str, task: str, n_shot: int) -> pd.DataFrame:
        path = self._sample_path(model, task, n_shot)
        return pd.read_feather(path) if path.exists() else pd.DataFrame()

    def get_aggregate(self, model: str | None = None, task: str | None = None) -> pd.DataFrame:
        df = self._load_aggregate()
        if model:
            df = df[df["model"] == model]
        if task:
            df = df[df["task"] == task]
        return df

    # ── Paths ─────────────────────────────────────────────────────────────────

    def _sample_path(self, model: str, task: str, n_shot: int) -> Path:
        safe_model = model.replace("/", "__")
        return self.outputs_dir / safe_model / f"{task}_{n_shot}shot.f"

    @property
    def _aggregate_path(self) -> Path:
        return self.output_dir / "aggregate.csv"

    @lockutils.synchronized(name="aggregate", external=True, fair=True)
    def _load_aggregate(self) -> pd.DataFrame:
        p = self._aggregate_path
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    # ── Display ───────────────────────────────────────────────────────────────

    def display(
        self,
        model: str | None = None,
        task: str | None = None,
        metric: str | None = None,
    ) -> pd.DataFrame:
        """Return a pivot table of performance: rows=models, columns=tasks.

        Pass ``metric`` to restrict to a single metric name (e.g. "acc",
        "exact_match"); by default the performance of every metric is shown.
        """
        df = self.get_aggregate(model=model, task=task)
        if df.empty or "performance" not in df.columns:
            return pd.DataFrame()
        if metric:
            df = df[df["metric"] == metric]
        if df.empty:
            return pd.DataFrame()
        return df.pivot_table(index="model", columns="task", values="performance", aggfunc="mean")

    def missing_jobs(
        self,
        models: list[str],
        tasks: list[str],
        n_shots: list[int],
    ) -> list[dict]:
        """Return (model, task, n_shot) combos that have no results yet."""
        missing = []
        for model in models:
            for task in tasks:
                for n_shot in n_shots:
                    if not self.is_completed(model, task, n_shot):
                        missing.append({"model": model, "task": task, "n_shot": n_shot})
        return missing
