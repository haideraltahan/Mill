"""SLURM job array scheduling for Mill.

Adapted from oellm-evals/oellm/main.py and template.sbatch.

Workflow:
    1. Build jobs DataFrame: models × tasks × n_shots
    2. Filter out already-completed (model, task, n_shot) from output cache
    3. Shuffle rows for even load distribution across array workers
    4. Compute dynamic array size respecting queue limits and time budgets
    5. Render template.sbatch and submit via sbatch (or run locally)
"""
from __future__ import annotations

import fnmatch
import logging
import math
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mill.constants import (BUNDLED_CLUSTERS_PATH, CACHE_DIR, JOBS_DIR, LOGS_DIR,
                            MAX_HOURS_PER_JOB, MINUTES_PER_EVAL, OUTPUT_DIR, TEMPLATE_PATH)

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(
        self,
        models: list[str],
        tasks: list[str],
        n_shots: list[int] = (0,),
        output_dir: str | Path = OUTPUT_DIR,
        cache_dir: str | Path = CACHE_DIR,
        cluster: str = "auto",
        limit: int | None = None,
        extra_task_paths: str = "",
        venv_path: str = "",
        log_dir: str = "",
        minutes_per_eval: int | None = None,
    ):
        """
        output_dir: where evaluation results (feather files) are written.
        cache_dir:  where clusters.yaml, SLURM job CSVs, and logs are stored.
                    On first run, clusters.yaml is copied there from the bundled
                    default so you can edit it without touching the mill package.
        """
        self.models = models
        self.tasks = tasks
        self.n_shots = list(n_shots)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.limit = limit
        self.minutes_per_eval = minutes_per_eval or MINUTES_PER_EVAL
        self.extra_task_paths = extra_task_paths
        self.cluster_cfg = self._detect_cluster(cluster)
        self.venv_path = venv_path or self.cluster_cfg.get("venv_path", "")
        self.log_dir = log_dir or str(LOGS_DIR)
        logger.info(f"Using cluster config: {self.cluster_cfg.get('_name', cluster)}")

        from mill.output import OutputHandler
        self._output_handler = OutputHandler(output_dir=output_dir)

    # ── Public interface ──────────────────────────────────────────────────────

    def _resolve_tasks(self, names: list[str]) -> list[str]:
        """Expand benchmark names to their constituent task names."""
        import mill.tasks  # trigger auto-discovery
        from mill.api.registry import get_benchmark_config, list_benchmarks
        benchmarks = set(list_benchmarks())
        resolved = []
        for name in names:
            if name in benchmarks:
                resolved.extend(get_benchmark_config(name).task_names)
            else:
                resolved.append(name)
        return resolved

    def build_jobs(self) -> pd.DataFrame:
        """Return a DataFrame of pending jobs (already-completed ones excluded)."""
        tasks = self._resolve_tasks(self.tasks)
        rows = []
        for model in self.models:
            abbr = _model_abbr(model)
            for task in tasks:
                for n_shot in self.n_shots:
                    if not self._output_handler.is_completed(abbr, task, n_shot):
                        rows.append({
                            "model_abbr": abbr,
                            "model_path": model,
                            "tasks": task,
                            "n_shot": n_shot,
                        })

        if not rows:
            logger.info("All requested jobs already complete.")
            return pd.DataFrame(columns=["model_abbr", "model_path", "tasks", "n_shot"])

        df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Pending jobs: {len(df)}")
        return df

    def print_jobs(self) -> None:
        df = self.build_jobs()
        print(df.to_string(index=True))

    def run_local(self) -> None:
        """Run all pending jobs sequentially in the current process (no SLURM)."""
        import dataclasses
        import mill.tasks  # trigger task auto-discovery
        from mill.api.registry import get_task_config
        from mill.api.task import ConfigurableTask
        from mill.evaluator import evaluate_task
        from mill.models.loader import load_model_from_file

        df = self.build_jobs()
        if df.empty:
            return

        for _, row in df.iterrows():
            model_path = row["model_path"]
            if Path(model_path).exists():
                model = load_model_from_file(model_path)
            else:
                from mill.models.loader import load_model_from_config
                model = load_model_from_config({"type": "hf", "path": model_path})

            config = dataclasses.replace(get_task_config(row["tasks"]), n_shots=int(row["n_shot"]))
            task = ConfigurableTask(config)
            evaluate_task(
                model=model,
                task=task,
                output_handler=self._output_handler,
                limit=self.limit,
            )

    def submit(self) -> str | None:
        """Write jobs.csv, render sbatch template, submit to SLURM."""
        df = self.build_jobs()
        if df.empty:
            return None

        total = len(df)
        queue_limit = self.cluster_cfg.get("queue_limit", 100)
        max_array = self.cluster_cfg.get("max_array_len", 50)
        array_size = self._compute_array_size(total, queue_limit, max_array)

        # Write jobs CSV and sbatch script under cache_dir (not output_dir)
        jobs_dir = self.cache_dir / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        csv_path = jobs_dir / "jobs.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Jobs CSV: {csv_path}")

        # Render sbatch script
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        script = self._render_template(
            csv_path=str(csv_path),
            total_evals=total,
            num_jobs=array_size,
            array_limit=array_size - 1,
        )

        script_path = jobs_dir / "submit.sbatch"  # lives in cache_dir/jobs/
        script_path.write_text(script)

        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            job_id = result.stdout.strip()
            logger.info(f"Submitted: {job_id}")
            return job_id
        else:
            logger.error(f"sbatch failed: {result.stderr}")
            return None

    # ── Internals ─────���─────────────────────────────���─────────────────────────

    def _compute_array_size(self, total: int, queue_limit: int, max_array: int) -> int:
        max_array_len = self.cluster_cfg.get("max_array_len", max_array)
        remaining_capacity = max(1, queue_limit - self._jobs_in_queue())

        total_minutes = total * self.minutes_per_eval
        max_minutes = MAX_HOURS_PER_JOB * 60
        min_for_time = max(1, math.ceil(total_minutes / max_minutes))

        desired = min(max_array_len, total)
        desired = max(desired, min_for_time)
        return min(desired, remaining_capacity, total)

    def _jobs_in_queue(self) -> int:
        try:
            result = subprocess.run(
                ["squeue", "-u", os.environ.get("USER", ""), "-t", "pending,running", "-h"],
                capture_output=True, text=True, timeout=10,
            )
            return len(result.stdout.strip().splitlines())
        except Exception:
            return 0

    def _compute_time_limit(self, total: int, array_size: int) -> str:
        evals_per_job = math.ceil(total / array_size)
        total_minutes = evals_per_job * self.minutes_per_eval
        total_minutes = min(total_minutes, MAX_HOURS_PER_JOB * 60)
        h, m = divmod(int(total_minutes), 60)
        return f"{h:02d}:{m:02d}:00"

    def _render_template(self, csv_path: str, total_evals: int, num_jobs: int, array_limit: int) -> str:
        template = TEMPLATE_PATH.read_text()
        account_line = (
            f"#SBATCH --account={self.cluster_cfg['account']}"
            if self.cluster_cfg.get("account")
            else "# no account specified"
        )

        gpus = self.cluster_cfg.get("gpus_per_node", 1)
        gpu_type = self.cluster_cfg.get("gpu_type", "")
        gres_spec = f"gpu:{gpu_type}:{gpus}" if gpu_type else f"gpu:{gpus}"

        mem_per_gpu = self.cluster_cfg.get("mem_per_gpu", "")
        mem = self.cluster_cfg.get("mem", "")
        if mem_per_gpu:
            mem_line = f"#SBATCH --mem-per-gpu={mem_per_gpu}"
        elif mem:
            mem_line = f"#SBATCH --mem={mem}"
        else:
            mem_line = "#SBATCH --mem=32G"

        cpus = self.cluster_cfg.get("cpus_per_task", "")
        cpus_line = f"#SBATCH --cpus-per-task={cpus}" if cpus else "# no cpus-per-task specified"

        return template.format(
            time_limit=self._compute_time_limit(total_evals, num_jobs),
            gres_spec=gres_spec,
            mem_line=mem_line,
            cpus_line=cpus_line,
            log_dir=self.log_dir,
            partition=self.cluster_cfg.get("partition", "gpu"),
            account_line=account_line,
            array_limit=array_limit,
            max_array_len=self.cluster_cfg.get("max_array_len", 50),
            csv_path=csv_path,
            num_jobs=num_jobs,
            total_evals=total_evals,
            output_dir=str(self.output_dir),
            limit=str(self.limit) if self.limit else "",
            venv_path=self.venv_path,
            extra_task_paths=self.extra_task_paths,
        )

    def _clusters_path(self) -> Path:
        """Return the clusters.yaml to use.

        Looks in {cache_dir}/clusters.yaml. On first run, copies the bundled
        default there so the user can edit it without touching the package.
        """
        user_path = self.cache_dir / "clusters.yaml"
        if not user_path.exists():
            import shutil
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(BUNDLED_CLUSTERS_PATH, user_path)
            logger.info(f"Created clusters.yaml at {user_path} — edit it to configure your cluster.")
        return user_path

    def _detect_cluster(self, cluster: str) -> dict:
        with open(self._clusters_path()) as f:
            all_clusters: dict = yaml.safe_load(f).get("clusters", {})

        if cluster != "auto":
            cfg = all_clusters.get(cluster, all_clusters.get("generic", {}))
            cfg["_name"] = cluster
            return cfg

        hostname = socket.gethostname()
        for name, cfg in all_clusters.items():
            pattern = cfg.get("hostname_pattern", "")
            if pattern and fnmatch.fnmatch(hostname, pattern):
                cfg["_name"] = name
                return cfg

        fallback = all_clusters.get("generic", {})
        fallback["_name"] = "generic"
        return fallback


def _model_abbr(model_path: str) -> str:
    p = Path(model_path)
    if p.exists():
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("_cfg", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = getattr(mod, "model", {})
        return cfg.get("abbr", p.stem)
    return model_path.replace("/", "__")
