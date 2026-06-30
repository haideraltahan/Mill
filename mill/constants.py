"""Central constants for Mill.

All path and numeric constants live here so they are defined exactly once.
Set environment variables to override directories without changing code:

    MILL_CACHE_DIR   — config, scheduler files, logs  (default: ~/.cache/mill)
    MILL_OUTPUT_DIR  — evaluation results              (default: ./mill_results)
"""
import os
from pathlib import Path

##################################################################
# PACKAGE PATHS
##################################################################

PACKAGE_DIR           = Path(__file__).parent
TASKS_DIR             = PACKAGE_DIR / "tasks"
SCHEDULER_DIR         = PACKAGE_DIR / "scheduler"
TEMPLATE_PATH         = SCHEDULER_DIR / "template.sbatch"
BUNDLED_CLUSTERS_PATH = SCHEDULER_DIR / "clusters.yaml"

##################################################################
# DIRECTORIES
##################################################################

CACHE_DIR = Path(os.getenv("MILL_CACHE_DIR", Path.home() / ".cache" / "mill"))
OUTPUT_DIR = Path(os.getenv("MILL_OUTPUT_DIR", Path.cwd() / "mill_results"))

LOGS_DIR   = CACHE_DIR / "logs"
JOBS_DIR   = CACHE_DIR / "jobs"
LOCK_DIR   = CACHE_DIR / "locks"

##################################################################
# OUTPUT / CACHING
##################################################################

DEFAULT_ROUND_VALUES = 4    # decimal places in stored metrics

##################################################################
# REPRODUCIBILITY
##################################################################

DEFAULT_SEED = 42   # seeds every source of randomness in an eval (shuffles,
                    # few-shot sampling, random-guess fallbacks). Override per
                    # run with `mill eval ... --seed N`.

##################################################################
# MODEL / BATCHING
##################################################################

FALLBACK_STARTING_BS = 64   # starting batch size when GPU info is unavailable

##################################################################
# SLURM SCHEDULER
##################################################################

MINUTES_PER_EVAL  = 60     # time budget per (model, task, n_shot) job
MAX_HOURS_PER_JOB = 18     # hard ceiling on a single SLURM array task
