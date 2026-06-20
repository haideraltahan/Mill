from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, type] = {}
_TASK_REGISTRY: dict[str, "MillTaskConfig"] = {}          # noqa: F821
_BENCHMARK_REGISTRY: dict[str, "MillBenchmarkConfig"] = {}  # noqa: F821


def register_model(*names: str):
    """Decorator: register a MillModel subclass under one or more names."""
    def decorator(cls):
        for name in names:
            _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str) -> type:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not registered. Available: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name]


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())


def register_task(config: "MillTaskConfig") -> None:  # noqa: F821
    _TASK_REGISTRY[config.name] = config


def get_task_config(name: str) -> "MillTaskConfig":  # noqa: F821
    if name not in _TASK_REGISTRY:
        raise KeyError(f"Task '{name}' not registered. Run `mill ls` to see available tasks.")
    return _TASK_REGISTRY[name]


def list_tasks() -> list[str]:
    return sorted(_TASK_REGISTRY.keys())


def register_benchmark(config: "MillBenchmarkConfig") -> None:  # noqa: F821
    _BENCHMARK_REGISTRY[config.name] = config


def get_benchmark_config(name: str) -> "MillBenchmarkConfig":  # noqa: F821
    if name not in _BENCHMARK_REGISTRY:
        raise KeyError(f"Benchmark '{name}' not registered. Run `mill ls` to see available benchmarks.")
    return _BENCHMARK_REGISTRY[name]


def list_benchmarks() -> list[str]:
    return sorted(_BENCHMARK_REGISTRY.keys())


class Registry:
    """Central registry — thin wrapper over the module-level dicts above."""
    register_model = staticmethod(register_model)
    get_model_class = staticmethod(get_model_class)
    list_models = staticmethod(list_models)
    register_task = staticmethod(register_task)
    get_task_config = staticmethod(get_task_config)
    list_tasks = staticmethod(list_tasks)
    register_benchmark = staticmethod(register_benchmark)
    get_benchmark_config = staticmethod(get_benchmark_config)
    list_benchmarks = staticmethod(list_benchmarks)


def load_tasks_from_path(path: str) -> None:
    """Discover and register all tasks and benchmarks in a directory or .py file."""
    p = Path(path)
    if p.is_file():
        _load_task_module(p)
    elif p.is_dir():
        for f in sorted(p.rglob("task.py")):
            _load_task_module(f)
    else:
        raise FileNotFoundError(f"Task path not found: {path}")


def _load_task_module(path: Path) -> None:
    module_name = f"_mill_task_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        for config in getattr(module, "TASKS_TABLE", []):
            register_task(config)
            logger.debug(f"Registered task: {config.name}")
        for config in getattr(module, "BENCHMARKS_TABLE", []):
            register_benchmark(config)
            logger.debug(f"Registered benchmark: {config.name}")
    except Exception as e:
        logger.warning(f"Failed to load task from {path}: {e}")
