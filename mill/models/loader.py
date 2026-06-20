"""Load a model from a config dict (opencompass-style) or a registry name."""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

from mill.api.model import MillModel
from mill.api.registry import get_model_class, list_models

logger = logging.getLogger(__name__)


def load_model_from_config(config: dict) -> MillModel:
    """Instantiate a MillModel from an opencompass-style config dict.

    Example config::

        dict(
            type=TransformersModel,          # class or "mill.models.transformers:TransformersModel"
            abbr="llama3-8b",
            path="meta-llama/Meta-Llama-3-8B-Instruct",
            dtype="bfloat16",
            run_cfg=dict(num_gpus=1, batch_size=16),
        )
    """
    cfg = dict(config)
    model_type = cfg.pop("type")
    cfg.pop("abbr", None)
    run_cfg = cfg.pop("run_cfg", {})

    # Merge run_cfg fields into top-level kwargs (batch_size, etc.)
    cfg.update(run_cfg)
    cfg.pop("num_gpus", None)  # used by scheduler, not model constructor

    if isinstance(model_type, str):
        if ":" in model_type:
            # "module.path:ClassName"
            module_path, cls_name = model_type.rsplit(":", 1)
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
        else:
            try:
                # registered alias e.g. "hf", "vllm", "litellm"
                cls = get_model_class(model_type)
            except KeyError:
                # plain HF model ID e.g. "Qwen/Qwen3-0.6B"
                from mill.models.transformers import TransformersModel
                cfg.setdefault("path", model_type)
                cls = TransformersModel
    else:
        cls = model_type

    return cls(**cfg)


def load_model_from_name(name: str, **kwargs) -> MillModel:
    """Instantiate a model by registry name, e.g. 'hf', 'vllm', 'litellm'."""
    cls = get_model_class(name)
    return cls(**kwargs)


def load_model_from_file(config_path: str) -> MillModel:
    """Load a model from a Python config file that exports `model = dict(...)`."""
    p = Path(config_path)
    spec = importlib.util.spec_from_file_location("_mill_model_cfg", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    config = getattr(mod, "model", None)
    if config is None:
        raise ValueError(f"Config file {p} must export a top-level `model = dict(...)` variable.")
    return load_model_from_config(config)


def model_name_from_config(config: dict) -> str:
    """Canonical model name (matching ``MillModel.model_name``) without loading.

    Mirrors how each backend derives its name — the HF/vLLM ``path`` or the
    LiteLLM ``model`` string — so the pipeline can consult the output cache and
    skip loading model weights when there is nothing pending.
    """
    if config.get("path"):
        return str(config["path"])
    if config.get("model"):
        return str(config["model"])
    type_val = config.get("type")
    if isinstance(type_val, str) and ":" not in type_val and type_val not in list_models():
        return type_val  # plain HF model ID resolves to its own path
    return model_abbr(config)


def model_abbr(config: dict) -> str:
    """Return the model abbreviation used for output filenames."""
    if config.get("abbr"):
        return config["abbr"]
    if config.get("path"):
        return str(config["path"]).replace("/", "__")
    # type may be a plain HF model ID
    type_val = config.get("type", "unknown")
    if isinstance(type_val, str) and ":" not in type_val:
        return type_val.replace("/", "__")
    return "unknown"
