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

    cls = resolve_model_class({"type": model_type})
    # Plain HF model ID (e.g. "Qwen/Qwen3-0.6B") routes to TransformersModel with path set.
    if isinstance(model_type, str) and ":" not in model_type and model_type not in list_models():
        cfg.setdefault("path", model_type)

    return cls(**cfg)


def resolve_model_class(config: dict) -> type:
    """Resolve a model config's ``type`` to its backend class without instantiating.

    Accepts a class, a registry alias ("hf"/"vllm"/"clip"/...), a
    "module.path:ClassName" string, or a plain HF model ID (-> TransformersModel).
    """
    model_type = config.get("type")
    if not isinstance(model_type, str):
        return model_type  # already a class
    if ":" in model_type:
        module_path, cls_name = model_type.rsplit(":", 1)
        return getattr(importlib.import_module(module_path), cls_name)
    try:
        return get_model_class(model_type)  # registered alias
    except KeyError:
        from mill.models.transformers import TransformersModel
        return TransformersModel  # plain HF model ID


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


def _path_with_pretrained(config: dict) -> str:
    """Combine ``path`` with an optional ``pretrained`` weights tag.

    Backends whose identity is (architecture, weights) — e.g. open_clip's
    ``ViT-B-32`` + ``laion2b_s34b_b79k`` — must keep both in the name so two
    weight sets of the same architecture don't collide in the output cache.
    """
    name = str(config["path"])
    if config.get("pretrained"):
        name = f"{name}/{config['pretrained']}"
    return name


def model_name_from_config(config: dict) -> str:
    """Canonical model name (matching ``MillModel.model_name``) without loading.

    Mirrors how each backend derives its name — the HF/vLLM ``path`` (plus an
    optional ``pretrained`` tag) or the LiteLLM ``model`` string — so the
    pipeline can consult the output cache and skip loading weights when there is
    nothing pending.
    """
    if config.get("path"):
        return _path_with_pretrained(config)
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
        return _path_with_pretrained(config).replace("/", "__")
    # type may be a plain HF model ID
    type_val = config.get("type", "unknown")
    if isinstance(type_val, str) and ":" not in type_val:
        return type_val.replace("/", "__")
    return "unknown"
