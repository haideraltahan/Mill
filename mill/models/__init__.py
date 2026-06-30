from mill.api.registry import register_model  # noqa: F401 — makes decorator available

# Import all backends eagerly so their @register_model decorators fire.
# Each import is wrapped to silently skip backends whose optional deps are missing.
def _try_import(module: str) -> None:
    try:
        __import__(module)
    except ImportError:
        pass


_try_import("mill.models.transformers")
_try_import("mill.models.vllm")
_try_import("mill.models.litellm")
_try_import("mill.models.clip")
_try_import("mill.models.timm")
