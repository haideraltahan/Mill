"""timm (pytorch-image-models) supervised classification backend.

Wraps https://github.com/huggingface/pytorch-image-models vision models as a
``SupervisedClassifier``: each request carries an image and the model predicts
over its fixed pretrained head (e.g. ImageNet-1k). The task's ``target_index``
must index into that same class space.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mill.api.model import ModelCapabilities, SupervisedClassifier
from mill.api.registry import register_model
from mill.models.base import chunk, load_pil_image

if TYPE_CHECKING:
    from mill.api.instance import Instance

logger = logging.getLogger(__name__)


@register_model("timm", "pytorch-image-models")
class TimmModel(SupervisedClassifier):
    """Vision-only supervised image classifier via timm.

    Config dict fields (opencompass style)::

        dict(
            type="timm",
            path="resnet50.a1_in1k",   # any timm model name
            run_cfg=dict(batch_size=64),
        )

    The model classifies over its fixed pretrained head; ``doc.choices`` (label
    text) is not consumed — predictions are argmax over the model's classes, so
    the task's ``target_index`` must use the same indexing (e.g. ImageNet-1k).
    """

    def __init__(
        self,
        path: str,
        pretrained: bool = True,
        device: str | None = None,
        batch_size: int = 64,
        num_classes: int | None = None,
        **kwargs,
    ):
        try:
            import timm
        except ImportError:
            raise ImportError("Install timm to use TimmModel: pip install timm")
        import torch

        self._path = path
        self._batch_size = batch_size
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.capabilities = ModelCapabilities(
            modalities={"image"},
            supports_logprobs=False,
            supports_chat_template=False,
        )

        create_kwargs: dict = {"pretrained": pretrained}
        if num_classes is not None:
            create_kwargs["num_classes"] = num_classes
        model = timm.create_model(path, **create_kwargs)
        model.eval().to(self._device)
        self._model = model

        # Build the eval-time preprocessing transform from the model's data config.
        try:
            data_config = timm.data.resolve_model_data_config(model)
        except AttributeError:  # older timm
            data_config = timm.data.resolve_data_config({}, model=model)
        self._transform = timm.data.create_transform(**data_config, is_training=False)
        logger.info(f"Loaded timm {path} on {self._device} ({model.num_classes} classes)")

    @property
    def model_name(self) -> str:
        return self._path

    # ── SupervisedClassifier hook ─────────────────────────────────────────────

    def classify(self, requests: list["Instance"]) -> list[int]:
        import torch

        preds: list[int] = []
        for batch in chunk(list(requests), self._batch_size):
            pixels = torch.stack([self._transform(load_pil_image(req)) for req in batch]).to(self._device)
            with torch.inference_mode():
                logits = self._model(pixels)
            preds.extend(logits.argmax(dim=-1).tolist())
        return preds

    def cleanup(self) -> None:
        import torch

        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
