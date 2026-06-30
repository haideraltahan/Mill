"""open_clip zero-shot classification backend.

Wraps https://github.com/mlfoundations/open_clip CLIP-family models as a
``ZeroShotClassifier``: each request carries an image and a list of candidate
text labels; the model returns the index of the best-matching label by
image–text cosine similarity.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mill.api.model import ModelCapabilities, ZeroShotClassifier
from mill.api.registry import register_model
from mill.models.base import chunk, load_pil_image

if TYPE_CHECKING:
    from mill.api.instance import Instance

logger = logging.getLogger(__name__)


@register_model("clip", "open_clip", "openclip")
class OpenCLIPModel(ZeroShotClassifier):
    """Zero-shot image classification via open_clip.

    Config dict fields (opencompass style)::

        dict(
            type="clip",
            path="ViT-B-32",                 # open_clip architecture name
            pretrained="laion2b_s34b_b79k",  # open_clip weights tag
            prompt_template="a photo of a {c}.",
            run_cfg=dict(batch_size=64),
        )

    ``path`` + ``pretrained`` together form the model identity used for output
    caching (see ``model_name``), so two weight sets of the same architecture
    stay distinct.
    """

    def __init__(
        self,
        path: str,
        pretrained: str | None = None,
        device: str | None = None,
        batch_size: int = 64,
        prompt_template: str = "a photo of a {c}.",
        max_context_length: int = 77,
        **kwargs,
    ):
        try:
            import open_clip
        except ImportError:
            raise ImportError("Install open_clip to use OpenCLIPModel: pip install open_clip_torch")
        import torch

        self._arch = path
        self._pretrained = pretrained
        self._model_name = f"{path}/{pretrained}" if pretrained else path
        self._batch_size = batch_size
        self._template = prompt_template
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.capabilities = ModelCapabilities(
            modalities={"image", "text"},
            max_context_length=max_context_length,
            supports_logprobs=False,
            supports_chat_template=False,
        )

        model, _, preprocess = open_clip.create_model_and_transforms(
            path, pretrained=pretrained, device=self._device
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(path)
        logger.info(f"Loaded open_clip {self._model_name} on {self._device}")

    @property
    def model_name(self) -> str:
        return self._model_name

    # ── ZeroShotClassifier hook ───────────────────────────────────────────────

    def zero_shot_classify(self, requests: list["Instance"]) -> list[int]:
        import torch

        preds: list[int] = []
        weight_cache: dict[tuple, "torch.Tensor"] = {}  # (labels, templates) -> class embeddings

        for batch in chunk(list(requests), self._batch_size):
            pixels = torch.stack([self._preprocess(load_pil_image(req)) for req in batch])
            pixels = pixels.to(self._device)
            with torch.inference_mode():
                img_feats = self._model.encode_image(pixels)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            for i, req in enumerate(batch):
                labels = tuple(req.arguments[1])
                if not labels:
                    raise ValueError(f"Zero-shot classification needs candidate labels (request {req.idx}).")
                templates = req.arguments[2] if len(req.arguments) > 2 and req.arguments[2] else [self._template]
                templates = tuple(templates)
                class_weights = weight_cache.get((labels, templates))
                if class_weights is None:
                    class_weights = self._class_embeddings(list(labels), list(templates))
                    weight_cache[(labels, templates)] = class_weights
                logits = img_feats[i : i + 1] @ class_weights.T  # (1, num_labels)
                preds.append(int(logits.argmax(dim=-1).item()))
        return preds

    # ── Internal ──────────────────────────────────────────────────────────────

    def _class_embeddings(self, labels: list[str], templates: list[str]) -> "torch.Tensor":
        """Build one classifier embedding per label = mean of its templated,
        normalized text embeddings (clip_benchmark zero-shot weights)."""
        import torch

        # Class-major, template-minor so we can reshape and average per class.
        prompts = [tmpl.format(c=label) for label in labels for tmpl in templates]
        tokens = self._tokenizer(prompts).to(self._device)
        with torch.inference_mode():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.reshape(len(labels), len(templates), -1).mean(dim=1)  # (num_labels, dim)
        return feats / feats.norm(dim=-1, keepdim=True)

    def cleanup(self) -> None:
        import torch

        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
