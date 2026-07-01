"""HuggingFace Transformers zero-shot classification backend (CLIP and CLAP).

Wraps any dual-encoder contrastive model exposed through the ``transformers``
API — image-text (``CLIPModel``) or audio-text (``ClapModel``) — as a single
``ZeroShotClassifier``. Both share the same shape: ``get_text_features`` plus a
media tower (``get_image_features`` / ``get_audio_features``) into one embedding
space, scored by cosine similarity against candidate label captions.

This is deliberately separate from :class:`mill.models.clip.OpenCLIPModel`
(the open_clip backend, alias ``clip``), which is left untouched — use ``hf_clip``
/ ``clap`` for the transformers path.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mill.api.model import ModelCapabilities, ZeroShotClassifier
from mill.api.registry import register_model
from mill.models.base import chunk, load_audio, load_pil_image

if TYPE_CHECKING:
    from mill.api.instance import Instance

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATES = {"image": "a photo of a {c}.", "audio": "This is a sound of {c}."}


@register_model("hf_clip", "clap", "hf_clap", "transformers_clip")
class HFContrastiveModel(ZeroShotClassifier):
    """Zero-shot classification via a transformers CLIP/CLAP model.

    Config dict fields (opencompass style)::

        dict(
            type="clap",
            path="laion/clap-htsat-unfused",   # HF checkpoint (CLIP or CLAP)
            prompt_template="This is a sound of {c}.",
            run_cfg=dict(batch_size=32),
        )

    The media modality (image vs audio) is inferred from the processor, so the
    same class serves ``openai/clip-vit-base-patch32`` and
    ``laion/clap-htsat-unfused`` unchanged.
    """

    def __init__(
        self,
        path: str,
        modality: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        prompt_template: str | None = None,
        max_context_length: int = 77,
        dtype: str = "float32",
        trust_remote_code: bool = True,
        **kwargs,
    ):
        import torch
        from transformers import AutoModel, AutoProcessor

        self._path = path
        self._batch_size = batch_size
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._processor = AutoProcessor.from_pretrained(path, trust_remote_code=trust_remote_code)
        torch_dtype = getattr(torch, dtype, torch.float32)
        self._model = AutoModel.from_pretrained(path, dtype=torch_dtype, trust_remote_code=trust_remote_code)
        self._model.eval().to(self._device)

        # Infer the media modality from the processor unless the caller pinned it.
        if modality is None:
            if getattr(self._processor, "image_processor", None) is not None:
                modality = "image"
            elif getattr(self._processor, "feature_extractor", None) is not None:
                modality = "audio"
            else:
                raise ValueError(f"Could not infer image/audio modality for '{path}'; pass modality=.")
        if modality not in ("image", "audio"):
            raise ValueError(f"modality must be 'image' or 'audio', got '{modality}'.")
        self._modality = modality
        self._template = prompt_template or _DEFAULT_TEMPLATES[modality]

        self.capabilities = ModelCapabilities(
            modalities={modality, "text"},
            max_context_length=max_context_length,
            supports_logprobs=False,
            supports_chat_template=False,
        )
        logger.info(f"Loaded transformers {modality}-text model {path} on {self._device}")

    @property
    def model_name(self) -> str:
        return self._path

    def _audio_sampling_rate(self) -> int:
        fe = getattr(self._processor, "feature_extractor", None)
        return int(getattr(fe, "sampling_rate", 48000) or 48000)

    def _audio_kwarg(self) -> str:
        """The processor's raw-audio argument name.

        transformers ≥5 standardised on the singular ``audio`` and deprecated
        ``audios`` (it now raises), so prefer ``audio``.
        """
        import inspect
        try:
            params = inspect.signature(self._processor.__call__).parameters
        except (TypeError, ValueError):
            return "audio"
        return "audio" if "audio" in params else ("audios" if "audios" in params else "audio")

    @staticmethod
    def _embeds(out):
        """Extract the joint-space embedding tensor from a ``get_*_features`` call.

        CLIP returns a plain tensor; transformers-5 CLAP returns an output object
        whose projected embedding is in ``pooler_output`` (equivalently
        ``audio_embeds`` / ``text_embeds``).
        """
        import torch
        if isinstance(out, torch.Tensor):
            return out
        for attr in ("pooler_output", "audio_embeds", "text_embeds", "image_embeds"):
            v = getattr(out, attr, None)
            if v is not None:
                return v
        raise TypeError(f"Could not extract an embedding tensor from {type(out).__name__}")

    # ── ZeroShotClassifier hook ───────────────────────────────────────────────

    def zero_shot_classify(self, requests: list["Instance"]) -> list[int]:
        import torch

        preds: list[int] = []
        weight_cache: dict[tuple, "torch.Tensor"] = {}  # (labels, templates) -> class embeddings

        for batch in chunk(list(requests), self._batch_size):
            with torch.inference_mode():
                feats = self._encode_media(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)

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
                logits = feats[i : i + 1] @ class_weights.T  # (1, num_labels)
                preds.append(int(logits.argmax(dim=-1).item()))
        return preds

    # ── Internal ──────────────────────────────────────────────────────────────

    def _encode_media(self, batch: list["Instance"]) -> "torch.Tensor":
        if self._modality == "image":
            inputs = self._processor(
                images=[load_pil_image(req) for req in batch], return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            return self._embeds(self._model.get_image_features(**inputs))
        target_sr = self._audio_sampling_rate()
        arrays = [load_audio(req, target_sr) for req in batch]
        inputs = self._processor(
            **{self._audio_kwarg(): arrays},
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        return self._embeds(self._model.get_audio_features(**inputs))

    def _class_embeddings(self, labels: list[str], templates: list[str]) -> "torch.Tensor":
        """One classifier embedding per label = mean of its templated, normalized
        text embeddings (clip_benchmark zero-shot weights)."""
        import torch

        prompts = [tmpl.format(c=label) for label in labels for tmpl in templates]
        inputs = self._processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            feats = self._embeds(self._model.get_text_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.reshape(len(labels), len(templates), -1).mean(dim=1)  # (num_labels, dim)
        return feats / feats.norm(dim=-1, keepdim=True)

    def cleanup(self) -> None:
        import torch

        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
