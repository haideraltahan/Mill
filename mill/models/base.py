"""Shared utilities for Mill model backends."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mill.api.instance import Instance

logger = logging.getLogger(__name__)


def collate_by_gen_kwargs(requests: list["Instance"]) -> list[list["Instance"]]:
    """Group requests that share the same generation kwargs so they can be batched."""
    from collections import defaultdict
    import json

    groups: dict[str, list] = defaultdict(list)
    for req in requests:
        gen_kwargs = req.arguments[1] if len(req.arguments) > 1 else {}
        key = json.dumps(gen_kwargs, sort_keys=True, default=str)
        groups[key].append(req)
    return list(groups.values())


def chunk(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def is_multimodal_request(request: "Instance") -> bool:
    from mill.api.protocol import ChatMessages
    return isinstance(request.arguments[0], ChatMessages)


def decode_audio_array(audio, target_sr: int):
    """Coerce one audio item to a 1-D mono float32 waveform at ``target_sr``.

    Accepts any shape audio tasks produce: a decoded ``datasets`` dict
    (``{"array","sampling_rate"}``), an un-decoded dict (``{"bytes","path"}`` —
    what Mill yields when audio decoding is disabled to avoid torchcodec), a file
    path/URL string, or a raw numpy array (assumed already at ``target_sr``).
    Shared by the generative (transformers) and contrastive (CLAP) backends.
    """
    import numpy as np

    src_sr = target_sr
    if isinstance(audio, dict) and audio.get("array") is not None:
        array = np.asarray(audio["array"], dtype=np.float32)
        src_sr = int(audio.get("sampling_rate") or target_sr)
    elif isinstance(audio, dict) and audio.get("bytes") is not None:
        import io
        import soundfile as sf
        array, src_sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    else:
        src = audio.get("path") if isinstance(audio, dict) else audio
        if isinstance(src, str):
            import librosa
            array, _ = librosa.load(src, sr=target_sr)  # librosa resamples + downmixes
            return np.asarray(array, dtype=np.float32)
        array = np.asarray(audio, dtype=np.float32)

    if array.ndim > 1:  # stereo -> mono
        array = array.mean(axis=1)
    if src_sr != target_sr:
        import librosa
        array = librosa.resample(np.asarray(array, dtype=np.float32), orig_sr=src_sr, target_sr=target_sr)
    return np.asarray(array, dtype=np.float32)


def load_audio(request: "Instance", target_sr: int):
    """Resolve a single mono waveform (at ``target_sr``) from a classification request.

    Prefers the doc's raw ``audios``, falling back to audio embedded in the
    ChatMessages context. The audio analog of :func:`load_pil_image`, used by the
    audio zero-shot classification backend (CLAP).
    """
    audio = None
    if getattr(request.doc, "audios", None):
        audio = request.doc.audios[0]
    else:
        context = request.arguments[0]
        if hasattr(context, "extract_media"):
            _, _, audios = context.extract_media()
            audio = audios[0] if audios else None
    if audio is None:
        raise ValueError(f"Audio classification requires audio; none found for request {request.idx}.")
    return decode_audio_array(audio, target_sr)


def load_pil_image(request: "Instance"):
    """Resolve a single RGB PIL image from a classification request.

    Prefers the doc's raw ``visuals``, falling back to media embedded in the
    ChatMessages context. Accepts PIL images or file paths. Shared by the
    image-classification backends (CLIP zero-shot, timm supervised).
    """
    from PIL import Image

    image = None
    if getattr(request.doc, "visuals", None):
        image = request.doc.visuals[0]
    else:
        context = request.arguments[0]
        if hasattr(context, "extract_media"):
            images, _, _ = context.extract_media()
            image = images[0] if images else None
    if image is None:
        raise ValueError(f"Image classification requires an image; none found for request {request.idx}.")
    if isinstance(image, str):
        image = Image.open(image)
    return image.convert("RGB")
