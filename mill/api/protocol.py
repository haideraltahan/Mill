"""Unified media protocol for multimodal evaluation.

Adapted from lmms-eval/lmms_eval/protocol.py. Provides a model-backend-agnostic
representation of multi-turn conversations containing text, images, video, and audio.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel


class ChatTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatImageContent(BaseModel):
    type: Literal["image"] = "image"
    url: Any  # PIL.Image, file path str, or URL


class ChatVideoContent(BaseModel):
    type: Literal["video"] = "video"
    url: Any  # file path str or URL


class ChatAudioContent(BaseModel):
    type: Literal["audio"] = "audio"
    url: Any  # file path str, bytes, or URL


ChatContent = Union[ChatTextContent, ChatImageContent, ChatVideoContent, ChatAudioContent]


class ChatMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: List[ChatContent]


class ChatMessages(BaseModel):
    """A multi-turn conversation with mixed media content.

    Use `extract_media()` to pull images/videos/audios for processor calls.
    Use `to_hf_messages()` or `to_openai_messages()` to format for a backend.
    """
    messages: List[ChatMessage]

    def extract_media(self) -> Tuple[list, list, list]:
        """Returns (images, videos, audios) lists for batch processor calls."""
        images, videos, audios = [], [], []
        for message in self.messages:
            for content in message.content:
                if content.type == "image":
                    images.append(content.url)
                elif content.type == "video":
                    videos.append(content.url)
                elif content.type == "audio":
                    audios.append(content.url)
        return images, videos, audios

    def to_hf_messages(self, video_kwargs: Optional[Dict[str, Any]] = None) -> List[dict]:
        """Format for HuggingFace processor (AutoProcessor / chat_template)."""
        video_kwargs = video_kwargs or {}
        hf_messages = []
        for message in self.messages:
            hf_msg: dict = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    hf_msg["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    hf_msg["content"].append({"type": "image", "image": content.url})
                elif content.type == "video":
                    hf_msg["content"].append({"type": "video", "video": content.url, **video_kwargs})
                elif content.type == "audio":
                    hf_msg["content"].append({"type": "audio", "audio": content.url})
            hf_messages.append(hf_msg)
        return hf_messages

    def to_openai_messages(self, video_kwargs: Optional[Dict[str, Any]] = None) -> List[dict]:
        """Format for OpenAI / LiteLLM API. Videos are exploded to base64 frames."""
        video_kwargs = video_kwargs or {}
        image_format = os.getenv("MILL_IMAGE_FORMAT", "PNG").upper()
        mime_type = f"image/{'jpeg' if image_format in ('JPG', 'JPEG') else image_format.lower()}"

        openai_messages = []
        for message in self.messages:
            oa_msg: dict = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    oa_msg["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    b64 = _encode_image_base64(content.url, image_format)
                    oa_msg["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                    })
                elif content.type == "video":
                    frames = _video_to_frames(content.url, **video_kwargs)
                    for frame in frames:
                        b64 = _encode_image_base64(frame, image_format)
                        oa_msg["content"].append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                        })
                elif content.type == "audio":
                    oa_msg["content"].append({"type": "audio_url", "audio_url": {"url": content.url}})
            openai_messages.append(oa_msg)
        return openai_messages

    @classmethod
    def from_text(cls, text: str, role: str = "user") -> "ChatMessages":
        """Convenience: wrap a plain text string in a single-turn conversation."""
        return cls(messages=[ChatMessage(role=role, content=[ChatTextContent(text=text)])])

    @classmethod
    def from_text_and_images(cls, text: str, images: list, role: str = "user") -> "ChatMessages":
        """Build a single user turn with images followed by text."""
        content: List[ChatContent] = [ChatImageContent(url=img) for img in images]
        content.append(ChatTextContent(text=text))
        return cls(messages=[ChatMessage(role=role, content=content)])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_image_base64(image: Any, fmt: str = "PNG") -> str:
    import base64
    import io
    from PIL import Image

    if isinstance(image, str):
        image = Image.open(image)
    if not isinstance(image, Image.Image):
        raise TypeError(f"Cannot encode image of type {type(image)}")
    buf = io.BytesIO()
    save_fmt = "JPEG" if fmt in ("JPG", "JPEG") else fmt
    image.save(buf, format=save_fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _video_to_frames(path: str, nframes: int = 16, **kwargs) -> list:
    """Extract up to `nframes` frames from a video file. Returns list of PIL Images."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0))
        total = len(vr)
        step = max(1, total // nframes)
        indices = list(range(0, total, step))[:nframes]
        frames = vr.get_batch(indices).asnumpy()
        from PIL import Image
        return [Image.fromarray(f) for f in frames]
    except ImportError:
        raise ImportError("Install decord to decode video files: pip install decord")
