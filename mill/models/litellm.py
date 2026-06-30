"""LiteLLM / OpenAI-compatible API backend."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from mill.api.model import GenerativeModel, ModelCapabilities
from mill.api.registry import register_model

if TYPE_CHECKING:
    from mill.api.instance import Instance

logger = logging.getLogger(__name__)


@register_model("litellm", "openai", "api")
class LiteLLMModel(GenerativeModel):
    """Model backend via LiteLLM — supports OpenAI, Anthropic, Gemini, etc.

    Config dict fields:
        model (str): LiteLLM model string, e.g. "gpt-4o", "claude-3-5-sonnet-20241022".
        api_key (str | None): API key (or set env var).
        max_context_length (int): Default 128000.
        max_retries (int): Retry on rate limit. Default 5.
        requests_per_minute (int | None): Rate limiting. Default None (no limit).
        modalities (list[str]): Default ["text"].
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_context_length: int = 128_000,
        max_retries: int = 5,
        requests_per_minute: int | None = None,
        modalities: list[str] | None = None,
        **kwargs,
    ):
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise ImportError("Install litellm: pip install litellm")

        self._model = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._rpm = requests_per_minute
        self._last_call: float = 0.0

        self.capabilities = ModelCapabilities(
            modalities=set(modalities or ["text"]),
            max_context_length=max_context_length,
            supports_logprobs=False,
            supports_chat_template=True,
        )
        logger.info(f"LiteLLM backend initialised for model: {model}")

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def batch_size(self) -> int:
        # API calls are per-request; progress updates after each one.
        return 1

    # ── MillModel hooks ───────────────────────────────────────────────────────

    def _generate_batch(self, batch: list["Instance"], gen_kwargs: dict) -> list[str]:
        req = batch[0]
        context = req.arguments[0]
        max_tokens = gen_kwargs.get("max_new_tokens", 256)

        if hasattr(context, "to_openai_messages"):
            messages = context.to_openai_messages()
        else:
            messages = [{"role": "user", "content": str(context)}]

        return [self._call(messages, max_tokens=max_tokens)]

    def _loglikelihood_batch(self, batch: list["Instance"]) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "Log-likelihood scoring is not available for API models. "
            "Use output_type=GENERATIVE."
        )

    def _loglikelihood_rolling_single(self, request: "Instance") -> float:
        raise NotImplementedError("Perplexity is not available for API models.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call(self, messages: list[dict], max_tokens: int = 256) -> str:
        import litellm
        if self._rpm:
            min_interval = 60.0 / self._rpm
            elapsed = time.time() - self._last_call
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

        kwargs: dict = {"model": self._model, "messages": messages, "max_tokens": max_tokens}
        if self._api_key:
            kwargs["api_key"] = self._api_key

        for attempt in range(self._max_retries):
            try:
                resp = litellm.completion(**kwargs)
                self._last_call = time.time()
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt == self._max_retries - 1:
                    logger.error(f"LiteLLM call failed after {self._max_retries} retries: {e}")
                    return ""
                wait = 2 ** attempt
                logger.warning(f"LiteLLM retry {attempt + 1}/{self._max_retries} after {wait}s: {e}")
                time.sleep(wait)
        return ""
