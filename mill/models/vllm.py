"""vLLM model backend for high-throughput batch inference."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mill.api.model import GenerativeModel, ModelCapabilities
from mill.api.registry import register_model

if TYPE_CHECKING:
    from mill.api.instance import Instance

logger = logging.getLogger(__name__)


@register_model("vllm")
class VLLMModel(GenerativeModel):
    """vLLM-backed model for text generation.

    Config dict fields:
        path (str): HuggingFace model ID or local path.
        dtype (str): "bfloat16" | "float16" | "auto". Default "bfloat16".
        max_context_length (int): Default 4096.
        gpu_memory_utilization (float): Default 0.9.
        tensor_parallel_size (int): Number of GPUs for tensor parallelism. Default 1.
        max_model_len (int | None): Override model's max position embeddings.
    """

    def __init__(
        self,
        path: str,
        dtype: str = "bfloat16",
        max_context_length: int = 4096,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError:
            raise ImportError("Install vLLM to use VLLMModel: pip install vllm")

        self._path = path
        self.capabilities = ModelCapabilities(
            modalities={"text"},
            max_context_length=max_context_length,
            supports_logprobs=True,
        )

        from vllm import LLM
        llm_kwargs: dict = {
            "model": path,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": trust_remote_code,
        }
        if max_model_len:
            llm_kwargs["max_model_len"] = max_model_len

        self._llm = LLM(**llm_kwargs)
        logger.info(f"Loaded {path} via vLLM (tp={tensor_parallel_size})")

    @property
    def model_name(self) -> str:
        return self._path

    @property
    def batch_size(self) -> None:
        # vLLM manages its own batching internally — pass all requests at once.
        return None

    # ── MillModel hooks ───────────────────────────────────────────────────────

    def _generate_batch(self, batch: list["Instance"], gen_kwargs: dict) -> list[str]:
        from vllm import SamplingParams
        prompts = [req.arguments[0] for req in batch]
        sampling_params = SamplingParams(
            max_tokens=gen_kwargs.get("max_new_tokens", 256),
            temperature=0.0,
            stop=gen_kwargs.get("stop", []),
        )
        outputs = self._llm.generate(prompts, sampling_params)
        results = [""] * len(batch)
        for i, out in enumerate(outputs):
            results[i] = out.outputs[0].text
        return results

    def _loglikelihood_batch(self, batch: list["Instance"]) -> list[tuple[float, bool]]:
        from vllm import SamplingParams
        prompts = [req.arguments[0] + req.arguments[1] for req in batch]
        ctx_lens = [
            len(self._llm.get_tokenizer().encode(req.arguments[0]))
            for req in batch
        ]
        sampling_params = SamplingParams(prompt_logprobs=1, max_tokens=1, temperature=0.0)
        outputs = self._llm.generate(prompts, sampling_params)

        results: list[tuple[float, bool]] = []
        for out, ctx_len in zip(outputs, ctx_lens):
            log_probs = out.prompt_logprobs or []
            cont_log_probs = log_probs[ctx_len:]
            total = sum(list(lp.values())[0].logprob for lp in cont_log_probs if lp)
            results.append((total, False))
        return results

    def _loglikelihood_rolling_single(self, request: "Instance") -> float:
        from vllm import SamplingParams
        sampling_params = SamplingParams(prompt_logprobs=1, max_tokens=1, temperature=0.0)
        outputs = self._llm.generate([request.arguments[0]], sampling_params)
        log_probs = outputs[0].prompt_logprobs or []
        valid = [list(lp.values())[0].logprob for lp in log_probs if lp]
        return sum(valid) / len(valid) if valid else 0.0

    def cleanup(self) -> None:
        import torch
        del self._llm
        torch.cuda.empty_cache()
