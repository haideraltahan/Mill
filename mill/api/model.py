from __future__ import annotations

import gc
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mill.api.instance import Instance

logger = logging.getLogger(__name__)

from mill.constants import FALLBACK_STARTING_BS


def _is_oom(e: Exception) -> bool:
    if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
        return True
    try:
        import torch
        return isinstance(e, torch.cuda.OutOfMemoryError)
    except (ImportError, AttributeError):
        return False


def _clear_cuda_cache() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except (ImportError, AttributeError):
        pass
    gc.collect()


@dataclass
class ModelCapabilities:
    modalities: set[str] = field(default_factory=lambda: {"text"})
    max_context_length: int = 4096
    supports_logprobs: bool = True
    supports_chat_template: bool = False

    def supports(self, modality: str) -> bool:
        return modality in self.modalities


class _RichProgress:
    """Rich live progress bar for interactive terminals."""

    def __init__(self, desc: str, total: int) -> None:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        self._progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        self._desc = desc
        self._total = total
        self._task_id = None

    def __enter__(self) -> "_RichProgress":
        self._progress.__enter__()
        self._task_id = self._progress.add_task(self._desc, total=self._total)
        return self

    def __exit__(self, *args) -> None:
        self._progress.__exit__(*args)

    def update(self, n: int) -> None:
        self._progress.advance(self._task_id, n)


class _LogProgress:
    """Plain-text progress for non-TTY environments (SLURM, file redirection)."""

    def __init__(self, desc: str, total: int, log_every_pct: int = 10) -> None:
        self._desc = desc
        self._total = total
        self._completed = 0
        self._log_every = max(1, total * log_every_pct // 100)
        self._last_milestone = -1

    def __enter__(self) -> "_LogProgress":
        logger.info("%s: 0/%d", self._desc, self._total)
        return self

    def __exit__(self, *args) -> None:
        if self._completed < self._total:
            logger.info("%s: %d/%d (done)", self._desc, self._total, self._total)

    def update(self, n: int) -> None:
        self._completed += n
        milestone = self._completed // self._log_every
        if milestone > self._last_milestone:
            self._last_milestone = milestone
            pct = min(100, 100 * self._completed // self._total)
            logger.info("%s: %d/%d (%d%%)", self._desc, self._completed, self._total, pct)


def _progress(desc: str, total: int):
    """Return a progress context manager suited to the current environment."""
    if sys.stdout.isatty():
        return _RichProgress(desc, total)
    return _LogProgress(desc, total)


class MillModel(ABC):
    """Abstract base class for all Mill model backends.

    Implement ``_generate_batch``, ``_loglikelihood_batch``, and
    ``_loglikelihood_rolling_single``. The public methods handle batching,
    result ordering, progress bars, and OOM recovery automatically.

    Batch size behaviour
    --------------------
    ``auto_batch_size = True`` (default for GPU backends)
        Starts at ``_AUTO_STARTING_BS`` and halves on OOM until a size fits.
        Set ``batch_size=<int>`` explicitly to disable auto and use a fixed size.

    ``auto_batch_size = False`` with ``batch_size = None``
        Passes all requests to the hook in one call (e.g. vLLM).

    ``auto_batch_size = False`` with ``batch_size = <int>``
        Fixed size, no OOM retry.
    """

    capabilities: ModelCapabilities = ModelCapabilities()

    @property
    def batch_size(self) -> int | None:
        return 1

    @property
    def auto_batch_size(self) -> bool:
        """If True, batch size is found automatically via OOM retry."""
        return False

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_until(self, requests: list["Instance"]) -> list[str]:
        from mill.models.base import collate_by_gen_kwargs
        outputs: list[str] = [""] * len(requests)
        groups = collate_by_gen_kwargs(requests)
        oom_retry = self.auto_batch_size
        effective_bs = self._estimate_starting_batch_size() if oom_retry else (self.batch_size or len(requests))
        logger.info("generate_until: batch_size=%d (auto=%s, requests=%d)", effective_bs, oom_retry, len(requests))

        with _progress("Generating", len(requests)) as pbar:
            for group in groups:
                gen_kwargs = group[0].arguments[1] if len(group[0].arguments) > 1 else {}
                i = 0
                while i < len(group):
                    batch = group[i:i + effective_bs]
                    try:
                        for req, text in zip(batch, self._generate_batch(batch, gen_kwargs)):
                            outputs[req.idx] = text
                        pbar.update(len(batch))
                        i += effective_bs
                    except Exception as e:
                        if not oom_retry or not _is_oom(e):
                            raise
                        if effective_bs == 1:
                            raise RuntimeError(
                                "OOM with batch_size=1 — reduce max_new_tokens or use a smaller model"
                            ) from e
                        _clear_cuda_cache()
                        effective_bs = max(1, effective_bs // 2)
                        logger.warning("OOM — reducing batch_size to %d and retrying", effective_bs)

        return outputs

    def loglikelihood(self, requests: list["Instance"]) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = [(0.0, False)] * len(requests)
        oom_retry = self.auto_batch_size
        effective_bs = self._estimate_starting_batch_size() if oom_retry else (self.batch_size or len(requests))
        logger.info("loglikelihood: batch_size=%d (auto=%s, requests=%d)", effective_bs, oom_retry, len(requests))

        with _progress("Scoring", len(requests)) as pbar:
            i = 0
            while i < len(requests):
                batch = requests[i:i + effective_bs]
                try:
                    for req, res in zip(batch, self._loglikelihood_batch(batch)):
                        results[req.idx] = res
                    pbar.update(len(batch))
                    i += effective_bs
                except Exception as e:
                    if not oom_retry or not _is_oom(e):
                        raise
                    if effective_bs == 1:
                        raise RuntimeError(
                            "OOM with batch_size=1 — reduce max_context_length or use a smaller model"
                        ) from e
                    _clear_cuda_cache()
                    effective_bs = max(1, effective_bs // 2)
                    logger.warning("OOM — reducing batch_size to %d and retrying", effective_bs)

        return results

    def loglikelihood_rolling(self, requests: list["Instance"]) -> list[float]:
        results = [0.0] * len(requests)
        with _progress("Perplexity", len(requests)) as pbar:
            for req in requests:
                results[req.idx] = self._loglikelihood_rolling_single(req)
                pbar.update(1)
        return results

    # ── Abstract hooks ────────────────────────────────────────────────────────

    @abstractmethod
    def _generate_batch(self, batch: list["Instance"], gen_kwargs: dict) -> list[str]:
        """Generate text for one batch. Return texts in the same order as batch."""

    @abstractmethod
    def _loglikelihood_batch(self, batch: list["Instance"]) -> list[tuple[float, bool]]:
        """Score one batch of (context, continuation) pairs."""

    @abstractmethod
    def _loglikelihood_rolling_single(self, request: "Instance") -> float:
        """Compute rolling log-probability for a single request."""

    def cleanup(self) -> None:
        """Release GPU memory / close connections. Called after evaluation."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical identifier used in output filenames and result tables."""

    @property
    def max_length(self) -> int:
        return self.capabilities.max_context_length

    def _estimate_starting_batch_size(self) -> int:
        """Estimate a starting batch size from model memory and free GPU memory.

        Uses ``torch.cuda.memory_allocated()`` as a proxy for model size when
        subclasses don't override.  Subclasses with direct access to parameter
        counts should override for a more accurate estimate.

        Heuristic: activation memory per sample ≈ model_bytes * (seq_len/2048) / 16.
        We budget 40 % of current free GPU memory for batch activations.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return FALLBACK_STARTING_BS
            free_bytes, _ = torch.cuda.mem_get_info()
            model_bytes = torch.cuda.memory_allocated()
            if model_bytes == 0:
                return FALLBACK_STARTING_BS
            ctx_scale = max(1, self.max_length) / 2048
            per_sample_bytes = model_bytes * ctx_scale / 16
            bs = max(1, int(free_bytes * 0.4 / per_sample_bytes))
            bs = 1 << max(0, bs.bit_length() - 1)   # round down to power of 2
            return min(bs, 512)
        except Exception:
            return FALLBACK_STARTING_BS
