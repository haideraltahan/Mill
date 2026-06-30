from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from mill.api.instance import OutputType

_METRIC_REGISTRY: dict[str, "Metric"] = {}


@dataclass
class Metric:
    name: str
    higher_is_better: bool
    output_type: OutputType
    sample_level_fn: Callable  # (doc, response) -> float
    corpus_level_fn: Callable = field(default=None)  # ([float]) -> float, default mean
    bootstrap_stderr: bool = True

    def __post_init__(self):
        if self.corpus_level_fn is None:
            self.corpus_level_fn = _mean

    def aggregate(self, values: list[float]) -> tuple[float, float | None]:
        """Returns (score, stderr). stderr is None when bootstrap_stderr=False."""
        score = self.corpus_level_fn(values)
        stderr = _bootstrap_stderr(values, self.corpus_level_fn) if self.bootstrap_stderr and len(values) > 1 else None
        return score, stderr


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bootstrap_stderr(values: list[float], fn: Callable, n_resamples: int = 1000, seed: int = 1234) -> float:
    rng = random.Random(seed)
    n = len(values)
    scores = []
    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        scores.append(fn(sample))
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return variance ** 0.5


def register_metric(
    name: str,
    higher_is_better: bool = True,
    output_type: OutputType = OutputType.GENERATIVE,
    corpus_level_fn: Callable | None = None,
    bootstrap_stderr: bool = True,
):
    """Decorator to register a sample-level metric function."""
    def decorator(fn: Callable) -> Callable:
        _METRIC_REGISTRY[name] = Metric(
            name=name,
            higher_is_better=higher_is_better,
            output_type=output_type,
            sample_level_fn=fn,
            corpus_level_fn=corpus_level_fn,
            bootstrap_stderr=bootstrap_stderr,
        )
        return fn
    return decorator


def get_metric(name: str) -> Metric:
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Metric '{name}' not found. Available: {list(_METRIC_REGISTRY)}")
    return _METRIC_REGISTRY[name]


def list_metrics() -> list[str]:
    return sorted(_METRIC_REGISTRY.keys())


# ── Built-in metrics ──────────────────────────────────────────────────────────

@register_metric("exact_match", higher_is_better=True, output_type=OutputType.GENERATIVE)
def exact_match(doc, response: str) -> float:
    gold = doc.target_index if isinstance(doc.target_index, str) else (doc.choices[doc.target_index] if doc.choices else "")
    return float(response.strip() == gold.strip())


@register_metric("acc", higher_is_better=True, output_type=OutputType.LOGPROBS)
def acc(doc, response: int) -> float:
    # response = index of the highest-logprob choice
    return float(response == doc.target_index)


@register_metric("acc_norm", higher_is_better=True, output_type=OutputType.LOGPROBS)
def acc_norm(doc, response: tuple) -> float:
    logprob, _ = response
    # Caller is responsible for computing length-normalized logprob before passing
    return float(logprob)


@register_metric("perplexity", higher_is_better=False, output_type=OutputType.PERPLEXITY,
                 corpus_level_fn=lambda vals: 2 ** (-sum(vals) / len(vals)) if vals else float("inf"),
                 bootstrap_stderr=False)
def perplexity(doc, response: float) -> float:
    return response  # log-likelihood per token, aggregated by corpus_level_fn


@register_metric("contains_answer", higher_is_better=True, output_type=OutputType.GENERATIVE)
def contains_answer(doc, response: str) -> float:
    gold = doc.target_index if isinstance(doc.target_index, str) else (doc.choices[doc.target_index] if doc.choices else "")
    return float(gold.strip().lower() in response.strip().lower())
