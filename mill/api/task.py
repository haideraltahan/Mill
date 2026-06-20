from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from mill.api.instance import Instance, OutputType
from mill.api.metrics import Metric

logger = logging.getLogger(__name__)


@dataclass
class Doc:
    """One evaluation sample."""
    query: str                                    # Assembled text prompt
    choices: list[str] | None = None             # Answer options (LOGPROBS tasks)
    target_index: int | list[int] | str | None = None  # Correct answer index or string
    visuals: list | None = None                  # PIL Images, paths, or URLs
    audios: list | None = None                   # Audio paths or bytes
    videos: list | None = None                   # Video paths
    instruction: str | None = None               # Optional system prompt
    metadata: dict = field(default_factory=dict) # Task-specific data (ids, splits, etc.)
    fewshot_samples: list["Doc"] = field(default_factory=list)
    task_name: str = ""

    @property
    def is_multimodal(self) -> bool:
        return bool(self.visuals or self.audios or self.videos)


@dataclass
class MillTaskConfig:
    """Full configuration for one Mill evaluation task.

    Task files export a TASKS_TABLE = [MillTaskConfig(...)] list.
    The registry auto-discovers these on import.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    name: str
    version: int = 0

    # ── Dataset ───────────────────────────────────────────────────────────────
    hf_repo: str = ""
    hf_subset: str | None = None
    hf_filter: Callable[[dict], bool] | None = None
    hf_revision: str | None = None
    hf_avail_splits: list[str] = field(default_factory=lambda: ["train", "validation", "test"])
    evaluation_splits: list[str] = field(default_factory=lambda: ["test"])
    few_shots_split: str | None = None

    # ── Prompt functions ──────────────────────────────────────────────────────
    # For text tasks: provide `prompt_function` (doc_dict -> Doc).
    # For multimodal: provide individual doc_to_* functions.
    prompt_function: Callable[[dict], Doc] | None = None
    doc_to_text: Callable[[dict], str] | str | None = None
    doc_to_target: Callable[[dict], str] | str | None = None
    doc_to_choices: Callable[[dict], list[str]] | None = None
    doc_to_visual: Callable[[dict], list] | str | None = None
    doc_to_audio: Callable[[dict], list] | str | None = None
    doc_to_video: Callable[[dict], list] | str | None = None

    # ── Generation config ─────────────────────────────────────────────────────
    output_type: OutputType = OutputType.GENERATIVE
    generation_size: int | None = 256
    stop_sequences: list[str] = field(default_factory=list)
    n_shots: int = 0

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics: list[Metric] = field(default_factory=list)

    # ── Documentation (shown in `mill ls` preview) ────────────────────────────
    description: str = ""
    categories: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    paper_url: str = ""
    approx_num_samples: dict[str, int] = field(default_factory=dict)


@dataclass
class MillBenchmarkConfig:
    """Groups one or more tasks under a named benchmark.

    Benchmark files export a BENCHMARKS_TABLE = [MillBenchmarkConfig(...)] list.
    A benchmark with a single task (e.g. gsm8k) is still a benchmark — it
    provides a stable public name that is decoupled from the internal task name.

    Attributes:
        name:               Public benchmark name (e.g. "gsm8k", "mmlu").
        task_names:         Registered task names that belong to this benchmark.
        metric_names:       Which metric(s) to aggregate across tasks and report.
        weighted_aggregate: False = unweighted mean across tasks (standard for
                            MMLU/MATH); True = weight by each task's sample count.
    """
    name: str
    task_names: list[str]
    metric_names: list[str] = field(default_factory=list)
    weighted_aggregate: bool = False
    description: str = ""
    categories: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    paper_url: str = ""


class MillTask(ABC):
    """Wraps a MillTaskConfig with dataset loading and request building."""

    def __init__(self, config: MillTaskConfig):
        self.config = config
        self._docs: dict[str, list[Doc]] = {}  # split -> docs

    @property
    def name(self) -> str:
        return self.config.name

    # ── Dataset ───────────────────────────────────────────────────────────────

    def download(self, limit: int | None = None) -> None:
        from datasets import load_dataset
        splits_to_load = list(self.config.evaluation_splits)
        if self.config.few_shots_split and self.config.few_shots_split not in splits_to_load:
            splits_to_load.append(self.config.few_shots_split)
        for split in splits_to_load:
            try:
                ds = load_dataset(
                    self.config.hf_repo,
                    name=self.config.hf_subset,
                    split=split,
                    revision=self.config.hf_revision,
                )
                if self.config.hf_filter:
                    ds = ds.filter(self.config.hf_filter)
                if limit is not None:
                    ds = ds.select(range(min(limit, len(ds))))
                self._docs[split] = [self._make_doc(row) for row in ds]
            except Exception as e:
                logger.warning(f"Could not load split '{split}' for task '{self.name}': {e}")
                self._docs[split] = []

    def _make_doc(self, row: dict) -> Doc:
        if self.config.prompt_function:
            return self.config.prompt_function(row)
        return Doc(
            query=self._apply_fn(self.config.doc_to_text, row, ""),
            target_index=self._apply_fn(self.config.doc_to_target, row, None),
            choices=self._apply_fn(self.config.doc_to_choices, row, None) if self.config.doc_to_choices else None,
            visuals=self._apply_fn(self.config.doc_to_visual, row, None) if self.config.doc_to_visual else None,
            audios=self._apply_fn(self.config.doc_to_audio, row, None) if self.config.doc_to_audio else None,
            videos=self._apply_fn(self.config.doc_to_video, row, None) if self.config.doc_to_video else None,
            metadata={"raw": row},
            task_name=self.name,
        )

    @staticmethod
    def _apply_fn(fn_or_field: Callable | str | None, row: dict, default: Any) -> Any:
        if fn_or_field is None:
            return default
        if callable(fn_or_field):
            return fn_or_field(row)
        return row.get(fn_or_field, default)

    def get_docs(self, split: str | None = None) -> list[Doc]:
        if not self._docs:
            self.download()
        if split:
            return self._docs.get(split, [])
        return [doc for docs in self._docs.values() for doc in docs]

    def fewshot_docs(self, n: int, split: str | None = None, seed: int = 42) -> list[Doc]:
        src_split = split or self.config.few_shots_split or self.config.evaluation_splits[0]
        all_docs = self._docs.get(src_split, [])
        rng = random.Random(seed)
        return rng.sample(all_docs, min(n, len(all_docs)))

    # ── Request building ──────────────────────────────────────────────────────

    def build_all_requests(self) -> list[Instance]:
        n_shot = self.config.n_shots
        instances = []
        fewshots = self.fewshot_docs(n_shot) if n_shot > 0 else []
        idx = 0
        doc_idx = 0
        for split in self.config.evaluation_splits:
            for doc in self._docs.get(split, []):
                doc.fewshot_samples = fewshots
                context = self._build_context(doc)
                if self.config.output_type == OutputType.LOGPROBS:
                    for choice_idx, choice in enumerate(doc.choices or []):
                        instances.append(Instance(
                            request_type=OutputType.LOGPROBS,
                            doc=doc,
                            arguments=(context, choice),
                            idx=idx,
                            metadata={"task": self.name, "split": split, "choice_idx": choice_idx, "n_shot": n_shot, "doc_id": doc_idx},
                        ))
                        idx += 1
                    doc_idx += 1
                else:
                    gen_kwargs = {"max_new_tokens": self.config.generation_size, "stop": self.config.stop_sequences}
                    instances.append(Instance(
                        request_type=self.config.output_type,
                        doc=doc,
                        arguments=(context, gen_kwargs),
                        idx=idx,
                        metadata={"task": self.name, "split": split, "n_shot": n_shot},
                    ))
                    idx += 1
        return instances

    def _build_context(self, doc: Doc) -> str | Any:
        prefix = doc.instruction or ""
        if doc.is_multimodal:
            from mill.api.protocol import ChatMessages
            return ChatMessages.from_text_and_images(
                text=prefix + self._format_fewshot(doc) + doc.query,
                images=doc.visuals or [],
            )
        return prefix + self._format_fewshot(doc) + doc.query

    def _format_fewshot(self, doc: Doc) -> str:
        if not doc.fewshot_samples:
            return ""
        parts = []
        for ex in doc.fewshot_samples:
            gold = ex.target_index if isinstance(ex.target_index, str) else (ex.choices[ex.target_index] if ex.choices else "")
            # Choices use a leading space for logprob scoring (e.g. " A"); appending
            # directly gives "Answer: A". Tasks whose gold has no leading space get a
            # newline instead so the demo reads naturally.
            sep = "" if (gold and gold[0] == " ") else "\n"
            parts.append(f"{ex.query}{sep}{gold}")
        return "\n\n".join(parts) + "\n\n"

    # ── Metrics ───────────────────────────────────────────────────────────────

    def process_results(self, doc: Doc, responses: list) -> dict[str, float]:
        results = {}
        for metric in self.config.metrics:
            if self.config.output_type == OutputType.LOGPROBS:
                best_idx = max(range(len(responses)), key=lambda i: responses[i][0])
                results[metric.name] = metric.sample_level_fn(doc, best_idx)
            else:
                results[metric.name] = metric.sample_level_fn(doc, responses[0] if responses else "")
        return results

    def aggregate_metrics(self, all_results: list[dict[str, float]]) -> dict[str, float | None]:
        aggregated = {}
        for metric in self.config.metrics:
            values = [r[metric.name] for r in all_results if metric.name in r]
            score, stderr = metric.aggregate(values)
            aggregated[metric.name] = score
            if stderr is not None:
                aggregated[f"{metric.name}_stderr"] = stderr
        return aggregated


class ConfigurableTask(MillTask):
    """A MillTask fully driven by a MillTaskConfig (no subclassing needed)."""

    def __init__(self, config: MillTaskConfig):
        super().__init__(config)
