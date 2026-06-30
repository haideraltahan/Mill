from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mill.api.taxonomy import TaskType


class OutputType(str, Enum):
    """How a generative response is produced/scored (a detail of the generative
    task family). The semantic task shape lives in ``TaskType``."""
    GENERATIVE = "generate_until"
    LOGPROBS = "loglikelihood"
    PERPLEXITY = "loglikelihood_rolling"


@dataclass
class TokenCounts:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Instance:
    """One model request — built from a Doc by a task."""

    request_type: "OutputType | TaskType"  # OutputType for generative, TaskType for classification
    doc: Any  # Doc — forward ref to avoid circular import
    arguments: tuple  # (context_or_ChatMessages, gen_kwargs_or_continuation_or_labels)
    idx: int
    metadata: dict = field(default_factory=dict)  # task_name, doc_id, split, n_shot
    resps: list = field(default_factory=list)
    filtered_resps: list = field(default_factory=list)
    token_counts: TokenCounts | None = None
