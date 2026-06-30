"""Model and task taxonomy.

Two orthogonal axes drive capability matching in Mill:

``Modality``
    What a model *ingests* (text, image, audio, video).

``TaskType``
    What a task *asks* — the semantic shape of the problem. This is the primary
    axis: a task declares its ``TaskType`` and the evaluator dispatches to the
    matching model-capability interface (see ``mill.api.model``).

``OutputType`` (in ``mill.api.instance``) is a *scoring* detail of the
generative family only — e.g. whether a multiple-choice task is graded by the
generated answer letter or by per-choice log-probability. It is not a task type.
"""
from __future__ import annotations

from enum import Enum


class Modality(str, Enum):
    """A media type a model can consume."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class TaskType(str, Enum):
    """The semantic shape of an evaluation task.

    Generative family (served by ``GenerativeModel`` — LLMs/VLMs):
        GENERATIVE_QA       Free-form text generation.
        MULTIPLE_CHOICE     Pick one option; graded by generated letter or by
                            per-choice log-probability (see ``OutputType``).
        PERPLEXITY          Rolling log-likelihood over a sequence.

    Classification family (served by dedicated encoders):
        ZERO_SHOT_CLASSIFICATION    CLIP-style: score inputs against candidate
                                    text labels via embedding similarity.
        SUPERVISED_CLASSIFICATION   Fixed label head (timm vision, audio-only).
    """
    GENERATIVE_QA = "generative_qa"
    MULTIPLE_CHOICE = "multiple_choice"
    PERPLEXITY = "perplexity"
    ZERO_SHOT_CLASSIFICATION = "zero_shot_classification"
    SUPERVISED_CLASSIFICATION = "supervised_classification"


# Task types served by a ``GenerativeModel`` (LLM/VLM) backend.
GENERATIVE_TASK_TYPES: frozenset[TaskType] = frozenset(
    {TaskType.GENERATIVE_QA, TaskType.MULTIPLE_CHOICE, TaskType.PERPLEXITY}
)

# Task types served by a classification encoder backend.
CLASSIFICATION_TASK_TYPES: frozenset[TaskType] = frozenset(
    {TaskType.ZERO_SHOT_CLASSIFICATION, TaskType.SUPERVISED_CLASSIFICATION}
)
