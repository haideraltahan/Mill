---
name: adding-a-model
description: Add a new model backend to Mill — pick the right capability interface, register it, expose its config, and document it. Use when a contributor wants to wire a new inference engine or model family (LLM/VLM, CLIP-style encoder, or supervised classifier) into Mill.
---

# Adding a model backend to Mill

Most "new models" need **no code** — a HuggingFace, vLLM, or LiteLLM model is just a
config (see `mill/models/configs/` and `docs/reference/models.mdx`). Write a new
backend only when the inference engine or model *interface* is genuinely new.

Worked examples — read the closest one first:
- Generative LLM/VLM: `mill/models/transformers.py`, `mill/models/vllm.py`
- API backend (generative only): `mill/models/litellm.py`
- CLIP-style zero-shot: `mill/models/clip.py`
- Supervised fixed-head classifier: `mill/models/timm.py`

---

## Phase 1 — Pick the capability interface

A backend's abilities come from which interface it subclasses (`mill/api/model.py`).
Pick by *what the model does*, which determines which `task_type`s it can serve:

| Interface | Implement | Serves task types | Examples |
|---|---|---|---|
| `GenerativeModel` | `_generate_batch`, `_loglikelihood_batch`, `_loglikelihood_rolling_single` | `GENERATIVE_QA`, `MULTIPLE_CHOICE`, `PERPLEXITY` | HF, vLLM, LiteLLM |
| `ZeroShotClassifier` | `zero_shot_classify` | `ZERO_SHOT_CLASSIFICATION` | open_clip |
| `SupervisedClassifier` | `classify` | `SUPERVISED_CLASSIFICATION` | timm |

The base class handles batching, progress bars, ordering, and automatic OOM
retry/back-off — implement only the batch hooks, not the public methods.

## Phase 2 — Implement and register

```python
from mill.api.model import GenerativeModel, ModelCapabilities  # or ZeroShotClassifier / SupervisedClassifier
from mill.api.registry import register_model

@register_model("my-backend", "alias")          # names usable as the model arg
class MyModel(GenerativeModel):
    def __init__(self, path: str, **kwargs):
        self._path = path
        self.capabilities = ModelCapabilities(
            modalities={"text"},                 # what it can ingest; gates multimodal tasks
            max_context_length=4096,
            supports_logprobs=True,              # False blocks LOGPROBS/PERPLEXITY tasks
            supports_chat_template=False,
        )
        # load weights / open the client here

    @property
    def model_name(self) -> str:                 # identity used for output caching
        return self._path
    # ... implement the batch hooks for your interface ...

    def cleanup(self) -> None:                   # free GPU memory / close clients
        ...
```

Rules that keep results correct and cacheable:
- `model_name` must be **unique per weight set** (open_clip uses
  `f"{arch}/{pretrained}"`), so two checkpoints don't collide in the output cache.
- Set `capabilities` honestly — `modalities` and `supports_logprobs` are how the
  evaluator rejects unsupported (model, task) pairs with a clear error instead of
  silently producing wrong numbers.
- Gate heavy/optional imports (e.g. `open_clip`, `timm`, `vllm`) inside `__init__`
  with a helpful `ImportError` naming the pip extra to install.
- Add the dependency as an optional extra in `pyproject.toml` if it isn't core.

## Phase 3 — Make it loadable from config

Add a per-family Python config under `mill/models/configs/<family>/<model>.py`
(opencompass style) so users can run it by file path. The `type` field maps to the
registry name:

```python
model = dict(type="my-backend", abbr="my-model", path="org/my-model", run_cfg=dict(batch_size=8))
```

## Phase 4 — Validate and document

- Run it on a benchmark its interface supports and sanity-check the score against a
  known number for that model.
- Add a backend row to the overview table **and** a section (config dict,
  `model_args`, requirements, *when to use it*) in `docs/reference/models.mdx`.
- Add a bullet to `docs/changelog.mdx`.

Live docs are at **pymill.com**; just commit the `.mdx` changes — the maintainer
publishes to Mintlify separately.

## Definition of done

- [ ] Correct interface chosen; only batch hooks implemented.
- [ ] `@register_model` with a unique, cache-safe `model_name`; honest `capabilities`.
- [ ] Optional deps gated with a clear `ImportError` + a `pyproject.toml` extra.
- [ ] Config file under `mill/models/configs/`; validated on a real benchmark.
- [ ] `docs/reference/models.mdx` section + changelog updated.
