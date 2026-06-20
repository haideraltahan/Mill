# Mill

**Unified multi-modal evaluation framework** — one tool for text, image, video, and audio benchmarks.

## Install

```bash
cd Mill
pip install -e ".[dev]"

# Optional extras
pip install -e ".[vllm]"     # vLLM backend
pip install -e ".[litellm]"  # OpenAI / Anthropic API backend
pip install -e ".[video]"    # video decoding (decord)
```

## Quick start

```bash
# Text evaluation — local HF model
mill --output_dir ./results eval \
     meta-llama/Meta-Llama-3-8B-Instruct mmlu,mmlu_pro \
     --model_args dtype=bfloat16,batch_size=8

# Chain-of-thought benchmark — instruction-tuned model config file
mill --output_dir ./results eval \
     mill/models/configs/qwen/qwen2_5_7b_instruct.py mmlu_pro

# API model (generative tasks only)
mill --output_dir ./results eval litellm mmlu_pro \
     --model_args model=gpt-4o

# Distributed SLURM scheduling
mill --output_dir /scratch/results schedule \
     meta-llama/Meta-Llama-3-8B-Instruct mmlu \
     --n_shots 0,5 --cluster auto

# Collect results after jobs finish
mill --output_dir /scratch/results collect --check

# Browse registered benchmarks and tasks (interactive TUI)
mill ls
```

## Architecture

```
mill/
├── api/
│   ├── task.py        MillTaskConfig dataclass + MillTask base class
│   ├── model.py       MillModel ABC (generate_until / loglikelihood / loglikelihood_rolling)
│   ├── instance.py    Instance (one model request) + OutputType enum
│   ├── metrics.py     Metric dataclass + @register_metric + bootstrap CI
│   ├── registry.py    Central registry for models, tasks, metrics
│   └── protocol.py    ChatMessages — multimodal message protocol
├── models/
│   ├── transformers.py  HuggingFace AutoModel (text + multimodal)
│   ├── vllm.py          vLLM backend
│   ├── litellm.py       OpenAI / Anthropic / LiteLLM API
│   ├── loader.py        load_model_from_config / load_model_from_file
│   └── configs/         Per-family Python config files (opencompass style)
│       ├── qwen/
│       ├── llama/
│       └── internvl/
├── tasks/
│   ├── mmlu/task.py      MMLU (57 subjects, log-prob scoring)
│   └── mmlu_pro/task.py  MMLU-Pro (10-option, generative chain-of-thought)
├── output.py       Feather-based caching (never recompute completed jobs)
├── evaluator.py    Core eval loop: requests → model → metrics → cache
├── pipeline.py     Orchestrator: skip done → load model → run → display
├── cli.py          mill eval / schedule / collect / ls
└── scheduler/
    ├── slurm.py        SLURM job array generation + submission
    ├── clusters.yaml   Per-cluster queue limits, partitions, GPU types
    └── template.sbatch SLURM array template
```

## Adding a new task

Create `mill/tasks/my_task/task.py`:

```python
from mill.api.instance import OutputType
from mill.api.metrics import get_metric
from mill.api.task import Doc, MillTaskConfig

def my_prompt(row: dict) -> Doc:
    return Doc(
        query=f"Question: {row['question']}\nAnswer:",
        target_index=row["answer"],
    )

TASKS_TABLE = [
    MillTaskConfig(
        name="my_task",
        hf_repo="org/my_dataset",
        evaluation_splits=["test"],
        prompt_function=my_prompt,
        output_type=OutputType.GENERATIVE,
        generation_size=128,
        metrics=[get_metric("exact_match")],
    )
]
```

Then: `mill --output_dir ./results eval hf my_task --model_args path=... --task_paths mill/tasks/my_task`

## Adding a new model

Create `mill/models/configs/my_family/my_model.py`:

```python
from mill.models.transformers import TransformersModel

model = dict(
    type=TransformersModel,
    abbr="my-model-7b",
    path="org/my-model-7b",
    modalities=["text", "image"],
    dtype="bfloat16",
    max_context_length=32768,
    run_cfg=dict(num_gpus=1, batch_size=4),
)
```

Then: `mill --output_dir ./results eval mill/models/configs/my_family/my_model.py <tasks>`

## Output cache

All results are written to `output_dir` (default `./mill_results`, set with `--output_dir`).
Configuration and SLURM files live separately under `cache_dir` (default `~/.cache/mill`).

```
<output_dir>/                        # e.g. ./mill_results
  outputs/
    {model_abbr}/
      {task_name}_{n_shot}shot.f     # Per-sample results (Apache Feather)
  aggregate.csv                       # Long-format summary: one row per
                                      # (model, task, n_shot, metric) with
                                      # columns metric, performance, stderr
```

Re-running `mill eval` with the same model + task skips inference automatically.
Use `mill collect --check` to see which jobs are missing.
