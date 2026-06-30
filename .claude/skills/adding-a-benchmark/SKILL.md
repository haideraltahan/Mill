---
name: adding-a-benchmark
description: Add a new benchmark/task to Mill end to end — locate the source benchmark, replicate how it scores, write the task.py, validate against the published baseline, and document it. Use when a contributor wants to add, port, or wire up a new evaluation (text or multimodal) into Mill.
---

# Adding a benchmark to Mill

A benchmark is **done** only when (1) its task code is registered, (2) Mill's score
is validated against the original implementation's published number, and (3) it is
documented with that comparison. Skipping (2) or (3) produces numbers nobody can
trust. Work through every phase below in order.

Worked examples already in the repo — read the closest one before you start:
- Text, log-prob MCQ: `mill/tasks/mmlu/task.py`
- Text, generative chain-of-thought MCQ: `mill/tasks/mmlu_pro/task.py`
- Multimodal, dual rendering (CLIP zero-shot **and** VLM generative-MCQ): `mill/tasks/cifar10/task.py`, `mill/tasks/mmmu_pro/task.py`

---

## Phase 1 — Find the source of truth (do this before writing code)

You are reproducing someone else's number. Find it first, or you have nothing to
validate against.

1. **Locate the original benchmark repo and paper.** Search for the official
   implementation (e.g. the `MMMU-Benchmark/MMMU` repo, `TIGER-Lab/MMLU-Pro`,
   `LAION-AI/CLIP_benchmark`). Note the URL — it goes in `paper_url` and in the docs.
2. **Read how the official metric is computed**, not just what it is. Open their
   eval/grader script and answer:
   - How is the prediction extracted from a raw model output (regex? "Answer:" line?
     option-text match? argmax over embeddings)?
   - How are ties / unparseable answers handled (random guess? zero credit)?
   - What exact prompt/template and n-shot setting produced the published score?
   - Is the metric an unweighted mean over subtasks, or weighted by sample count?
   - Which split and subset is scored?
   If Mill's grading diverges from theirs, your number will diverge — port their
   logic faithfully (see `mill/tasks/mmmu_pro/utils.py`, a line-for-line port of the
   official MMMU-Pro grader).
3. **Record the published baseline** for at least one specific model
   (number + source URL). This is the target you must hit in Phase 4. If no public
   per-model number exists, plan to record Mill's own number as the *initial*
   baseline and say so explicitly.
4. **Find the dataset on the HF Hub.** Note `hf_repo`, `hf_subset`, the split names,
   and the column names you'll read. For non-standard formats (WebDataset /
   clip_benchmark exports) you'll set `hf_builder="webdataset"` and
   `hf_data_files={split: "hf://datasets/<repo>/<split>/*.tar"}` (see `cifar10`).

> Bring the published number + the grading details back to the maintainer before
> coding if anything is ambiguous. A wrong baseline is worse than no baseline.

---

## Phase 2 — Write the task

Create `mill/tasks/<name>/task.py`. It is auto-discovered on import (any file under
`mill/tasks/` that exports `TASKS_TABLE`). Keep heavy/ported helpers in a sibling
`utils.py`.

A task file exports:
- `TASKS_TABLE: list[MillTaskConfig]` — one entry per concrete task.
- `BENCHMARKS_TABLE: list[MillBenchmarkConfig]` — the public benchmark name(s).

### Prompt function `(row: dict) -> Doc`

Assemble the prompt and the gold answer:

```python
from mill.api.task import Doc

def my_prompt(row: dict) -> Doc:
    return Doc(
        query="...assembled prompt text...",
        choices=["A", "B", "C", "D"],     # options (MCQ / zero-shot)
        target_index=row["answer"],       # gold index, or gold letter as a str
        visuals=[row["image"]],           # PIL.Image / path / URL, multimodal only
        metadata={"id": row["id"]},       # anything the metric needs at grading time
        task_name="my_task",
    )
```

### Choose the task shape

`task_type` (in `mill/api/taxonomy.py`) is the **primary axis** — it decides which
model interface serves the task. `output_type` is only a *scoring* detail of the
generative family.

| `task_type` | Served by | `output_type` | Use for |
|---|---|---|---|
| `MULTIPLE_CHOICE` | LLM/VLM | `LOGPROBS` | rank options by log-prob (MMLU) |
| `MULTIPLE_CHOICE` | LLM/VLM | `GENERATIVE` | model writes a letter, you parse it (MMLU-Pro) |
| `GENERATIVE_QA` | LLM/VLM | `GENERATIVE` | free-form answer + scorer |
| `PERPLEXITY` | LLM | `PERPLEXITY` | rolling log-likelihood |
| `ZERO_SHOT_CLASSIFICATION` | CLIP-style | — | image↔text similarity over labels (CIFAR-10) |
| `SUPERVISED_CLASSIFICATION` | timm | — | fixed pretrained head (ImageNet via ResNet) |

Set `input_modalities=["image", "text"]` for anything multimodal — the evaluator
rejects models that can't ingest those inputs with a clear error instead of garbage
scores.

### Write the metric

Reuse a built-in metric (`get_metric("acc")`) when grading is a plain index/letter
match. Otherwise register one — and make it a faithful copy of the official grader:

```python
from mill.api.instance import OutputType
from mill.api.metrics import get_metric, register_metric

@register_metric("my_acc", higher_is_better=True, output_type=OutputType.GENERATIVE)
def my_acc(doc: Doc, response: str) -> float:
    pred = parse_answer(response)          # mirror the upstream extraction logic
    return float(pred == str(doc.target_index).strip().upper())
```

For reproducible random fallbacks (unparseable answers, option shuffles) use
`mill.utils.sample_rng(...)` keyed by a stable per-sample id — never the bare
`random` module — so scores are deterministic under `--seed`. There are already
three answer-letter parsers in the repo (`mill/utils.py`,
`mill/tasks/mmmu_pro/utils.py`, `mill/tasks/mmlu_pro/utils.py`); reuse
`mill.utils.parse_multi_choice_response` for a generic MCQ rather than adding a
fourth, unless fidelity to a specific upstream grader requires its own port.

### Assemble `MillTaskConfig` / `MillBenchmarkConfig`

Fill the documentation fields (`description`, `categories`, `capabilities`,
`paper_url`, `approx_num_samples`) — they power `mill ls` and the docs. For a
benchmark that has both a CLIP and a VLM rendering, register both tasks and set
`pick_variant_by_model=True` so the pipeline runs the one the model supports (see
`cifar10`/`mmmu_pro`).

---

## Phase 3 — Smoke test

```bash
# Tiny run to confirm it loads, prompts, and grades without error
mill --output_dir /tmp/mill_smoke eval <small-model> <task> --limit 16
```

Confirm in `mill ls` that the task and benchmark appear with the right metadata.
A task that lives outside `mill/tasks/` is loaded with `--task_paths /path/to/dir`.

---

## Phase 4 — Validate against the baseline (non-negotiable)

```bash
mill --output_dir ./results eval <model> <task> --model_args dtype=bfloat16
mill --output_dir ./results collect --metric <metric>
```

Read the rollup row from `./results/aggregate.csv` (the row whose `task` equals the
benchmark name, e.g. `mmlu,mmlu`). Compare Mill's score to the Phase 1 published
number:
- **Within ~1 standard error** → reproduced. Record it.
- **Off by more than a couple of points** → stop. The prompt, n-shot, split, or
  grader almost certainly differs from the original. Re-read the upstream script
  before "fixing" the number. Do not paper over a gap.

If no published per-model number exists, record Mill's measured number as the
**initial baseline** and label it as such — don't invent a "Reported" figure.

---

## Phase 5 — Document it (part of the task, not optional)

1. **Reproducibility page** `docs/reproducibility/<name>.mdx`. Copy the template in
   `docs/reproducibility/overview.mdx` ("Adding a new benchmark"). Fill the
   evaluation-config table, the reproduce command, and the Mill-vs-Reported cards +
   per-model table from your `aggregate.csv` and the Phase 1 baseline.
2. **Overview** — add a `<Card>` for the benchmark in
   `docs/reproducibility/overview.mdx`.
3. **Task reference** — add a row to the built-in benchmarks table in
   `docs/reference/tasks.mdx` (name, modality, output type, default n-shot, metric).
4. **Nav** — add the new page path to the `Reproducibility` group in `docs.json`.
5. **Changelog** — add a bullet under the current version in `docs/changelog.mdx`.

Numbers are written as fractions in `aggregate.csv` (e.g. `0.5378`); the docs show
percentages to match how papers report them, and the `±` column is the bootstrap
standard error in percentage points.

Live docs are at **pymill.com**; the maintainer publishes them to Mintlify
separately, so just commit the `.mdx` changes.

---

## Phase 6 — Submit a PR

Put the work on its own branch, commit only the files this benchmark touches
(task, model config, docs), and open a pull request. Keep the PR description
**concise** and write it in the **first person** — describe what I did and how I
validated it, not a generic feature blurb:

```bash
git checkout -b add-<name>
git add mill/tasks/<name> docs/... mill/models/configs/...
git commit  # concise, first-person message
git push -u origin add-<name>
gh pr create --title "Add <name> benchmark" --body "<concise, first-person summary>"
```

In the body, state the score I measured and the baseline I compared against
(e.g. "I reproduced X within one standard error" or "I recorded X as the initial
baseline") in one or two sentences. No filler.

---

## Definition of done

- [ ] Source repo + paper located; official grading logic understood and mirrored.
- [ ] `mill/tasks/<name>/task.py` exports `TASKS_TABLE` (+ `BENCHMARKS_TABLE`); shows in `mill ls`.
- [ ] Reproducible randomness via `sample_rng`; metric matches upstream.
- [ ] Score validated against the published baseline (or recorded as an explicit initial baseline).
- [ ] Reproducibility page + overview card + tasks-table row + `docs.json` nav + changelog updated.
- [ ] Changes committed on a branch and opened as a PR with a concise, first-person summary.
