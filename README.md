# Mill

**One tool to evaluate any model on any modality.**

Mill is a unified multi-modal evaluation framework for **text, image, video, and audio**
benchmarks. Running a benchmark, reproducing a published number, and scaling across a
cluster all work the same way — whatever the model, whatever the modality.

## Why Mill

The evaluation ecosystem is fragmented: text benchmarks live in one harness, vision in
another, each with its own model abstraction, caching story, and result format. Stitching
them together — and trusting the numbers — is the hard part.

Mill brings the best ideas from across that ecosystem into a single, consistent interface:

- **Multi-modal, one interface.** Text LLMs, vision-language models, CLIP-style encoders,
  and supervised vision models all run through `mill eval`. Mill matches each model to the
  task renderings it can actually serve.
- **Reproducible by design.** Seeded, deterministic runs with bootstrap confidence
  intervals — and a [reproducibility suite](https://pymill.com) that validates Mill's
  scores against published baselines.
- **Never recompute.** Feather output caching skips any `(model, task, n-shot)` job that
  already finished, so reruns and sweeps are cheap.
- **Scales to a cluster.** Built-in SLURM scheduling fans a `(models × tasks × n-shots)`
  sweep across a cluster, then collects the results into one table.
- **Easy to extend.** Add a new benchmark or model backend with a single Python file.

## Install

```bash
git clone https://github.com/haideraltahan/Mill
cd Mill
pip install -e ".[dev]"
```

Add the backends you need:

```bash
pip install -e ".[vllm]"        # vLLM — high-throughput local generation
pip install -e ".[litellm]"     # OpenAI, Anthropic, and 100+ API providers
pip install -e ".[vision]"      # CLIP (open_clip) + timm vision backends
pip install -e ".[video]"       # video decoding (decord)
```

## Quick start

```bash
# Text — local HF model on MMLU
mill --output_dir ./results eval "Qwen/Qwen3-0.6B-Base[dtype=bfloat16]" mmlu

# Vision — CLIP zero-shot on CIFAR-10
mill --output_dir ./results eval \
     "clip[path=ViT-B-32,pretrained=laion2b_s34b_b79k]" cifar10

# Browse benchmarks and tasks (interactive TUI)
mill ls
```

## Documentation

Full documentation — installation, guides, CLI/API reference, reproducibility, and
contributing — lives at **[pymill.com](https://pymill.com)**.

## Contributing

Mill ships [Claude Code](https://claude.com/claude-code) skills that guide adding a
benchmark or a model backend end to end (`.claude/skills/`). Start with the
[Contributing guide](https://pymill.com/docs/contributing/add-a-benchmark).

## License

MIT — see [LICENSE](LICENSE).
