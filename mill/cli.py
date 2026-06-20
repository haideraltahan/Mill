"""Mill CLI — powered by python-fire.

Each method on the Mill class becomes a subcommand automatically.
Shared state (cache_dir, output_dir, limit) is set on __init__.

Directory layout::

    cache_dir   (~/.cache/mill)   — configuration: clusters.yaml, SLURM job files, logs
    output_dir  (required)        — evaluation results: feather files, aggregate.f

Usage examples::

    mill --output_dir /scratch/results eval meta-llama/Meta-Llama-3-8B-Instruct gsm8k,mmlu
    mill --output_dir ./results eval configs/qwen/qwen2_5_vl_7b.py vqav2
    mill --output_dir /scratch/results --cache_dir /home/user/.mill schedule llama3-8b gsm8k
    mill --output_dir /scratch/results collect
    mill ls
"""
from __future__ import annotations

import logging
from pathlib import Path  # still used in eval() for Path(model).exists()
from typing import List, Optional, Union

from mill.constants import CACHE_DIR, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


class Mill:
    """Mill — unified multi-modal evaluation framework.

    Args:
        output_dir: Where evaluation results are written (feather files, aggregate.f).
                    Default: ./mill_results
        cache_dir:  Where Mill stores configuration and scheduler files (clusters.yaml,
                    SLURM job CSVs, logs). Default: ~/.cache/mill
        limit:      Cap samples per task (useful for smoke tests). Default: no limit
    """

    def __init__(
        self,
        output_dir: str = str(OUTPUT_DIR),
        cache_dir: str = str(CACHE_DIR),
        limit: Optional[int] = None,
    ):
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.limit = limit

    # ── eval ──────────────────────────────────────────────────────────────────

    def eval(
        self,
        model: str,
        tasks: Union[str, List[str]],
        model_args: str = "",
        task_paths: Optional[str] = None,
    ):
        """Run evaluation locally.

        Args:
            model: HF model ID, registry name (hf/vllm/litellm), or path to a
                   Python config file (mill/models/configs/.../*.py).
            tasks: Task name(s). A string is treated as comma-separated.
                   e.g. "gsm8k,mmlu" or ["gsm8k", "mmlu"]
            model_args: Extra key=value pairs forwarded to the model constructor,
                        e.g. "dtype=bfloat16,batch_size=8".
            task_paths: Comma-separated extra directories to scan for custom tasks.
        """
        import mill.tasks  # trigger built-in task auto-discovery
        from mill.pipeline import Pipeline

        task_list = _to_list(tasks)
        extra_paths = [p.strip() for p in task_paths.split(",")] if task_paths else None

        if Path(model).exists():
            from mill.models.loader import load_model_from_file
            resolved_model = load_model_from_file(model)
        else:
            kw = _parse_kv(model_args)
            resolved_model = {"type": model, **kw}

        Pipeline(
            model=resolved_model,
            tasks=task_list,
            output_dir=self.output_dir,
            limit=self.limit,
            task_paths=extra_paths,
        ).run()

    # ── schedule ──────────────────────────────────────────────────────────────

    def schedule(
        self,
        models: Union[str, List[str]],
        tasks: Union[str, List[str]],
        n_shots: Union[str, List[int]] = "0",
        cluster: str = "auto",
        local: bool = False,
        dry_run: bool = False,
        venv_path: str = "",
        extra_task_paths: str = "",
        minutes_per_eval: int = 0,
    ):
        """Generate and submit a SLURM job array (or run locally).

        Args:
            models: Model path(s) or abbreviation(s). Comma-separated string or list.
            tasks: Task name(s). Comma-separated string or list.
            n_shots: Few-shot values to evaluate. Comma-separated string or list of ints.
            cluster: Cluster name defined in clusters.yaml, or "auto" for hostname detection.
            local: Run all jobs sequentially in the current process (no SLURM).
            dry_run: Print the jobs table without submitting.
            venv_path: Path to virtual environment to activate inside SLURM jobs.
            extra_task_paths: Comma-separated extra task directories for SLURM workers.
            minutes_per_eval: Walltime budget per (model, task, n_shot) eval, used to
                size the SLURM array and time limit. 0 = use the built-in default.
                Raise it for heavy generative tasks (e.g. mmlu_pro) that run far
                longer than the default per eval.

        Cluster config:
            clusters.yaml is read from {cache_dir}/clusters.yaml (default ~/.cache/mill/).
            On first run it is copied there from the bundled default — edit it to add your
            cluster. Change cache_dir on Mill.__init__ to use a different location.
        """
        from mill.scheduler.slurm import Scheduler

        model_list = _to_list(models)
        task_list = _to_list(tasks)
        n_shot_list = [int(n) for n in _to_list(n_shots)]

        scheduler = Scheduler(
            models=model_list,
            tasks=task_list,
            n_shots=n_shot_list,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            cluster=cluster,
            limit=self.limit,
            venv_path=venv_path,
            extra_task_paths=extra_task_paths,
            minutes_per_eval=minutes_per_eval,
        )

        if local:
            scheduler.run_local()
        elif dry_run:
            scheduler.print_jobs()
        else:
            scheduler.submit()

    # ── collect ───────────────────────────────────────────────────────────────

    def collect(
        self,
        models: Optional[Union[str, List[str]]] = None,
        tasks: Optional[Union[str, List[str]]] = None,
        metric: Optional[str] = None,
        check: bool = True,
        n_shots: Union[str, List[int]] = "0",
    ):
        """Display aggregated performance and optionally report missing jobs.

        The aggregate table is long-format (one row per metric), so results are
        compared by the generic ``performance`` column across any benchmark.

        Args:
            models: Filter to specific model(s). None = show all.
            tasks: Filter to specific task(s). None = show all.
            metric: Restrict the table to one metric name (e.g. "acc",
                    "exact_match"). None = show every metric's performance.
            check: If True, also print which (model, task, n_shot) combos are missing.
            n_shots: n_shot values to check for completeness.
        """
        from mill.output import OutputHandler
        from rich.console import Console
        from rich.table import Table

        handler = OutputHandler(output_dir=self.output_dir)
        model_list = _to_list(models) if models else None
        task_list = _to_list(tasks) if tasks else None

        df = handler.display(
            model=model_list[0] if model_list and len(model_list) == 1 else None,
            task=task_list[0] if task_list and len(task_list) == 1 else None,
            metric=metric,
        )

        console = Console()
        if df.empty:
            console.print("[yellow]No results found.[/yellow]")
        else:
            title = "Mill results — performance" + (f" ({metric})" if metric else "")
            table = Table(title=title)
            table.add_column("model", style="cyan")
            for col in df.columns:
                table.add_column(str(col))
            for idx, row in df.iterrows():
                table.add_row(str(idx), *[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])
            console.print(table)

        if check:
            agg = handler.get_aggregate()
            if agg.empty:
                console.print("\n[red]No results at all.[/red]")
            else:
                n_done = len(agg.drop_duplicates(["model", "task", "n_shot"])) \
                    if {"model", "task", "n_shot"}.issubset(agg.columns) else len(agg)
                console.print(f"\n[green]Completed jobs: {n_done}[/green]")
                if model_list and task_list:
                    n_shot_list = [int(n) for n in _to_list(n_shots)]
                    missing = handler.missing_jobs(model_list, task_list, n_shot_list)
                    if missing:
                        console.print(f"[red]Missing {len(missing)} jobs:[/red]")
                        for m in missing:
                            console.print(f"  model={m['model']}  task={m['task']}  n_shot={m['n_shot']}")
                    else:
                        console.print("[green]All requested jobs are complete.[/green]")

    # ── ls ────────────────────────────────────────────────────────────────────

    def ls(self):
        """Full-screen tabbed browser for benchmarks and tasks.

        Two tabs: Benchmarks (left) and Tasks (right).
        Switch tabs: Tab / Shift+Tab / ← →
        Navigate list: ↑ / ↓
        Scroll detail panel: Shift+↑ / Shift+↓
        Copy name to clipboard and exit: Enter
        Exit without copying: Escape / Ctrl-C
        """
        import mill.tasks  # trigger auto-discovery
        from mill.api.registry import (
            get_benchmark_config, get_task_config,
            list_benchmarks, list_tasks,
        )

        all_benchmarks = sorted(list_benchmarks())
        all_tasks = sorted(list_tasks())
        TABS = ["Benchmarks", "Tasks"]

        from prompt_toolkit import Application
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import HSplit, VSplit, Window
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.layout.processors import BeforeInput
        from prompt_toolkit.styles import Style

        search_buf = Buffer(name="search")
        state: dict = {
            "tab": 0,       # 0 = Benchmarks, 1 = Tasks
            "sel": 0,
            "scroll": 0,
            "rscroll": 0,
            "rmax": 0,
            "cache_q": None,
            "cache_res": [],
        }

        def _current_all() -> list[str]:
            return all_benchmarks if state["tab"] == 0 else all_tasks

        def _fuzzy(name: str, q: str) -> tuple[bool, int, list[int]]:
            t = name.lower()
            indices: list[int] = []
            ti = 0
            for ch in q:
                while ti < len(t) and t[ti] != ch:
                    ti += 1
                if ti >= len(t):
                    return False, 0, []
                indices.append(ti)
                ti += 1
            score = (indices[-1] - indices[0]) - (100 if indices[0] == 0 else 0)
            return True, score, indices

        def get_filtered() -> list[tuple[str, list[int]]]:
            q = search_buf.text.lower()
            cache_key = (state["tab"], q)
            if cache_key == state["cache_q"]:
                return state["cache_res"]
            items = _current_all()
            if q:
                scored = []
                for t in items:
                    ok, score, idxs = _fuzzy(t, q)
                    if ok:
                        scored.append((score, t, idxs))
                scored.sort(key=lambda x: (x[0], x[1]))
                result: list[tuple[str, list[int]]] = [(t, idxs) for _, t, idxs in scored]
            else:
                result = [(t, []) for t in items]
            state["cache_q"] = cache_key
            state["cache_res"] = result
            return result

        # ── tab bar ───────────────────────────────────────────────────────────

        def get_tab_bar() -> FormattedText:
            parts: list[tuple[str, str]] = []
            for i, name in enumerate(TABS):
                style = "class:tab-active" if i == state["tab"] else "class:tab-inactive"
                parts.append((style, f"  {name}  "))
            parts.append(("class:hint", "   ← → Tab S-Tab to switch  ↑↓ to select  Enter to copy"))
            return FormattedText(parts)

        # ── left panel ────────────────────────────────────────────────────────

        def get_left_panel() -> FormattedText:
            import shutil
            filtered = get_filtered()
            n = len(filtered)
            all_items = _current_all()
            label = "benchmarks" if state["tab"] == 0 else "tasks"

            sel = min(state["sel"], max(0, n - 1))
            state["sel"] = sel

            visible_count = max(5, shutil.get_terminal_size().lines - 6)
            scroll = state["scroll"]
            if sel < scroll:
                scroll = sel
            elif n > 0 and sel >= scroll + visible_count:
                scroll = sel - visible_count + 1
            state["scroll"] = scroll

            lines: list[tuple[str, str]] = [
                ("class:info", f"  {n}/{len(all_items)} {label}\n"),
                ("class:sep", "  " + "─" * 28 + "\n"),
            ]
            for i, (name, idxs) in enumerate(filtered[scroll: scroll + visible_count]):
                is_sel = (i + scroll) == sel
                pfx = "> " if is_sel else "  "
                base = "class:selected" if is_sel else ""
                if idxs:
                    idx_set = set(idxs)
                    row: list[tuple[str, str]] = [(base, pfx)]
                    for j, ch in enumerate(name):
                        s = ("class:match-sel" if is_sel else "class:match") if j in idx_set else base
                        row.append((s, ch))
                    row.append((base, "\n"))
                    lines.extend(row)
                else:
                    lines.append((base, f"{pfx}{name}\n"))
            return FormattedText(lines)

        # ── right panel ───────────────────────────────────────────────────────

        def get_right_panel() -> FormattedText:
            filtered = get_filtered()
            if not filtered:
                state["rmax"] = 0
                return FormattedText([("class:dim", "\n  No items match.\n")])
            name = filtered[min(state["sel"], len(filtered) - 1)][0]
            rows = _render_benchmark(name) if state["tab"] == 0 else _render_task(name)
            state["rmax"] = max(0, len(rows) - 1)
            rscroll = min(state["rscroll"], state["rmax"])
            state["rscroll"] = rscroll
            lines: list[tuple[str, str]] = []
            for r in rows[rscroll:]:
                lines.extend(r)
                lines.append(("", "\n"))
            return FormattedText(lines)

        def _render_benchmark(name: str) -> list[list[tuple[str, str]]]:
            import textwrap
            cfg = get_benchmark_config(name)
            rows: list[list[tuple[str, str]]] = []

            def R(*frags: tuple[str, str]) -> None:
                rows.append(list(frags))

            def KV(label: str, value: str, vstyle: str = "") -> None:
                rows.append([("class:label", f"  {label:<18}"), (vstyle or "class:value", value)])

            R(("class:title", f"  {name}"), ("class:version", "  benchmark"))
            R(("class:sep", "  " + "─" * 42))

            if cfg.description:
                for wline in textwrap.wrap(cfg.description, 54):
                    R(("class:desc", f"  {wline}"))
                R(("", ""))

            # Tasks — the key section for benchmarks
            R(("class:label", f"  {'Tasks':<18}"),
              ("class:dim", f"({len(cfg.task_names)} total)"))
            for t in cfg.task_names:
                R(("class:task-item", f"    • {t}"))

            if cfg.metric_names:
                R(("", ""))
                R(("class:label", "  Metrics"))
                for m in cfg.metric_names:
                    R(("class:metric", f"    {m}"))

            KV("Aggregation", "weighted" if cfg.weighted_aggregate else "unweighted mean")

            if cfg.categories:
                R(("", ""))
                cat: list[tuple[str, str]] = [("class:label", f"  {'Categories':<18}")]
                cat += [("class:tag", f"[{c}]  ") for c in cfg.categories]
                rows.append(cat)

            if cfg.capabilities:
                cap: list[tuple[str, str]] = [("class:label", f"  {'Capabilities':<18}")]
                cap += [("class:tag", f"[{c}]  ") for c in cfg.capabilities]
                rows.append(cap)

            if cfg.paper_url:
                R(("", ""))
                KV("Paper", cfg.paper_url, "class:link")

            return rows

        def _render_task(name: str) -> list[list[tuple[str, str]]]:
            import textwrap
            cfg = get_task_config(name)

            output_label = {
                "generate_until": "Generative (free text)",
                "loglikelihood": "Log-prob (MCQ)",
                "loglikelihood_rolling": "Perplexity",
            }.get(cfg.output_type.value, cfg.output_type.value)

            rows: list[list[tuple[str, str]]] = []

            def R(*frags: tuple[str, str]) -> None:
                rows.append(list(frags))

            def KV(label: str, value: str, vstyle: str = "") -> None:
                rows.append([("class:label", f"  {label:<18}"), (vstyle or "class:value", value)])

            R(("class:title", f"  {name}"), ("class:version", f"  v{cfg.version}"))
            R(("class:sep", "  " + "─" * 42))

            if cfg.description:
                for wline in textwrap.wrap(cfg.description, 54):
                    R(("class:desc", f"  {wline}"))
                R(("", ""))

            KV("Output type", output_label)
            KV("Eval splits", ", ".join(cfg.evaluation_splits))
            KV("Avail splits", ", ".join(cfg.hf_avail_splits))
            if cfg.few_shots_split:
                KV("Few-shot split", cfg.few_shots_split)
            KV("Default n-shot", str(cfg.n_shots))
            if cfg.output_type.value == "generate_until":
                if cfg.generation_size is not None:
                    KV("Max new tokens", str(cfg.generation_size))
                if cfg.stop_sequences:
                    KV("Stop sequences", ", ".join(repr(s) for s in cfg.stop_sequences))

            if cfg.hf_repo:
                R(("", ""))
                KV("HF dataset", cfg.hf_repo + (f"  /  {cfg.hf_subset}" if cfg.hf_subset else ""))
            if cfg.approx_num_samples:
                KV("Samples (approx)", "  |  ".join(f"{s}: {n:,}" for s, n in cfg.approx_num_samples.items()))

            if cfg.categories:
                R(("", ""))
                cat: list[tuple[str, str]] = [("class:label", f"  {'Categories':<18}")]
                cat += [("class:tag", f"[{c}]  ") for c in cfg.categories]
                rows.append(cat)

            if cfg.capabilities:
                cap: list[tuple[str, str]] = [("class:label", f"  {'Capabilities':<18}")]
                cap += [("class:tag", f"[{c}]  ") for c in cfg.capabilities]
                rows.append(cap)

            if cfg.metrics:
                R(("", ""))
                R(("class:label", "  Metrics"))
                for m in cfg.metrics:
                    R(("class:metric", f"    {m.name}  {'↑' if m.higher_is_better else '↓'}"))

            if cfg.paper_url:
                R(("", ""))
                KV("Paper", cfg.paper_url, "class:link")
            if cfg.hf_repo:
                KV("HF URL", f"https://huggingface.co/datasets/{cfg.hf_repo}", "class:link")

            return rows

        # ── key bindings ──────────────────────────────────────────────────────

        kb = KeyBindings()

        def _switch_tab(delta: int) -> None:
            state["tab"] = (state["tab"] + delta) % len(TABS)
            state["sel"] = 0
            state["scroll"] = 0
            state["rscroll"] = 0
            state["cache_q"] = None

        @kb.add("escape")
        @kb.add("c-c")
        def _exit(event):
            event.app.exit()

        @kb.add("enter")
        def _copy_and_exit(event):
            filtered = get_filtered()
            if filtered:
                selected = filtered[min(state["sel"], len(filtered) - 1)][0]
                event.app.exit(result=selected)
            else:
                event.app.exit()

        @kb.add("tab")
        @kb.add("right")
        def _next_tab(event):
            _switch_tab(1)

        @kb.add("s-tab")
        @kb.add("left")
        def _prev_tab(event):
            _switch_tab(-1)

        @kb.add("up")
        def _up(event):
            state["sel"] = max(0, state["sel"] - 1)
            state["rscroll"] = 0

        @kb.add("down")
        def _down(event):
            state["sel"] = min(max(0, len(get_filtered()) - 1), state["sel"] + 1)
            state["rscroll"] = 0

        @kb.add("s-up")
        def _r_up(event):
            state["rscroll"] = max(0, state["rscroll"] - 3)

        @kb.add("s-down")
        def _r_down(event):
            state["rscroll"] = min(state["rmax"], state["rscroll"] + 3)

        def _on_text_changed(_buf) -> None:
            state["sel"] = 0
            state["scroll"] = 0
            state["rscroll"] = 0
            state["cache_q"] = None

        search_buf.on_text_changed += _on_text_changed

        # ── layout & style ────────────────────────────────────────────────────

        style = Style.from_dict({
            "tab-active":   "bold reverse",
            "tab-inactive": "ansigray",
            "hint":         "ansigray",
            "title":        "bold",
            "version":      "ansigray",
            "desc":         "ansigray",
            "label":        "ansiblue",
            "value":        "",
            "link":         "ansicyan underline",
            "tag":          "ansiyellow",
            "metric":       "ansigreen",
            "task-item":    "ansicyan",
            "info":         "ansigreen",
            "sep":          "ansigray",
            "match":        "bold ansicyan",
            "match-sel":    "bold ansicyan reverse",
            "selected":     "reverse",
            "dim":          "ansigray",
            "prompt-label": "bold",
        })

        LEFT_W = 36
        layout = Layout(
            HSplit([
                Window(
                    content=FormattedTextControl(get_tab_bar),
                    height=1,
                    dont_extend_height=True,
                ),
                VSplit([
                    HSplit([
                        Window(
                            content=BufferControl(
                                buffer=search_buf,
                                input_processors=[BeforeInput("Filter: ", style="class:prompt-label")],
                            ),
                            height=1,
                            dont_extend_height=True,
                            width=LEFT_W,
                        ),
                        Window(content=FormattedTextControl(get_left_panel), width=LEFT_W),
                    ]),
                    Window(width=1, char="│", style="class:sep"),
                    Window(content=FormattedTextControl(get_right_panel)),
                ]),
            ])
        )

        selected = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=True,
            mouse_support=False,
        ).run()

        if selected:
            _copy_to_clipboard(selected)
            print(f"Copied: {selected}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _copy_to_clipboard(text: str) -> bool:
    import subprocess
    for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"], ["pbcopy"]):
        try:
            subprocess.run(cmd, input=text.encode(), check=True, capture_output=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return False


def _to_list(value: Union[str, list, None]) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    return [v.strip() for v in str(value).split(",")]


def _parse_kv(s: str) -> dict:
    if not s:
        return {}
    result = {}
    for pair in s.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k.strip()] = _coerce(v.strip())
    return result


def _coerce(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def main():
    import fire
    fire.Fire(Mill)


if __name__ == "__main__":
    main()
