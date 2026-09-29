"""Summarize EvalScope reasoning results across seeds as a markdown table.

Reads ``<run>/runs/seed_<seed>/reports/<model_id>/<dataset>.json`` for one or
more runs and prints one row per run, one column per dataset, each cell the
mean +/- sample standard deviation over seeds. Touches no server.

Run it from inside eval_reasoning/ (absolute ``utils`` imports, like the two
drivers next to it).

Usage:
    python summarize_results.py
    python summarize_results.py outputs/Qwen3.8-Flash-Next-GPTQ-W4A16
    python summarize_results.py --per_seed -o results.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.args import DATASETS, SEEDS
from utils.summary import (
    collect_run,
    discover_runs,
    order_datasets,
    provenance,
    render_per_seed_table,
    render_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize EvalScope results across seeds into a markdown table.",
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help=(
            "Run directories, e.g. outputs/<model>/evalscope (the model directory "
            "or its runs/ also work). Default: every run under --outputs_dir."
        ),
    )
    parser.add_argument(
        "--outputs_dir",
        default="./outputs",
        help="Root scanned when no run is given (default: %(default)s).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=f"Restrict to these seeds (default: every seed found; runs use {list(SEEDS)}).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Restrict and order the dataset columns (default: every dataset found).",
    )
    parser.add_argument(
        "--per_seed",
        action="store_true",
        help="Also print a second table with one row per (run, seed).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Also write the table(s) to this file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.runs:
        paths = [Path(run) for run in args.runs]
    else:
        paths = discover_runs(Path(args.outputs_dir))
        if not paths:
            sys.exit(
                f"[error] no EvalScope runs under {args.outputs_dir}/ "
                f"(looked for */evalscope/runs)"
            )

    runs = []
    for path in paths:
        try:
            runs.append(collect_run(path, args.seeds, args.datasets))
        except FileNotFoundError as exc:
            print(f"[warn] skipped {path}: {exc}", file=sys.stderr)
    if not runs:
        sys.exit("[error] no run directory resolved")

    # An explicit --datasets is a column order; otherwise fall back to the
    # driver's own dataset order so tables stay comparable across invocations.
    preferred = args.datasets if args.datasets else DATASETS
    datasets = order_datasets(runs, preferred)
    if not datasets:
        sys.exit("[error] no readable reports in any run")

    blocks = [render_table(runs, datasets)]
    if args.per_seed:
        blocks.append("**Per seed**\n\n" + render_per_seed_table(runs, datasets))
    body = "\n\n".join(blocks)
    print(body)

    # stderr keeps stdout clean for piping the table.
    print("\nSource reports:", file=sys.stderr)
    for run in runs:
        print("\n".join(provenance(run, datasets)), file=sys.stderr)
    print(
        f"\nCells are mean±std over seeds; a bare value is a single seed. "
        f"'*' marks an incomplete report, or an Avg. over fewer datasets.",
        file=sys.stderr,
    )

    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(body + "\n", encoding="utf-8")
        print(f"\nWrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
