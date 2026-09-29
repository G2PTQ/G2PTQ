"""EvalScope task construction and per-seed execution."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from evalscope import TaskConfig, run_task

from .config import generation_config


def build_task_config(
    args: argparse.Namespace,
    data_args: dict[str, dict[str, Any]],
    seed: int,
    seed_root: Path,
) -> Any:
    """Create the isolated EvalScope task for one seed."""
    return TaskConfig(
        model=args.model_path,
        model_id=args.model_id,
        model_task="text_generation",
        datasets=list(args.datasets),
        dataset_args=data_args,
        eval_type="openai_api",
        eval_backend="Native",
        eval_batch_size=args.eval_batch_size,
        generation_config=generation_config(args, seed),
        api_url=args.api_url,
        api_key=args.api_key,
        seed=seed,
        work_dir=str(seed_root),
        no_timestamp=True,
        ignore_errors=True,
        collect_perf=False,
        sandbox={"enabled": False},
        limit=args.limit,
    )


def run_seed(
    args: argparse.Namespace,
    data_args: dict[str, dict[str, Any]],
    run_root: Path,
    seed: int,
) -> int:
    """Run one seed into its own work_dir and return its exit status."""
    seed_root = run_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    task_config = build_task_config(args, data_args, seed, seed_root)

    print(f"Starting seed {seed}; EvalScope work_dir: {seed_root}", flush=True)
    print(f"TaskConfig for seed {seed}: {task_config.model_dump_json(indent=2)}", flush=True)
    try:
        run_task(task_cfg=task_config)
    except Exception:
        # One bad seed must not cost the seeds that have not run yet; EvalScope
        # has already written whatever it completed under ``seed_root/logs``.
        traceback.print_exc()
        print(f"Seed {seed} failed; continuing with remaining seeds", flush=True)
        return 1
    print(f"Seed {seed} completed", flush=True)
    return 0


def run_seeds(
    args: argparse.Namespace,
    data_args: dict[str, dict[str, Any]],
    run_root: Path,
) -> int:
    """Run all requested seeds, continuing after an individual seed fails."""
    overall_status = 0
    for seed in args.seeds:
        if run_seed(args, data_args, run_root, seed):
            overall_status = 1
    return overall_status


def create_run_paths(output_dir: Path) -> Path:
    """Create and return the ``runs/`` root holding one work_dir per seed.

    There is no timestamp layer: a rerun of the same seed replaces its previous
    work_dir rather than accumulating one directory per invocation.
    """
    run_root = output_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root
