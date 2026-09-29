"""EvalScope dataset and generation configuration builders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

#: LiveCodeBench release to evaluate. Bump deliberately when adopting a newer
#: upstream release; numbers are not comparable across releases.
LIVE_CODE_BENCH_RELEASE = "release_latest"


def dataset_args(
    dataset_root: Path, datasets: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Build EvalScope's offline dataset configuration."""
    result: dict[str, dict[str, Any]] = {}
    for name in datasets:
        path = dataset_root / name
        if not path.is_dir():
            raise FileNotFoundError(
                f"offline dataset path does not exist for {name}: {path}"
            )
        entry: dict[str, Any] = {"dataset_id": str(path)}
        if name == "live_code_bench":
            entry["subset_list"] = [LIVE_CODE_BENCH_RELEASE]
        result[name] = entry
    return result


def generation_config(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    """Build the per-seed vLLM generation configuration."""
    return {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "seed": seed,
        "retries": args.request_retries,
        "timeout": args.request_timeout,
        "extra_body": {
            "top_k": args.top_k,
            "min_p": args.min_p,
            "thinking_token_budget": args.thinking_budget,
            "chat_template_kwargs": {
                "enable_thinking": args.enable_thinking,
                "preserve_thinking": args.preserve_thinking,
            },
        },
    }
