"""Score lm-eval QA tasks against a running vLLM server.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from utils.api import wait_for_api
from utils.lm_eval_args import parse_args
from utils.lm_eval_runner import (
    build_model,
    format_markdown_table,
    resolve_tasks,
    run_tasks,
    summarize,
)


#: Leaf directory holding this driver's results, under ``--output_dir``.
RESULTS_SUBDIR = "lm_eval"


def _write_results(
    output_dir: Path, payload: dict, raw: dict, log_samples: bool, table: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (output_dir / "results.md").write_text(
        f"# QA eval: {payload['model_id']}\n\n"
        f"- model_path: `{payload['model_path']}`\n"
        f"- api_url: `{payload['api_url']}`\n"
        f"- limit: {payload['limit']}\n\n"
        f"{table}\n",
        encoding="utf-8",
    )

    provenance = {
        task: {key: value for key, value in blob.items() if key != "samples"}
        for task, blob in raw.items()
    }
    (output_dir / "raw_results.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    if not log_samples:
        return
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for task, blob in raw.items():
        records = blob.get("samples")
        if not records:
            continue
        with (samples_dir / f"{task}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    args = parse_args()

    # LocalCompletionsAPI.api_key reads this env var and has no constructor arg.
    os.environ.setdefault("OPENAI_API_KEY", args.api_key)

    task_manager, task_names = resolve_tasks(args)
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "model_path": args.model_path,
                "api_url": args.api_url,
                "tasks": task_names,
                "limit": args.limit,
            },
            indent=2,
        ),
        flush=True,
    )

    if args.api_wait > 0:
        wait_for_api(args.api_url, args.api_wait, args.api_wait_interval)

    lm = build_model(args)
    accuracies, raw, status = run_tasks(args, lm, task_manager, task_names)
    acc_avg, n_ok = summarize(accuracies)

    payload = {
        "model_id": args.model_id,
        "model_path": args.model_path,
        "api_url": args.api_url,
        "tokenizer": args.tokenizer,
        "limit": args.limit,
        "max_length": args.max_length,
        "num_concurrent": args.num_concurrent,
        "tasks": accuracies,
        "acc_avg": acc_avg,
        "n_tasks_ok": n_ok,
        "n_tasks": len(task_names),
        "status": status,
    }

    table = format_markdown_table(accuracies, acc_avg)
    output_dir = Path(args.output_dir) / RESULTS_SUBDIR
    _write_results(output_dir, payload, raw, args.log_samples, table)

    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print("\n" + table, flush=True)
    print(f"\nlm-eval results are under {output_dir}", flush=True)
    return status


if __name__ == "__main__":
    main()
