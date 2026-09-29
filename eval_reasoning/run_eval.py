from __future__ import annotations

import json
from pathlib import Path

from utils.api import wait_for_api
from utils.args import parse_args
from utils.config import dataset_args
from utils.runner import create_run_paths, run_seeds


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    data_args = dataset_args(dataset_root, args.datasets)

    print(json.dumps({"dataset_args": data_args, "seeds": args.seeds}, indent=2))

    if args.api_wait > 0:
        wait_for_api(args.api_url, args.api_wait, args.api_wait_interval)

    run_root = create_run_paths(output_dir)
    overall_status = run_seeds(args, data_args, run_root)

    print(f"EvalScope results are under {run_root}")
    return overall_status


if __name__ == "__main__":
    main()
