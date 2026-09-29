"""Command-line arguments for the lm-eval QA driver.

Deliberately separate from :mod:`utils.args`: that module's YAML sampling profile
exists for *generative* benchmarks, and every task here is loglikelihood-scored at
temperature 0, so there is no sampling recipe to layer. Only the two derivation
helpers are shared.
"""

from __future__ import annotations

import argparse

from utils.args import _derive_model_id


QA_TASKS = (
    "arc_challenge",
    "arc_easy",
    "ceval-valid",
    "hellaswag",
    "lambada_openai",
    "piqa",
    "winogrande",
)

#: The repo's own offline task YAMLs.
TASK_INCLUDE_PATH = "./datasets/lm_eval_configs/tasks"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score lm-eval QA tasks against a running vLLM server over the "
            "OpenAI-compatible /v1/completions endpoint."
        ),
    )

    endpoint = parser.add_argument_group("model and endpoint")
    endpoint.add_argument(
        "--model_path",
        default="./modelzoo/Qwen4/Qwen3.8-Flash-Next",
        help="Model path sent as the request 'model' field (default: %(default)s).",
    )
    endpoint.add_argument(
        "--model_id",
        default=None,
        help="Label used in results.json (default: the --model_path basename).",
    )
    endpoint.add_argument(
        "--api_url",
        default="http://127.0.0.1:8000/v1",
        help=(
            "OpenAI-compatible API base URL; '/completions' is appended for the "
            "loglikelihood requests (default: %(default)s)."
        ),
    )
    endpoint.add_argument(
        "--api_key",
        default="EMPTY",
        help=(
            "Exported as OPENAI_API_KEY, which is the only route lm-eval reads a "
            "key from (default: %(default)s)."
        ),
    )
    endpoint.add_argument(
        "--api_wait",
        type=float,
        default=1800.0,
        help=(
            "Seconds to wait for the endpoint to become ready before giving up; "
            "0 skips the check (default: %(default)s)."
        ),
    )
    endpoint.add_argument(
        "--api_wait_interval",
        type=float,
        default=10.0,
        help="Seconds between readiness probes (default: %(default)s).",
    )
    endpoint.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer used to measure context lengths; must match the served "
            "checkpoint (default: --model_path)."
        ),
    )

    run = parser.add_argument_group("run")
    run.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Run root; results are written to <output_dir>/lm_eval/ "
            "(default: ./outputs/<model_id>). Pass explicitly for quantized "
            "exports, which are all named export_model and would otherwise collide."
        ),
    )
    run.add_argument(
        "--tasks",
        nargs="+",
        default=list(QA_TASKS),
        help="lm-eval task or group names (default: %(default)s).",
    )
    run.add_argument(
        "--task_include_path",
        default=TASK_INCLUDE_PATH,
        help="Directory of offline task YAMLs (default: %(default)s).",
    )
    run.add_argument(
        "--num_concurrent",
        type=int,
        default=32,
        help="Concurrent in-flight requests; the throughput knob (default: %(default)s).",
    )
    run.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "Prompts per request. Left at 1 because batching sends a list of "
            "varying-length prompts in one call; use --num_concurrent instead "
            "(default: %(default)s)."
        ),
    )
    run.add_argument(
        "--max_length",
        type=int,
        default=32768,
        help=(
            "Context budget for truncation; too low silently truncates long "
            "hellaswag/ceval prompts (default: %(default)s)."
        ),
    )
    run.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Per-request retry count (default: %(default)s).",
    )
    run.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional samples per task; omit to evaluate every sample.",
    )
    run.add_argument(
        "--log_samples",
        action="store_true",
        help="Also write per-sample records under <output_dir>/samples/.",
    )

    return parser


def _derive_output_dir(model_id: str) -> str:
    """The run root for a checkpoint, mirroring run_eval.py's ``outputs/<model_id>``.
    """
    return f"./outputs/{model_id}"


def parse_args() -> argparse.Namespace:
    args = _parser().parse_args()
    if args.model_id is None:
        args.model_id = _derive_model_id(args.model_path)
    if args.output_dir is None:
        args.output_dir = _derive_output_dir(args.model_id)
    if args.tokenizer is None:
        args.tokenizer = args.model_path
    return args
