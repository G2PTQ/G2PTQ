"""Command-line arguments and the YAML sampling profile.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


# ``configs/`` sits next to run_eval.py, one level above this module.
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"

# The only keys a file in ``configs/`` may set.
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
)

DATASETS = (
    "aime24",
    "aime25",
    "aime26",
    "math_500",
    "gsm8k",
    "gpqa_diamond",
    "live_code_bench",
    "arxivmath",
    "ifbench",
    "erqa",
)
SEEDS = (42, 43, 44)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EvalScope reasoning benchmarks against a vLLM server.",
    )
    parser.add_argument(
        "--config",
        dest="config_file",
        default=None,
        help=(
            "Sampling profile to layer over configs/default.yaml "
            "(default: configs/<model-id>.yaml when it exists, else the default profile)."
        ),
    )

    endpoint = parser.add_argument_group("model and endpoint")
    endpoint.add_argument(
        "--model_path",
        default="./modelzoo/Qwen4/Qwen3.8-Flash-Next",
        help="Model path sent to the OpenAI-compatible endpoint (default: %(default)s).",
    )
    endpoint.add_argument(
        "--model_id",
        default=None,
        help=(
            "EvalScope report/model label "
            "(default: the --model_path basename, or its parent when the path ends "
            "in export_model/)."
        ),
    )
    endpoint.add_argument(
        "--api_url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible API base URL (default: %(default)s).",
    )
    endpoint.add_argument(
        "--api_key",
        default="EMPTY",
        help="OpenAI-compatible API key (default: %(default)s).",
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

    run = parser.add_argument_group("run")
    run.add_argument(
        "--dataset_root",
        default="./datasets",
        help="Root directory containing the offline datasets (default: %(default)s).",
    )
    run.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory holding runs/seed_<seed>/, one EvalScope work_dir per seed "
            "(default: ./outputs/<model_id>/evalscope)."
        ),
    )
    run.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        help="EvalScope dataset names (default: %(default)s).",
    )
    run.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS),
        help="One or more integer seeds, run sequentially (default: %(default)s).",
    )
    run.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Concurrent EvalScope requests (default: %(default)s).",
    )
    run.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of samples per subset; omit to evaluate every sample.",
    )
    run.add_argument(
        "--request_retries",
        type=int,
        default=3,
        help=(
            "Per-request retry count."
        ),
    )
    run.add_argument(
        "--request_timeout",
        type=float,
        default=3600,
        help=(
            "Per-request timeout in seconds; omit for no limit."
        ),
    )

    limits = parser.add_argument_group("output limits and thinking mode")
    limits.add_argument(
        "--max_tokens",
        type=int,
        default=196608,     # 192K
        help=(
            "vLLM total output cap; must stay below the server's max_model_len "
            "(default: %(default)s)."
        ),
    )
    limits.add_argument(
        "--thinking_budget",
        type=int,
        default=131072,     # 128K
        help="vLLM thinking_token_budget (default: %(default)s).",
    )
    # BooleanOptionalAction would derive "--no-enable_thinking" from the flag
    # name, mixing separators; declare the underscored negations explicitly.
    limits.add_argument(
        "--enable_thinking",
        action="store_true",
        default=True,
        help="Chat-template enable_thinking flag (default: enabled).",
    )
    limits.add_argument(
        "--no_enable_thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable the chat-template enable_thinking flag.",
    )
    limits.add_argument(
        "--preserve_thinking",
        action="store_true",
        default=True,
        help="Chat-template preserve_thinking flag (default: enabled).",
    )
    limits.add_argument(
        "--no_preserve_thinking",
        dest="preserve_thinking",
        action="store_false",
        help="Disable the chat-template preserve_thinking flag.",
    )

    # Left at None so that an explicit flag is distinguishable from "take the
    # value from the sampling profile"; parse_args fills these in.
    sampling = parser.add_argument_group(
        "sampling", "Defaults come from the YAML profile; a flag here overrides the file."
    )
    sampling.add_argument("--temperature", type=float, default=None)
    sampling.add_argument("--top_p", type=float, default=None)
    sampling.add_argument("--top_k", type=int, default=None)
    sampling.add_argument("--min_p", type=float, default=None)
    sampling.add_argument("--presence_penalty", type=float, default=None)
    sampling.add_argument("--repetition_penalty", type=float, default=None)

    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
        load = yaml.safe_load
        yaml_error = yaml.YAMLError
    except ImportError:  # pragma: no cover - depends on environment
        try:
            from ruamel.yaml import YAML
        except ImportError as exc:
            raise SystemExit(
                "A YAML parser is required to read evaluation config files. "
                "Install it with `pip install PyYAML`."
            ) from exc
        load = YAML(typ="safe").load
        yaml_error = Exception
    try:
        with path.open(encoding="utf-8") as stream:
            values = load(stream) or {}
    except OSError as exc:
        raise SystemExit(f"cannot read evaluation config file {path}: {exc}") from exc
    except yaml_error as exc:
        raise SystemExit(f"invalid YAML in evaluation config file {path}: {exc}") from exc
    if not isinstance(values, dict):
        raise SystemExit(f"evaluation config must contain a mapping: {path}")
    unknown = sorted((str(key) for key in values if key not in SAMPLING_KEYS))
    if unknown:
        raise SystemExit(
            f"evaluation config {path} may contain sampling parameters only; "
            f"unsupported key(s): {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(SAMPLING_KEYS)}"
        )
    return values


def _derive_model_id(model_path: str) -> str:
    """Label a checkpoint by its directory name.
    """
    path = Path(model_path.rstrip("/"))
    return path.name


def _derive_output_dir(model_id: str) -> str:
    return str(Path("./outputs") / model_id / "evalscope")


def _resolve_config(config_file: str | None, model_id: str) -> Path:
    """Pick the sampling profile: the flag, a model-named file, or the default.
    """
    if config_file:
        return Path(config_file)
    model_config = CONFIG_DIR / f"{model_id}.yaml"
    return model_config if model_config.is_file() else DEFAULT_CONFIG


def _sampling_values(config_file: Path) -> dict[str, Any]:
    """Read ``default.yaml``, then let the selected profile override its keys."""
    values = _load_yaml(DEFAULT_CONFIG)
    if config_file != DEFAULT_CONFIG:
        values.update(_load_yaml(config_file))
    missing = [key for key in SAMPLING_KEYS if key not in values]
    if missing:
        raise SystemExit(
            f"{DEFAULT_CONFIG} must define every sampling parameter; "
            f"missing: {', '.join(missing)}"
        )
    return values


def parse_args() -> argparse.Namespace:
    """Parse the command line, derive the label/output paths, then fill sampling values."""
    args = _parser().parse_args()

    # model_id names the run; output_dir defaults to a directory named after it.
    # Both are derived before the profile lookup, which keys off model_id.
    if args.model_id is None:
        args.model_id = _derive_model_id(args.model_path)
    if args.output_dir is None:
        args.output_dir = _derive_output_dir(args.model_id)

    config_file = _resolve_config(args.config_file, args.model_id)
    for key, value in _sampling_values(config_file).items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.config_file = str(config_file)
    return args
