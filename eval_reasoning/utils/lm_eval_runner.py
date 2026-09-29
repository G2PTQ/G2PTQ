"""lm-eval model construction and the per-task scoring loop.
"""

from __future__ import annotations

import argparse
import traceback
from typing import Any


def _patched_parse_logprobs(
    outputs: Any,
    tokens: Any = None,
    ctxlens: Any = None,
    **kwargs: Any,
) -> list[tuple[float, bool]]:
    """``LocalCompletionsAPI.parse_logprobs`` with the ``is_greedy`` bug fixed.

    lm-eval 0.4.4 assigns the *logprob floats* to a local named ``tokens`` and
    then compares each one against ``max(top, key=top.get)``, which is a token
    *string*. A float never equals a string, so ``is_greedy`` comes back ``False``
    for every request on this backend.

    That silently zeroes any task whose ``acc`` is ``int(is_greedy)`` -- i.e. the
    ``output_type: loglikelihood`` tasks, of which ``lambada_openai`` is one.
    ``multiple_choice`` tasks pick their answer with ``argmax`` over the summed
    logprobs and are unaffected, which is why only LAMBADA read 0.00 while the
    other six looked sane.

    Compare logprob-to-logprob instead: a continuation token was greedy iff its
    own logprob is the maximum of the returned ``top_logprobs`` at that position.
    """
    res: list[tuple[float, bool]] = []
    if not isinstance(outputs, list):
        outputs = [outputs]
    for out in outputs:
        for choice, ctxlen in zip(out["choices"], ctxlens):
            assert ctxlen > 0, "Context length must be greater than 0"
            logprobs = choice["logprobs"]["token_logprobs"][ctxlen:-1]
            top_logprobs = choice["logprobs"]["top_logprobs"][ctxlen:-1]
            is_greedy = True
            for logprob, top in zip(logprobs, top_logprobs):
                # vLLM includes the sampled token in top_logprobs alongside the
                # top-1 when they differ, so the max over values is the top-1.
                if top and logprob < max(top.values()):
                    is_greedy = False
                    break
            res.append((sum(logprobs), is_greedy))
    return res


def build_model(args: argparse.Namespace) -> Any:
    """Construct the ``local-completions`` backend pointed at the vLLM server.
    """
    from lm_eval.models.openai_completions import LocalCompletionsAPI

    # base_url is used verbatim as the POST target, so it needs the full
    # /v1/completions path -- not the /v1 base that --api_url carries.
    base_url = f"{args.api_url.rstrip('/')}/completions"

    # Fix upstream's always-False is_greedy before any request is scored.
    LocalCompletionsAPI.parse_logprobs = staticmethod(_patched_parse_logprobs)

    return LocalCompletionsAPI(
        model=args.model_path,
        base_url=base_url,
        tokenizer=args.tokenizer,
        tokenizer_backend="huggingface",
        num_concurrent=args.num_concurrent,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_retries=args.max_retries,
    )


def resolve_tasks(args: argparse.Namespace) -> tuple[Any, list[str]]:
    """Build the offline TaskManager and expand the requested task patterns."""
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import pattern_match

    # include_defaults=False keeps this to the repo's own YAMLs: no bundled task
    # set, no network access.
    task_manager = TaskManager(
        include_path=args.task_include_path, include_defaults=False
    )
    task_names = pattern_match(list(args.tasks), task_manager.all_tasks)
    if not task_names:
        raise SystemExit(
            f"no lm-eval tasks matched {list(args.tasks)} under "
            f"{args.task_include_path}"
        )
    return task_manager, task_names


def _task_accuracy(result: dict[str, Any]) -> float:
    """Prefer ``acc_norm`` when the task reports it, matching ``qa_eval``."""
    acc = result.get("acc_norm,none", result.get("acc,none"))
    if acc is None:
        raise KeyError(
            f"neither 'acc_norm,none' nor 'acc,none' in task result keys: "
            f"{sorted(result)}"
        )
    return round(acc * 100, 2)


def run_tasks(
    args: argparse.Namespace, lm: Any, task_manager: Any, task_names: list[str]
) -> tuple[dict[str, float | None], dict[str, Any], int]:
    """Score each task in its own ``simple_evaluate`` call.

    One call per task mirrors ``qa_eval``. A failing task is recorded as ``None``
    and does not abort the ones that have not run yet, following the same
    "one bad unit must not cost the rest" rule as ``utils.runner.run_seed``.
    """
    from lm_eval import simple_evaluate

    accuracies: dict[str, float | None] = {}
    raw: dict[str, Any] = {}
    status = 0

    for task_name in task_names:
        print(f"Evaluating {task_name}...", flush=True)
        try:
            output = simple_evaluate(
                lm,
                tasks=[task_name],
                task_manager=task_manager,
                limit=args.limit,
                log_samples=args.log_samples,
            )
            result = output["results"][task_name]
            accuracies[task_name] = _task_accuracy(result)
            raw[task_name] = {
                "results": output["results"],
                "n-samples": output.get("n-samples"),
            }
            if args.log_samples:
                raw[task_name]["samples"] = output.get("samples", {}).get(task_name)
            print(f"acc: {accuracies[task_name]}%", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"Task {task_name} failed; continuing with remaining tasks", flush=True)
            accuracies[task_name] = None
            status = 1

    return accuracies, raw, status


def format_markdown_table(
    accuracies: dict[str, float | None], acc_avg: float | None
) -> str:
    cells = {
        task: "-" if acc is None else f"{acc:.2f}"
        for task, acc in accuracies.items()
    }
    cells["acc_avg"] = "-" if acc_avg is None else f"{acc_avg:.2f}"

    headers = list(cells)
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_row = "| " + " | ".join(cells[key] for key in headers) + " |"
    return "\n".join([header_row, separator_row, data_row])


def summarize(accuracies: dict[str, float | None]) -> tuple[float | None, int]:
    """Average over the tasks that scored.

    A failed task stays visibly ``None`` rather than being counted as zero, which
    would quietly drag the average down.
    """
    scored = [acc for acc in accuracies.values() if acc is not None]
    if not scored:
        return None, 0
    return round(sum(scored) / len(scored), 2), len(scored)
