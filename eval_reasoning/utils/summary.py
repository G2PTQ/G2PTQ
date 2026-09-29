"""Aggregate EvalScope reports across seeds into markdown tables.

A read-only pass over ``<run>/runs/seed_<seed>/reports/<model_id>/<dataset>.json``.
Imports nothing from evalscope, so it runs wherever a stdlib Python does.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable, NamedTuple

MISSING = "-"
#: Appended to a cell whose report did not score every requested sample.
INCOMPLETE_MARK = "*"

RUNS_DIRNAME = "runs"
SEED_PREFIX = "seed_"
#: The leaf ``_derive_output_dir`` puts a run root under, i.e.
#: ``outputs/<model_id>/evalscope``.
RUN_LEAF = "evalscope"


class Report(NamedTuple):
    """One scored dataset for one seed."""

    dataset: str
    label: str
    value: float
    complete: bool
    detail: str


class Run(NamedTuple):
    """One checkpoint's results, keyed by dataset then seed."""

    name: str
    root: Path
    #: ``{dataset: {seed: Report}}``
    reports: dict[str, dict[int, Report]]
    labels: dict[str, str]
    seeds: list[int]


def _metric_key(identity: dict[str, Any]) -> tuple:
    """Hashable form of an EvalScope metric identity.

    Name alone does not identify a metric: ``live_code_bench`` reports two
    metrics both named ``accuracy``, one ``aggregation: mean`` and one
    ``pass_at_k`` with ``dimensions: {k: 1}``, and only the latter is primary.
    """
    dimensions = identity.get("dimensions") or {}
    return (
        identity.get("name"),
        identity.get("aggregation"),
        tuple(sorted((str(k), str(v)) for k, v in dimensions.items())),
    )


def _primary_metric(report: dict[str, Any]) -> dict[str, Any] | None:
    """The ``metrics[]`` entry matching ``primary_metric_identity`` exactly."""
    primary = report.get("primary_metric_identity")
    metrics = report.get("metrics") or []
    if not primary:
        return metrics[0] if len(metrics) == 1 else None
    wanted = _metric_key(primary)
    for metric in metrics:
        if _metric_key(metric.get("identity") or {}) == wanted:
            return metric
    return None


def _completeness(report: dict[str, Any]) -> tuple[bool, str]:
    """Whether every requested sample scored, and a human-readable detail.

    EvalScope writes a scored-looking report even when every request failed, so
    an unchecked score can silently be "0 of N succeeded".
    """
    summary = report.get("execution_summary") or {}
    requested = summary.get("requested")
    succeeded = summary.get("succeeded")
    errored = summary.get("errored") or 0
    if summary.get("incomplete"):
        return False, f"marked incomplete ({succeeded}/{requested} succeeded)"
    if errored:
        return False, f"{errored} errored ({succeeded}/{requested} succeeded)"
    if requested is not None and succeeded != requested:
        return False, f"{succeeded}/{requested} succeeded"
    return True, ""


def read_report(path: Path) -> Report | None:
    """Parse one dataset report; ``None`` when it carries no usable score."""
    try:
        with path.open(encoding="utf-8") as stream:
            report = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict):
        return None

    dataset = report.get("dataset_name") or path.stem
    label = report.get("dataset_pretty_name") or dataset

    metric = _primary_metric(report)
    if metric is None or metric.get("score") is None:
        return None
    # Scale by the metric's own declared multiplier rather than assuming a
    # ratio, so a non-ratio metric cannot be silently mis-scaled.
    semantics = metric.get("semantics") or {}
    multiplier = semantics.get("display_multiplier")
    if multiplier is None:
        multiplier = 1.0
    value = float(metric["score"]) * float(multiplier)

    complete, detail = _completeness(report)
    return Report(dataset, label, value, complete, detail)


def resolve_runs_root(path: Path) -> Path:
    """Find the ``runs/`` directory for a user-supplied path.

    Accepts a run root (``outputs/<id>/evalscope``), the ``runs`` directory
    itself, or a model directory holding ``evalscope/runs``.
    """
    candidates = [path / RUNS_DIRNAME, path / RUN_LEAF / RUNS_DIRNAME]
    if path.name == RUNS_DIRNAME:
        candidates.insert(0, path)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"no EvalScope runs under {path}; expected one of: "
        + ", ".join(str(c) for c in candidates)
    )


def _run_name(runs_root: Path) -> str:
    """Label a run by its model directory: the parent of ``evalscope/runs``."""
    run_root = runs_root.parent
    if run_root.name == RUN_LEAF and run_root.parent.name:
        return run_root.parent.name
    return run_root.name or str(run_root)


def discover_runs(outputs_dir: Path) -> list[Path]:
    """Every ``outputs/*/evalscope`` holding a ``runs/`` directory."""
    if not outputs_dir.is_dir():
        return []
    found = [
        child / RUN_LEAF
        for child in sorted(outputs_dir.iterdir())
        if (child / RUN_LEAF / RUNS_DIRNAME).is_dir()
    ]
    return found


def collect_run(
    path: Path,
    seeds: Iterable[int] | None = None,
    datasets: Iterable[str] | None = None,
) -> Run:
    """Read every seed's reports under one run root."""
    runs_root = resolve_runs_root(path)
    seed_filter = set(seeds) if seeds else None
    dataset_filter = set(datasets) if datasets else None

    reports: dict[str, dict[int, Report]] = {}
    labels: dict[str, str] = {}
    found_seeds: list[int] = []

    for seed_dir in sorted(runs_root.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith(SEED_PREFIX):
            continue
        try:
            seed = int(seed_dir.name[len(SEED_PREFIX):])
        except ValueError:
            continue
        if seed_filter is not None and seed not in seed_filter:
            continue
        found_seeds.append(seed)

        # reports/<model_id>/<dataset>.json; report.html sits alongside the
        # model directory, so glob one level down rather than rglob.
        for report_path in sorted((seed_dir / "reports").glob("*/*.json")):
            report = read_report(report_path)
            if report is None:
                continue
            if dataset_filter is not None and report.dataset not in dataset_filter:
                continue
            reports.setdefault(report.dataset, {})[seed] = report
            labels.setdefault(report.dataset, report.label)

    return Run(_run_name(runs_root), runs_root, reports, labels, sorted(found_seeds))


def aggregate(values: list[float]) -> tuple[float | None, float | None, int]:
    """``(mean, sample std, n)``; std is ``None`` below two seeds."""
    if not values:
        return None, None, 0
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return mean, std, len(values)


def _format_cell(reports: dict[int, Report] | None) -> str:
    if not reports:
        return MISSING
    ordered = [reports[seed] for seed in sorted(reports)]
    mean, std, _ = aggregate([r.value for r in ordered])
    cell = f"{mean:.2f}" if std is None else f"{mean:.2f}±{std:.2f}"
    if any(not r.complete for r in ordered):
        cell += INCOMPLETE_MARK
    return cell


def seed_averages(run: Run, datasets: list[str]) -> dict[int, float]:
    """Each seed's mean across the datasets it scored.

    Averaging within a seed and *then* over seeds is what makes the ``Avg.``
    spread mean the same thing as every other column's: seed-to-seed
    variability. With every dataset scored on every seed this returns the same
    mean as averaging the per-dataset means, just with a std attached.
    """
    per_seed: dict[int, list[float]] = {}
    for dataset in datasets:
        for seed, report in run.reports.get(dataset, {}).items():
            per_seed.setdefault(seed, []).append(report.value)
    return {seed: statistics.fmean(values) for seed, values in sorted(per_seed.items())}


def is_ragged(run: Run, datasets: list[str]) -> bool:
    """Whether the run's datasets disagree about which seeds they scored.

    When they do, each seed's average covers a different set of datasets, so the
    ``Avg.`` mean and std are not strictly comparable and the cell is marked.
    """
    seed_sets = {
        frozenset(run.reports[dataset])
        for dataset in datasets
        if run.reports.get(dataset)
    }
    return len(seed_sets) > 1


def order_datasets(runs: list[Run], preferred: Iterable[str]) -> list[str]:
    """Datasets present in any run, in ``preferred`` order, extras appended."""
    present = {dataset for run in runs for dataset in run.reports}
    ordered = [name for name in preferred if name in present]
    ordered += sorted(present - set(ordered))
    return ordered


def _labels_for(runs: list[Run], datasets: list[str]) -> list[str]:
    labels = []
    for dataset in datasets:
        label = next(
            (run.labels[dataset] for run in runs if dataset in run.labels), dataset
        )
        labels.append(label)
    return labels


def _render(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] * len(headers)) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render_table(runs: list[Run], datasets: list[str]) -> str:
    """One row per run, one column per dataset, cells as ``mean±std``.

    ``Avg.`` averages each seed across datasets, then reports mean±std over
    seeds — same units as every other column. It is only comparable across runs
    covering the same datasets, so a run with a narrower set is marked.
    """
    headers = ["Run"] + _labels_for(runs, datasets) + ["Avg."]
    widest = max((len(run.reports) for run in runs), default=0)

    rows = []
    for run in runs:
        cells = [_format_cell(run.reports.get(dataset)) for dataset in datasets]

        averages = list(seed_averages(run, datasets).values())
        mean, std, _ = aggregate(averages)
        if mean is None:
            average = MISSING
        else:
            average = f"{mean:.2f}" if std is None else f"{mean:.2f}±{std:.2f}"
            if len(run.reports) < widest or is_ragged(run, datasets):
                average += INCOMPLETE_MARK
        rows.append([run.name] + cells + [average])
    return _render(headers, rows)


def render_per_seed_table(runs: list[Run], datasets: list[str]) -> str:
    """One row per ``(run, seed)``, with no aggregation."""
    headers = ["Run", "Seed"] + _labels_for(runs, datasets) + ["Avg."]

    rows = []
    for run in runs:
        for seed in run.seeds:
            cells, values = [], []
            for dataset in datasets:
                report = run.reports.get(dataset, {}).get(seed)
                if report is None:
                    cells.append(MISSING)
                    continue
                cell = f"{report.value:.2f}"
                if not report.complete:
                    cell += INCOMPLETE_MARK
                cells.append(cell)
                values.append(report.value)
            average = f"{statistics.fmean(values):.2f}" if values else MISSING
            rows.append([run.name, str(seed)] + cells + [average])
    return _render(headers, rows)


def provenance(run: Run, datasets: list[str]) -> list[str]:
    """Per-run stderr notes: seeds found, gaps, and every incomplete report."""
    lines = [f"  {run.name}  {run.root}"]
    if run.seeds:
        lines.append(f"    seeds found: {', '.join(str(s) for s in run.seeds)}")
    else:
        lines.append("    seeds found: none")

    if not run.reports:
        lines.append("    no readable reports")
        return lines

    missing = [dataset for dataset in datasets if not run.reports.get(dataset)]
    if missing:
        lines.append(f"    no reports for: {', '.join(missing)}")
    for dataset in datasets:
        for seed, report in sorted(run.reports.get(dataset, {}).items()):
            gap = set(run.seeds) - set(run.reports.get(dataset, {}))
            if gap and seed == min(run.reports[dataset]):
                lines.append(
                    f"    {dataset}: missing seed(s) "
                    f"{', '.join(str(s) for s in sorted(gap))}"
                )
            if not report.complete:
                lines.append(f"    seed {seed} {dataset}: {report.detail}")
    return lines
