from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METRIC_SUFFIXES = (1, 5, 10)
NDCG_KEYS = tuple(f"ndcg@{value}" for value in METRIC_SUFFIXES)
RECALL_KEYS = tuple(f"recall@{value}" for value in METRIC_SUFFIXES)
METRIC_LABELS = {
    "ndcg@1": "@1",
    "ndcg@5": "@5",
    "ndcg@10": "@10",
    "recall@1": "@1",
    "recall@5": "@5",
    "recall@10": "@10",
}
METRIC_TITLES = {
    "ndcg@1": "NDCG@1",
    "ndcg@5": "NDCG@5",
    "ndcg@10": "NDCG@10",
    "recall@1": "Recall@1",
    "recall@5": "Recall@5",
    "recall@10": "Recall@10",
}
STAGE_COLORS = {
    "baseline": "#A0AEC0",
    "direct": "#F4A261",
    "refresh": "#2A9D8F",
    "optimized": "#2A9D8F",
}
STEP_LINE_STYLES = {
    "ndcg@1": {"color": "#D1495B", "marker": "o", "linestyle": "-"},
    "ndcg@5": {"color": "#F79256", "marker": "s", "linestyle": "-"},
    "ndcg@10": {"color": "#EDAe49", "marker": "D", "linestyle": "-"},
    "recall@1": {"color": "#00798C", "marker": "o", "linestyle": "--"},
    "recall@5": {"color": "#3066BE", "marker": "s", "linestyle": "--"},
    "recall@10": {"color": "#2A9D8F", "marker": "D", "linestyle": "--"},
}
STEP_PLOT_METRIC_KEYS = ("ndcg@5", "ndcg@10", "recall@5", "recall@10")
STEP_PLOT_FAMILIES = (
    ("NDCG", ("ndcg@5", "ndcg@10"), "Ranking quality"),
    ("Recall", ("recall@5", "recall@10"), "Document coverage"),
)


@dataclass(frozen=True, slots=True)
class PlotStage:
    key: str
    label: str
    metrics: dict[str, float]
    style_key: str


@dataclass(frozen=True, slots=True)
class StepPlotPoint:
    step: int
    label: str
    metrics: dict[str, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a README-ready SVG chart from a run report.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("artifacts/reproduce/run_report.json"),
        help="Path to the JSON run report.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output SVG path. Defaults to <report_dir>/run_report_plot.svg.",
    )
    return parser


def build_steps_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a step-wise SVG chart from a run report.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("artifacts/reproduce/run_report.json"),
        help="Path to the JSON run report.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output SVG path. Defaults to <report_dir>/run_report_steps_plot.svg.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = args.output_path or args.report_path.with_name("run_report_plot.svg")
    generate_run_report_plot(args.report_path, output_path)
    print(f"Wrote plot to {output_path}", flush=True)
    return 0


def main_steps(argv: list[str] | None = None) -> int:
    args = build_steps_parser().parse_args(argv)
    output_path = args.output_path or args.report_path.with_name("run_report_steps_plot.svg")
    generate_run_report_steps_plot(args.report_path, output_path)
    print(f"Wrote plot to {output_path}", flush=True)
    return 0


def generate_run_report_plot(report_path: Path, output_path: Path) -> Path:
    report = load_run_report(report_path)
    stages = normalize_report_stages(report)
    validate_metric_families(stages)
    render_run_report_plot(stages, output_path=output_path)
    return output_path


def generate_run_report_steps_plot(report_path: Path, output_path: Path) -> Path:
    report = load_run_report(report_path)
    checkpoints = normalize_report_checkpoints(report)
    validate_checkpoint_metric_families(checkpoints)
    render_run_report_steps_plot(checkpoints, output_path=output_path)
    return output_path


def load_run_report(report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        raise FileNotFoundError(f"Run report not found: {report_path}")
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Run report must deserialize to a JSON object.")
    return raw


def normalize_report_stages(report: dict[str, Any]) -> list[PlotStage]:
    raw_stages = report.get("stages")
    if not isinstance(raw_stages, dict):
        raise ValueError("Run report is missing a valid 'stages' mapping.")

    normalized: list[PlotStage] = []
    baseline = _normalize_single_stage(
        raw_stages,
        ("baseline", "direct retrieval"),
        "Direct Retrieval",
        "baseline",
    )
    if baseline is not None:
        normalized.append(baseline)

    direct_transform = _normalize_single_stage(
        raw_stages,
        ("doc-tranform", "direct transformation"),
        "Document Transformation",
        "direct",
    )
    if direct_transform is not None:
        normalized.append(direct_transform)

    optimized = _normalize_single_stage(
        raw_stages,
        ("optimized", "document optimization"),
        "Document Optimization",
        "optimized",
    )
    if optimized is not None:
        normalized.append(optimized)
    else:
        refresh = _select_latest_refresh_stage(raw_stages.get("refresh"))
        if refresh is not None:
            normalized.append(refresh)

    plottable: list[PlotStage] = []
    for stage in normalized:
        if any(key in stage.metrics for key in (*NDCG_KEYS, *RECALL_KEYS)):
            plottable.append(stage)
        else:
            _warn(f"Skipping stage '{stage.label}' because it does not include plottable metrics.")

    if not plottable:
        raise ValueError("No plottable stages found in the run report.")
    return plottable


def validate_metric_families(stages: list[PlotStage]) -> None:
    has_ndcg = any(any(key in stage.metrics for key in NDCG_KEYS) for stage in stages)
    has_recall = any(any(key in stage.metrics for key in RECALL_KEYS) for stage in stages)
    if not has_ndcg:
        raise ValueError("No NDCG metrics are available to plot.")
    if not has_recall:
        raise ValueError("No Recall metrics are available to plot.")


def normalize_report_checkpoints(report: dict[str, Any]) -> list[StepPlotPoint]:
    raw_stages = report.get("stages")
    if not isinstance(raw_stages, dict):
        raise ValueError("Run report is missing a valid 'stages' mapping.")

    checkpoints: list[StepPlotPoint] = []
    direct_transform = _normalize_single_stage(
        raw_stages,
        ("doc-tranform", "direct transformation"),
        "Document Transformation",
        "direct",
    )
    if direct_transform is not None:
        checkpoints.append(StepPlotPoint(step=0, label=direct_transform.label, metrics=direct_transform.metrics))
    else:
        baseline = _normalize_single_stage(
            raw_stages,
            ("baseline", "direct retrieval"),
            "Direct Retrieval",
            "baseline",
        )
        if baseline is not None:
            checkpoints.append(StepPlotPoint(step=0, label=baseline.label, metrics=baseline.metrics))

    checkpoints.extend(_normalize_refresh_checkpoints(raw_stages.get("refresh")))

    optimized_stage = raw_stages.get("document optimization") or raw_stages.get("optimized")
    optimized_step = _extract_completed_steps(optimized_stage)
    optimized_metrics = _extract_metrics(optimized_stage)
    if optimized_step is not None and optimized_metrics:
        checkpoints.append(
            StepPlotPoint(
                step=optimized_step,
                label="Document Optimization",
                metrics=optimized_metrics,
            )
        )

    if not checkpoints:
        raise ValueError("No optimization checkpoints found in the run report.")
    return sorted(checkpoints, key=lambda checkpoint: checkpoint.step)


def validate_checkpoint_metric_families(checkpoints: list[StepPlotPoint]) -> None:
    has_ndcg = any(any(key in checkpoint.metrics for key in NDCG_KEYS) for checkpoint in checkpoints)
    has_recall = any(any(key in checkpoint.metrics for key in RECALL_KEYS) for checkpoint in checkpoints)
    if not has_ndcg:
        raise ValueError("No NDCG metrics are available to plot.")
    if not has_recall:
        raise ValueError("No Recall metrics are available to plot.")


def render_run_report_plot(
    stages: list[PlotStage],
    *,
    output_path: Path,
) -> None:
    plt = _load_pyplot(output_path)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.6, 5.4),
        dpi=220,
    )
    figure.set_facecolor("#FBFAF7")

    _render_metric_panel(
        axes[0],
        stages=stages,
        metric_keys=NDCG_KEYS,
        title="NDCG",
    )
    _render_metric_panel(
        axes[1],
        stages=stages,
        metric_keys=RECALL_KEYS,
        title="Recall",
    )

    legend_handles = _build_stage_legend_handles(stages)

    figure.suptitle(
        "Same Retriever, Better Documents",
        fontsize=20,
        fontweight="bold",
        color="#153243",
        y=0.982,
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.928),
        ncol=max(1, len(legend_handles)),
        frameon=False,
        fontsize=10.5,
        handlelength=1.3,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    figure.text(
        0.06,
        0.035,
        "Dataset: DS1000Retrieval (MTEB). Retriever is a black-box OpenAI embedding API "
        "(text-embedding-3-small).",
        ha="left",
        va="center",
        fontsize=14.0,
        fontweight="bold",
        color="#54606B",
    )
    figure.subplots_adjust(left=0.06, right=0.98, top=0.73, bottom=0.19, wspace=0.12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg", facecolor=figure.get_facecolor())
    plt.close(figure)


def render_run_report_steps_plot(
    checkpoints: list[StepPlotPoint],
    *,
    output_path: Path,
) -> None:
    plt = _load_pyplot(output_path)

    figure, axes = plt.subplots(1, 2, figsize=(15.8, 7.8), dpi=220)
    figure.set_facecolor("#FBFAF7")

    steps = [checkpoint.step for checkpoint in checkpoints]
    end_label_dx = max((steps[-1] - steps[0]) * 0.08, 10.0) if len(steps) > 1 else 10.0

    for axis, (title, metric_keys, subtitle) in zip(axes, STEP_PLOT_FAMILIES):
        _render_step_metric_panel(
            axis,
            checkpoints=checkpoints,
            steps=steps,
            metric_keys=metric_keys,
            title=title,
            subtitle=subtitle,
            end_label_dx=end_label_dx,
            show_ylabel=(axis is axes[0]),
        )

    figure.suptitle("Retrieval Improves through Training", fontsize=19, fontweight="bold", color="#153243", y=0.965)
    figure.text(
        0.5,
        0.05,
        "Dataset: DS1000Retrieval (MTEB). Step 0 = direct document transformation before optimization begins. Code retrieval uses text-embedding-3-small.",
        ha="center",
        va="center",
        fontsize=14.0,
        fontweight="bold",
        color="#54606B",
    )
    figure.subplots_adjust(left=0.07, right=0.975, top=0.83, bottom=0.19, wspace=0.11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg", bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def _render_step_metric_panel(
    axis,
    *,
    checkpoints: list[StepPlotPoint],
    steps: list[int],
    metric_keys: tuple[str, ...],
    title: str,
    subtitle: str,
    end_label_dx: float,
    show_ylabel: bool,
) -> None:
    axis.set_facecolor("#FFFDF8")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#CCD5DD")
    axis.spines["bottom"].set_color("#CCD5DD")
    axis.grid(axis="y", color="#E2E8ED", linewidth=0.9, alpha=0.95)
    axis.grid(axis="x", color="#EFE4D2", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)

    if len(steps) > 1:
        for index, (start, end) in enumerate(zip(steps, steps[1:])):
            if index % 2 == 0:
                axis.axvspan(start, end, color="#F4A261", alpha=0.07, zorder=0)

    panel_metric_keys = [key for key in metric_keys if any(key in checkpoint.metrics for checkpoint in checkpoints)]
    max_value = max(
        (
            checkpoint.metrics[key] * 100
            for checkpoint in checkpoints
            for key in panel_metric_keys
            if key in checkpoint.metrics
        ),
        default=0.0,
    )
    min_value = min(
        (
            checkpoint.metrics[key] * 100
            for checkpoint in checkpoints
            for key in panel_metric_keys
            if key in checkpoint.metrics
        ),
        default=0.0,
    )
    y_lower, y_upper = _step_plot_limits(min_value, max_value)
    axis.set_ylim(y_lower, y_upper)

    endpoint_specs: list[tuple[float, str, str, float, float]] = []
    for metric_key in panel_metric_keys:
        style = STEP_LINE_STYLES[metric_key]
        values = [checkpoint.metrics.get(metric_key, math.nan) * 100 for checkpoint in checkpoints]
        axis.plot(
            steps,
            values,
            color=style["color"],
            linewidth=2.6,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=7.5,
            markerfacecolor="#FBFAF7",
            markeredgewidth=2.0,
            markeredgecolor=style["color"],
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=4,
        )

        finite_points = [
            (checkpoint.step, checkpoint.metrics[metric_key] * 100)
            for checkpoint in checkpoints
            if metric_key in checkpoint.metrics and math.isfinite(checkpoint.metrics[metric_key])
        ]
        if finite_points:
            end_x, end_y = finite_points[-1]
            endpoint_specs.append((end_y, METRIC_TITLES[metric_key], style["color"], end_x, end_y))

    for adjusted_y, label, color, end_x, end_y in _spread_endpoint_labels(
        endpoint_specs,
        lower_bound=y_lower,
        upper_bound=y_upper,
    ):
        axis.plot(
            [end_x, end_x + end_label_dx * 0.46],
            [end_y, adjusted_y],
            color=color,
            linewidth=1.2,
            alpha=0.9,
            zorder=5,
        )
        axis.text(
            end_x + end_label_dx * 0.56,
            adjusted_y,
            label,
            ha="left",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color=color,
        )

    axis.set_title(title, fontsize=17, fontweight="bold", color="#153243", pad=18)
    axis.text(
        0.5,
        1.02,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.2,
        color="#6B7782",
    )
    axis.set_xlabel("GRPO steps", fontsize=12.0, color="#40515C", labelpad=13)
    if show_ylabel:
        axis.set_ylabel("Performance (%)", fontsize=12.0, color="#40515C", labelpad=10)
    axis.set_xticks(steps)
    axis.set_xlim(steps[0] - max(end_label_dx * 0.18, 3.0), steps[-1] + end_label_dx * 1.1)
    axis.tick_params(axis="x", labelsize=11.0, colors="#40515C")
    axis.tick_params(axis="y", labelsize=10.8, colors="#40515C")


def _normalize_single_stage(
    raw_stages: dict[str, Any],
    candidate_keys: tuple[str, ...],
    label: str,
    style_key: str,
) -> PlotStage | None:
    stage_key = next((key for key in candidate_keys if key in raw_stages), None)
    if stage_key is None:
        return None
    raw_stage = raw_stages.get(stage_key)
    if raw_stage is None:
        return None
    metrics = _extract_metrics(raw_stage)
    if not metrics:
        _warn(f"Skipping stage '{label}' because its metrics payload is empty.")
        return None
    return PlotStage(key=stage_key, label=label, metrics=metrics, style_key=style_key)


def _select_latest_refresh_stage(raw_refresh: Any) -> PlotStage | None:
    if raw_refresh is None:
        return None
    if not isinstance(raw_refresh, list):
        raise ValueError("Expected 'refresh' to be a list of refresh checkpoints.")

    sortable_entries: list[tuple[int, int, dict[str, Any]]] = []
    for index, entry in enumerate(raw_refresh):
        if not isinstance(entry, dict):
            _warn(f"Skipping refresh entry {index + 1} because it is not an object.")
            continue
        completed_steps = int(entry.get("completed_steps", index + 1))
        sortable_entries.append((completed_steps, index, entry))

    latest_stage: PlotStage | None = None
    for completed_steps, index, entry in sorted(sortable_entries):
        metrics = _extract_metrics(entry)
        if not metrics:
            _warn(f"Skipping refresh entry {index + 1} because its metrics payload is empty.")
            continue
        latest_stage = PlotStage(
            key=f"refresh-{completed_steps}",
            label="Document Optimization",
            metrics=metrics,
            style_key="refresh",
        )
    return latest_stage


def _normalize_refresh_checkpoints(raw_refresh: Any) -> list[StepPlotPoint]:
    if raw_refresh is None:
        return []
    if not isinstance(raw_refresh, list):
        raise ValueError("Expected 'refresh' to be a list of refresh checkpoints.")

    checkpoints: list[StepPlotPoint] = []
    sortable_entries: list[tuple[int, int, dict[str, Any]]] = []
    for index, entry in enumerate(raw_refresh):
        if not isinstance(entry, dict):
            _warn(f"Skipping refresh entry {index + 1} because it is not an object.")
            continue
        completed_steps = int(entry.get("completed_steps", index + 1))
        sortable_entries.append((completed_steps, index, entry))

    for completed_steps, index, entry in sorted(sortable_entries):
        metrics = _extract_metrics(entry)
        if not metrics:
            _warn(f"Skipping refresh entry {index + 1} because its metrics payload is empty.")
            continue
        checkpoints.append(
            StepPlotPoint(
                step=completed_steps,
                label=f"Refresh {entry.get('refresh_round', index + 1)}",
                metrics=metrics,
            )
        )
    return checkpoints


def _extract_completed_steps(raw_stage: Any) -> int | None:
    if not isinstance(raw_stage, dict):
        return None
    raw_value = raw_stage.get("completed_steps")
    if raw_value is None:
        return None
    return int(raw_value)


def _extract_metrics(raw_stage: Any) -> dict[str, float]:
    if not isinstance(raw_stage, dict):
        return {}
    raw_metrics = raw_stage.get("metrics")
    if not isinstance(raw_metrics, dict):
        return {}

    metrics: dict[str, float] = {}
    for key in (*NDCG_KEYS, *RECALL_KEYS):
        value = raw_metrics.get(key)
        if value is None:
            continue
        metrics[key] = float(value)
    return metrics


def _render_metric_panel(axis, *, stages: list[PlotStage], metric_keys: tuple[str, ...], title: str) -> None:
    import numpy as np

    group_positions = np.arange(len(metric_keys), dtype=float)
    axis.set_facecolor("#FFFFFF")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#CCD5DD")
    axis.spines["bottom"].set_color("#CCD5DD")
    axis.grid(axis="y", color="#E2E8ED", linewidth=0.9, alpha=0.95)
    axis.set_axisbelow(True)
    bar_width = min(0.72 / max(len(stages), 1), 0.22)
    value_matrix = [
        [stage.metrics.get(metric_key, math.nan) * 100 for metric_key in metric_keys]
        for stage in stages
    ]
    max_value = max(
        (value for row in value_matrix for value in row if math.isfinite(value)),
        default=0.0,
    )
    upper_bound = _rounded_upper_bound(max_value)

    stage_offset_center = (len(stages) - 1) / 2
    for stage_index, stage in enumerate(stages):
        offsets = group_positions + (stage_index - stage_offset_center) * bar_width
        values = np.asarray([stage.metrics.get(metric_key, math.nan) * 100 for metric_key in metric_keys], dtype=float)
        bars = axis.bar(
            offsets,
            values,
            width=bar_width * 0.9,
            color=STAGE_COLORS[stage.style_key],
            edgecolor="#F6F7F9",
            linewidth=1.0,
            zorder=3,
            label=stage.label,
        )
        for bar, value in zip(bars, values):
            if not math.isfinite(value):
                continue
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + upper_bound * 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.6,
                fontweight="bold",
                color="#33414B",
            )

    subtitle = "Ranking quality" if title == "NDCG" else "Document coverage"
    axis.set_title(title, fontsize=16, fontweight="bold", color="#153243", pad=18)
    axis.text(
        0.5,
        1.02,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#6B7782",
    )
    axis.set_xticks(group_positions)
    axis.set_xticklabels([METRIC_LABELS[key] for key in metric_keys], fontsize=11, color="#23343E")
    axis.set_ylabel("Percent", fontsize=10.5, color="#40515C")
    axis.set_ylim(0, upper_bound)
    axis.set_xlim(-0.55, len(metric_keys) - 0.45)
    axis.tick_params(axis="y", labelsize=10, colors="#40515C")
    axis.tick_params(axis="x", pad=8, length=0)


def _build_stage_legend_handles(stages: list[PlotStage]):
    from matplotlib.patches import Patch

    seen: set[str] = set()
    handles: list[Patch] = []
    for stage in stages:
        if stage.label in seen:
            continue
        seen.add(stage.label)
        handles.append(
            Patch(
                facecolor=STAGE_COLORS[stage.style_key],
                edgecolor="none",
                label=stage.label,
            )
        )
    return handles


def _rounded_upper_bound(max_value: float) -> float:
    if max_value <= 0:
        return 10.0
    padded = max_value + 4.5
    rounded = math.ceil(padded / 5.0) * 5.0
    return max(rounded, 35.0 if max_value <= 35 else rounded)


def _rounded_lower_bound(min_value: float) -> float:
    if min_value <= 8:
        return 0.0
    return max(0.0, math.floor((min_value - 4.5) / 5.0) * 5.0)


def _spread_endpoint_labels(
    endpoints: list[tuple[float, str, str, float, float]],
    *,
    min_gap: float = 2.2,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> list[tuple[float, str, str, float, float]]:
    if not endpoints:
        return []

    sorted_endpoints = sorted(endpoints, key=lambda item: item[0])
    adjusted: list[list[float | str]] = []
    for index, (y_value, label, color, end_x, end_y) in enumerate(sorted_endpoints):
        adjusted_y = y_value
        if index > 0:
            previous_y = float(adjusted[-1][0])
            adjusted_y = max(adjusted_y, previous_y + min_gap)
        adjusted.append([adjusted_y, label, color, end_x, end_y])

    for index in range(len(adjusted) - 2, -1, -1):
        current_y = float(adjusted[index][0])
        next_y = float(adjusted[index + 1][0])
        if next_y - current_y < min_gap:
            adjusted[index][0] = next_y - min_gap

    if upper_bound is not None:
        overflow = float(adjusted[-1][0]) - upper_bound
        if overflow > 0:
            for item in adjusted:
                item[0] = float(item[0]) - overflow

    if lower_bound is not None:
        underflow = lower_bound - float(adjusted[0][0])
        if underflow > 0:
            for item in adjusted:
                item[0] = float(item[0]) + underflow

    return [(float(item[0]), str(item[1]), str(item[2]), float(item[3]), float(item[4])) for item in adjusted]


def _step_plot_limits(min_value: float, max_value: float) -> tuple[float, float]:
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        return 0.0, 1.0
    if math.isclose(min_value, max_value):
        padding = max(1.0, abs(max_value) * 0.05)
        return min_value - padding, max_value + padding
    return min_value - 3.0, max_value + 3.0


def _load_pyplot(output_path: Path):
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = output_path.parent / ".matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_config_dir.as_posix()

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "DejaVu Sans",
            "axes.titlepad": 10,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def _warn(message: str) -> None:
    print(f"[doc-opt plot] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
