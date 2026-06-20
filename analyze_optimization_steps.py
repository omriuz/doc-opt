#!/usr/bin/env python3
"""
Does training longer keep helping, or would an earlier checkpoint have been better?

Averages a retrieval metric (default `ndcg@5`) across datasets at each
optimization step — 0 (direct retrieval), the 100/200/300/400-step refresh-round
checkpoints, and the final document-optimization result (typically step 500) —
and plots the trend separately for code-retrieval and visual-retrieval
datasets/models, so we can see whether an intermediate checkpoint would have
scored higher than the final one.

Usage:
    python analyze_optimization_steps.py [--metric ndcg@5] [--output FILE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from run_status import (
    ARTIFACTS_ROOT, CLOSED_WEIGHT_MODELS, OPEN_WEIGHT_MODELS,
    CODE_DATASETS, VISUAL_DATASETS, MODEL_DISPLAY,
    _load_config_max_steps,
)

DEFAULT_OUTPUT = Path("optimization_steps_analysis.png")
ALL_MODELS = CLOSED_WEIGHT_MODELS + OPEN_WEIGHT_MODELS


def _load_step_series(dataset: str, model: str, metric: str) -> dict[int, float]:
    """Return {optimization_step: metric_value} for a single (dataset, model) run."""
    report_path = ARTIFACTS_ROOT / dataset / model / "run_report.json"
    if not report_path.exists():
        return {}
    with open(report_path) as f:
        report = json.load(f)
    stages = report.get("stages", {})

    series: dict[int, float] = {}

    direct = stages.get("direct retrieval")
    if direct and metric in direct.get("metrics", {}):
        series[0] = direct["metrics"][metric]

    for round_info in stages.get("refresh") or []:
        value = round_info.get("metrics", {}).get(metric)
        if value is not None:
            series[round_info["completed_steps"]] = value

    # The final "document optimization" result lands at the configured max_steps —
    # it isn't itself stamped with a step count, so look it up via the run's config.
    doc_opt = stages.get("document optimization")
    if doc_opt and metric in doc_opt.get("metrics", {}):
        max_steps = _load_config_max_steps(report.get("config_path", ""))
        if max_steps:
            series[max_steps] = doc_opt["metrics"][metric]

    return series


def _average_series(series_list: list[dict[int, float]]) -> list[tuple[int, float, int]]:
    """Average several {step: value} series, returning sorted (step, mean, n_runs)."""
    all_steps = sorted({step for series in series_list for step in series})
    result = []
    for step in all_steps:
        values = [series[step] for series in series_list if step in series]
        if values:
            result.append((step, sum(values) / len(values), len(values)))
    return result


def _build_domain_data(dataset_keys: list[str], metric: str):
    """Return (per_model_series, overall_series) for a retrieval domain (code or visual)."""
    per_model: dict[str, list[tuple[int, float, int]]] = {}
    pooled: list[dict[int, float]] = []

    for model in ALL_MODELS:
        series_list = [s for ds in dataset_keys if (s := _load_step_series(ds, model, metric))]
        if series_list:
            per_model[model] = _average_series(series_list)
            pooled.extend(series_list)

    overall = _average_series(pooled)
    return per_model, overall


def _best_step(series: list[tuple[int, float, int]]) -> Optional[tuple[int, float]]:
    if not series:
        return None
    step, value, _ = max(series, key=lambda item: item[1])
    return step, value


def render(code_data, visual_data, metric: str, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    for axis, (title, (per_model, overall)) in zip(
        axes, [("Code Retrieval", code_data), ("Visual Retrieval", visual_data)]
    ):
        for model, series in per_model.items():
            steps = [s for s, _, _ in series]
            values = [v * 100 for _, v, _ in series]
            axis.plot(steps, values, marker="o", linewidth=1.4, alpha=0.55,
                      label=MODEL_DISPLAY.get(model, model))

        if overall:
            steps = [s for s, _, _ in overall]
            values = [v * 100 for _, v, _ in overall]
            axis.plot(steps, values, marker="o", linewidth=3.2, color="#153243",
                      label="All models (avg)", zorder=5)

            best = _best_step(overall)
            if best:
                best_step, best_value = best
                axis.axvline(best_step, color="#D1495B", linestyle="--", alpha=0.6, zorder=1)
                axis.annotate(
                    f"best: step {best_step}\n({best_value * 100:.1f})",
                    xy=(best_step, best_value * 100),
                    xytext=(10, -24), textcoords="offset points",
                    fontsize=9, color="#D1495B", fontweight="bold",
                )

        axis.set_title(title, fontsize=14, fontweight="bold")
        axis.set_xlabel("Optimization step")
        axis.set_ylabel(f"avg {metric} (%)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, loc="lower right")

    fig.suptitle("Does training longer keep helping?", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)
    print(f"\n→ Saved plot to {output_path.resolve()}")


def _print_summary(label: str, overall: list[tuple[int, float, int]], metric: str) -> None:
    print(f"\n{label} retrieval — avg {metric} by step (n = number of runs averaged):")
    for step, value, n in overall:
        print(f"  step {step:>4d}: {value * 100:6.2f}  (n={n})")
    best = _best_step(overall)
    if best:
        print(f"  -> best on average at step {best[0]} ({best[1] * 100:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze metric trends across optimization steps")
    parser.add_argument("--metric", default="ndcg@5", help="Metric to analyze (default: ndcg@5)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output plot path")
    args = parser.parse_args()

    code_data = _build_domain_data(CODE_DATASETS, args.metric)
    visual_data = _build_domain_data(VISUAL_DATASETS, args.metric)

    _print_summary("Code", code_data[1], args.metric)
    _print_summary("Visual", visual_data[1], args.metric)

    render(code_data, visual_data, args.metric, Path(args.output))


if __name__ == "__main__":
    main()
