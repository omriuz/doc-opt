"""
Run status and results summary for document-optimization experiments.

Usage:
    python run_status.py [--artifacts DIR] [--metric METRIC] [--output FILE]

Scans artifacts/ and writes a markdown report with:
  1. Status of each run (not started / initialized / in progress / completed)
  2. Results table: model x dataset with direct retrieval / naive / doc-opt scores
  3. Deep-dive on incomplete runs — last activity, trainer state, stuck detection

Output is saved to run_status.md by default and also printed to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

ARTIFACTS_ROOT = Path("artifacts")
CONFIGS_ROOT   = Path("configs")
# Visual-dataset runs reference config files that only live in the sibling
# document-optimization-visual checkout, not this repo's configs/ dir.
VISUAL_CONFIGS_ROOT = Path("/juice2/scr2/uzan/document-optimization-visual/configs")
DEFAULT_OUTPUT = Path("run_status.md")

STALE_HOURS   = 4.0   # hours without file activity → possibly stuck (each training round takes ~3h)
ACTIVE_MINUTES = 60   # minutes → consider actively running (artifact files only update at round boundaries)

STATUS_NOT_STARTED = "not started"
STATUS_INITIALIZED = "initialized"
STATUS_IN_PROGRESS = "in progress"
STATUS_CRASHED     = "crashed"
STATUS_COMPLETED   = "completed"

STATUS_EMOJI = {
    STATUS_NOT_STARTED: "⬜",
    STATUS_INITIALIZED: "🟡",
    STATUS_IN_PROGRESS: "🟠",
    STATUS_CRASHED:     "🔴",
    STATUS_COMPLETED:   "🟢",
}


@dataclass
class RunInfo:
    dataset: str
    model: str
    status: str
    run_dir: Path
    completed_steps: int = 0
    max_steps: int = 0
    refresh_rounds_done: int = 0
    direct_retrieval: Optional[dict] = None
    naive_transform: Optional[dict] = None
    doc_opt: Optional[dict] = None
    best_refresh: Optional[dict] = None
    refresh_by_step: dict = None
    config_path: str = ""


def _load_config_max_steps(config_path: str) -> int:
    p = Path(config_path)
    if not p.exists():
        p = CONFIGS_ROOT / p.name
    if not p.exists():
        p = VISUAL_CONFIGS_ROOT / Path(config_path).name
    if not p.exists():
        return 0
    with open(p) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("doc-opt", {}).get("max_steps", 0)


def scan_runs() -> list[RunInfo]:
    runs = []
    now = datetime.now()
    if not ARTIFACTS_ROOT.exists():
        return runs

    for dataset_dir in sorted(ARTIFACTS_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            report_path = model_dir / "run_report.json"

            if not report_path.exists():
                runs.append(RunInfo(
                    dataset=dataset, model=model,
                    status=STATUS_NOT_STARTED, run_dir=model_dir,
                ))
                continue

            with open(report_path) as f:
                report = json.load(f)

            stages         = report.get("stages", {})
            refresh_rounds = stages.get("refresh") or []
            doc_opt_stage  = stages.get("document optimization")
            direct_stage   = stages.get("direct retrieval")
            naive_stage    = stages.get("direct transformation")
            config_path    = report.get("config_path", "")

            max_steps       = _load_config_max_steps(config_path)
            completed_steps = refresh_rounds[-1]["completed_steps"] if refresh_rounds else 0

            if doc_opt_stage is not None:
                status = STATUS_COMPLETED
            elif refresh_rounds:
                status = STATUS_IN_PROGRESS
            else:
                status = STATUS_INITIALIZED

            if status in (STATUS_IN_PROGRESS, STATUS_INITIALIZED):
                newest = _newest_mtime(model_dir)
                if newest is not None:
                    delta_h = (now - newest).total_seconds() / 3600
                    if delta_h >= STALE_HOURS:
                        status = STATUS_CRASHED

            best_refresh = (
                max(refresh_rounds, key=lambda r: r["metrics"].get("ndcg@10", 0.0))
                if refresh_rounds else None
            )
            refresh_by_step = {rr["completed_steps"]: rr["metrics"] for rr in refresh_rounds}

            runs.append(RunInfo(
                dataset=dataset,
                model=model,
                status=status,
                run_dir=model_dir,
                completed_steps=completed_steps,
                max_steps=max_steps,
                refresh_rounds_done=len(refresh_rounds),
                direct_retrieval=direct_stage.get("metrics") if direct_stage else None,
                naive_transform=naive_stage.get("metrics") if naive_stage else None,
                doc_opt=doc_opt_stage.get("metrics") if doc_opt_stage else None,
                best_refresh=best_refresh["metrics"] if best_refresh else None,
                refresh_by_step=refresh_by_step,
                config_path=config_path,
            ))

    return runs


MODEL_DISPLAY = {
    "text-embedding-3-small":    "OpenAI v3-small",
    "text-embedding-3-large":    "OpenAI v3-large",
    "Qwen_Qwen3-Embedding-0.6B": "Qwen3-Emb-0.6B",
    "Qwen_Qwen3-Embedding-4B":   "Qwen3-Emb-4B",
}

CLOSED_WEIGHT_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]
OPEN_WEIGHT_MODELS   = ["Qwen_Qwen3-Embedding-0.6B", "Qwen_Qwen3-Embedding-4B"]

# Methods shown for closed-weights retrievers stop at "Policy Optimization" — the
# remaining two require access to the retriever's weights.
EXTENDED_METHOD_LABELS = [
    "Direct Retrieval", "Zero-shot Transformation", "Policy Optimization",
    "Retriever Finetuning", "Joint Adaptation",
]

CODE_DATASETS = ["humaneval", "mbpp", "ds1000", "freshstack"]
CODE_DATASET_DISPLAY = {
    "humaneval": "Human Eval", "mbpp": "MBPP", "ds1000": "DS10K", "freshstack": "Fresh Stack",
}

VISUAL_DATASETS = ["biomedical", "economics_reports", "esg_reports_human_labeled", "esg_reports"]
VISUAL_DATASET_DISPLAY = {
    "biomedical": "Biomed", "economics_reports": "Econ",
    "esg_reports_human_labeled": "ESG Human", "esg_reports": "ESG Full",
}


def _base_dataset(dataset: str) -> str:
    return dataset.removesuffix("_local")


def _fmt(val: Optional[float]) -> str:
    return f"{val:.3f}" if val is not None else "—"


def _fmt_pct(val: Optional[float]) -> str:
    return f"{val * 100:.1f}" if val is not None else "—"


def _fmt_delta(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"+{val * 100:.1f}" if val >= 0 else f"{val * 100:.1f}"


def _m(metrics: Optional[dict], key: str) -> Optional[float]:
    return metrics.get(key) if metrics else None


def _avg(vals: list) -> Optional[float]:
    valid = [v for v in vals if v is not None]
    return sum(valid) / len(valid) if valid else None


def _load_finetuned_metric(dataset: str, model: str, metric: str) -> Optional[float]:
    """Look up the post-finetuning metric for `model` from <dataset>_finetune/finetune_report.json."""
    report_path = ARTIFACTS_ROOT / f"{dataset}_finetune" / "finetune_report.json"
    if not report_path.exists():
        return None
    with open(report_path) as f:
        entries = json.load(f)
    for entry in entries:
        if entry.get("model", "").replace("/", "_") == model:
            return entry.get("finetuned_metrics", {}).get(metric)
    return None


def _build_results_index(runs: list[RunInfo], metric: str,
                          checkpoint_step: Optional[int] = None) -> dict:
    """Build index: (base_dataset, model) → (direct, naive, docopt, approx) values.

    When the same combination appears in both a _local and non-local dir,
    prefer the one with more progress (completed > in_progress > initialized).

    If `checkpoint_step` is given, the "policy optimization" value is taken
    from that refresh round's checkpoint instead of the run's final result —
    useful for asking "what if we'd stopped early?".
    """
    STATUS_RANK = {STATUS_COMPLETED: 3, STATUS_IN_PROGRESS: 2, STATUS_CRASHED: 2,
                   STATUS_INITIALIZED: 1, STATUS_NOT_STARTED: 0}
    best: dict = {}
    for r in runs:
        key = (_base_dataset(r.dataset), r.model)
        if key not in best or STATUS_RANK[r.status] > STATUS_RANK[best[key].status]:
            best[key] = r

    index = {}
    for (base_ds, model), r in best.items():
        direct = _m(r.direct_retrieval, metric)
        naive  = _m(r.naive_transform,  metric)
        if checkpoint_step is not None:
            checkpoint = (r.refresh_by_step or {}).get(checkpoint_step)
            if checkpoint:
                docopt = _m(checkpoint, metric)
                approx = False
            elif r.doc_opt:
                docopt = _m(r.doc_opt, metric)
                approx = True
            elif r.best_refresh:
                docopt = _m(r.best_refresh, metric)
                approx = True
            else:
                docopt = None
                approx = False
        elif r.doc_opt:
            docopt = _m(r.doc_opt, metric)
            approx = False
        elif r.best_refresh:
            docopt = _m(r.best_refresh, metric)
            approx = True
        else:
            docopt = None
            approx = False
        index[(base_ds, model)] = (direct, naive, docopt, approx)
    return index


def _newest_mtime(run_dir: Path) -> Optional[datetime]:
    """Return the most recent modification time of any file under run_dir."""
    newest: Optional[float] = None
    for root, _, files in os.walk(run_dir):
        for fname in files:
            try:
                mt = os.path.getmtime(os.path.join(root, fname))
                if newest is None or mt > newest:
                    newest = mt
            except OSError:
                pass
    if newest is None:
        return None
    return datetime.fromtimestamp(newest)


def _load_trainer_state(run_dir: Path) -> Optional[dict]:
    p = run_dir / "checkpoints" / "current" / "trainer_state.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _latest_checkpoint_step(run_dir: Path) -> Optional[int]:
    """Return the highest step number from saved checkpoint dirs."""
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return None
    steps = []
    for d in ckpt_root.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint_step_"):
            try:
                steps.append(int(d.name.split("_")[-1]))
            except ValueError:
                pass
    return max(steps) if steps else None


def _activity_label(dt: Optional[datetime], now: datetime) -> str:
    if dt is None:
        return "unknown"
    delta = now - dt
    minutes = delta.total_seconds() / 60
    hours   = minutes / 60
    if minutes < ACTIVE_MINUTES:
        return f"🟢 {int(minutes)}m ago (active)"
    if hours < STALE_HOURS:
        return f"🟡 {int(hours)}h {int(minutes % 60)}m ago"
    return f"🔴 {int(hours)}h {int(minutes % 60)}m ago (possibly stuck)"


def _grouped_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _build_grouped_results_table(index: dict, metric: str, title: str,
                                 dataset_keys: list[str], dataset_display: dict[str, str],
                                 include_finetune: bool) -> list[str]:
    """Render a paper-style results table grouped into Closed-weights / Open-weights
    sections, with one method row per (retriever, method).

    Closed-weights retrievers only show Direct Retrieval / Zero-shot Transformation /
    Policy Optimization — they expose no weights to finetune or jointly adapt.
    Open-weights retrievers additionally show Retriever Finetuning (populated from
    <dataset>_finetune/finetune_report.json when `include_finetune`) and Joint
    Adaptation, which is left blank pending those experiments.
    """
    lines: list[str] = []
    n_ds = len(dataset_keys)

    lines.append(f"## {title} — `{metric}` (as %)\n")

    h1_cells = ["**Retriever**", "**Method**"]
    for ds_key in dataset_keys:
        h1_cells += [f"**{dataset_display[ds_key]}**", ""]
    h1_cells += ["**AVG**", ""]
    lines.append(_grouped_table_row(h1_cells))

    h2_cells = ["", ""] + ["val", "Δ"] * (n_ds + 1)
    lines.append(_grouped_table_row(h2_cells))

    sep_cells = ["---", "---"] + ["---", "---"] * (n_ds + 1)
    lines.append(_grouped_table_row(sep_cells))

    n_cols = 2 + 2 * (n_ds + 1)
    weight_groups = [("Closed-weights", CLOSED_WEIGHT_MODELS), ("Open-weights", OPEN_WEIGHT_MODELS)]

    for group_label, group_models in weight_groups:
        models = [m for m in group_models if any((ds, m) in index for ds in dataset_keys)]
        if not models:
            continue

        lines.append(_grouped_table_row([f"_{group_label}_"] + [""] * (n_cols - 1)))

        for model in models:
            display_model = MODEL_DISPLAY.get(model, model)
            is_open = model in OPEN_WEIGHT_MODELS
            method_labels = EXTENDED_METHOD_LABELS if is_open else EXTENDED_METHOD_LABELS[:3]

            direct_vals = [index.get((ds, model), (None, None, None, False))[0] for ds in dataset_keys]
            direct_avg  = _avg(direct_vals)

            # rows_data: (method_label, [(val, is_approx)], avg)
            rows_data = []
            for method_idx, method_label in enumerate(method_labels):
                row_vals = []
                for ds_key in dataset_keys:
                    direct, naive, docopt, approx = index.get((ds_key, model), (None, None, None, False))
                    if method_idx == 0:
                        val, is_approx = direct, False
                    elif method_idx == 1:
                        val, is_approx = naive, False
                    elif method_idx == 2:
                        val, is_approx = docopt, approx
                    elif method_idx == 3:
                        val = _load_finetuned_metric(ds_key, model, metric) if include_finetune else None
                        is_approx = False
                    else:  # Joint Adaptation — left blank pending those experiments
                        val, is_approx = None, False
                    row_vals.append((val, is_approx))
                rows_data.append((method_label, row_vals, _avg(v for v, _ in row_vals)))

            # Bold the best value per column among this retriever's own methods.
            col_best = [
                max((row_vals[ds_idx][0] for _, row_vals, _ in rows_data if row_vals[ds_idx][0] is not None),
                    default=None)
                for ds_idx in range(n_ds)
            ]
            avg_best = max((avg for _, _, avg in rows_data if avg is not None), default=None)

            for row_idx, (method_label, row_vals, avg) in enumerate(rows_data):
                retriever_cell = f"**{display_model}**" if row_idx == 0 else ""
                cells = [retriever_cell, method_label]

                for ds_idx, (val, is_approx) in enumerate(row_vals):
                    direct = direct_vals[ds_idx]
                    delta  = (val - direct) if (val is not None and direct is not None) else None
                    val_str   = _fmt_pct(val)
                    delta_str = "—" if row_idx == 0 else _fmt_delta(delta)

                    if val is not None and col_best[ds_idx] is not None and abs(val - col_best[ds_idx]) < 1e-9:
                        val_str = f"**{val_str}**"
                    if is_approx:
                        val_str += "\\*"

                    cells += [val_str, delta_str]

                avg_delta = (avg - direct_avg) if (avg is not None and direct_avg is not None and row_idx > 0) else None
                avg_str   = _fmt_pct(avg)
                if avg is not None and avg_best is not None and abs(avg - avg_best) < 1e-9:
                    avg_str = f"**{avg_str}**"
                avg_delta_str = "—" if row_idx == 0 else _fmt_delta(avg_delta)
                cells += [avg_str, avg_delta_str]

                lines.append(_grouped_table_row(cells))

            lines.append(_grouped_table_row([""] * n_cols))  # spacer row between retrievers

    lines.append("")
    lines.append("_\\* best refresh round (doc-opt not yet complete)_\n")
    return lines


def build_markdown(runs: list[RunInfo], metric: str,
                   checkpoint_step: Optional[int] = None) -> str:
    lines: list[str] = []
    now = datetime.now()
    ts  = now.strftime("%Y-%m-%d %H:%M")

    lines.append("# DocOpt Run Status\n")
    lines.append(f"_Generated: {ts}_\n")
    if checkpoint_step is not None:
        lines.append(
            f"_Results tables below use the **step-{checkpoint_step}** refresh-round "
            f"checkpoint instead of each run's final result._\n"
        )

    # ── 1. Status table ───────────────────────────────────────────────────
    lines.append("## Run Status\n")
    lines.append("| Dataset | Model | Status | Progress |")
    lines.append("|---------|-------|--------|----------|")

    for r in runs:
        emoji = STATUS_EMOJI[r.status]
        if r.status in (STATUS_IN_PROGRESS, STATUS_CRASHED) and r.completed_steps:
            pct      = int(100 * r.completed_steps / r.max_steps) if r.max_steps else 0
            progress = f"{r.completed_steps}/{r.max_steps} steps ({pct}%, {r.refresh_rounds_done} refresh rounds)"
        elif r.status == STATUS_COMPLETED:
            progress = f"{r.max_steps}/{r.max_steps} steps"
        elif r.status in (STATUS_INITIALIZED, STATUS_CRASHED):
            progress = "direct retrieval done, training not started"
        else:
            progress = ""
        lines.append(f"| `{r.dataset}` | `{r.model}` | {emoji} {r.status} | {progress} |")

    lines.append("")
    legend_items = "  ".join(f"{v} {k}" for k, v in STATUS_EMOJI.items())
    lines.append(f"_{legend_items}_\n")

    # ── 2. Results tables (paper-style, by retrieval domain) ──────────────
    index = _build_results_index(runs, metric, checkpoint_step=checkpoint_step)
    lines += _build_grouped_results_table(index, metric, "Code Retrieval Results",
                                           CODE_DATASETS, CODE_DATASET_DISPLAY,
                                           include_finetune=True)
    lines += _build_grouped_results_table(index, metric, "Visual Retrieval Results",
                                           VISUAL_DATASETS, VISUAL_DATASET_DISPLAY,
                                           include_finetune=False)

    # ── 3. Deep-dive: incomplete runs ─────────────────────────────────────
    incomplete = [r for r in runs if r.status != STATUS_COMPLETED]

    lines.append("## Incomplete Runs — Deep Dive\n")

    if not incomplete:
        lines.append("All runs completed. Nothing to investigate. ✅\n")
        return "\n".join(lines)

    for r in incomplete:
        lines.append(f"### `{r.dataset}` / `{r.model}`\n")
        lines.append(f"**Status:** {STATUS_EMOJI[r.status]} {r.status}\n")

        newest   = _newest_mtime(r.run_dir)
        activity = _activity_label(newest, now)
        lines.append(f"**Last file activity:** {newest.strftime('%Y-%m-%d %H:%M') if newest else 'unknown'} — {activity}\n")

        if r.status == STATUS_NOT_STARTED:
            lines.append("Run directory exists but no `run_report.json` found. Training has not been launched.\n")
            lines.append("**Action required:** launch the run.\n")

        elif r.status == STATUS_INITIALIZED:
            lines.append(f"**Stages done:** direct retrieval ✅  naive transform ✅  training ❌\n")
            if newest:
                delta_h = (now - newest).total_seconds() / 3600
                if delta_h > STALE_HOURS:
                    lines.append(f"⚠️ No activity for **{delta_h:.1f}h** — training was never started or the launch script exited early.\n")
                    lines.append("**Action required:** re-launch the run (setup stages will be skipped).\n")
                else:
                    lines.append("Run was recently set up. Training may be starting shortly.\n")

        elif r.status == STATUS_IN_PROGRESS:
            pct = int(100 * r.completed_steps / r.max_steps) if r.max_steps else 0
            lines.append(f"**Progress:** {r.completed_steps}/{r.max_steps} steps ({pct}%), {r.refresh_rounds_done} refresh rounds completed\n")

            # Trainer state
            trainer = _load_trainer_state(r.run_dir)
            if trainer:
                history  = trainer.get("log_history", [])
                last_log = history[-1] if history else None
                lines.append(f"**Trainer global step:** {trainer.get('global_step', '?')}")
                if last_log:
                    loss   = last_log.get("train_loss") or last_log.get("loss")
                    reward = last_log.get("reward")
                    parts  = []
                    if loss   is not None: parts.append(f"loss={loss:.4f}")
                    if reward is not None: parts.append(f"reward={reward:.4f}")
                    lines.append(f"  ({', '.join(parts)})\n")
                else:
                    lines.append("\n")

            # Most recent saved checkpoint
            latest_ckpt = _latest_checkpoint_step(r.run_dir)
            if latest_ckpt:
                lines.append(f"**Latest saved checkpoint:** step {latest_ckpt}\n")

            if newest:
                delta_h = (now - newest).total_seconds() / 3600
                delta_m = delta_h * 60
                if delta_m < ACTIVE_MINUTES:
                    lines.append(f"✅ Actively running — last file written {int(delta_m)}m ago.\n")
                else:
                    lines.append(f"🟡 No activity for {int(delta_m)}m — may be between refresh rounds or slow epoch.\n")

        elif r.status == STATUS_CRASHED:
            if r.completed_steps:
                pct = int(100 * r.completed_steps / r.max_steps) if r.max_steps else 0
                lines.append(f"**Progress before crash:** {r.completed_steps}/{r.max_steps} steps ({pct}%), {r.refresh_rounds_done} refresh rounds completed\n")

            trainer = _load_trainer_state(r.run_dir)
            if trainer:
                history  = trainer.get("log_history", [])
                last_log = history[-1] if history else None
                lines.append(f"**Last trainer global step:** {trainer.get('global_step', '?')}")
                if last_log:
                    loss   = last_log.get("train_loss") or last_log.get("loss")
                    reward = last_log.get("reward")
                    parts  = []
                    if loss   is not None: parts.append(f"loss={loss:.4f}")
                    if reward is not None: parts.append(f"reward={reward:.4f}")
                    lines.append(f"  ({', '.join(parts)})\n")
                else:
                    lines.append("\n")

            latest_ckpt = _latest_checkpoint_step(r.run_dir)
            if latest_ckpt:
                lines.append(f"**Latest saved checkpoint:** step {latest_ckpt}\n")

            if newest:
                delta_h = (now - newest).total_seconds() / 3600
                lines.append(f"⚠️ **No file activity for {delta_h:.1f}h.** Run has crashed or been pre-empted.\n")
            lines.append("**Action required:** check job logs and re-launch (will resume from last checkpoint).\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Doc-opt run status and results summary")
    parser.add_argument("--artifacts", default="artifacts",       help="Artifacts root directory")
    parser.add_argument("--metric",    default="ndcg@5",          help="Metric for results table (default: ndcg@5)")
    parser.add_argument("--output",    default=str(DEFAULT_OUTPUT), help="Output markdown file (default: run_status.md)")
    parser.add_argument("--checkpoint-step", type=int, default=None,
                        help="Use this refresh-round checkpoint's metrics for the "
                             "'policy optimization' results instead of each run's final result")
    args = parser.parse_args()

    global ARTIFACTS_ROOT
    ARTIFACTS_ROOT = Path(args.artifacts)

    runs = scan_runs()
    if not runs:
        print("No runs found.")
        return

    md = build_markdown(runs, metric=args.metric, checkpoint_step=args.checkpoint_step)

    out_path = Path(args.output)
    out_path.write_text(md)
    print(md)
    print(f"\n→ Saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
