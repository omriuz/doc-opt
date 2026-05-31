from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .config import ExperimentConfig, artifact_layout
from .data import DatasetBundle, load_ds1000_dataset
from .doc_opt import seed_everything
from .embeddings import (
    embed_texts,
    load_or_compute_doc_embeddings,
    load_or_compute_embeddings,
    load_or_compute_query_embeddings,
)
from .evaluation import evaluate_from_embeddings
from .splits import split_query_indices


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    train_bundle: DatasetBundle
    eval_bundle: DatasetBundle
    docs_embeddings: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    test_query_embeddings: np.ndarray
    test_query2doc: dict[int, int | dict[int, float]]


METRIC_KS = (1, 5, 10)


def _model_slug(model: str) -> str:
    return model.replace("/", "_")


def run_pipeline(config: ExperimentConfig, *, config_path: Path) -> list[dict[str, object]]:
    results = []
    for model in config.embedding_models:
        model_config = replace(
            config,
            embedding_models=[model],
            output_dir=config.output_dir / _model_slug(model),
        )
        results.append(_run_pipeline_single(model_config, config_path=config_path))
    return results


def _run_pipeline_single(config: ExperimentConfig, *, config_path: Path) -> dict[str, object]:
    layout = artifact_layout(config.output_dir)
    context = load_runtime_context(config)
    run_report = _initialize_run_report(config=config, config_path=config_path, context=context, metric_ks=METRIC_KS)
    _write_run_report(layout, run_report)
    _print_dataset_summary(context)

    baseline_metrics = evaluate_from_embeddings(
        context.docs_embeddings,
        context.test_query_embeddings,
        context.test_query2doc,
        context.train_bundle.doc_ids,
        ks=METRIC_KS,
    )
    print(f"Direct retrieval metrics: {baseline_metrics}", flush=True)
    run_report["stages"]["direct retrieval"] = {"metrics": baseline_metrics}
    _write_run_report(layout, run_report)

    transformed_docs_path = layout.doc_tranform_dir / "transformed_docs.json"
    if not transformed_docs_path.exists():
        _run_cli_subprocess(
            "doc-tranform",
            config_path=config_path,
            config=config,
            extra_args=["--output-path", transformed_docs_path.as_posix()],
        )

    transformed_docs = json.loads(transformed_docs_path.read_text(encoding="utf-8"))
    transformed_docs_embeddings = embed_texts(transformed_docs, model=config.embedding_model, device=config.embedding_device)
    transformed_metrics = evaluate_from_embeddings(
        transformed_docs_embeddings,
        context.test_query_embeddings,
        context.test_query2doc,
        context.train_bundle.doc_ids,
        ks=METRIC_KS,
    )
    print(f"Direct transformation metrics: {transformed_metrics}", flush=True)
    run_report["stages"]["direct transformation"] = {
        "metrics": transformed_metrics,
        "docs_path": transformed_docs_path.as_posix(),
    }
    _write_run_report(layout, run_report)

    current_docs = list(transformed_docs)
    current_doc_embeddings = np.asarray(transformed_docs_embeddings, dtype=np.float32).copy()
    completed_steps = 0
    refresh_round = 0
    latest_checkpoint_dir = layout.checkpoints_dir / "optimized_policy"
    model_source = config.policy_model

    while completed_steps < config.doc_opt.max_steps:
        refresh_round += 1
        steps_this_round = min(config.doc_opt.refresh_rate, config.doc_opt.max_steps - completed_steps)
        steps_after = completed_steps + steps_this_round
        is_last = steps_after >= config.doc_opt.max_steps
        if is_last:
            checkpoint_dir = latest_checkpoint_dir
        elif steps_after % config.doc_opt.checkpoint_save_rate == 0:
            checkpoint_dir = layout.checkpoints_dir / f"checkpoint_step_{steps_after}"
        else:
            checkpoint_dir = layout.checkpoints_dir / "current"
        round_output_dir = layout.doc_opt_dir / f"refresh_{refresh_round}"
        current_embeddings_path = layout.runtime_dir / "current_doc_embs.npy"
        current_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(current_embeddings_path, current_doc_embeddings)
        _run_cli_subprocess(
            "doc-opt",
            config_path=config_path,
            config=config,
            extra_args=[
                "--checkpoint-dir",
                checkpoint_dir.as_posix(),
                "--grpo-output-dir",
                round_output_dir.as_posix(),
                "--baseline-embs-path",
                current_embeddings_path.as_posix(),
                "--model-source",
                model_source,
                "--max-steps",
                str(steps_this_round),
            ],
        )
        completed_steps += steps_this_round
        latest_checkpoint_dir = checkpoint_dir
        model_source = checkpoint_dir.as_posix()

        if completed_steps >= config.doc_opt.max_steps:
            break

        current_docs_path = layout.runtime_dir / "current_docs.json"
        _run_cli_subprocess(
            "doc-tranform",
            config_path=config_path,
            config=config,
            extra_args=[
                "--output-path",
                current_docs_path.as_posix(),
                "--model-source",
                model_source,
            ],
        )
        current_docs = json.loads(current_docs_path.read_text(encoding="utf-8"))
        current_doc_embeddings = embed_texts(current_docs, model=config.embedding_model, device=config.embedding_device)
        refresh_metrics = evaluate_from_embeddings(
            current_doc_embeddings,
            context.test_query_embeddings,
            context.test_query2doc,
            context.train_bundle.doc_ids,
            ks=METRIC_KS,
        )
        print(f"Refresh metrics after {completed_steps} steps: {refresh_metrics}", flush=True)
        run_report["stages"]["refresh"].append(
            {
                "refresh_round": refresh_round,
                "completed_steps": completed_steps,
                "metrics": refresh_metrics,
                "checkpoint_dir": checkpoint_dir.as_posix(),
                "docs_path": current_docs_path.as_posix(),
            }
        )
        _write_run_report(layout, run_report)

    optimized_docs_path = layout.doc_opt_dir / "optimized_docs.json"
    final_policy_source = latest_checkpoint_dir if latest_checkpoint_dir.exists() else Path(model_source)
    _run_cli_subprocess(
        "doc-tranform",
        config_path=config_path,
        config=config,
        extra_args=[
            "--output-path",
            optimized_docs_path.as_posix(),
            "--model-source",
            final_policy_source.as_posix(),
        ],
    )
    optimized_docs = json.loads(optimized_docs_path.read_text(encoding="utf-8"))
    optimized_docs_embeddings = embed_texts(optimized_docs, model=config.embedding_model, device=config.embedding_device)
    optimized_metrics = evaluate_from_embeddings(
        optimized_docs_embeddings,
        context.test_query_embeddings,
        context.test_query2doc,
        context.train_bundle.doc_ids,
        ks=METRIC_KS,
    )
    print(f"Document optimization metrics: {optimized_metrics}", flush=True)
    run_report["stages"]["document optimization"] = {
        "metrics": optimized_metrics,
        "docs_path": optimized_docs_path.as_posix(),
        "model_source": final_policy_source.as_posix(),
    }
    _write_run_report(layout, run_report)

    return run_report


def load_runtime_context(config: ExperimentConfig) -> RuntimeContext:
    seed_everything(config.seed)
    layout = artifact_layout(config.output_dir)
    train_bundle = load_ds1000_dataset(config.dataset_root)
    if config.eval_dataset_root is None:
        docs_embeddings, query_embeddings = load_or_compute_embeddings(
            docs=train_bundle.docs,
            queries=train_bundle.queries,
            cache_dir=layout.cache_dir,
            model=config.embedding_model,
            device=config.embedding_device,
        )
        query_split = split_query_indices(
            train_bundle.queries,
            seed=config.seed,
            test_size=config.test_size,
        )
        test_query_embeddings = np.vstack([query_embeddings[index] for index in query_split.test_indices])
        test_query2doc = {
            new_index: train_bundle.query2doc[old_index]
            for new_index, old_index in enumerate(query_split.test_indices)
        }
        train_indices = query_split.train_indices
        test_indices = query_split.test_indices
        eval_bundle = train_bundle
    else:
        eval_bundle = load_ds1000_dataset(config.eval_dataset_root)
        if train_bundle.doc_ids != eval_bundle.doc_ids:
            raise ValueError(
                "Train/eval dataset roots must share identical corpus document ids for evaluation."
            )
        docs_embeddings = load_or_compute_doc_embeddings(
            docs=train_bundle.docs,
            cache_dir=layout.cache_dir,
            model=config.embedding_model,
            device=config.embedding_device,
        )
        test_query_embeddings = load_or_compute_query_embeddings(
            queries=eval_bundle.queries,
            cache_dir=layout.cache_dir,
            model=config.embedding_model,
            device=config.embedding_device,
            cache_name="eval_queries_embs.npy",
        )
        train_indices = np.arange(len(train_bundle.queries), dtype=np.int64)
        test_indices = np.arange(len(eval_bundle.queries), dtype=np.int64)
        test_query2doc = eval_bundle.query2doc
    return RuntimeContext(
        train_bundle=train_bundle,
        eval_bundle=eval_bundle,
        docs_embeddings=docs_embeddings,
        train_indices=train_indices,
        test_indices=test_indices,
        test_query_embeddings=test_query_embeddings,
        test_query2doc=test_query2doc,
    )


def _run_cli_subprocess(
    subcommand: str,
    *,
    config_path: Path,
    config: ExperimentConfig,
    extra_args: list[str],
) -> None:
    env = os.environ.copy()
    src_dir = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = (
        src_dir.as_posix()
        if "PYTHONPATH" not in env or not env["PYTHONPATH"]
        else f"{src_dir.as_posix()}:{env['PYTHONPATH']}"
    )
    cmd = [
        sys.executable,
        "-m",
        "doc_opt.cli",
        subcommand,
        "--config",
        config_path.as_posix(),
        "--dataset-root",
        config.dataset_root.as_posix(),
        "--output-dir",
        config.output_dir.as_posix(),
        "--embedding-model",
        config.embedding_model,
        *extra_args,
    ]
    if config.eval_dataset_root is not None:
        cmd.extend(["--eval-dataset-root", config.eval_dataset_root.as_posix()])
    subprocess.run(cmd, check=True, env=env)


def _print_dataset_summary(context: RuntimeContext) -> None:
    print(f"Docs loaded:            {len(context.train_bundle.docs)}", flush=True)
    print(f"All queries loaded:     {len(context.train_bundle.queries)}", flush=True)
    print(f"Train queries loaded:   {len(context.train_indices)}", flush=True)
    print(f"Test queries loaded:    {len(context.test_indices)}", flush=True)


def _initialize_run_report(
    *,
    config: ExperimentConfig,
    config_path: Path,
    context: RuntimeContext,
    metric_ks: tuple[int, ...],
) -> dict[str, object]:
    return {
        "config_path": config_path.as_posix(),
        "output_dir": config.output_dir.as_posix(),
        "metric_ks": list(metric_ks),
        "dataset": {
            "dataset_root": config.dataset_root.as_posix(),
            "eval_dataset_root": (
                config.eval_dataset_root.as_posix() if config.eval_dataset_root is not None else None
            ),
            "doc_count": len(context.train_bundle.docs),
            "query_count": len(context.train_bundle.queries),
            "train_query_count": len(context.train_indices),
            "test_query_count": len(context.test_indices),
        },
        "stages": {
            "direct retrieval": None,
            "direct transformation": None,
            "refresh": [],
            "document optimization": None,
        },
    }


def _write_run_report(layout: ArtifactLayout, run_report: dict[str, object]) -> None:
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.run_report_path.write_text(json.dumps(run_report, indent=2), encoding="utf-8")
