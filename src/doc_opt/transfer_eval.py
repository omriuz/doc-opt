"""
Evaluate a doc-opt trained checkpoint on a (possibly different) dataset and retriever.

Usage:
    python -m doc_opt.transfer_eval \
        --config configs/humaneval.yaml \
        --checkpoint artifacts/ds1000/text-embedding-3-small/checkpoints/optimized_policy
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

METRIC_KS = (1, 5, 10)


def _eval_single_model(
    *,
    model: str,
    config,
    checkpoint_path: Path,
    gpu_memory_utilization: float | None = None,
) -> dict[str, object]:
    from .config import artifact_layout
    from .data import load_ds1000_dataset
    from .doc_transform import configure_vllm_environment, doc_tranform_documents
    from .embeddings import embed_texts, load_or_compute_embeddings
    from .evaluation import evaluate_from_embeddings
    from .splits import split_query_indices

    model_config = replace(
        config,
        embedding_models=[model],
        output_dir=config.output_dir / model.replace("/", "_"),
    )
    transfer_dir = model_config.output_dir / "transfer" / checkpoint_path.name
    transfer_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_ds1000_dataset(model_config.dataset_root)
    layout = artifact_layout(model_config.output_dir)

    docs_embeddings, query_embeddings = load_or_compute_embeddings(
        docs=bundle.docs,
        queries=bundle.queries,
        cache_dir=layout.cache_dir,
        model=model_config.embedding_model,
        device=model_config.embedding_device,
    )
    query_split = split_query_indices(
        bundle.queries, seed=model_config.seed, test_size=model_config.test_size
    )
    test_query_embeddings = np.vstack(
        [query_embeddings[i] for i in query_split.test_indices]
    )
    test_query2doc = {
        new_i: bundle.query2doc[old_i]
        for new_i, old_i in enumerate(query_split.test_indices)
    }

    baseline_metrics = evaluate_from_embeddings(
        docs_embeddings, test_query_embeddings, test_query2doc, bundle.doc_ids, ks=METRIC_KS
    )
    print(f"[{model}] Baseline: {baseline_metrics}", flush=True)

    transformed_docs_path = transfer_dir / "transformed_docs.json"
    if not transformed_docs_path.exists():
        from transformers import AutoTokenizer
        from vllm import LLM

        configure_vllm_environment(model_config)
        # Load tokenizer from the original Hub model name to avoid HF path-validation
        # errors when pointing from_pretrained at a local checkpoint directory.
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.policy_model, trust_remote_code=True
        )
        vllm_model = LLM(
            model=checkpoint_path.as_posix(),
            dtype=model_config.vllm.dtype,
            gpu_memory_utilization=gpu_memory_utilization or float(model_config.vllm.gpu_memory_utilization),
            max_model_len=model_config.vllm.max_model_len,
            enforce_eager=model_config.vllm.enforce_eager,
        )
        transformed_docs = doc_tranform_documents(
            bundle.docs, tokenizer=tokenizer, model=vllm_model, config=model_config
        )
        del vllm_model
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        transformed_docs_path.write_text(json.dumps(transformed_docs), encoding="utf-8")
    else:
        print(f"[{model}] Reusing existing transformed docs at {transformed_docs_path}", flush=True)
        transformed_docs = json.loads(transformed_docs_path.read_text(encoding="utf-8"))

    transformed_embeddings = embed_texts(
        transformed_docs,
        model=model_config.embedding_model,
        device=model_config.embedding_device,
    )
    transfer_metrics = evaluate_from_embeddings(
        transformed_embeddings, test_query_embeddings, test_query2doc, bundle.doc_ids, ks=METRIC_KS
    )
    print(f"[{model}] Transfer ({checkpoint_path.name}): {transfer_metrics}", flush=True)

    report = {
        "checkpoint_path": checkpoint_path.as_posix(),
        "target_dataset": model_config.dataset_root.as_posix(),
        "embedding_model": model_config.embedding_model,
        "metric_ks": list(METRIC_KS),
        "stages": {
            "baseline": {"metrics": baseline_metrics},
            "transfer": {
                "metrics": transfer_metrics,
                "docs_path": transformed_docs_path.as_posix(),
            },
        },
    }
    report_path = transfer_dir / "transfer_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{model}] Report saved to {report_path}", flush=True)
    return report


def run_transfer_eval(
    config_path: Path,
    checkpoint_path: Path,
    gpu_memory_utilization: float | None = None,
) -> list[dict[str, object]]:
    from dotenv import load_dotenv

    from .config import load_config

    load_dotenv()
    config = load_config(config_path)
    checkpoint_path = checkpoint_path.resolve()
    return [
        _eval_single_model(
            model=model,
            config=config,
            checkpoint_path=checkpoint_path,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        for model in config.embedding_models
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a doc-opt checkpoint on a target dataset/retriever."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                        help="Override vllm.gpu_memory_utilization from config.")
    args = parser.parse_args()
    run_transfer_eval(args.config, args.checkpoint, args.gpu_memory_utilization)


if __name__ == "__main__":
    main()
