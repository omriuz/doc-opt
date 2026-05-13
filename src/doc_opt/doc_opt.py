from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .data import load_ds1000_dataset
from .doc_transform import build_doc_tranform_prompt
from .embeddings import embed_texts_openai
from .rewards import build_reward_function, normalize_qrels
from .splits import split_query_indices


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_doc_opt(
    config: ExperimentConfig,
    *,
    model_source: str | None,
    checkpoint_dir: Path,
    grpo_output_dir: Path,
    baseline_embs_path: Path,
) -> None:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    seed_everything(config.seed)
    bundle = load_ds1000_dataset(config.dataset_root)
    queries = bundle.queries
    docs = bundle.docs
    query2doc = bundle.query2doc

    tokenizer = AutoTokenizer.from_pretrained(model_source or config.policy_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_source or config.policy_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")

    query_id_to_embedding = load_query_embeddings(config, queries)
    train_indices = split_query_indices(
        queries,
        seed=config.seed,
        test_size=config.test_size,
    ).train_indices
    train_queries_by_doc: dict[int, list[str]] = {}
    for query_index in train_indices:
        for doc_index in normalize_qrels(query2doc[query_index]):
            train_queries_by_doc.setdefault(int(doc_index), []).append(queries[query_index])

    baseline_doc_embeddings = np.load(baseline_embs_path)
    reward = build_reward_function(
        config,
        train_indices=train_indices,
        query_embeddings=query_id_to_embedding,
        baseline_doc_embeddings=baseline_doc_embeddings,
        query2doc=query2doc,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=GRPOConfig(
            output_dir=grpo_output_dir.as_posix(),
            max_steps=config.doc_opt.max_steps,
            per_device_train_batch_size=config.doc_opt.per_device_train_batch_size,
            num_generations=config.doc_opt.num_generations,
            learning_rate=config.doc_opt.learning_rate,
            beta=config.doc_opt.kl_beta,
            temperature=config.doc_opt.temperature,
            top_p=config.doc_tranform.top_p,
            top_k=config.doc_tranform.top_k,
            bf16=True,
            remove_unused_columns=False,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            max_completion_length=config.doc_opt.max_completion_length,
        ),
        train_dataset=Dataset.from_list(
            build_doc_opt_rows(
                docs=docs,
                tokenizer=tokenizer,
                train_queries_by_doc=train_queries_by_doc,
                transform_instruction=config.doc_tranform.transform_instruction,
            )
        ),
    )
    trainer.train()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trained_model = _unwrap_model_for_saving(trainer.model)
    trained_model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)


def build_doc_opt_rows(
    *,
    docs: list[str],
    tokenizer,
    train_queries_by_doc: dict[int, list[str]],
    transform_instruction: str,
) -> list[dict[str, str]]:
    return [
        {
            "prompt": tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": build_doc_tranform_prompt(
                            docs[doc_index],
                            transform_instruction,
                        ),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            ),
            "document": str(doc_index),
        }
        for doc_index in sorted(train_queries_by_doc)
    ]


def load_query_embeddings(config: ExperimentConfig, queries: list[str]) -> np.ndarray:
    from .config import artifact_layout

    cache_dir = artifact_layout(config.output_dir).cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    queries_cache = cache_dir / "queries_embs.npy"
    if queries_cache.exists():
        return np.load(queries_cache)
    query_embeddings = embed_texts_openai(queries, model=config.embedding_model)
    np.save(queries_cache, query_embeddings)
    return query_embeddings


def _unwrap_model_for_saving(model):
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped
