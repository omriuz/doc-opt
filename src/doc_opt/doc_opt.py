from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .data import load_ds1000_dataset
from .doc_transform import build_doc_tranform_prompt
from .embeddings import embed_texts_openai


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
    train_indices = _split_indices(len(queries), seed=config.seed, test_size=config.test_size)[0]
    train_queries_by_doc: dict[int, list[str]] = {}
    for query_index in train_indices:
        for doc_index in _normalize_qrels(query2doc[query_index]):
            train_queries_by_doc.setdefault(int(doc_index), []).append(queries[query_index])

    train_query_embeddings = np.vstack([query_id_to_embedding[index] for index in train_indices])
    baseline_doc_embeddings = np.load(baseline_embs_path)
    reward_doc_indices = sorted(train_queries_by_doc)
    reward_logging_state = {"step": 0}

    train_query_qrels = [_normalize_qrels(query2doc[index]) for index in train_indices]
    train_query_baseline_scores = train_query_embeddings @ baseline_doc_embeddings.T
    train_query_baseline_ndcg = np.asarray(
        [
            _ndcg_at_k_from_scores(scores, qrels, k=5)
            for scores, qrels in zip(train_query_baseline_scores, train_query_qrels)
        ],
        dtype=np.float32,
    )
    train_positive_query_indices_by_doc = {
        doc_index: np.asarray(
            [
                query_index
                for query_index, qrels in enumerate(train_query_qrels)
                if doc_index in qrels
            ],
            dtype=np.int64,
        )
        for doc_index in reward_doc_indices
    }
    train_negative_query_indices_by_doc = {
        doc_index: _mine_negative_query_indices_for_doc(
            doc_index=doc_index,
            train_query_qrels=train_query_qrels,
            train_query_baseline_scores=train_query_baseline_scores,
            top_k=5,
        )
        for doc_index in reward_doc_indices
    }

    def reward(completions, document, **kwargs):
        candidate_embeddings = embed_texts_openai(
            [str(completion) for completion in completions],
            model=config.embedding_model,
            verbose=False,
        )
        doc_keys = document if isinstance(document, list) else [document]
        rewards: list[float] = []
        positive_deltas: list[float] = []
        negative_deltas: list[float] = []
        for doc_key, candidate_embedding in zip(doc_keys, candidate_embeddings):
            doc_index = int(doc_key)
            positive_delta = _mean_counterfactual_ndcg_delta_for_query_indices(
                query_indices=train_positive_query_indices_by_doc.get(
                    doc_index,
                    np.empty(0, dtype=np.int64),
                ),
                doc_index=doc_index,
                candidate_embedding=candidate_embedding,
                train_query_baseline_scores=train_query_baseline_scores,
                train_query_baseline_ndcg=train_query_baseline_ndcg,
                train_query_embeddings=train_query_embeddings,
                train_query_qrels=train_query_qrels,
                k=5,
            )
            negative_delta = _mean_counterfactual_ndcg_delta_for_query_indices(
                query_indices=train_negative_query_indices_by_doc.get(
                    doc_index,
                    np.empty(0, dtype=np.int64),
                ),
                doc_index=doc_index,
                candidate_embedding=candidate_embedding,
                train_query_baseline_scores=train_query_baseline_scores,
                train_query_baseline_ndcg=train_query_baseline_ndcg,
                train_query_embeddings=train_query_embeddings,
                train_query_qrels=train_query_qrels,
                k=5,
            )
            positive_deltas.append(float(positive_delta))
            negative_deltas.append(float(negative_delta))
            rewards.append(float(positive_delta - negative_delta))

        reward_logging_state["step"] += 1
        print(
            "Reward step "
            f"{reward_logging_state['step']}: "
            f"mean_pos={np.mean(positive_deltas):.6f}, "
            f"mean_neg={np.mean(negative_deltas):.6f}, "
            f"mean_total={np.mean(rewards):.6f}",
            flush=True,
        )
        return rewards

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


def _split_indices(size: int, *, seed: int, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(size)
    np.random.default_rng(seed).shuffle(indices)
    split_index = int(len(indices) * (1 - test_size))
    return indices[:split_index], indices[split_index:]


def _normalize_qrels(raw_target):
    if isinstance(raw_target, dict):
        return {int(doc_id): float(relevance) for doc_id, relevance in raw_target.items()}
    if isinstance(raw_target, (list, tuple, set, np.ndarray)):
        return {int(doc_id): 1.0 for doc_id in raw_target}
    return {int(raw_target): 1.0}


def _ndcg_at_k_from_scores(scores: np.ndarray, qrels: dict[int, float], *, k: int) -> float:
    if k <= 0 or scores.size == 0 or not qrels:
        return 0.0
    top_k = min(k, int(scores.shape[0]))
    if top_k <= 0:
        return 0.0
    top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    gains = np.asarray([qrels.get(int(doc_index), 0.0) for doc_index in top_indices], dtype=np.float32)
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2, dtype=np.float32))
    dcg = float(np.sum(gains * discounts))
    ideal_gains = np.asarray(
        sorted((float(relevance) for relevance in qrels.values()), reverse=True)[:top_k],
        dtype=np.float32,
    )
    if ideal_gains.size == 0:
        return 0.0
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_gains.size + 2, dtype=np.float32))
    idcg = float(np.sum(ideal_gains * ideal_discounts))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _mine_negative_query_indices_for_doc(
    *,
    doc_index: int,
    train_query_qrels: list[dict[int, float]],
    train_query_baseline_scores: np.ndarray,
    top_k: int,
) -> np.ndarray:
    candidate_query_indices = np.asarray(
        [query_index for query_index, qrels in enumerate(train_query_qrels) if doc_index not in qrels],
        dtype=np.int64,
    )
    if candidate_query_indices.size == 0:
        return candidate_query_indices
    candidate_scores = train_query_baseline_scores[candidate_query_indices, doc_index]
    top_k = min(top_k, int(candidate_query_indices.size))
    if top_k <= 0:
        return np.empty(0, dtype=np.int64)
    top_positions = np.argpartition(-candidate_scores, top_k - 1)[:top_k]
    top_positions = top_positions[np.argsort(candidate_scores[top_positions])[::-1]]
    return candidate_query_indices[top_positions]


def _mean_counterfactual_ndcg_delta_for_query_indices(
    *,
    query_indices: np.ndarray,
    doc_index: int,
    candidate_embedding: np.ndarray,
    train_query_baseline_scores: np.ndarray,
    train_query_baseline_ndcg: np.ndarray,
    train_query_embeddings: np.ndarray,
    train_query_qrels: list[dict[int, float]],
    k: int,
) -> float:
    local_query_indices = np.asarray(query_indices, dtype=np.int64)
    if local_query_indices.size == 0:
        return 0.0
    counterfactual_scores = train_query_baseline_scores[local_query_indices].copy()
    counterfactual_scores[:, doc_index] = train_query_embeddings[local_query_indices] @ candidate_embedding
    baseline_ndcgs = train_query_baseline_ndcg[local_query_indices]
    counterfactual_ndcgs = np.asarray(
        [
            _ndcg_at_k_from_scores(counterfactual_scores[row_index], train_query_qrels[query_index], k=k)
            for row_index, query_index in enumerate(local_query_indices)
        ],
        dtype=np.float32,
    )
    return float(np.mean(counterfactual_ndcgs - baseline_ndcgs))


def _unwrap_model_for_saving(model):
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped
