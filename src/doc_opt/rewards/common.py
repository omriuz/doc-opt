from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RewardState:
    train_query_embeddings: np.ndarray
    train_query_qrels: list[dict[int, float]]
    train_query_baseline_scores: np.ndarray
    train_positive_query_indices_by_doc: dict[int, np.ndarray]
    train_negative_query_indices_by_doc: dict[int, np.ndarray]


def normalize_qrels(raw_target) -> dict[int, float]:
    if isinstance(raw_target, dict):
        return {int(doc_id): float(relevance) for doc_id, relevance in raw_target.items()}
    if isinstance(raw_target, (list, tuple, set, np.ndarray)):
        return {int(doc_id): 1.0 for doc_id in raw_target}
    return {int(raw_target): 1.0}


def build_reward_state(
    *,
    train_indices: np.ndarray,
    query_embeddings: np.ndarray,
    baseline_doc_embeddings: np.ndarray,
    query2doc: dict[int, int | dict[int, float] | dict[str, float]],
    negative_top_k: int,
) -> RewardState:
    train_query_embeddings = np.vstack([query_embeddings[index] for index in train_indices])
    train_query_qrels = [normalize_qrels(query2doc[index]) for index in train_indices]
    train_query_baseline_scores = train_query_embeddings @ baseline_doc_embeddings.T
    reward_doc_indices = sorted({doc_index for qrels in train_query_qrels for doc_index in qrels})
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
        doc_index: mine_negative_query_indices_for_doc(
            doc_index=doc_index,
            train_query_qrels=train_query_qrels,
            train_query_baseline_scores=train_query_baseline_scores,
            top_k=negative_top_k,
        )
        for doc_index in reward_doc_indices
    }
    return RewardState(
        train_query_embeddings=train_query_embeddings,
        train_query_qrels=train_query_qrels,
        train_query_baseline_scores=train_query_baseline_scores,
        train_positive_query_indices_by_doc=train_positive_query_indices_by_doc,
        train_negative_query_indices_by_doc=train_negative_query_indices_by_doc,
    )


def mine_negative_query_indices_for_doc(
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


def mean_counterfactual_score_delta_for_query_indices(
    *,
    query_indices: np.ndarray,
    doc_index: int,
    candidate_embedding: np.ndarray,
    train_query_baseline_scores: np.ndarray,
    train_query_embeddings: np.ndarray,
) -> float:
    local_query_indices = np.asarray(query_indices, dtype=np.int64)
    if local_query_indices.size == 0:
        return 0.0
    baseline_scores = train_query_baseline_scores[local_query_indices, doc_index]
    candidate_scores = train_query_embeddings[local_query_indices] @ candidate_embedding
    return float(np.mean(candidate_scores - baseline_scores))
