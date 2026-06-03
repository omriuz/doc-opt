from __future__ import annotations

import numpy as np

from ..config import ExperimentConfig
from ..doc_transform import extract_rewritten_document
from ..embeddings import embed_texts
from ..multivec import maxsim as _maxsim
from .common import RewardState


def build_ranking_reward(*, config: ExperimentConfig, reward_state: RewardState, k: int):
    train_query_baseline_ndcg = np.asarray(
        [
            _ndcg_at_k_from_scores(scores, qrels, k=k)
            for scores, qrels in zip(
                reward_state.train_query_baseline_scores,
                reward_state.train_query_qrels,
            )
        ],
        dtype=np.float32,
    )
    reward_logging_state = {"step": 0}

    def reward(completions, document, **kwargs):
        candidate_embeddings = embed_texts(
            [extract_rewritten_document(str(completion)) for completion in completions],
            model=config.embedding_model,
            device=config.embedding_device,
            verbose=False,
        )
        doc_keys = document if isinstance(document, list) else [document]
        rewards: list[float] = []
        positive_deltas: list[float] = []
        negative_deltas: list[float] = []
        for doc_key, candidate_embedding in zip(doc_keys, candidate_embeddings):
            doc_index = int(doc_key)
            positive_delta, negative_delta, reward_value = ranking_reward_components_for_doc(
                reward_state=reward_state,
                doc_index=doc_index,
                candidate_embedding=candidate_embedding,
                train_query_baseline_ndcg=train_query_baseline_ndcg,
                k=k,
            )
            positive_deltas.append(float(positive_delta))
            negative_deltas.append(float(negative_delta))
            rewards.append(float(reward_value))

        reward_logging_state["step"] += 1
        print(
            "Reward step "
            f"{reward_logging_state['step']} "
            f"(ranking): "
            f"mean_pos={np.mean(positive_deltas):.6f}, "
            f"mean_neg={np.mean(negative_deltas):.6f}, "
            f"mean_total={np.mean(rewards):.6f}",
            flush=True,
        )
        return rewards

    return reward


def ranking_reward_components_for_doc(
    *,
    reward_state: RewardState,
    doc_index: int,
    candidate_embedding: np.ndarray,
    train_query_baseline_ndcg: np.ndarray,
    k: int,
) -> tuple[float, float, float]:
    positive_delta = _mean_counterfactual_ndcg_delta_for_query_indices(
        query_indices=reward_state.train_positive_query_indices_by_doc.get(
            doc_index,
            np.empty(0, dtype=np.int64),
        ),
        doc_index=doc_index,
        candidate_embedding=candidate_embedding,
        train_query_baseline_scores=reward_state.train_query_baseline_scores,
        train_query_baseline_ndcg=train_query_baseline_ndcg,
        train_query_embeddings=reward_state.train_query_embeddings,
        train_query_qrels=reward_state.train_query_qrels,
        k=k,
    )
    negative_delta = _mean_counterfactual_ndcg_delta_for_query_indices(
        query_indices=reward_state.train_negative_query_indices_by_doc.get(
            doc_index,
            np.empty(0, dtype=np.int64),
        ),
        doc_index=doc_index,
        candidate_embedding=candidate_embedding,
        train_query_baseline_scores=reward_state.train_query_baseline_scores,
        train_query_baseline_ndcg=train_query_baseline_ndcg,
        train_query_embeddings=reward_state.train_query_embeddings,
        train_query_qrels=reward_state.train_query_qrels,
        k=k,
    )
    reward_value = float(positive_delta + negative_delta)
    return float(positive_delta), float(negative_delta), reward_value


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


def _mean_counterfactual_ndcg_delta_for_query_indices(
    *,
    query_indices: np.ndarray,
    doc_index: int,
    candidate_embedding: np.ndarray,
    train_query_baseline_scores: np.ndarray,
    train_query_baseline_ndcg: np.ndarray,
    train_query_embeddings: np.ndarray | list[np.ndarray],
    train_query_qrels: list[dict[int, float]],
    k: int,
) -> float:
    local_query_indices = np.asarray(query_indices, dtype=np.int64)
    if local_query_indices.size == 0:
        return 0.0
    counterfactual_scores = train_query_baseline_scores[local_query_indices].copy()
    if isinstance(train_query_embeddings, list):
        counterfactual_scores[:, doc_index] = np.array(
            [_maxsim(train_query_embeddings[int(i)], candidate_embedding) for i in local_query_indices],
            dtype=np.float32,
        )
    else:
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
