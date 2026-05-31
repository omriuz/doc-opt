from __future__ import annotations

import numpy as np

from ..config import ExperimentConfig
from ..doc_transform import extract_rewritten_document
from ..embeddings import embed_texts
from .common import RewardState
from .dense import dense_reward_components_for_doc
from .ranking import ranking_reward_components_for_doc


def build_hybrid_reward(*, config: ExperimentConfig, reward_state: RewardState, k: int):
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
        ranking_rewards: list[float] = []
        dense_rewards: list[float] = []
        for doc_key, candidate_embedding in zip(doc_keys, candidate_embeddings):
            doc_index = int(doc_key)
            _, _, ranking_reward = ranking_reward_components_for_doc(
                reward_state=reward_state,
                doc_index=doc_index,
                candidate_embedding=candidate_embedding,
                train_query_baseline_ndcg=train_query_baseline_ndcg,
                k=k,
            )
            _, _, dense_reward = dense_reward_components_for_doc(
                reward_state=reward_state,
                doc_index=doc_index,
                candidate_embedding=candidate_embedding,
            )
            ranking_rewards.append(float(ranking_reward))
            dense_rewards.append(float(dense_reward))
            rewards.append(float(0.5 * (ranking_reward + dense_reward)))

        reward_logging_state["step"] += 1
        print(
            "Reward step "
            f"{reward_logging_state['step']} "
            f"(hybrid): "
            f"mean_ranking={np.mean(ranking_rewards):.6f}, "
            f"mean_dense={np.mean(dense_rewards):.6f}, "
            f"mean_total={np.mean(rewards):.6f}",
            flush=True,
        )
        return rewards

    return reward


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
