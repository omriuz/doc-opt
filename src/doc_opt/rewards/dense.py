from __future__ import annotations

import numpy as np

from ..config import ExperimentConfig
from ..doc_transform import extract_rewritten_document
from ..embeddings import embed_texts
from .common import RewardState, mean_counterfactual_score_delta_for_query_indices


def build_dense_reward(*, config: ExperimentConfig, reward_state: RewardState):
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
            positive_delta, negative_delta, reward_value = dense_reward_components_for_doc(
                reward_state=reward_state,
                doc_index=doc_index,
                candidate_embedding=candidate_embedding,
            )
            positive_deltas.append(float(positive_delta))
            negative_deltas.append(float(negative_delta))
            rewards.append(float(reward_value))

        reward_logging_state["step"] += 1
        print(
            "Reward step "
            f"{reward_logging_state['step']} "
            f"(dense): "
            f"mean_pos={np.mean(positive_deltas):.6f}, "
            f"mean_neg={np.mean(negative_deltas):.6f}, "
            f"mean_total={np.mean(rewards):.6f}",
            flush=True,
        )
        return rewards

    return reward


def dense_reward_components_for_doc(
    *,
    reward_state: RewardState,
    doc_index: int,
    candidate_embedding: np.ndarray,
) -> tuple[float, float, float]:
    positive_delta = mean_counterfactual_score_delta_for_query_indices(
        query_indices=reward_state.train_positive_query_indices_by_doc.get(
            doc_index,
            np.empty(0, dtype=np.int64),
        ),
        doc_index=doc_index,
        candidate_embedding=candidate_embedding,
        train_query_baseline_scores=reward_state.train_query_baseline_scores,
        train_query_embeddings=reward_state.train_query_embeddings,
    )
    negative_delta = mean_counterfactual_score_delta_for_query_indices(
        query_indices=reward_state.train_negative_query_indices_by_doc.get(
            doc_index,
            np.empty(0, dtype=np.int64),
        ),
        doc_index=doc_index,
        candidate_embedding=candidate_embedding,
        train_query_baseline_scores=reward_state.train_query_baseline_scores,
        train_query_embeddings=reward_state.train_query_embeddings,
    )
    reward_value = float(positive_delta - negative_delta)
    return float(positive_delta), float(negative_delta), reward_value
