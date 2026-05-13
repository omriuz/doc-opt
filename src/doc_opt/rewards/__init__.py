from __future__ import annotations

import numpy as np

from ..config import ExperimentConfig
from .common import build_reward_state, normalize_qrels
from .dense import build_dense_reward
from .hybrid import build_hybrid_reward
from .ranking import build_ranking_reward

REWARD_NEGATIVE_TOP_K = 5
REWARD_RANKING_K = 5


def build_reward_function(
    config: ExperimentConfig,
    *,
    train_indices: np.ndarray,
    query_embeddings: np.ndarray,
    baseline_doc_embeddings: np.ndarray,
    query2doc: dict[int, int | dict[int, float] | dict[str, float]],
):
    reward_state = build_reward_state(
        train_indices=train_indices,
        query_embeddings=query_embeddings,
        baseline_doc_embeddings=baseline_doc_embeddings,
        query2doc=query2doc,
        negative_top_k=REWARD_NEGATIVE_TOP_K,
    )
    if config.reward_function == "dense":
        return build_dense_reward(config=config, reward_state=reward_state)
    if config.reward_function == "hybrid":
        return build_hybrid_reward(
            config=config,
            reward_state=reward_state,
            k=REWARD_RANKING_K,
        )
    if config.reward_function == "ranking":
        return build_ranking_reward(
            config=config,
            reward_state=reward_state,
            k=REWARD_RANKING_K,
        )
    raise ValueError(f"Unsupported reward function: {config.reward_function}")


__all__ = ["build_reward_function", "normalize_qrels"]
