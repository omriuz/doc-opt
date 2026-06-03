from __future__ import annotations

import numpy as np

_MULTIVEC_KEYWORDS = ("colbert",)


def is_multivec_model(model: str) -> bool:
    """Return True if the model produces multi-vector (token-level) embeddings."""
    lower = model.lower()
    return any(kw in lower for kw in _MULTIVEC_KEYWORDS)


def maxsim(query_embs: np.ndarray, doc_embs: np.ndarray) -> float:
    """Late-interaction MaxSim score: sum_i max_j dot(q_i, d_j)."""
    return float(np.sum(np.max(query_embs @ doc_embs.T, axis=1)))


def compute_scores(
    query_embs_list: list[np.ndarray],
    doc_embs_list: list[np.ndarray],
) -> np.ndarray:
    """All-pairs MaxSim score matrix of shape [n_queries, n_docs]."""
    scores = np.zeros((len(query_embs_list), len(doc_embs_list)), dtype=np.float32)
    for qi, qe in enumerate(query_embs_list):
        for di, de in enumerate(doc_embs_list):
            scores[qi, di] = maxsim(qe, de)
    return scores
