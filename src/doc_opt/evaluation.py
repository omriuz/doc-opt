from __future__ import annotations

import numpy as np
import pytrec_eval
import torch


def rank_topk(query_embeddings: np.ndarray, doc_embeddings: np.ndarray, k: int) -> np.ndarray:
    k = min(k, int(doc_embeddings.shape[0]))
    scores = torch.from_numpy(query_embeddings) @ torch.from_numpy(doc_embeddings).T
    return torch.topk(scores, k=k, dim=1).indices.cpu().numpy()


def evaluate_from_embeddings(
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    gold_qrels: dict[int, str | dict[int, float] | dict[str, float]],
    doc_ids: list[str],
    *,
    k: int | None = None,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    metric_ks = _normalize_metric_ks(k=k, ks=ks)
    max_k = max(metric_ks)
    ranked = rank_topk(query_embeddings, doc_embeddings, max_k)
    qrel_dict = {str(query_index): _normalize_qrel(target) for query_index, target in gold_qrels.items()}
    run_dict = {
        str(query_index): {
            doc_ids[doc_index]: float(max_k - rank)
            for rank, doc_index in enumerate(doc_indices)
        }
        for query_index, doc_indices in enumerate(ranked)
    }
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel_dict,
        {f"ndcg_cut_{metric_k}" for metric_k in metric_ks},
    )
    scores = evaluator.evaluate(run_dict)
    metrics: dict[str, float] = {}
    for metric_k in metric_ks:
        ndcgs = [score[f"ndcg_cut_{metric_k}"] for score in scores.values()]
        metrics[f"ndcg@{metric_k}"] = float(np.mean(ndcgs))
        metrics[f"recall@{metric_k}"] = _mean_recall_at_k(
            ranked=ranked,
            qrel_dict=qrel_dict,
            doc_ids=doc_ids,
            k=metric_k,
        )
    return metrics


def _normalize_qrel(target: str | dict[int, float] | dict[str, float]) -> dict[str, float]:
    if isinstance(target, dict):
        return {str(doc_id): int(relevance) for doc_id, relevance in target.items()}
    return {str(target): 1}


def _normalize_metric_ks(*, k: int | None, ks: tuple[int, ...]) -> tuple[int, ...]:
    values = (k,) if k is not None else ks
    normalized = tuple(dict.fromkeys(int(value) for value in values if int(value) > 0))
    if not normalized:
        raise ValueError("At least one positive k must be provided.")
    return normalized


def _mean_recall_at_k(
    *,
    ranked: np.ndarray,
    qrel_dict: dict[str, dict[str, float]],
    doc_ids: list[str],
    k: int,
) -> float:
    recalls: list[float] = []
    for query_index, doc_indices in enumerate(ranked):
        retrieved_ids = {doc_ids[doc_index] for doc_index in doc_indices[:k]}
        relevant_ids = {
            doc_id
            for doc_id, relevance in qrel_dict[str(query_index)].items()
            if float(relevance) > 0.0
        }
        if not relevant_ids:
            recalls.append(0.0)
            continue
        recalls.append(len(retrieved_ids & relevant_ids) / len(relevant_ids))
    return float(np.mean(recalls))
