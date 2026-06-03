from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from doc_opt.multivec import compute_scores, is_multivec_model, maxsim
from doc_opt.rewards.common import (
    RewardState,
    build_reward_state,
    mean_counterfactual_score_delta_for_query_indices,
)
from doc_opt.evaluation import evaluate_from_embeddings, rank_topk


class IsMultivecModelTest(unittest.TestCase):
    def test_colbert_variants_detected(self) -> None:
        self.assertTrue(is_multivec_model("jinaai/jina-colbert-v2"))
        self.assertTrue(is_multivec_model("colbert-base"))
        self.assertTrue(is_multivec_model("ColBERT-v2"))

    def test_single_vec_models_not_detected(self) -> None:
        self.assertFalse(is_multivec_model("text-embedding-3-small"))
        self.assertFalse(is_multivec_model("Qwen/Qwen3-Embedding-0.6B"))
        self.assertFalse(is_multivec_model("jinaai/jina-embeddings-v3"))


class MaxsimTest(unittest.TestCase):
    def test_identical_vectors_give_max_score(self) -> None:
        q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        d = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        # Each q_i matches its own d_i perfectly → sum = 2.0
        self.assertAlmostEqual(maxsim(q, d), 2.0, places=5)

    def test_each_query_token_picks_best_doc_token(self) -> None:
        # q has one token, d has two tokens
        q = np.array([[1.0, 0.0]], dtype=np.float32)
        d = np.array([[0.5, 0.0], [0.9, 0.0]], dtype=np.float32)
        # max(0.5, 0.9) = 0.9, sum = 0.9
        self.assertAlmostEqual(maxsim(q, d), 0.9, places=5)

    def test_zero_doc_gives_zero_score(self) -> None:
        q = np.array([[1.0, 0.0]], dtype=np.float32)
        d = np.array([[0.0, 0.0]], dtype=np.float32)
        self.assertAlmostEqual(maxsim(q, d), 0.0, places=5)


class ComputeScoresTest(unittest.TestCase):
    def test_shape_is_n_queries_by_n_docs(self) -> None:
        q_list = [np.random.randn(3, 4).astype(np.float32) for _ in range(5)]
        d_list = [np.random.randn(6, 4).astype(np.float32) for _ in range(7)]
        scores = compute_scores(q_list, d_list)
        self.assertEqual(scores.shape, (5, 7))

    def test_known_values(self) -> None:
        # Single-token queries and docs → MaxSim reduces to dot product
        q_list = [np.array([[1.0, 0.0]], dtype=np.float32),
                  np.array([[0.0, 1.0]], dtype=np.float32)]
        d_list = [np.array([[0.5, 0.5]], dtype=np.float32),
                  np.array([[1.0, 0.0]], dtype=np.float32)]
        scores = compute_scores(q_list, d_list)
        np.testing.assert_allclose(scores[0, 0], 0.5, atol=1e-5)
        np.testing.assert_allclose(scores[0, 1], 1.0, atol=1e-5)
        np.testing.assert_allclose(scores[1, 0], 0.5, atol=1e-5)
        np.testing.assert_allclose(scores[1, 1], 0.0, atol=1e-5)


class MultivecRewardStateTest(unittest.TestCase):
    def _make_multivec_embeddings(self) -> list[np.ndarray]:
        rng = np.random.default_rng(0)
        # 3 queries, each with 2 token embeddings of dim 4
        return [rng.random((2, 4)).astype(np.float32) for _ in range(3)]

    def _make_multivec_doc_embeddings(self) -> list[np.ndarray]:
        rng = np.random.default_rng(1)
        # 2 docs, each with 3 token embeddings of dim 4
        return [rng.random((3, 4)).astype(np.float32) for _ in range(2)]

    def test_build_reward_state_multivec_baseline_scores_shape(self) -> None:
        q_embs = self._make_multivec_embeddings()
        d_embs = self._make_multivec_doc_embeddings()
        state = build_reward_state(
            train_indices=np.asarray([0, 1, 2], dtype=np.int64),
            query_embeddings=q_embs,
            baseline_doc_embeddings=d_embs,
            query2doc={0: {0: 1.0}, 1: {1: 1.0}, 2: {0: 1.0}},
            negative_top_k=5,
        )
        self.assertEqual(state.train_query_baseline_scores.shape, (3, 2))
        self.assertIsInstance(state.train_query_embeddings, list)
        self.assertEqual(len(state.train_query_embeddings), 3)

    def test_build_reward_state_multivec_positive_indices(self) -> None:
        q_embs = self._make_multivec_embeddings()
        d_embs = self._make_multivec_doc_embeddings()
        state = build_reward_state(
            train_indices=np.asarray([0, 1, 2], dtype=np.int64),
            query_embeddings=q_embs,
            baseline_doc_embeddings=d_embs,
            query2doc={0: {0: 1.0}, 1: {1: 1.0}, 2: {0: 1.0}},
            negative_top_k=5,
        )
        np.testing.assert_array_equal(
            state.train_positive_query_indices_by_doc[0], np.asarray([0, 2])
        )
        np.testing.assert_array_equal(
            state.train_positive_query_indices_by_doc[1], np.asarray([1])
        )


class MultivecCounterfactualDeltaTest(unittest.TestCase):
    def test_counterfactual_delta_multivec(self) -> None:
        # 3 queries with 1 token each (dim 2), so MaxSim == dot product
        q_embs = [
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 1.0]], dtype=np.float32),
        ]
        baseline_scores = np.array(
            [[0.9, 0.0], [0.0, 0.8], [0.9, 0.8]], dtype=np.float32
        )
        # Candidate doc embedding: 1 token of dim 2
        candidate = np.array([[1.1, -0.1]], dtype=np.float32)
        # query 0 score: maxsim([[1,0]], [[1.1,-0.1]]) = 1.1
        # query 2 score: maxsim([[1,1]], [[1.1,-0.1]]) = 1.0
        # deltas: (1.1 - 0.9) and (1.0 - 0.9) → mean = 0.15
        delta = mean_counterfactual_score_delta_for_query_indices(
            query_indices=np.asarray([0, 2], dtype=np.int64),
            doc_index=0,
            candidate_embedding=candidate,
            train_query_baseline_scores=baseline_scores,
            train_query_embeddings=q_embs,
        )
        self.assertAlmostEqual(delta, 0.15, places=5)

    def test_empty_query_indices_returns_zero(self) -> None:
        q_embs = [np.array([[1.0, 0.0]], dtype=np.float32)]
        baseline_scores = np.array([[0.9, 0.0]], dtype=np.float32)
        delta = mean_counterfactual_score_delta_for_query_indices(
            query_indices=np.empty(0, dtype=np.int64),
            doc_index=0,
            candidate_embedding=np.array([[1.1, 0.0]], dtype=np.float32),
            train_query_baseline_scores=baseline_scores,
            train_query_embeddings=q_embs,
        )
        self.assertEqual(delta, 0.0)


class MultivecEvaluationTest(unittest.TestCase):
    def test_rank_topk_multivec(self) -> None:
        # 2 queries (1 token each), 3 docs (1 token each), dim=2
        q_list = [
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 1.0]], dtype=np.float32),
        ]
        d_list = [
            np.array([[0.9, 0.1]], dtype=np.float32),  # closest to q0
            np.array([[0.1, 0.9]], dtype=np.float32),  # closest to q1
            np.array([[0.5, 0.5]], dtype=np.float32),
        ]
        ranked = rank_topk(q_list, d_list, k=2)
        self.assertEqual(ranked.shape, (2, 2))
        # q0 best match should be doc 0
        self.assertEqual(ranked[0, 0], 0)
        # q1 best match should be doc 1
        self.assertEqual(ranked[1, 0], 1)

    def test_evaluate_from_embeddings_multivec(self) -> None:
        q_list = [np.array([[1.0, 0.0]], dtype=np.float32)]
        d_list = [
            np.array([[0.9, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.9]], dtype=np.float32),
        ]
        doc_ids = ["doc0", "doc1"]
        metrics = evaluate_from_embeddings(
            d_list, q_list, {0: {0: 1}}, doc_ids, ks=(1,)
        )
        self.assertAlmostEqual(metrics["ndcg@1"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
