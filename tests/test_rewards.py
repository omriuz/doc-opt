from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from doc_opt.config import load_config
from doc_opt.rewards import build_reward_function
from doc_opt.rewards.common import build_reward_state, mean_counterfactual_score_delta_for_query_indices


class RewardConfigTest(unittest.TestCase):
    def test_load_config_defaults_to_ranking_reward(self) -> None:
        config_yaml = textwrap.dedent(
            """
            seed: 42
            dataset_root: data/example
            output_dir: artifacts/example
            test_size: 0.2
            embedding_model: text-embedding-3-small
            reward_function: ranking
            policy_model: Qwen/Qwen3-4B-Instruct-2507

            doc-tranform:
              transform-instruction: rewrite
              batch_size: 1
              max_tokens: 64
              temperature: 0.3
              top_p: 0.8
              top_k: 20
              repetition_penalty: 1.0
              presence_penalty: 0.0

            doc-opt:
              max_steps: 10
              refresh_rate: 5
              temperature: 0.7
              per_device_train_batch_size: 2
              num_generations: 2
              learning_rate: 1.0e-5
              kl_beta: 0.1
              max_completion_length: 128

            vllm:
              dtype: bfloat16
              gpu_memory_utilization: 0.7
              max_model_len: 4096
              enforce_eager: true
              use_deep_gemm: false
              enable_v1_multiprocessing: false
            """
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(config_yaml, encoding="utf-8")
            config = load_config(config_path)

        self.assertEqual(config.reward_function, "ranking")

    def test_load_config_accepts_legacy_nested_reward_function(self) -> None:
        config_yaml = textwrap.dedent(
            """
            seed: 42
            dataset_root: data/example
            output_dir: artifacts/example
            test_size: 0.2
            embedding_model: text-embedding-3-small
            policy_model: Qwen/Qwen3-4B-Instruct-2507

            doc-tranform:
              transform-instruction: rewrite
              batch_size: 1
              max_tokens: 64
              temperature: 0.3
              top_p: 0.8
              top_k: 20
              repetition_penalty: 1.0
              presence_penalty: 0.0

            doc-opt:
              reward_function: dense
              max_steps: 10
              refresh_rate: 5
              temperature: 0.7
              per_device_train_batch_size: 2
              num_generations: 2
              learning_rate: 1.0e-5
              kl_beta: 0.1
              max_completion_length: 128

            vllm:
              dtype: bfloat16
              gpu_memory_utilization: 0.7
              max_model_len: 4096
              enforce_eager: true
              use_deep_gemm: false
              enable_v1_multiprocessing: false
            """
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(config_yaml, encoding="utf-8")
            config = load_config(config_path)

        self.assertEqual(config.reward_function, "dense")

    def test_load_config_accepts_hybrid_reward(self) -> None:
        config_yaml = textwrap.dedent(
            """
            seed: 42
            dataset_root: data/example
            output_dir: artifacts/example
            test_size: 0.2
            embedding_model: text-embedding-3-small
            reward_function: hybrid
            policy_model: Qwen/Qwen3-4B-Instruct-2507

            doc-tranform:
              transform-instruction: rewrite
              batch_size: 1
              max_tokens: 64
              temperature: 0.3
              top_p: 0.8
              top_k: 20
              repetition_penalty: 1.0
              presence_penalty: 0.0

            doc-opt:
              max_steps: 10
              refresh_rate: 5
              temperature: 0.7
              per_device_train_batch_size: 2
              num_generations: 2
              learning_rate: 1.0e-5
              kl_beta: 0.1
              max_completion_length: 128

            vllm:
              dtype: bfloat16
              gpu_memory_utilization: 0.7
              max_model_len: 4096
              enforce_eager: true
              use_deep_gemm: false
              enable_v1_multiprocessing: false
            """
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(config_yaml, encoding="utf-8")
            config = load_config(config_path)

        self.assertEqual(config.reward_function, "hybrid")


class RewardStateTest(unittest.TestCase):
    def test_build_reward_state_tracks_positive_and_hard_negative_queries(self) -> None:
        state = build_reward_state(
            train_indices=np.asarray([0, 1, 2], dtype=np.int64),
            query_embeddings=np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float32,
            ),
            baseline_doc_embeddings=np.asarray(
                [
                    [0.9, 0.0],
                    [0.0, 0.8],
                ],
                dtype=np.float32,
            ),
            query2doc={
                0: {0: 1.0},
                1: {1: 1.0},
                2: {0: 1.0},
            },
            negative_top_k=5,
        )

        np.testing.assert_array_equal(state.train_positive_query_indices_by_doc[0], np.asarray([0, 2]))
        np.testing.assert_array_equal(state.train_negative_query_indices_by_doc[0], np.asarray([1]))
        np.testing.assert_array_equal(state.train_positive_query_indices_by_doc[1], np.asarray([1]))
        np.testing.assert_array_equal(state.train_negative_query_indices_by_doc[1], np.asarray([2, 0]))


class DenseRewardTest(unittest.TestCase):
    def test_mean_counterfactual_score_delta_matches_expected_average(self) -> None:
        delta = mean_counterfactual_score_delta_for_query_indices(
            query_indices=np.asarray([0, 2], dtype=np.int64),
            doc_index=0,
            candidate_embedding=np.asarray([1.1, -0.1], dtype=np.float32),
            train_query_baseline_scores=np.asarray(
                [
                    [0.9, 0.0],
                    [0.0, 0.8],
                    [0.9, 0.8],
                ],
                dtype=np.float32,
            ),
            train_query_embeddings=np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

        self.assertAlmostEqual(delta, 0.15, places=6)

    def test_dense_reward_uses_positive_minus_negative_score_delta(self) -> None:
        config_yaml = textwrap.dedent(
            """
            seed: 42
            dataset_root: data/example
            output_dir: artifacts/example
            test_size: 0.2
            embedding_model: text-embedding-3-small
            reward_function: dense
            policy_model: Qwen/Qwen3-4B-Instruct-2507

            doc-tranform:
              transform-instruction: rewrite
              batch_size: 1
              max_tokens: 64
              temperature: 0.3
              top_p: 0.8
              top_k: 20
              repetition_penalty: 1.0
              presence_penalty: 0.0

            doc-opt:
              max_steps: 10
              refresh_rate: 5
              temperature: 0.7
              per_device_train_batch_size: 2
              num_generations: 2
              learning_rate: 1.0e-5
              kl_beta: 0.1
              max_completion_length: 128

            vllm:
              dtype: bfloat16
              gpu_memory_utilization: 0.7
              max_model_len: 4096
              enforce_eager: true
              use_deep_gemm: false
              enable_v1_multiprocessing: false
            """
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(config_yaml, encoding="utf-8")
            config = load_config(config_path)

        reward = build_reward_function(
            config,
            train_indices=np.asarray([0, 1, 2], dtype=np.int64),
            query_embeddings=np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float32,
            ),
            baseline_doc_embeddings=np.asarray(
                [
                    [0.9, 0.0],
                    [0.0, 0.8],
                ],
                dtype=np.float32,
            ),
            query2doc={
                0: {0: 1.0},
                1: {1: 1.0},
                2: {0: 1.0},
            },
        )

        with patch(
            "doc_opt.rewards.dense.embed_texts_openai",
            return_value=np.asarray([[1.1, -0.1]], dtype=np.float32),
        ):
            rewards = reward(["rewritten doc"], "0")

        self.assertEqual(len(rewards), 1)
        self.assertAlmostEqual(rewards[0], 0.25, places=6)


class HybridRewardTest(unittest.TestCase):
    def test_hybrid_reward_averages_ranking_and_dense_rewards(self) -> None:
        config_yaml = textwrap.dedent(
            """
            seed: 42
            dataset_root: data/example
            output_dir: artifacts/example
            test_size: 0.2
            embedding_model: text-embedding-3-small
            reward_function: hybrid
            policy_model: Qwen/Qwen3-4B-Instruct-2507

            doc-tranform:
              transform-instruction: rewrite
              batch_size: 1
              max_tokens: 64
              temperature: 0.3
              top_p: 0.8
              top_k: 20
              repetition_penalty: 1.0
              presence_penalty: 0.0

            doc-opt:
              max_steps: 10
              refresh_rate: 5
              temperature: 0.7
              per_device_train_batch_size: 2
              num_generations: 2
              learning_rate: 1.0e-5
              kl_beta: 0.1
              max_completion_length: 128

            vllm:
              dtype: bfloat16
              gpu_memory_utilization: 0.7
              max_model_len: 4096
              enforce_eager: true
              use_deep_gemm: false
              enable_v1_multiprocessing: false
            """
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(config_yaml, encoding="utf-8")
            config = load_config(config_path)

        reward = build_reward_function(
            config,
            train_indices=np.asarray([0, 1], dtype=np.int64),
            query_embeddings=np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            baseline_doc_embeddings=np.asarray(
                [
                    [0.2, 0.0],
                    [0.8, 1.0],
                ],
                dtype=np.float32,
            ),
            query2doc={
                0: {0: 1.0},
                1: {1: 1.0},
            },
        )

        with patch(
            "doc_opt.rewards.hybrid.embed_texts_openai",
            return_value=np.asarray([[1.5, 0.0]], dtype=np.float32),
        ):
            rewards = reward(["rewritten doc"], "0")

        self.assertEqual(len(rewards), 1)
        self.assertAlmostEqual(rewards[0], 0.834535, places=5)


if __name__ == "__main__":
    unittest.main()
