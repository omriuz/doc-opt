"""
Smoke / integration tests — one toy run per retriever × doc_type combination.

Each test:
  1. Creates a tiny synthetic dataset (6 docs/images, 6 queries) in a temp dir
  2. Embeds docs and queries with the target retriever (real model call, no mocks)
  3. Builds the ranking reward state and reward function
  4. Calls the reward function once with a synthetic completion
  5. Asserts the reward output is a valid list[float]

The vLLM / GRPO training stages are NOT tested here — they are exercised by
actual training runs. These tests focus on the embedding + reward path, which
is what varies across retriever configurations.

Markers
-------
  smoke   all tests in this file
  openai  requires OPENAI_API_KEY in the environment
  slow    loads a large local HF model (Qwen, Jina); may take minutes on CPU

Run all smoke tests:
    pytest tests/test_smoke.py -v

Skip when no OpenAI key is available:
    pytest tests/test_smoke.py -v -m "not openai"

Skip slow local-model tests:
    pytest tests/test_smoke.py -v -m "not slow"
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from doc_opt.config import DocOptConfig, DocTranformConfig, ExperimentConfig, VLLMConfig
from doc_opt.data import load_ds1000_dataset, load_image_dataset
from doc_opt.embeddings import embed_texts, load_or_compute_embeddings, load_or_compute_query_embeddings
from doc_opt.multivec import is_multivec_model
from doc_opt.rewards import build_reward_function
from doc_opt.splits import split_query_indices


HAS_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))
_N = 6  # docs/images and queries; gives a non-trivial 3+3 train/test split


# ---------------------------------------------------------------------------
# Synthetic dataset factories
# ---------------------------------------------------------------------------

def _make_text_dataset(root: Path) -> None:
    (root / "texts").mkdir(parents=True)
    (root / "benchmark").mkdir(parents=True)
    for i in range(_N):
        (root / "texts" / f"doc{i}.txt").write_text(
            f"Document {i}: unique information about subject {i}. "
            f"This document explains topic {i} with examples and details.",
            encoding="utf-8",
        )
    questions = [f"What information does the document about subject {i} contain?" for i in range(_N)]
    qrels     = [f"{{'doc{i}': 1.0}}" for i in range(_N)]
    pd.DataFrame({"question": questions, "correct_answer_document_ids": qrels}).to_csv(
        root / "benchmark" / "benchmark.csv", index=False
    )


def _make_image_dataset(root: Path) -> None:
    from PIL import Image

    (root / "images").mkdir(parents=True)
    (root / "benchmark").mkdir(parents=True)
    for i in range(_N):
        color = ((i * 40) % 256, (i * 80) % 256, (i * 120) % 256)
        Image.new("RGB", (32, 32), color=color).save(root / "images" / f"img{i}.png")
    questions = [f"Which image shows color pattern number {i}?" for i in range(_N)]
    qrels     = [f"{{'img{i}': 1.0}}" for i in range(_N)]
    pd.DataFrame({"question": questions, "correct_answer_document_ids": qrels}).to_csv(
        root / "benchmark" / "benchmark.csv", index=False
    )


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def _make_config(
    output_dir: Path,
    dataset_root: Path,
    embedding_model: str,
    *,
    doc_type: str = "text",
    device: str = "cpu",
) -> ExperimentConfig:
    return ExperimentConfig(
        seed=42,
        dataset_root=dataset_root,
        eval_dataset_root=None,
        output_dir=output_dir,
        test_size=0.5,
        embedding_models=[embedding_model],
        embedding_device=device,
        embedding_device_grpo=device,
        reward_function="ranking",
        policy_model="Qwen/Qwen3-4B-Instruct-2507",
        doc_type=doc_type,
        doc_tranform=DocTranformConfig(
            transform_instruction="Rewrite this document to be more informative.",
            batch_size=1,
            max_tokens=64,
            temperature=0.3,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
            presence_penalty=0.0,
        ),
        doc_opt=DocOptConfig(
            max_steps=1,
            refresh_rate=1,
            checkpoint_save_rate=1,
            temperature=0.7,
            per_device_train_batch_size=1,
            num_generations=2,
            learning_rate=1e-5,
            kl_beta=0.1,
            max_completion_length=128,
        ),
        vllm=VLLMConfig(
            dtype="bfloat16",
            gpu_memory_utilization=0.7,
            max_model_len=4096,
            enforce_eager=True,
            use_deep_gemm=False,
            enable_v1_multiprocessing=False,
        ),
    )


# ---------------------------------------------------------------------------
# Core smoke runners
# ---------------------------------------------------------------------------

def _smoke_text(embedding_model: str, device: str = "cpu") -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path  = Path(tmp)
        ds_root   = tmp_path / "dataset"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        _make_text_dataset(ds_root)
        config = _make_config(tmp_path / "output", ds_root, embedding_model, device=device)

        bundle = load_ds1000_dataset(ds_root)

        # Real embedding call — exercises the full embed_texts path for this retriever.
        docs_embs, query_embs = load_or_compute_embeddings(
            docs=bundle.docs,
            queries=bundle.queries,
            cache_dir=cache_dir,
            model=embedding_model,
            device=device,
        )

        multivec = is_multivec_model(embedding_model)
        if multivec:
            assert isinstance(docs_embs, list) and len(docs_embs) == _N
            assert isinstance(query_embs, list) and len(query_embs) == _N
        else:
            assert docs_embs.shape[0] == _N
            assert query_embs.shape[0] == _N

        split  = split_query_indices(bundle.queries, seed=42, test_size=0.5)
        reward = build_reward_function(
            config,
            train_indices=split.train_indices,
            query_embeddings=query_embs,
            baseline_doc_embeddings=docs_embs,
            query2doc=bundle.query2doc,
        )

        # One full reward call: embed_texts is called inside on the completion text.
        doc_key = str(int(split.train_indices[0]))
        out = reward(
            ["<answer>A rewritten version of the document with improved clarity.</answer>"],
            document=doc_key,
        )
        assert isinstance(out, list) and len(out) == 1
        assert isinstance(out[0], float)


def _smoke_visual(embedding_model: str, device: str = "cpu") -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path  = Path(tmp)
        ds_root   = tmp_path / "dataset"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        _make_image_dataset(ds_root)
        config = _make_config(
            tmp_path / "output", ds_root, embedding_model, doc_type="image", device=device
        )

        bundle = load_image_dataset(ds_root)
        assert len(bundle.docs) == _N
        for path_str in bundle.docs:
            assert Path(path_str).exists(), f"Image not found: {path_str}"

        # Image pipeline: only queries are embedded from text.  Doc embeddings come
        # from VLM-generated captions at doc-transform time; we simulate them here.
        query_embs = load_or_compute_query_embeddings(
            queries=bundle.queries,
            cache_dir=cache_dir,
            model=embedding_model,
            device=device,
        )
        assert isinstance(query_embs, np.ndarray) and query_embs.shape[0] == _N

        vlm_descriptions = [
            f"An image with color pattern {i} showing abstract geometric shapes."
            for i in range(_N)
        ]
        docs_embs = embed_texts(vlm_descriptions, model=embedding_model, device=device)
        assert isinstance(docs_embs, np.ndarray) and docs_embs.shape[0] == _N

        split  = split_query_indices(bundle.queries, seed=42, test_size=0.5)
        reward = build_reward_function(
            config,
            train_indices=split.train_indices,
            query_embeddings=query_embs,
            baseline_doc_embeddings=docs_embs,
            query2doc=bundle.query2doc,
        )

        doc_key = str(int(split.train_indices[0]))
        out = reward(
            ["<answer>A colorful image displaying geometric shapes in vivid patterns.</answer>"],
            document=doc_key,
        )
        assert isinstance(out, list) and len(out) == 1
        assert isinstance(out[0], float)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.mark.smoke
@pytest.mark.openai
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY is not set")
def test_text_openai():
    """Text dataset · OpenAI text-embedding-3-small retriever."""
    _smoke_text("text-embedding-3-small")


@pytest.mark.smoke
@pytest.mark.slow
def test_text_qwen():
    """Text dataset · Qwen3-Embedding-0.6B local retriever."""
    _smoke_text("Qwen/Qwen3-Embedding-0.6B")


@pytest.mark.smoke
@pytest.mark.slow
def test_text_colbert():
    """Text dataset · Jina-ColBERT-v2 multi-vector retriever."""
    _smoke_text("jinaai/jina-colbert-v2")


@pytest.mark.smoke
@pytest.mark.openai
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY is not set")
def test_visual_openai():
    """Image dataset · OpenAI text-embedding-3-small retriever (VLM captions simulated)."""
    _smoke_visual("text-embedding-3-small")


@pytest.mark.smoke
@pytest.mark.slow
def test_visual_qwen():
    """Image dataset · Qwen3-Embedding-0.6B local retriever (VLM captions simulated)."""
    _smoke_visual("Qwen/Qwen3-Embedding-0.6B")
