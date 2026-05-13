from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openai import OpenAI


def require_openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Export it before running.")
    return api_key


def get_openai_client() -> OpenAI:
    from openai import OpenAI

    require_openai_api_key()
    return OpenAI()


def normalize_embeddings(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return array / norms


def embed_texts_openai(
    texts: list[str],
    *,
    model: str,
    batch_size: int = 128,
    verbose: bool = True,
) -> np.ndarray:
    client = get_openai_client()
    all_embeddings: list[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
        if verbose:
            print(f"Embedding batch {batch_number}/{total_batches}", flush=True)
        batch_texts = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch_texts)
        ordered = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        all_embeddings.append(np.asarray(ordered, dtype=np.float32))
    return normalize_embeddings(np.vstack(all_embeddings))


def load_or_compute_embeddings(
    *,
    docs: list[str],
    queries: list[str],
    cache_dir: Path,
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    docs_cache = cache_dir / "docs_embs.npy"
    queries_cache = cache_dir / "queries_embs.npy"

    if docs_cache.exists():
        docs_embeddings = np.load(docs_cache)
    else:
        docs_embeddings = embed_texts_openai(docs, model=model)
        np.save(docs_cache, docs_embeddings)

    if queries_cache.exists():
        queries_embeddings = np.load(queries_cache)
    else:
        queries_embeddings = embed_texts_openai(queries, model=model)
        np.save(queries_cache, queries_embeddings)

    return docs_embeddings, queries_embeddings
