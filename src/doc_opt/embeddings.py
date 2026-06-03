from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .multivec import is_multivec_model

if TYPE_CHECKING:
    from openai import OpenAI

# Cache for locally loaded HuggingFace embedding models (model_name -> (model, tokenizer))
_LOCAL_MODEL_CACHE: dict[str, tuple] = {}
# Cache for multi-vector ColBERT models (model_name -> (model, tokenizer))
_PYLATE_MODEL_CACHE: dict[str, Any] = {}


def _stub_torchvision_if_broken() -> None:
    """
    jina-colbert-v2's custom model code imports transformers.modeling_utils
    directly, which unconditionally pulls in transformers' vision utilities,
    which import torchvision. If torchvision is installed but built against a
    different torch version, it crashes with RuntimeError on import. Intercept
    that crash and replace the module with a hollow stub so the import chain
    completes without error.

    Only activates when torchvision raises RuntimeError (version mismatch).
    A correctly installed torchvision is left untouched.
    """
    if "torchvision" in sys.modules:
        return
    try:
        import torchvision  # noqa: F401
    except RuntimeError:
        from unittest.mock import MagicMock
        for key in [k for k in sys.modules if k.startswith("torchvision")]:
            del sys.modules[key]
        stub = MagicMock()
        sys.modules["torchvision"] = stub
        for sub in ("transforms", "ops", "datasets", "io", "models", "utils"):
            sys.modules[f"torchvision.{sub}"] = MagicMock()


def _is_openai_model(model: str) -> bool:
    return "/" not in model


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


def embed_texts(
    texts: list[str],
    *,
    model: str,
    device: str = "cpu",
    batch_size: int | None = None,
    verbose: bool = True,
    is_query: bool = False,
) -> np.ndarray | list[np.ndarray]:
    if is_multivec_model(model):
        return embed_texts_multivec(
            texts,
            model=model,
            device=device,
            batch_size=batch_size or 32,
            verbose=verbose,
            is_query=is_query,
        )
    if _is_openai_model(model):
        return embed_texts_openai(texts, model=model, batch_size=batch_size or 128, verbose=verbose)
    return embed_texts_local(texts, model=model, device=device, batch_size=batch_size or 32, verbose=verbose)


def embed_texts_multivec(
    texts: list[str],
    *,
    model: str,
    device: str = "cpu",
    batch_size: int = 32,
    verbose: bool = True,
    is_query: bool = False,
) -> list[np.ndarray]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    _stub_torchvision_if_broken()
    if model not in _PYLATE_MODEL_CACHE:
        print(f"Loading ColBERT model {model}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        hf_model = AutoModel.from_pretrained(model, trust_remote_code=True).to(device)
        hf_model.eval()
        _PYLATE_MODEL_CACHE[model] = (hf_model, tokenizer)

    hf_model, tokenizer = _PYLATE_MODEL_CACHE[model]
    all_embeddings: list[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_num, start in enumerate(range(0, len(texts), batch_size), start=1):
        if verbose:
            print(f"Embedding batch {batch_num}/{total_batches}", flush=True)
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = hf_model(**encoded)
        token_embs = outputs.last_hidden_state  # [batch, seq_len, dim]
        attention_mask = encoded["attention_mask"]
        for i in range(len(batch_texts)):
            seq_len = int(attention_mask[i].sum())
            emb = token_embs[i, :seq_len].cpu().float().numpy()
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            all_embeddings.append(emb / np.where(norms == 0, 1.0, norms))

    return all_embeddings


def embed_texts_openai(
    texts: list[str],
    *,
    model: str,
    batch_size: int = 128,
    verbose: bool = True,
) -> np.ndarray:
    client = get_openai_client()
    cleaned_texts = []
    empty_indices: list[int] = []
    for index, text in enumerate(texts):
        if text is None:
            text = ""
        text = str(text)
        if not text.strip():
            empty_indices.append(index)
            cleaned_texts.append("[EMPTY]")
        else:
            cleaned_texts.append(text)
    if empty_indices:
        preview = ", ".join(str(idx) for idx in empty_indices[:5])
        print(
            f"[doc-opt] Warning: {len(empty_indices)} empty texts encountered; "
            f"replaced with '[EMPTY]'. First indices: {preview}",
            file=sys.stderr,
            flush=True,
        )
    all_embeddings: list[np.ndarray] = []
    total_batches = (len(cleaned_texts) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(cleaned_texts), batch_size), start=1):
        if verbose:
            print(f"Embedding batch {batch_number}/{total_batches}", flush=True)
        batch_texts = cleaned_texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch_texts)
        ordered = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        all_embeddings.append(np.asarray(ordered, dtype=np.float32))
    return normalize_embeddings(np.vstack(all_embeddings))


def embed_texts_local(
    texts: list[str],
    *,
    model: str,
    device: str = "cpu",
    batch_size: int = 32,
    verbose: bool = True,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if model not in _LOCAL_MODEL_CACHE:
        print(f"Loading local embedding model {model}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        hf_model = AutoModel.from_pretrained(model, trust_remote_code=True).to(device)
        hf_model.eval()
        _LOCAL_MODEL_CACHE[model] = (hf_model, tokenizer)

    hf_model, tokenizer = _LOCAL_MODEL_CACHE[model]
    all_embeddings: list[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
        if verbose:
            print(f"Embedding batch {batch_number}/{total_batches}", flush=True)
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        import torch
        with torch.no_grad():
            outputs = hf_model(**encoded)
        embeddings = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        all_embeddings.append(embeddings.cpu().float().numpy())

    return normalize_embeddings(np.vstack(all_embeddings))


def _last_token_pool(last_hidden_states, attention_mask):
    import torch
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def save_embeddings(path: Path, embeddings: np.ndarray | list[np.ndarray]) -> None:
    if isinstance(embeddings, list):
        np.save(path, np.array(embeddings, dtype=object), allow_pickle=True)
    else:
        np.save(path, embeddings)


def load_embeddings(path: Path, *, multivec: bool) -> np.ndarray | list[np.ndarray]:
    if multivec:
        return np.load(path, allow_pickle=True).tolist()
    return np.load(path)


def load_or_compute_embeddings(
    *,
    docs: list[str],
    queries: list[str],
    cache_dir: Path,
    model: str,
    device: str = "cpu",
) -> tuple[np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray]]:
    docs_embeddings = load_or_compute_doc_embeddings(docs=docs, cache_dir=cache_dir, model=model, device=device)
    queries_embeddings = load_or_compute_query_embeddings(queries=queries, cache_dir=cache_dir, model=model, device=device)
    return docs_embeddings, queries_embeddings


def load_or_compute_doc_embeddings(
    *,
    docs: list[str],
    cache_dir: Path,
    model: str,
    device: str = "cpu",
    cache_name: str = "docs_embs.npy",
) -> np.ndarray | list[np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    docs_cache = cache_dir / cache_name
    multivec = is_multivec_model(model)

    if docs_cache.exists():
        return load_embeddings(docs_cache, multivec=multivec)
    docs_embeddings = embed_texts(docs, model=model, device=device)
    save_embeddings(docs_cache, docs_embeddings)
    return docs_embeddings


def load_or_compute_query_embeddings(
    *,
    queries: list[str],
    cache_dir: Path,
    model: str,
    device: str = "cpu",
    cache_name: str = "queries_embs.npy",
) -> np.ndarray | list[np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    queries_cache = cache_dir / cache_name
    multivec = is_multivec_model(model)

    if queries_cache.exists():
        return load_embeddings(queries_cache, multivec=multivec)
    queries_embeddings = embed_texts(queries, model=model, device=device, is_query=True)
    save_embeddings(queries_cache, queries_embeddings)
    return queries_embeddings
