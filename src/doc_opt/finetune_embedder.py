from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .config import EmbeddingFinetuneConfig, ExperimentConfig
from .data import load_ds1000_dataset, load_image_dataset
from .evaluation import evaluate_from_embeddings
from .splits import split_query_indices

METRIC_KS = (1, 5, 10)


class _PairDataset(Dataset):
    def __init__(self, all_queries: list[str], all_docs: list[str], pairs: list[tuple[int, int]]):
        self.all_queries = all_queries
        self.all_docs = all_docs
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        q_idx, d_idx = self.pairs[idx]
        return self.all_queries[q_idx], self.all_docs[d_idx]


def _encode(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: str,
    max_length: int,
) -> torch.Tensor:
    """Encode texts into normalized embeddings using last-token pooling."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    outputs = model(**encoded)
    hidden = outputs.last_hidden_state  # [batch, seq, dim]
    mask = encoded["attention_mask"]
    left_padding = mask[:, -1].sum() == mask.shape[0]
    if left_padding:
        embs = hidden[:, -1]
    else:
        seq_lens = mask.sum(dim=1) - 1
        embs = hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_lens]
    return F.normalize(embs.float(), dim=-1)


def _embed_corpus(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: str,
    max_length: int,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            embs = _encode(model, tokenizer, texts[start : start + batch_size], device, max_length)
            chunks.append(embs.cpu().numpy())
    return np.vstack(chunks)


def run_finetune_embedder(config: ExperimentConfig) -> None:
    if config.doc_type not in {"text", "image"}:
        raise ValueError("finetune-embedder only supports doc_type='text' or doc_type='image'")

    ft_cfg = config.embedding_finetune
    if ft_cfg is None:
        raise ValueError("Config must include an 'embedding-finetune' section to run finetune-embedder.")

    if config.doc_type == "image":
        dataset = load_image_dataset(config.dataset_root)
        # Image documents aren't text-embeddable; train/eval against the cached
        # VLM-generated descriptions produced by a prior doc-tranform run instead.
        transformed_docs = json.loads(ft_cfg.transformed_docs_path.read_text(encoding="utf-8"))
        if len(transformed_docs) != len(dataset.docs):
            raise ValueError(
                f"transformed_docs_path has {len(transformed_docs)} entries but the dataset "
                f"has {len(dataset.docs)} documents; they must be aligned 1:1."
            )
        docs = [str(d) for d in transformed_docs]
    else:
        dataset = load_ds1000_dataset(config.dataset_root)
        docs = dataset.docs

    query_split = split_query_indices(dataset.queries, seed=config.seed, test_size=config.test_size)

    train_pairs: list[tuple[int, int]] = []
    for q_idx in query_split.train_indices.tolist():
        target = dataset.query2doc[int(q_idx)]
        if isinstance(target, dict):
            for d_idx in target:
                train_pairs.append((int(q_idx), int(d_idx)))
        else:
            train_pairs.append((int(q_idx), int(target)))

    test_queries = [dataset.queries[int(i)] for i in query_split.test_indices]
    # Remap test qrels to contiguous 0-based indices matching test_queries rows
    test_query2doc = {
        i: dataset.query2doc[int(q)]
        for i, q in enumerate(query_split.test_indices)
    }

    device = config.embedding_device
    report: list[dict] = []

    for model_name in config.embedding_models:
        model_slug = model_name.replace("/", "_")
        model_output_dir = config.output_dir / model_slug
        model_output_dir.mkdir(parents=True, exist_ok=True)

        result = _finetune_and_eval(
            model_name=model_name,
            model_output_dir=model_output_dir,
            all_queries=dataset.queries,
            docs=docs,
            doc_ids=dataset.doc_ids,
            train_pairs=train_pairs,
            test_queries=test_queries,
            test_query2doc=test_query2doc,
            ft_cfg=ft_cfg,
            device=device,
        )
        report.append(result)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.output_dir / "finetune_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFine-tune report written to {report_path}", flush=True)


def _finetune_and_eval(
    *,
    model_name: str,
    model_output_dir: Path,
    all_queries: list[str],
    docs: list[str],
    doc_ids: list[str],
    train_pairs: list[tuple[int, int]],
    test_queries: list[str],
    test_query2doc: dict[int, int | dict[int, float]],
    ft_cfg: EmbeddingFinetuneConfig,
    device: str,
) -> dict:
    print(f"\n=== Fine-tuning {model_name} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, dtype=torch.bfloat16
    ).to(device)

    print("Evaluating baseline (pre-finetune)...", flush=True)
    doc_embs = _embed_corpus(model, tokenizer, docs, device, ft_cfg.max_length)
    q_embs = _embed_corpus(model, tokenizer, test_queries, device, ft_cfg.max_length)
    baseline_metrics = evaluate_from_embeddings(doc_embs, q_embs, test_query2doc, doc_ids, ks=METRIC_KS)
    print(f"Baseline metrics: {baseline_metrics}", flush=True)

    # --- Training ---
    effective_batch = min(ft_cfg.batch_size, len(train_pairs))
    if effective_batch < 2:
        raise ValueError(f"Too few training pairs ({len(train_pairs)}) for InfoNCE — need at least 2.")
    if effective_batch < ft_cfg.batch_size:
        print(
            f"Warning: only {len(train_pairs)} train pairs; reducing batch size "
            f"from {ft_cfg.batch_size} to {effective_batch}.",
            flush=True,
        )
    print(f"Training on {len(train_pairs)} pairs, batch_size={effective_batch}", flush=True)

    # Gradient checkpointing trades compute for memory by recomputing activations
    # during the backward pass instead of storing them all at once.  This is
    # critical for larger models (4B+) where storing all layer activations for
    # backprop would OOM.
    model.gradient_checkpointing_enable()

    train_loader = DataLoader(
        _PairDataset(all_queries, docs, train_pairs),
        batch_size=effective_batch,
        shuffle=True,
        drop_last=False,
    )
    total_steps = ft_cfg.epochs * len(train_loader)
    warmup_steps = int(total_steps * ft_cfg.warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=ft_cfg.learning_rate, weight_decay=ft_cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    step = 0
    log_interval = max(1, total_steps // 20)
    for epoch in range(ft_cfg.epochs):
        for query_batch, doc_batch in train_loader:
            q_embs_t = _encode(model, tokenizer, list(query_batch), device, ft_cfg.max_length)
            d_embs_t = _encode(model, tokenizer, list(doc_batch), device, ft_cfg.max_length)

            # InfoNCE: diagonal entries are positives, all others are in-batch negatives
            sim = (q_embs_t @ d_embs_t.T) / ft_cfg.temperature  # [batch, batch]
            labels = torch.arange(len(q_embs_t), device=device)
            loss = F.cross_entropy(sim, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % log_interval == 0 or step == total_steps:
                print(
                    f"Epoch {epoch + 1}/{ft_cfg.epochs} "
                    f"step {step}/{total_steps} "
                    f"loss {loss.item():.4f}",
                    flush=True,
                )

    # --- Post-finetune eval ---
    model.gradient_checkpointing_disable()
    print("Evaluating post-finetune...", flush=True)
    doc_embs = _embed_corpus(model, tokenizer, docs, device, ft_cfg.max_length)
    q_embs = _embed_corpus(model, tokenizer, test_queries, device, ft_cfg.max_length)
    finetuned_metrics = evaluate_from_embeddings(doc_embs, q_embs, test_query2doc, doc_ids, ks=METRIC_KS)
    print(f"Fine-tuned metrics: {finetuned_metrics}", flush=True)

    save_path = model_output_dir / "finetuned"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved fine-tuned model to {save_path}", flush=True)

    # Explicitly free GPU memory before the next model is loaded.
    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "train_pairs": len(train_pairs),
        "test_queries": len(test_queries),
        "baseline_metrics": baseline_metrics,
        "finetuned_metrics": finetuned_metrics,
    }
