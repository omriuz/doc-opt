from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .data import load_ds1000_dataset, load_image_dataset
from .doc_transform import build_doc_tranform_prompt
from .embeddings import embed_texts, load_embeddings, save_embeddings
from .multivec import is_multivec_model
from .rewards import build_reward_function, normalize_qrels
from .splits import split_query_indices


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_doc_opt(
    config: ExperimentConfig,
    *,
    model_source: str | None,
    checkpoint_dir: Path,
    grpo_output_dir: Path,
    baseline_embs_path: Path,
) -> None:
    if config.doc_type == "image":
        _run_doc_opt_image(
            config,
            model_source=model_source,
            checkpoint_dir=checkpoint_dir,
            grpo_output_dir=grpo_output_dir,
            baseline_embs_path=baseline_embs_path,
        )
    else:
        _run_doc_opt_text(
            config,
            model_source=model_source,
            checkpoint_dir=checkpoint_dir,
            grpo_output_dir=grpo_output_dir,
            baseline_embs_path=baseline_embs_path,
        )


def _resolve_policy_source(raw_source: str) -> tuple[str, bool]:
    local = Path(raw_source)
    policy_source = local.resolve().as_posix() if not local.is_absolute() and local.exists() else raw_source
    return policy_source, Path(policy_source).exists()


def _grpo_config(config: ExperimentConfig, grpo_output_dir: Path) -> "GRPOConfig":
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=grpo_output_dir.as_posix(),
        max_steps=config.doc_opt.max_steps,
        per_device_train_batch_size=config.doc_opt.per_device_train_batch_size,
        num_generations=config.doc_opt.num_generations,
        learning_rate=config.doc_opt.learning_rate,
        beta=config.doc_opt.kl_beta,
        temperature=config.doc_opt.temperature,
        top_p=config.doc_tranform.top_p,
        top_k=config.doc_tranform.top_k,
        bf16=True,
        remove_unused_columns=False,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        max_completion_length=config.doc_opt.max_completion_length,
    )


def _save_checkpoint(trainer, *, checkpoint_dir: Path, processing_class) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Persist optimizer/scheduler/rng state alongside the final weights so a run can be resumed.
    original_output_dir = trainer.args.output_dir
    trainer.args.output_dir = checkpoint_dir.as_posix()
    trainer.save_state()
    trainer.args.output_dir = original_output_dir
    trained_model = _unwrap_model_for_saving(trainer.model)
    trained_model.save_pretrained(checkpoint_dir, safe_serialization=True)
    processing_class.save_pretrained(checkpoint_dir)


def _run_doc_opt_text(
    config: ExperimentConfig,
    *,
    model_source: str | None,
    checkpoint_dir: Path,
    grpo_output_dir: Path,
    baseline_embs_path: Path,
) -> None:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer

    seed_everything(config.seed)
    bundle = load_ds1000_dataset(config.dataset_root)
    policy_source, is_local = _resolve_policy_source(model_source or config.policy_model)
    tokenizer = AutoTokenizer.from_pretrained(
        policy_source, trust_remote_code=True, local_files_only=is_local
    )
    model = AutoModelForCausalLM.from_pretrained(
        policy_source, torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=is_local,
    ).to("cuda")

    train_queries_by_doc, train_indices = _build_train_queries_by_doc(config, bundle)
    reward = build_reward_function(
        config,
        train_indices=train_indices,
        query_embeddings=load_query_embeddings(config, bundle.queries),
        baseline_doc_embeddings=load_embeddings(
            baseline_embs_path, multivec=is_multivec_model(config.embedding_model)
        ),
        query2doc=bundle.query2doc,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=_grpo_config(config, grpo_output_dir),
        train_dataset=Dataset.from_list(
            build_doc_opt_rows(
                docs=bundle.docs,
                tokenizer=tokenizer,
                train_queries_by_doc=train_queries_by_doc,
                transform_instruction=config.doc_tranform.transform_instruction,
            )
        ),
    )
    trainer.train()
    _save_checkpoint(trainer, checkpoint_dir=checkpoint_dir, processing_class=tokenizer)


def _run_doc_opt_image(
    config: ExperimentConfig,
    *,
    model_source: str | None,
    checkpoint_dir: Path,
    grpo_output_dir: Path,
    baseline_embs_path: Path,
) -> None:
    import torch
    from datasets import Dataset, Image as HFImage
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from trl import GRPOTrainer

    seed_everything(config.seed)
    bundle = load_image_dataset(config.dataset_root)
    policy_source, is_local = _resolve_policy_source(model_source or config.policy_model)
    processor = AutoProcessor.from_pretrained(
        policy_source, trust_remote_code=True, local_files_only=is_local
    )
    model = AutoModelForVision2Seq.from_pretrained(
        policy_source, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=is_local,
    ).to("cuda")

    train_queries_by_doc, train_indices = _build_train_queries_by_doc(config, bundle)
    reward = build_reward_function(
        config,
        train_indices=train_indices,
        query_embeddings=load_query_embeddings(config, bundle.queries),
        baseline_doc_embeddings=load_embeddings(
            baseline_embs_path, multivec=is_multivec_model(config.embedding_model)
        ),
        query2doc=bundle.query2doc,
    )

    # Dataset rows carry image paths; cast to HF Image so TRL's collator loads pixels.
    rows = build_image_doc_opt_rows(
        image_paths=bundle.docs,
        train_queries_by_doc=train_queries_by_doc,
        transform_instruction=config.doc_tranform.transform_instruction,
    )
    dataset = Dataset.from_list(rows).cast_column("image", HFImage())

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=_grpo_config(config, grpo_output_dir),
        processing_class=processor,
        train_dataset=dataset,
    )
    trainer.train()
    _save_checkpoint(trainer, checkpoint_dir=checkpoint_dir, processing_class=processor)


def _build_train_queries_by_doc(
    config: ExperimentConfig,
    bundle,
) -> tuple[dict[int, list[str]], np.ndarray]:
    if config.eval_dataset_root is None:
        train_indices = split_query_indices(
            bundle.queries, seed=config.seed, test_size=config.test_size
        ).train_indices
    else:
        train_indices = np.arange(len(bundle.queries), dtype=np.int64)
    train_queries_by_doc: dict[int, list[str]] = {}
    for query_index in train_indices:
        for doc_index in normalize_qrels(bundle.query2doc[query_index]):
            train_queries_by_doc.setdefault(int(doc_index), []).append(bundle.queries[query_index])
    return train_queries_by_doc, train_indices


def build_doc_opt_rows(
    *,
    docs: list[str],
    tokenizer,
    train_queries_by_doc: dict[int, list[str]],
    transform_instruction: str,
) -> list[dict[str, str]]:
    return [
        {
            "prompt": tokenizer.apply_chat_template(
                [{"role": "user", "content": build_doc_tranform_prompt(docs[doc_index], transform_instruction)}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            "document": str(doc_index),
        }
        for doc_index in sorted(train_queries_by_doc)
    ]


def build_image_doc_opt_rows(
    *,
    image_paths: list[str],
    train_queries_by_doc: dict[int, list[str]],
    transform_instruction: str,
) -> list[dict]:
    # Prompt uses message-list format; TRL's VLM collator applies the processor.
    # The {"type": "image"} placeholder is matched to the "image" column by TRL.
    return [
        {
            "prompt": [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": transform_instruction},
            ]}],
            "image": image_paths[doc_index],
            "document": str(doc_index),
        }
        for doc_index in sorted(train_queries_by_doc)
    ]


def load_query_embeddings(
    config: ExperimentConfig, queries: list[str]
) -> np.ndarray | list[np.ndarray]:
    from .config import artifact_layout

    cache_dir = artifact_layout(config.output_dir).cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    queries_cache = cache_dir / "queries_embs.npy"
    multivec = is_multivec_model(config.embedding_model)
    if queries_cache.exists():
        return load_embeddings(queries_cache, multivec=multivec)
    query_embeddings = embed_texts(
        queries, model=config.embedding_model, device=config.embedding_device, is_query=True
    )
    save_embeddings(queries_cache, query_embeddings)
    return query_embeddings


def _unwrap_model_for_saving(model):
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped
