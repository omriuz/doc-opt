from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

from tqdm import tqdm

from .config import ExperimentConfig
from .data import load_ds1000_dataset, load_image_dataset

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def build_doc_tranform_prompt(text: str, transform_instruction: str) -> str:
    return (
        "Transform-instruction:\n"
        f"{transform_instruction}\n"
        f"Document:\n{text}\n\nRewritten document:"
    )


def extract_rewritten_document(text: str) -> str:
    match = ANSWER_PATTERN.search(text)
    if match is not None:
        return match.group(1).strip()
    return text.strip()


def configure_vllm_environment(config: ExperimentConfig) -> None:
    os.environ["VLLM_USE_DEEP_GEMM"] = "1" if config.vllm.use_deep_gemm else "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = (
        "1" if config.vllm.enable_v1_multiprocessing else "0"
    )


def load_policy_vllm(config: ExperimentConfig, model_source: str, *, image: bool = False):
    from vllm import LLM

    configure_vllm_environment(config)
    print(f"Initializing vLLM engine for {model_source}...", flush=True)
    start_time = time.time()
    kwargs: dict = dict(
        model=model_source,
        dtype=config.vllm.dtype,
        gpu_memory_utilization=float(config.vllm.gpu_memory_utilization),
        max_model_len=config.vllm.max_model_len,
        enforce_eager=config.vllm.enforce_eager,
    )
    if image:
        kwargs["limit_mm_per_prompt"] = {"image": 1}
    model = LLM(**kwargs)
    print(f"vLLM engine initialized in {time.time() - start_time:.1f}s.", flush=True)
    return model


def doc_tranform_documents(
    docs: list[str],
    *,
    tokenizer,
    model,
    config: ExperimentConfig,
) -> list[str]:
    from vllm import SamplingParams

    prompts = [
        build_doc_tranform_prompt(
            text=doc,
            transform_instruction=config.doc_tranform.transform_instruction,
        )
        for doc in docs
    ]
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    sampling_params = SamplingParams(
        max_tokens=config.doc_tranform.max_tokens,
        temperature=config.doc_tranform.temperature,
        top_p=config.doc_tranform.top_p,
        top_k=config.doc_tranform.top_k,
        repetition_penalty=config.doc_tranform.repetition_penalty,
        presence_penalty=config.doc_tranform.presence_penalty,
    )
    rewrites: list[str] = []
    total_batches = math.ceil(len(chat_prompts) / config.doc_tranform.batch_size) if chat_prompts else 0
    print(
        f"Starting generation for {len(chat_prompts)} documents across {total_batches} batches...",
        flush=True,
    )
    start_time = time.time()
    for batch_start in tqdm(
        range(0, len(chat_prompts), config.doc_tranform.batch_size),
        total=total_batches,
        desc="Doc-tranforming docs",
    ):
        batch_prompts = chat_prompts[batch_start : batch_start + config.doc_tranform.batch_size]
        outputs = model.generate(batch_prompts, sampling_params, use_tqdm=True)
        for output in outputs:
            rewrites.append(extract_rewritten_document(output.outputs[0].text))
    print(f"Finished generation in {time.time() - start_time:.1f}s.", flush=True)
    return rewrites


def doc_tranform_images(
    image_paths: list[str],
    *,
    processor,
    model,
    config: ExperimentConfig,
) -> list[str]:
    """Run VLM inference on a list of image paths and return text descriptions."""
    from PIL import Image
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=config.doc_tranform.max_tokens,
        temperature=config.doc_tranform.temperature,
        top_p=config.doc_tranform.top_p,
        top_k=config.doc_tranform.top_k,
        repetition_penalty=config.doc_tranform.repetition_penalty,
        presence_penalty=config.doc_tranform.presence_penalty,
    )
    messages_template = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": config.doc_tranform.transform_instruction},
        ]}
    ]
    captions: list[str] = []
    total_batches = math.ceil(len(image_paths) / config.doc_tranform.batch_size) if image_paths else 0
    print(
        f"Starting VLM generation for {len(image_paths)} images across {total_batches} batches...",
        flush=True,
    )
    start_time = time.time()
    for batch_start in tqdm(
        range(0, len(image_paths), config.doc_tranform.batch_size),
        total=total_batches,
        desc="Captioning images",
    ):
        batch_paths = image_paths[batch_start : batch_start + config.doc_tranform.batch_size]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
        prompt_text = processor.apply_chat_template(
            messages_template, tokenize=False, add_generation_prompt=True
        )
        inputs = [
            {"prompt": prompt_text, "multi_modal_data": {"image": img}}
            for img in batch_images
        ]
        outputs = model.generate(inputs, sampling_params, use_tqdm=True)
        for output in outputs:
            captions.append(output.outputs[0].text.strip())
    print(f"Finished VLM generation in {time.time() - start_time:.1f}s.", flush=True)
    return captions


def run_doc_tranform(
    config: ExperimentConfig,
    *,
    output_path: Path,
    model_source: str | None = None,
) -> None:
    policy_source = model_source or config.policy_model
    local = Path(policy_source)
    if not local.is_absolute() and local.exists():
        policy_source = local.resolve().as_posix()
    is_local = Path(policy_source).exists()

    if config.doc_type == "image":
        from transformers import AutoProcessor

        bundle = load_image_dataset(config.dataset_root)
        processor = AutoProcessor.from_pretrained(
            policy_source, trust_remote_code=True, local_files_only=is_local
        )
        policy = load_policy_vllm(config, policy_source, image=True)
        transformed_docs = doc_tranform_images(
            bundle.docs,
            processor=processor,
            model=policy,
            config=config,
        )
    else:
        from transformers import AutoTokenizer

        bundle = load_ds1000_dataset(config.dataset_root)
        tokenizer = AutoTokenizer.from_pretrained(
            policy_source, trust_remote_code=True, local_files_only=is_local
        )
        policy = load_policy_vllm(config, policy_source)
        transformed_docs = doc_tranform_documents(
            bundle.docs,
            tokenizer=tokenizer,
            model=policy,
            config=config,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(transformed_docs), encoding="utf-8")
