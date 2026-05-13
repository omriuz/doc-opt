from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class DocTranformConfig:
    transform_instruction: str
    batch_size: int
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    presence_penalty: float


@dataclass(frozen=True, slots=True)
class DocOptConfig:
    max_steps: int
    refresh_rate: int
    temperature: float
    per_device_train_batch_size: int
    num_generations: int
    learning_rate: float
    kl_beta: float
    max_completion_length: int


@dataclass(frozen=True, slots=True)
class VLLMConfig:
    dtype: str
    gpu_memory_utilization: float
    max_model_len: int
    enforce_eager: bool
    use_deep_gemm: bool
    enable_v1_multiprocessing: bool


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    seed: int
    dataset_root: Path
    output_dir: Path
    test_size: float
    embedding_model: str
    reward_function: str
    policy_model: str
    doc_tranform: DocTranformConfig
    doc_opt: DocOptConfig
    vllm: VLLMConfig

    def with_overrides(
        self,
        *,
        dataset_root: Path | None = None,
        output_dir: Path | None = None,
        model_source: str | None = None,
        max_steps: int | None = None,
    ) -> "ExperimentConfig":
        updated = self
        if dataset_root is not None:
            updated = replace(updated, dataset_root=dataset_root)
        if output_dir is not None:
            updated = replace(updated, output_dir=output_dir)
        if model_source is not None:
            updated = replace(updated, policy_model=model_source)
        if max_steps is not None:
            updated = replace(
                updated,
                doc_opt=replace(updated.doc_opt, max_steps=max_steps),
            )
        validate_config(updated)
        return updated


@dataclass(frozen=True, slots=True)
class ArtifactLayout:
    root: Path
    cache_dir: Path
    doc_tranform_dir: Path
    doc_opt_dir: Path
    checkpoints_dir: Path
    runtime_dir: Path
    run_report_path: Path


def artifact_layout(output_dir: Path) -> ArtifactLayout:
    return ArtifactLayout(
        root=output_dir,
        cache_dir=output_dir / "cache",
        doc_tranform_dir=output_dir / "doc-tranform",
        doc_opt_dir=output_dir / "doc-opt",
        checkpoints_dir=output_dir / "checkpoints",
        runtime_dir=output_dir / "runtime",
        run_report_path=output_dir / "run_report.json",
    )


def load_config(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_source: str | None = None,
    max_steps: int | None = None,
) -> ExperimentConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config file must deserialize to a mapping.")

    cfg = ExperimentConfig(
        seed=int(_require(raw, "seed")),
        dataset_root=_resolve_path(_require(raw, "dataset_root")),
        output_dir=_resolve_path(_require(raw, "output_dir")),
        test_size=float(_require(raw, "test_size")),
        embedding_model=str(_require(raw, "embedding_model")),
        reward_function=_load_reward_function(raw),
        policy_model=str(_require(raw, "policy_model")),
        doc_tranform=_load_doc_tranform_config(_require(raw, "doc-tranform")),
        doc_opt=_load_doc_opt_config(_require(raw, "doc-opt")),
        vllm=_load_vllm_config(_require(raw, "vllm")),
    )
    cfg = cfg.with_overrides(
        dataset_root=Path(dataset_root) if dataset_root is not None else None,
        output_dir=Path(output_dir) if output_dir is not None else None,
        model_source=model_source,
        max_steps=max_steps,
    )
    return cfg


def validate_config(config: ExperimentConfig) -> None:
    if config.reward_function not in {"ranking", "dense", "hybrid"}:
        raise ValueError("reward_function must be one of: ranking, dense, hybrid.")
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if config.doc_tranform.batch_size <= 0:
        raise ValueError("doc-tranform.batch_size must be positive.")
    if config.doc_tranform.max_tokens <= 0:
        raise ValueError("doc-tranform.max_tokens must be positive.")
    if not config.doc_tranform.transform_instruction.strip():
        raise ValueError("doc-tranform.transform-instruction must not be empty.")
    if config.doc_opt.max_steps <= 0:
        raise ValueError("doc-opt.max_steps must be positive.")
    if config.doc_opt.refresh_rate <= 0:
        raise ValueError("doc-opt.refresh_rate must be positive.")
    if config.doc_opt.per_device_train_batch_size <= 0:
        raise ValueError("doc-opt.per_device_train_batch_size must be positive.")
    if config.doc_opt.num_generations <= 0:
        raise ValueError("doc-opt.num_generations must be positive.")
    if config.doc_opt.max_completion_length <= 0:
        raise ValueError("doc-opt.max_completion_length must be positive.")
    if config.vllm.gpu_memory_utilization <= 0:
        raise ValueError("vllm.gpu_memory_utilization must be positive.")
    if config.vllm.max_model_len <= 0:
        raise ValueError("vllm.max_model_len must be positive.")


def _load_doc_tranform_config(raw: Any) -> DocTranformConfig:
    if not isinstance(raw, dict):
        raise ValueError("doc-tranform must be a mapping.")
    return DocTranformConfig(
        transform_instruction=str(_require(raw, "transform-instruction")),
        batch_size=int(_require(raw, "batch_size")),
        max_tokens=int(_require(raw, "max_tokens")),
        temperature=float(_require(raw, "temperature")),
        top_p=float(_require(raw, "top_p")),
        top_k=int(_require(raw, "top_k")),
        repetition_penalty=float(_require(raw, "repetition_penalty")),
        presence_penalty=float(_require(raw, "presence_penalty")),
    )


def _load_doc_opt_config(raw: Any) -> DocOptConfig:
    if not isinstance(raw, dict):
        raise ValueError("doc-opt must be a mapping.")
    return DocOptConfig(
        max_steps=int(_require(raw, "max_steps")),
        refresh_rate=int(_require(raw, "refresh_rate")),
        temperature=float(_require(raw, "temperature")),
        per_device_train_batch_size=int(_require(raw, "per_device_train_batch_size")),
        num_generations=int(_require(raw, "num_generations")),
        learning_rate=float(_require(raw, "learning_rate")),
        kl_beta=float(_require(raw, "kl_beta")),
        max_completion_length=int(_require(raw, "max_completion_length")),
    )


def _load_reward_function(raw: dict[str, Any]) -> str:
    if "reward_function" in raw:
        return str(raw["reward_function"])
    doc_opt = raw.get("doc-opt")
    if isinstance(doc_opt, dict) and "reward_function" in doc_opt:
        return str(doc_opt["reward_function"])
    return "ranking"


def _load_vllm_config(raw: Any) -> VLLMConfig:
    if not isinstance(raw, dict):
        raise ValueError("vllm must be a mapping.")
    return VLLMConfig(
        dtype=str(_require(raw, "dtype")),
        gpu_memory_utilization=float(_require(raw, "gpu_memory_utilization")),
        max_model_len=int(_require(raw, "max_model_len")),
        enforce_eager=bool(_require(raw, "enforce_eager")),
        use_deep_gemm=bool(_require(raw, "use_deep_gemm")),
        enable_v1_multiprocessing=bool(_require(raw, "enable_v1_multiprocessing")),
    )


def _require(raw: dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise ValueError(f"Missing required config key: {key}")
    return raw[key]


def _resolve_path(value: str | Path) -> Path:
    return Path(value)
