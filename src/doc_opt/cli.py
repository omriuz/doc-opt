from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from .config import load_config

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doc-opt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-dataset",
        help="Download a Hugging Face retrieval dataset and convert it to the local repo layout.",
    )
    prepare_parser.add_argument("--dataset-id", default="mteb/DS1000Retrieval")
    prepare_parser.add_argument("--config")
    prepare_parser.add_argument("--output-dir", required=True, type=Path)
    prepare_parser.add_argument("--split", default="test")
    prepare_parser.add_argument("--overwrite", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run the full reproduction pipeline.")
    _add_shared_arguments(run_parser)

    doc_tranform_parser = subparsers.add_parser(
        "doc-tranform",
        help="Doc-tranform the dataset documents.",
    )
    _add_shared_arguments(doc_tranform_parser)
    doc_tranform_parser.add_argument("--output-path", required=True, type=Path)

    doc_opt_parser = subparsers.add_parser("doc-opt", help="Run the GRPO doc-opt stage.")
    _add_shared_arguments(doc_opt_parser)
    doc_opt_parser.add_argument("--checkpoint-dir", required=True, type=Path)
    doc_opt_parser.add_argument("--grpo-output-dir", required=True, type=Path)
    doc_opt_parser.add_argument("--baseline-embs-path", required=True, type=Path)
    # --max-steps (shared) is the full training budget and fixes the LR-schedule
    # horizon; --steps-this-round caps how many steps this invocation runs, and
    # --resume-from restores optimizer/scheduler/step state from a prior round.
    doc_opt_parser.add_argument("--steps-this-round", type=int)
    doc_opt_parser.add_argument("--resume-from", type=Path)

    transfer_parser = subparsers.add_parser(
        "transfer",
        help="Evaluate a trained checkpoint on a different dataset/retriever.",
    )
    _add_shared_arguments(transfer_parser)
    transfer_parser.add_argument("--checkpoint", required=True, type=Path, help="Path to the trained policy checkpoint.")

    finetune_parser = subparsers.add_parser(
        "finetune-embedder",
        help="Fine-tune embedding models directly on the retrieval dataset (supervised baseline).",
    )
    _add_shared_arguments(finetune_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-dataset":
        from .prepare_dataset import prepare_hf_dataset

        prepare_hf_dataset(
            args.dataset_id,
            config_name=args.config,
            output_dir=args.output_dir,
            split=args.split,
            overwrite=args.overwrite,
        )
        return 0

    config = load_config(
        args.config,
        dataset_root=args.dataset_root,
        eval_dataset_root=args.eval_dataset_root,
        output_dir=args.output_dir,
        model_source=args.model_source,
        max_steps=args.max_steps,
        embedding_model=args.embedding_model,
    )

    if args.command == "run":
        from .pipeline import run_pipeline

        run_pipeline(config, config_path=Path(args.config))
        return 0

    if args.command == "doc-tranform":
        from .doc_transform import run_doc_tranform

        run_doc_tranform(config, output_path=args.output_path, model_source=args.model_source)
        return 0

    if args.command == "doc-opt":
        from .doc_opt import run_doc_opt

        run_doc_opt(
            config,
            model_source=args.model_source,
            checkpoint_dir=args.checkpoint_dir,
            grpo_output_dir=args.grpo_output_dir,
            baseline_embs_path=args.baseline_embs_path,
            steps_this_round=args.steps_this_round,
            resume_from=args.resume_from,
        )
        return 0

    if args.command == "transfer":
        from .transfer_eval import run_transfer_eval

        run_transfer_eval(config, checkpoint_path=args.checkpoint, config_path=Path(args.config))
        return 0

    if args.command == "finetune-embedder":
        from .finetune_embedder import run_finetune_embedder

        run_finetune_embedder(config)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the YAML config file (default: {DEFAULT_CONFIG_PATH.as_posix()})",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--eval-dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-source")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--embedding-model")


if __name__ == "__main__":
    raise SystemExit(main())
