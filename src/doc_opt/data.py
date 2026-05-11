from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DOCS_SUBDIR = "texts"
BENCHMARK_CSV = "benchmark/benchmark.csv"


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    query2doc: dict[int, str]
    queries: list[str]
    docs: list[str]
    doc_ids: list[str]


def validate_dataset_root(dataset_root: Path) -> None:
    docs_dir = dataset_root / DOCS_SUBDIR
    benchmark_path = dataset_root / BENCHMARK_CSV
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Document directory not found: {docs_dir}")
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark CSV not found: {benchmark_path}")


def load_ds1000_dataset(dataset_root: Path) -> DatasetBundle:
    validate_dataset_root(dataset_root)
    doc_paths = sorted((dataset_root / DOCS_SUBDIR).glob("*.txt"))
    docs = [path.read_text(encoding="utf-8") for path in doc_paths]
    doc_ids = [path.stem for path in doc_paths]
    query2doc: dict[int, str] = {}
    queries: list[str] = []
    dataframe = pd.read_csv(dataset_root / BENCHMARK_CSV)

    for query_index, (question, raw_ids) in enumerate(
        zip(dataframe["question"], dataframe["correct_answer_document_ids"])
    ):
        queries.append(str(question).strip())
        query2doc[query_index] = list(ast.literal_eval(raw_ids).keys())[0]

    return DatasetBundle(query2doc=query2doc, queries=queries, docs=docs, doc_ids=doc_ids)
