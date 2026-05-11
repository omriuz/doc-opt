from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import pandas as pd


DOCS_SUBDIR = "texts"
BENCHMARK_CSV = "benchmark/benchmark.csv"


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    query2doc: dict[int, int | dict[int, float]]
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
    doc_ids = [unquote(path.stem) for path in doc_paths]
    doc_index_by_id = {doc_id: doc_index for doc_index, doc_id in enumerate(doc_ids)}
    query2doc: dict[int, int | dict[int, float]] = {}
    queries: list[str] = []
    dataframe = pd.read_csv(dataset_root / BENCHMARK_CSV)

    for query_index, (question, raw_ids) in enumerate(
        zip(dataframe["question"], dataframe["correct_answer_document_ids"])
    ):
        queries.append(str(question).strip())
        parsed_qrels = ast.literal_eval(raw_ids)
        if not isinstance(parsed_qrels, dict):
            raise ValueError("correct_answer_document_ids must deserialize to a mapping.")
        qrels_by_index: dict[int, float] = {}
        for doc_id, relevance in parsed_qrels.items():
            doc_id_str = str(doc_id)
            if doc_id_str not in doc_index_by_id:
                raise ValueError(f"Query {query_index} references unknown document id: {doc_id_str}")
            qrels_by_index[doc_index_by_id[doc_id_str]] = float(relevance)
        if not qrels_by_index:
            raise ValueError(f"Query {query_index} has no positive document ids.")
        if len(qrels_by_index) == 1:
            query2doc[query_index] = next(iter(qrels_by_index))
        else:
            query2doc[query_index] = qrels_by_index

    return DatasetBundle(query2doc=query2doc, queries=queries, docs=docs, doc_ids=doc_ids)
