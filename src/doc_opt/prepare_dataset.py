from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


DEFAULT_DATASET_ID = "mteb/DS1000Retrieval"
DEFAULT_SPLIT = "test"
BENCHMARK_FIELDNAMES = ("question", "correct_answer_document_ids")
VIDORE_IMAGE_COLUMN = "image"


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    documents: list[tuple[str, str]]
    benchmark_rows: list[dict[str, str]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doc-opt prepare-dataset")
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"Hugging Face dataset id (default: {DEFAULT_DATASET_ID})",
    )
    parser.add_argument(
        "--config",
        help="Dataset config/subset to materialize. Required for multi-config datasets such as OBLIQ-Bench.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the local repo dataset layout.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split to materialize from each config (default: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing texts/ directory and benchmark CSV in the output directory.",
    )
    return parser


def prepare_hf_dataset(
    dataset_id: str,
    *,
    config_name: str | None = None,
    output_dir: Path,
    split: str = DEFAULT_SPLIT,
    overwrite: bool = False,
) -> None:
    if _is_vidore_dataset(dataset_id):
        prepare_vidore_dataset(dataset_id, output_dir=output_dir, overwrite=overwrite)
        return

    metadata = _fetch_hf_dataset_metadata(dataset_id)

    if _uses_split_configs(metadata):
        corpus = _load_hf_rows(dataset_id, config_name="corpus", split=split)
        queries = _load_hf_rows(dataset_id, config_name="queries", split=split)
        qrels = _load_hf_rows(dataset_id, config_name="qrels", split=split)
    elif _uses_beir_separate_qrels_layout(dataset_id=dataset_id, metadata=metadata):
        corpus = _load_hf_rows(dataset_id, config_name="corpus", split="corpus")
        queries = _load_hf_rows(dataset_id, config_name="queries", split="queries")
        qrels = _load_hf_rows(f"{dataset_id}-qrels", split=split)
        queries = _filter_query_rows_to_qrels(query_rows=queries, qrel_rows=qrels)
    else:
        if not config_name:
            raise ValueError(
                f"Dataset {dataset_id!r} requires --config. "
                f"Available configs: {', '.join(_list_dataset_configs(metadata))}"
            )
        corpus = _load_hf_rows(dataset_id, config_name=config_name, split="corpus")
        queries = _load_hf_rows(dataset_id, config_name=config_name, split="queries")
        qrels = _load_qrels_rows(dataset_id=dataset_id, config_name=config_name, metadata=metadata)

    prepared = convert_retrieval_rows(
        corpus_rows=corpus,
        query_rows=queries,
        qrel_rows=qrels,
    )
    write_prepared_dataset(prepared, output_dir=output_dir, overwrite=overwrite)

    print(
        f"Prepared {len(prepared.documents)} documents and "
        f"{len(prepared.benchmark_rows)} benchmark rows in {output_dir}",
        flush=True,
    )


def _is_vidore_dataset(dataset_id: str) -> bool:
    return dataset_id.startswith("vidore/")


def _require_vidore_id(row: Mapping[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    raise KeyError(f"Cannot find ID field in row. Tried: {keys}. Available keys: {list(row.keys())}")


def _load_vidore_config(dataset_id: str, config_name: str):
    from datasets import load_dataset

    for split in (config_name, "test"):
        try:
            return load_dataset(dataset_id, config_name, split=split)
        except ValueError:
            continue
    raise ValueError(f"Could not find a usable split for {dataset_id!r} config {config_name!r}.")


def prepare_vidore_dataset(
    dataset_id: str,
    *,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """Download a ViDoRe image-retrieval dataset and write it to the local layout.

    Output layout:
        <output_dir>/images/<doc_id>.jpg   — one JPEG per corpus document
        <output_dir>/benchmark/benchmark.csv
    """
    from datasets import load_dataset

    images_dir = output_dir / "images"
    benchmark_dir = output_dir / "benchmark"
    benchmark_path = benchmark_dir / "benchmark.csv"

    if not overwrite and benchmark_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset in {output_dir}. Pass --overwrite to replace it."
        )
    if overwrite and images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    corpus = _load_vidore_config(dataset_id, "corpus")
    queries_ds = _load_vidore_config(dataset_id, "queries")
    qrels_ds = _load_vidore_config(dataset_id, "qrels")

    doc_id_to_stem: dict[str, str] = {}
    for row in corpus:
        doc_id = _require_vidore_id(row, ("corpus-id", "doc-id", "image_id", "doc_id", "image-id", "id"))
        image = row[VIDORE_IMAGE_COLUMN]
        stem = _encode_doc_id_for_filename(doc_id)
        image.save(images_dir / f"{stem}.jpg", format="JPEG")
        doc_id_to_stem[doc_id] = stem

    qrels_by_query: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrels_ds:
        qid = _require_vidore_id(row, ("query-id", "query_id", "qid"))
        did = _require_vidore_id(row, ("corpus-id", "corpus_id", "doc-id", "doc_id", "pid"))
        score = int(row.get("score", 1))
        if score > 0 and did in doc_id_to_stem:
            qrels_by_query[qid][did] = score

    benchmark_rows: list[dict[str, str]] = []
    _query_id_keys = ("query-id", "query_id", "id")
    for row in sorted(queries_ds, key=lambda r: _require_vidore_id(r, ("query-id", "query_id", "id"))):
        qid = _require_vidore_id(row, _query_id_keys)
        query_text = str(row.get("query") or row.get("text") or "").strip()
        relevant = qrels_by_query.get(qid)
        if not relevant:
            continue
        benchmark_rows.append({
            "question": query_text,
            "correct_answer_document_ids": _format_qrel_mapping(relevant),
        })

    with benchmark_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDNAMES)
        writer.writeheader()
        writer.writerows(benchmark_rows)

    print(
        f"Prepared {len(doc_id_to_stem)} images and {len(benchmark_rows)} benchmark rows in {output_dir}",
        flush=True,
    )


def convert_retrieval_rows(
    *,
    corpus_rows: Iterable[Mapping[str, object]],
    query_rows: Iterable[Mapping[str, object]],
    qrel_rows: Iterable[Mapping[str, object]],
) -> PreparedDataset:
    corpus_by_id: dict[str, str] = {}
    for row in corpus_rows:
        doc_id = _require_identifier_with_aliases(row, "document id", "id", "_id")
        doc_text = _compose_text(row)
        if doc_id in corpus_by_id:
            raise ValueError(f"Duplicate corpus id: {doc_id}")
        corpus_by_id[doc_id] = doc_text

    qrels_by_query: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrel_rows:
        query_id = _require_identifier(row, "query-id")
        corpus_id = _require_identifier(row, "corpus-id")
        score = int(row["score"])
        if score <= 0:
            continue
        qrels_by_query[query_id][corpus_id] = score

    documents = [(doc_id, corpus_by_id[doc_id]) for doc_id in sorted(corpus_by_id)]
    benchmark_rows: list[dict[str, str]] = []
    query_ids: set[str] = set()

    for row in sorted(query_rows, key=lambda entry: _require_identifier_with_aliases(entry, "query id", "id", "_id")):
        query_id = _require_identifier_with_aliases(row, "query id", "id", "_id")
        query_text = _compose_text(row)
        if query_id in query_ids:
            raise ValueError(f"Duplicate query id: {query_id}")
        query_ids.add(query_id)

        relevant_docs = qrels_by_query.get(query_id)
        if not relevant_docs:
            raise ValueError(f"Missing positive qrels for query id: {query_id}")

        unknown_doc_ids = sorted(doc_id for doc_id in relevant_docs if doc_id not in corpus_by_id)
        if unknown_doc_ids:
            raise ValueError(
                f"Query id {query_id} references missing corpus ids: {', '.join(unknown_doc_ids)}"
            )

        benchmark_rows.append(
            {
                "question": query_text,
                "correct_answer_document_ids": _format_qrel_mapping(relevant_docs),
            }
        )

    extra_qrel_queries = sorted(query_id for query_id in qrels_by_query if query_id not in query_ids)
    if extra_qrel_queries:
        raise ValueError(
            "Found qrels for query ids missing from the queries split: "
            f"{', '.join(extra_qrel_queries[:5])}"
        )

    return PreparedDataset(documents=documents, benchmark_rows=benchmark_rows)


def write_prepared_dataset(
    prepared: PreparedDataset,
    *,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    texts_dir = output_dir / "texts"
    benchmark_dir = output_dir / "benchmark"
    benchmark_path = benchmark_dir / "benchmark.csv"

    if overwrite:
        if texts_dir.exists():
            shutil.rmtree(texts_dir)
        if benchmark_path.exists():
            benchmark_path.unlink()
    elif _has_existing_dataset_output(texts_dir=texts_dir, benchmark_path=benchmark_path):
        raise FileExistsError(
            f"Refusing to overwrite existing dataset output in {output_dir}. "
            "Pass --overwrite to replace it."
        )

    texts_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    for doc_id, doc_text in prepared.documents:
        (texts_dir / f"{_encode_doc_id_for_filename(doc_id)}.txt").write_text(doc_text, encoding="utf-8")

    with benchmark_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDNAMES)
        writer.writeheader()
        writer.writerows(prepared.benchmark_rows)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    prepare_hf_dataset(
        args.dataset_id,
        config_name=args.config,
        output_dir=args.output_dir,
        split=args.split,
        overwrite=args.overwrite,
    )
    return 0


def _format_qrel_mapping(relevant_docs: Mapping[str, int]) -> str:
    items = ", ".join(
        f"'{doc_id}': {int(relevant_docs[doc_id])}" for doc_id in sorted(relevant_docs)
    )
    return "{" + items + "}"


def _has_existing_dataset_output(*, texts_dir: Path, benchmark_path: Path) -> bool:
    if benchmark_path.exists():
        return True
    if not texts_dir.exists():
        return False
    return any(texts_dir.iterdir())


def _encode_doc_id_for_filename(doc_id: str) -> str:
    return quote(doc_id, safe="")


def _load_hf_rows(
    dataset_id: str,
    *,
    split: str,
    config_name: str | None = None,
) -> list[dict[str, object]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, config_name, split=split)
    return [dict(row) for row in dataset]


def _uses_split_configs(metadata: Mapping[str, object]) -> bool:
    config_names = set(_list_dataset_configs(metadata))
    return {"corpus", "queries", "qrels"}.issubset(config_names)


def _uses_beir_separate_qrels_layout(
    *,
    dataset_id: str,
    metadata: Mapping[str, object],
) -> bool:
    config_names = set(_list_dataset_configs(metadata))
    return dataset_id.startswith("BeIR/") and {"corpus", "queries"}.issubset(config_names) and "qrels" not in config_names


def _list_dataset_configs(metadata: Mapping[str, object]) -> list[str]:
    configs = metadata.get("cardData", {}).get("configs", [])
    if not isinstance(configs, list):
        return []
    return [
        str(config["config_name"])
        for config in configs
        if isinstance(config, Mapping) and "config_name" in config
    ]


def _load_qrels_rows(
    *,
    dataset_id: str,
    config_name: str,
    metadata: Mapping[str, object],
) -> list[dict[str, object]]:
    from datasets import load_dataset

    try:
        qrels_split = load_dataset(dataset_id, config_name, split="qrels")
        return [dict(row) for row in qrels_split]
    except Exception:
        qrels_path = _derive_qrels_path(metadata=metadata, config_name=config_name)
        if not qrels_path:
            raise ValueError(
                f"Could not find qrels for dataset {dataset_id!r} config {config_name!r}. "
                "The dataset does not expose a qrels split and no qrels.tsv path was found in Hub metadata."
            ) from None
        return _read_tsv_from_hub(dataset_id=dataset_id, relative_path=qrels_path)


def _derive_qrels_path(*, metadata: Mapping[str, object], config_name: str) -> str | None:
    configs = metadata.get("cardData", {}).get("configs", [])
    if not isinstance(configs, list):
        return None
    for config in configs:
        if not isinstance(config, Mapping):
            continue
        if str(config.get("config_name")) != config_name:
            continue
        data_files = config.get("data_files", [])
        if not isinstance(data_files, list):
            continue
        for data_file in data_files:
            if not isinstance(data_file, Mapping):
                continue
            if str(data_file.get("split")) != "queries":
                continue
            query_path = data_file.get("path")
            if not isinstance(query_path, str):
                continue
            return query_path.rsplit("/", 1)[0] + "/qrels.tsv"
    return None


def _read_tsv_from_hub(*, dataset_id: str, relative_path: str) -> list[dict[str, object]]:
    dataset_path = quote(dataset_id, safe="/")
    encoded_relative_path = "/".join(quote(part, safe="") for part in relative_path.split("/"))
    url = (
        f"https://huggingface.co/datasets/{dataset_path}/resolve/main/"
        f"{encoded_relative_path}?download=true"
    )
    with urlopen(url) as response:
        content = response.read().decode("utf-8")
    return list(csv.DictReader(content.splitlines(), delimiter="\t"))


def _fetch_hf_dataset_metadata(dataset_id: str) -> dict[str, object]:
    dataset_path = quote(dataset_id, safe="/")
    url = f"https://huggingface.co/api/datasets/{dataset_path}"
    try:
        with urlopen(url) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        raise ValueError(f"Failed to fetch Hugging Face metadata for dataset {dataset_id!r}: {exc}") from exc
    except URLError as exc:
        raise ValueError(f"Failed to reach Hugging Face for dataset {dataset_id!r}: {exc}") from exc
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected metadata payload for dataset {dataset_id!r}.")
    return data


def _filter_query_rows_to_qrels(
    *,
    query_rows: Iterable[Mapping[str, object]],
    qrel_rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    qrel_query_ids = {_require_identifier(row, "query-id") for row in qrel_rows}
    filtered = [
        dict(row)
        for row in query_rows
        if _require_identifier_with_aliases(row, "query id", "id", "_id") in qrel_query_ids
    ]
    if not filtered:
        raise ValueError("No query rows matched the selected qrels split.")
    return filtered


def _compose_text(row: Mapping[str, object]) -> str:
    title = _optional_string(row, "title").strip()
    text = _require_string(row, "text").strip()
    if title and text:
        return f"{title}\n\n{text}"
    if title:
        return title
    return text


def _require_string(row: Mapping[str, object], key: str) -> str:
    value = row[key]
    if not isinstance(value, str):
        raise ValueError(f"Expected {key!r} to be a string, got {type(value).__name__}")
    return value


def _require_identifier(row: Mapping[str, object], key: str) -> str:
    value = row[key]
    if not isinstance(value, (str, int)):
        raise ValueError(f"Expected {key!r} to be a string or int, got {type(value).__name__}")
    return str(value)


def _optional_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"Expected {key!r} to be a string, got {type(value).__name__}")
    return value


def _require_string_with_aliases(row: Mapping[str, object], label: str, *keys: str) -> str:
    for key in keys:
        if key in row:
            return _require_string(row, key)
    raise ValueError(f"Expected {label} in one of: {', '.join(keys)}")


def _require_identifier_with_aliases(row: Mapping[str, object], label: str, *keys: str) -> str:
    for key in keys:
        if key in row:
            return _require_identifier(row, key)
    raise ValueError(f"Expected {label} in one of: {', '.join(keys)}")


if __name__ == "__main__":
    raise SystemExit(main())
