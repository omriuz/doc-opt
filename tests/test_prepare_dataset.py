from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from doc_opt.data import load_ds1000_dataset
from doc_opt.prepare_dataset import PreparedDataset, convert_retrieval_rows, prepare_hf_dataset, write_prepared_dataset


class PrepareDatasetTest(unittest.TestCase):
    def test_convert_retrieval_rows_matches_repo_layout(self) -> None:
        prepared = convert_retrieval_rows(
            corpus_rows=[
                {"id": "000000001", "text": "doc one"},
                {"id": "000000000", "text": "doc zero"},
            ],
            query_rows=[
                {"id": "000000001", "text": "second query"},
                {"id": "000000000", "text": "first query"},
            ],
            qrel_rows=[
                {"query-id": "000000001", "corpus-id": "000000001", "score": 1},
                {"query-id": "000000000", "corpus-id": "000000000", "score": 1},
            ],
        )

        self.assertEqual(
            prepared.documents,
            [("000000000", "doc zero"), ("000000001", "doc one")],
        )
        self.assertEqual(
            prepared.benchmark_rows,
            [
                {
                    "question": "first query",
                    "correct_answer_document_ids": "{'000000000': 1}",
                },
                {
                    "question": "second query",
                    "correct_answer_document_ids": "{'000000001': 1}",
                },
            ],
        )

    def test_convert_retrieval_rows_accepts_underscore_ids(self) -> None:
        prepared = convert_retrieval_rows(
            corpus_rows=[{"_id": "doc-a", "text": "alpha"}],
            query_rows=[{"_id": "query-a", "text": "needle"}],
            qrel_rows=[{"query-id": "query-a", "corpus-id": "doc-a", "score": 1}],
        )

        self.assertEqual(prepared.documents, [("doc-a", "alpha")])
        self.assertEqual(
            prepared.benchmark_rows,
            [{"question": "needle", "correct_answer_document_ids": "{'doc-a': 1}"}],
        )

    def test_convert_retrieval_rows_requires_positive_qrel(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing positive qrels"):
            convert_retrieval_rows(
                corpus_rows=[{"id": "000000000", "text": "doc zero"}],
                query_rows=[{"id": "000000000", "text": "first query"}],
                qrel_rows=[],
            )

    def test_convert_retrieval_rows_preserves_multiple_positive_docs(self) -> None:
        prepared = convert_retrieval_rows(
            corpus_rows=[
                {"id": "000000000", "text": "doc zero"},
                {"id": "000000001", "text": "doc one"},
            ],
            query_rows=[{"id": "000000000", "text": "first query"}],
            qrel_rows=[
                {"query-id": "000000000", "corpus-id": "000000001", "score": 1},
                {"query-id": "000000000", "corpus-id": "000000000", "score": 1},
            ],
        )

        self.assertEqual(
            prepared.benchmark_rows,
            [
                {
                    "question": "first query",
                    "correct_answer_document_ids": "{'000000000': 1, '000000001': 1}",
                }
            ],
        )

    def test_convert_retrieval_rows_accepts_integer_qrel_ids(self) -> None:
        prepared = convert_retrieval_rows(
            corpus_rows=[{"id": "31715818", "text": "doc body"}],
            query_rows=[{"id": "0", "text": "claim text"}],
            qrel_rows=[{"query-id": 0, "corpus-id": 31715818, "score": 1}],
        )

        self.assertEqual(prepared.documents, [("31715818", "doc body")])
        self.assertEqual(
            prepared.benchmark_rows,
            [{"question": "claim text", "correct_answer_document_ids": "{'31715818': 1}"}],
        )

    def test_convert_retrieval_rows_includes_title_in_flattened_text(self) -> None:
        prepared = convert_retrieval_rows(
            corpus_rows=[{"id": "doc-a", "title": "Alpha title", "text": "alpha body"}],
            query_rows=[{"id": "query-a", "title": "", "text": "needle"}],
            qrel_rows=[{"query-id": "query-a", "corpus-id": "doc-a", "score": 1}],
        )

        self.assertEqual(prepared.documents, [("doc-a", "Alpha title\n\nalpha body")])
        self.assertEqual(
            prepared.benchmark_rows,
            [{"question": "needle", "correct_answer_document_ids": "{'doc-a': 1}"}],
        )

    def test_write_and_load_roundtrip_for_slash_doc_ids(self) -> None:
        prepared = convert_retrieval_rows(
            corpus_rows=[{"id": "american-math-monthly___2008___11212/11220", "text": "proof text"}],
            query_rows=[{"id": "q1", "text": "math query"}],
            qrel_rows=[
                {
                    "query-id": "q1",
                    "corpus-id": "american-math-monthly___2008___11212/11220",
                    "score": 1,
                }
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            write_prepared_dataset(prepared, output_dir=output_dir, overwrite=False)
            bundle = load_ds1000_dataset(output_dir)

        self.assertEqual(bundle.doc_ids, ["american-math-monthly___2008___11212/11220"])
        self.assertEqual(bundle.docs, ["proof text"])
        self.assertEqual(bundle.query2doc, {0: 0})

    def test_prepare_hf_dataset_supports_beir_sibling_qrels_dataset(self) -> None:
        metadata = {"cardData": {"configs": [{"config_name": "corpus"}, {"config_name": "queries"}]}}
        corpus_rows = [{"_id": "doc-a", "title": "Alpha title", "text": "alpha body"}]
        query_rows = [
            {"_id": "query-a", "title": "", "text": "needle"},
            {"_id": "query-b", "title": "", "text": "unused"},
        ]
        qrel_rows = [{"query-id": "query-a", "corpus-id": "doc-a", "score": 1}]

        with patch("doc_opt.prepare_dataset._fetch_hf_dataset_metadata", return_value=metadata), patch(
            "doc_opt.prepare_dataset._load_hf_rows",
            side_effect=[corpus_rows, query_rows, qrel_rows],
        ), patch(
            "doc_opt.prepare_dataset.convert_retrieval_rows",
            return_value=PreparedDataset(documents=[], benchmark_rows=[]),
        ) as convert_mock, patch("doc_opt.prepare_dataset.write_prepared_dataset"):
            prepare_hf_dataset("BeIR/nfcorpus", output_dir=Path("data/nfcorpus/test"), split="test")

        convert_mock.assert_called_once_with(
            corpus_rows=corpus_rows,
            query_rows=[{"_id": "query-a", "title": "", "text": "needle"}],
            qrel_rows=qrel_rows,
        )

    def test_prepare_hf_dataset_filters_integer_qrel_query_ids(self) -> None:
        metadata = {"cardData": {"configs": [{"config_name": "corpus"}, {"config_name": "queries"}]}}
        corpus_rows = [{"_id": "31715818", "title": "Alpha title", "text": "alpha body"}]
        query_rows = [
            {"_id": "0", "title": "", "text": "needle"},
            {"_id": "1", "title": "", "text": "unused"},
        ]
        qrel_rows = [{"query-id": 0, "corpus-id": 31715818, "score": 1}]

        with patch("doc_opt.prepare_dataset._fetch_hf_dataset_metadata", return_value=metadata), patch(
            "doc_opt.prepare_dataset._load_hf_rows",
            side_effect=[corpus_rows, query_rows, qrel_rows],
        ), patch(
            "doc_opt.prepare_dataset.convert_retrieval_rows",
            return_value=PreparedDataset(documents=[], benchmark_rows=[]),
        ) as convert_mock, patch("doc_opt.prepare_dataset.write_prepared_dataset"):
            prepare_hf_dataset("BeIR/scifact", output_dir=Path("data/scifact/test"), split="test")

        convert_mock.assert_called_once_with(
            corpus_rows=corpus_rows,
            query_rows=[{"_id": "0", "title": "", "text": "needle"}],
            qrel_rows=qrel_rows,
        )


if __name__ == "__main__":
    unittest.main()
