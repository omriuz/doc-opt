from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from doc_opt.data import load_ds1000_dataset
from doc_opt.prepare_dataset import convert_retrieval_rows, write_prepared_dataset


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


if __name__ == "__main__":
    unittest.main()
