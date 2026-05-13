from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from doc_opt.splits import split_query_indices


class QuerySplitTest(unittest.TestCase):
    def test_groups_identical_query_texts_on_one_side_of_split(self) -> None:
        queries = [
            "duplicate alpha",
            "beta",
            "duplicate alpha",
            "gamma",
            "beta",
            "delta",
        ]

        split = split_query_indices(queries, seed=42, test_size=0.5)

        train_indices = set(map(int, split.train_indices))
        test_indices = set(map(int, split.test_indices))
        self.assertFalse(train_indices & test_indices)

        train_queries = {queries[index] for index in train_indices}
        test_queries = {queries[index] for index in test_indices}
        self.assertFalse(train_queries & test_queries)
        self.assertTrue({0, 2}.issubset(train_indices) or {0, 2}.issubset(test_indices))
        self.assertTrue({1, 4}.issubset(train_indices) or {1, 4}.issubset(test_indices))

    def test_split_is_deterministic_for_same_seed(self) -> None:
        queries = ["q0", "q1", "q0", "q2", "q3", "q4"]

        first = split_query_indices(queries, seed=7, test_size=0.4)
        second = split_query_indices(queries, seed=7, test_size=0.4)

        self.assertListEqual(first.train_indices.tolist(), second.train_indices.tolist())
        self.assertListEqual(first.test_indices.tolist(), second.test_indices.tolist())

    def test_requires_two_distinct_query_texts(self) -> None:
        queries = ["same query", "same query"]

        with self.assertRaisesRegex(ValueError, "distinct query texts"):
            split_query_indices(queries, seed=42, test_size=0.5)


if __name__ == "__main__":
    unittest.main()
