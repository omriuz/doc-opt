from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class QuerySplit:
    train_indices: np.ndarray
    test_indices: np.ndarray


def split_query_indices(
    queries: list[str],
    *,
    seed: int,
    test_size: float,
) -> QuerySplit:
    query_indices_by_text: dict[str, list[int]] = defaultdict(list)
    for query_index, query in enumerate(queries):
        query_indices_by_text[str(query).strip()].append(query_index)

    grouped_indices = [query_indices_by_text[text] for text in sorted(query_indices_by_text)]
    if len(grouped_indices) < 2:
        raise ValueError("Need at least two distinct query texts for a query-disjoint train/test split.")

    shuffled_group_positions = np.random.default_rng(seed).permutation(len(grouped_indices))
    shuffled_groups = [grouped_indices[position] for position in shuffled_group_positions]
    target_train_size = int(len(queries) * (1 - test_size))
    split_index = _choose_split_index(shuffled_groups, target_train_size=target_train_size)

    train_indices = np.asarray(
        [query_index for group in shuffled_groups[:split_index] for query_index in group],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [query_index for group in shuffled_groups[split_index:] for query_index in group],
        dtype=np.int64,
    )
    if train_indices.size == 0 or test_indices.size == 0:
        raise ValueError("Query-disjoint splitting produced an empty train or test split.")

    train_queries = {queries[int(query_index)].strip() for query_index in train_indices}
    test_queries = {queries[int(query_index)].strip() for query_index in test_indices}
    if train_queries & test_queries:
        raise ValueError("Train/test split is contaminated by overlapping query texts.")

    return QuerySplit(train_indices=train_indices, test_indices=test_indices)


def _choose_split_index(groups: list[list[int]], *, target_train_size: int) -> int:
    best_split_index = 1
    best_distance = None
    running_train_size = 0

    for split_index in range(1, len(groups)):
        running_train_size += len(groups[split_index - 1])
        distance = abs(running_train_size - target_train_size)
        if best_distance is None or distance < best_distance:
            best_split_index = split_index
            best_distance = distance

    return best_split_index
