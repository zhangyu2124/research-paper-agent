"""Paper-level retrieval metrics with support for equivalent document IDs."""

from __future__ import annotations

import math
from collections.abc import Sequence


def unique_paper_ids(hits: Sequence[dict[str, object]]) -> list[str]:
    """Collapse chunk results to a paper ranking while preserving first rank."""
    seen: set[str] = set()
    ranked: list[str] = []
    for hit in hits:
        paper_id = str(hit["paper_id"])
        if paper_id not in seen:
            seen.add(paper_id)
            ranked.append(paper_id)
    return ranked


def _matched_group_indexes(
    ranked_paper_ids: Sequence[str],
    relevant_groups: Sequence[Sequence[str]],
    cutoff: int,
) -> list[int | None]:
    groups = [set(group) for group in relevant_groups]
    matched: set[int] = set()
    matches: list[int | None] = []
    for paper_id in ranked_paper_ids[:cutoff]:
        group_index = next(
            (
                index
                for index, group in enumerate(groups)
                if index not in matched and paper_id in group
            ),
            None,
        )
        if group_index is not None:
            matched.add(group_index)
        matches.append(group_index)
    return matches


def recall_at_k(
    ranked_paper_ids: Sequence[str],
    relevant_groups: Sequence[Sequence[str]],
    k: int,
) -> float:
    """Measure how many logical relevant papers occur in the top-k ranking."""
    if not relevant_groups:
        return 0.0
    matches = _matched_group_indexes(ranked_paper_ids, relevant_groups, k)
    return len({match for match in matches if match is not None}) / len(relevant_groups)


def reciprocal_rank(
    ranked_paper_ids: Sequence[str],
    relevant_groups: Sequence[Sequence[str]],
    k: int,
) -> float:
    """Return reciprocal rank of the first relevant paper within k results."""
    matches = _matched_group_indexes(ranked_paper_ids, relevant_groups, k)
    for rank, match in enumerate(matches, 1):
        if match is not None:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_paper_ids: Sequence[str],
    relevant_groups: Sequence[Sequence[str]],
    k: int,
) -> float:
    """Compute binary nDCG while counting equivalent IDs only once."""
    if not relevant_groups:
        return 0.0
    matches = _matched_group_indexes(ranked_paper_ids, relevant_groups, k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, match in enumerate(matches, 1)
        if match is not None
    )
    ideal_count = min(len(relevant_groups), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0
