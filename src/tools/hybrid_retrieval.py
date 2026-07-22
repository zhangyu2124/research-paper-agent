"""Pure ranking helpers for local paper hybrid retrieval."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

RankedItems = list[tuple[str, float]]


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize English terms, identifiers, and contiguous Chinese text."""
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", text.lower())


def chunk_retrieval_text(chunk: Mapping[str, Any]) -> str:
    """Build the passage used by sparse retrieval and cross-encoder reranking."""
    title = str(chunk.get("title") or "")
    section = str(chunk.get("section") or "")
    content = str(chunk.get("content") or "")
    return f"Title: {title}\nSection: {section}\nContent: {content}"


def dense_ranking(
    chunk_ids: Sequence[str],
    similarities: Sequence[float],
    top_n: int,
) -> RankedItems:
    """Return chunk IDs ordered by descending embedding similarity."""
    pairs = zip(chunk_ids, similarities, strict=True)
    return sorted(
        ((str(chunk_id), float(score)) for chunk_id, score in pairs),
        key=lambda item: (-item[1], item[0]),
    )[:top_n]


def bm25_ranking(
    *,
    query_tokens: Sequence[str],
    chunk_ids: Sequence[str],
    bm25: Any,
    top_n: int,
) -> RankedItems:
    """Return positive-scoring BM25 results in deterministic order."""
    scores = bm25.get_scores(list(query_tokens))
    ranked = dense_ranking(chunk_ids, scores, len(chunk_ids))
    return [item for item in ranked if item[1] > 0][:top_n]


def reciprocal_rank_fusion(
    rankings: Mapping[str, RankedItems],
    *,
    rrf_k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Fuse rankings without comparing their incompatible raw score scales."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1.")

    weights = weights or {}
    fused: dict[str, dict[str, Any]] = {}
    for source, ranking in rankings.items():
        weight = float(weights.get(source, 1.0))
        for rank, (chunk_id, raw_score) in enumerate(ranking, 1):
            hit = fused.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "rrf_score": 0.0,
                    "source_ranks": {},
                    "source_scores": {},
                },
            )
            hit["rrf_score"] += weight / (rrf_k + rank)
            hit["source_ranks"][source] = rank
            hit["source_scores"][source] = float(raw_score)

    return sorted(
        fused.values(),
        key=lambda hit: (-hit["rrf_score"], hit["chunk_id"]),
    )


def content_deduplication_key(chunk: Mapping[str, Any]) -> str:
    """Identify duplicate PDF chunks while ignoring whitespace differences."""
    title = re.sub(r"\s+", " ", str(chunk.get("title") or "").lower()).strip()
    content = re.sub(r"\s+", " ", str(chunk.get("content") or "").lower()).strip()
    return hashlib.sha256(f"{title}\n{content}".encode()).hexdigest()


def deduplicate_ranked_hits(
    hits: Sequence[dict[str, Any]],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Keep the highest-ranked copy and record equivalent chunk IDs."""
    kept: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for source_hit in hits:
        chunk_id = str(source_hit["chunk_id"])
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        key = content_deduplication_key(chunk)
        existing = by_key.get(key)
        if existing is not None:
            existing.setdefault("duplicate_chunk_ids", []).append(chunk_id)
            continue

        hit = dict(source_hit)
        hit["duplicate_chunk_ids"] = []
        by_key[key] = hit
        kept.append(hit)
        if len(kept) == limit:
            break
    return kept
