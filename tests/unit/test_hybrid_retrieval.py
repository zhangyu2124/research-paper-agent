"""Unit tests for hybrid paper retrieval ranking helpers."""

from rank_bm25 import BM25Okapi

from src.tools.hybrid_retrieval import (
    bm25_ranking,
    deduplicate_ranked_hits,
    dense_ranking,
    reciprocal_rank_fusion,
    tokenize_for_bm25,
)


def test_bm25_ranking_prefers_exact_research_terms():
    chunk_ids = ["chunk-agent", "chunk-rag", "chunk-summary"]
    corpus = [
        tokenize_for_bm25("multi agent collaboration and planning"),
        tokenize_for_bm25("recursive retrieval augmented generation"),
        tokenize_for_bm25("long document summarization"),
    ]
    bm25 = BM25Okapi(corpus)

    ranked = bm25_ranking(
        query_tokens=tokenize_for_bm25("recursive retrieval"),
        chunk_ids=chunk_ids,
        bm25=bm25,
        top_n=3,
    )

    assert ranked[0][0] == "chunk-rag"
    assert all(score > 0 for _, score in ranked)


def test_dense_ranking_is_deterministic_for_equal_scores():
    ranked = dense_ranking(
        ["chunk-b", "chunk-a", "chunk-c"],
        [0.8, 0.8, 0.2],
        top_n=3,
    )

    assert [chunk_id for chunk_id, _ in ranked] == [
        "chunk-a",
        "chunk-b",
        "chunk-c",
    ]


def test_rrf_rewards_chunks_retrieved_by_both_sources():
    fused = reciprocal_rank_fusion(
        {
            "dense": [("shared", 0.82), ("dense-only", 0.81)],
            "bm25": [("bm25-only", 7.2), ("shared", 6.8)],
        }
    )

    assert fused[0]["chunk_id"] == "shared"
    assert fused[0]["source_ranks"] == {"dense": 1, "bm25": 2}
    assert fused[0]["source_scores"] == {"dense": 0.82, "bm25": 6.8}


def test_duplicate_pdf_chunks_do_not_fill_multiple_result_slots():
    hits = [
        {"chunk_id": "duplicate-a"},
        {"chunk_id": "duplicate-b"},
        {"chunk_id": "different"},
    ]
    chunks = {
        "duplicate-a": {"title": "RAPTOR", "content": "same evidence"},
        "duplicate-b": {"title": "RAPTOR", "content": "same   evidence"},
        "different": {"title": "LongAgent", "content": "other evidence"},
    }

    deduplicated = deduplicate_ranked_hits(hits, chunks, limit=3)

    assert [hit["chunk_id"] for hit in deduplicated] == [
        "duplicate-a",
        "different",
    ]
    assert deduplicated[0]["duplicate_chunk_ids"] == ["duplicate-b"]
