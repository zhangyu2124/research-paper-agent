from src.evaluation.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    unique_paper_ids,
)


def test_unique_paper_ids_collapses_chunk_results() -> None:
    hits = [
        {"paper_id": "paper-a", "chunk_id": "a-1"},
        {"paper_id": "paper-a", "chunk_id": "a-2"},
        {"paper_id": "paper-b", "chunk_id": "b-1"},
    ]
    assert unique_paper_ids(hits) == ["paper-a", "paper-b"]


def test_metrics_accept_equivalent_duplicate_document_ids() -> None:
    ranked = ["irrelevant", "paper-copy-2"]
    relevant_groups = [["paper-copy-1", "paper-copy-2"]]

    assert recall_at_k(ranked, relevant_groups, 5) == 1.0
    assert reciprocal_rank(ranked, relevant_groups, 10) == 0.5
    assert round(ndcg_at_k(ranked, relevant_groups, 10), 6) == 0.63093


def test_ndcg_does_not_count_equivalent_ids_twice() -> None:
    ranked = ["paper-copy-1", "paper-copy-2", "paper-b"]
    relevant_groups = [["paper-copy-1", "paper-copy-2"], ["paper-b"]]

    assert ndcg_at_k(ranked, relevant_groups, 10) < 1.0
    assert recall_at_k(ranked, relevant_groups, 5) == 1.0
