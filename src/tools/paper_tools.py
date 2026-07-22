"""Local research paper search tools for the paper agent."""

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

# Retrieval models are installed under models/ and must never fetch at request time.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain.tools import tool

from src.tools.hybrid_retrieval import (
    bm25_ranking,
    chunk_retrieval_text,
    deduplicate_ranked_hits,
    dense_ranking,
    reciprocal_rank_fusion,
    tokenize_for_bm25,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER_LIBRARY_PATH = PROJECT_ROOT / "data" / "papers" / "papers.jsonl"
DEFAULT_PAPER_CHUNKS_PATH = PROJECT_ROOT / "data" / "papers" / "paper_chunks.jsonl"
DEFAULT_PAPER_VECTOR_INDEX_PATH = PROJECT_ROOT / "data" / "papers" / "paper_vector_index.npz"
DEFAULT_EMBEDDING_MODEL_PATH = PROJECT_ROOT / "models" / "gte-multilingual-base"
DEFAULT_RERANKER_MODEL_PATH = PROJECT_ROOT / "models" / "bge-reranker-v2-m3"
DEFAULT_EMBEDDING_MAX_SEQ_LENGTH = 512
DEFAULT_RERANKER_MAX_SEQ_LENGTH = 512
MAX_TOP_K = 10
MAX_RETRIEVAL_CANDIDATES = 100
MAX_CHUNK_TEXT_CHARS = 1100

QUERY_ALIASES = {
    "\u8bba\u6587": "paper",
    "\u68c0\u7d22": "retrieval",
    "\u589e\u5f3a": "augmented",
    "\u751f\u6210": "generation",
    "\u667a\u80fd\u4f53": "agent",
    "\u591a\u667a\u80fd\u4f53": "multi-agent",
    "\u5de5\u5177\u8c03\u7528": "tool use",
    "\u53cd\u601d": "reflection",
    "\u8bb0\u5fc6": "memory",
    "\u89c4\u5212": "planning",
    "\u7efc\u8ff0": "survey",
    "\u5e7b\u89c9": "factuality",
    "\u7ea0\u9519": "corrective",
    "\u81ea\u9002\u5e94": "adaptive",
    "\u76f8\u5173\u5de5\u4f5c": "related work",
}


def _paper_library_path() -> Path:
    """Return the configured paper library path."""
    configured = os.getenv("PAPER_LIBRARY_PATH")
    if not configured:
        return DEFAULT_PAPER_LIBRARY_PATH

    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _paper_chunks_path() -> Path:
    """Return the configured paper chunk path."""
    configured = os.getenv("PAPER_CHUNKS_PATH")
    if not configured:
        return DEFAULT_PAPER_CHUNKS_PATH

    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _paper_vector_index_path() -> Path:
    """Return the configured vector index path."""
    configured = os.getenv("PAPER_VECTOR_INDEX_PATH")
    if not configured:
        return DEFAULT_PAPER_VECTOR_INDEX_PATH

    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _embedding_model_path() -> Path:
    """Return the configured local embedding model path."""
    configured = os.getenv("EMBEDDING_MODEL_PATH")
    if not configured:
        return DEFAULT_EMBEDDING_MODEL_PATH

    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _reranker_model_path() -> Path:
    """Return the configured local cross-encoder reranker path."""
    configured = os.getenv("RERANKER_MODEL_PATH")
    if not configured:
        return DEFAULT_RERANKER_MODEL_PATH

    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _configure_huggingface_cache() -> None:
    """Keep embedding model caches inside the project by default."""
    cache_dir = PROJECT_ROOT / "models" / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir / "sentence_transformers"))


def _configure_offline_model_loading() -> None:
    """Prevent locally installed retrieval models from checking the network."""
    _configure_huggingface_cache()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


@lru_cache(maxsize=1)
def _load_papers() -> list[dict[str, Any]]:
    """Load papers from a JSONL file."""
    path = _paper_library_path()
    if not path.exists():
        raise FileNotFoundError(f"Paper library not found: {path}")

    papers: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                paper = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in paper library line {line_number}: {exc}"
                ) from exc
            papers.append(paper)

    return papers


@lru_cache(maxsize=1)
def _load_chunks() -> list[dict[str, Any]]:
    """Load PDF text chunks from a JSONL file."""
    path = _paper_chunks_path()
    if not path.exists():
        raise FileNotFoundError(f"Paper chunks not found: {path}")

    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                chunk = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in paper chunks line {line_number}: {exc}"
                ) from exc
            chunks.append(chunk)

    return chunks


@lru_cache(maxsize=1)
def _load_chunks_by_id() -> dict[str, dict[str, Any]]:
    """Load PDF chunks keyed by chunk_id."""
    return {str(chunk.get("chunk_id")): chunk for chunk in _load_chunks()}


@lru_cache(maxsize=1)
def _load_vector_index() -> tuple[list[str], Any]:
    """Load the local vector index arrays."""
    path = _paper_vector_index_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Paper vector index not found: {path}. "
            "Run scripts/build_paper_vector_index.py first."
        )

    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        chunk_ids = [str(item) for item in data["chunk_ids"].tolist()]
        embeddings = data["embeddings"].astype("float32")
    return chunk_ids, embeddings


@lru_cache(maxsize=1)
def _load_embedding_model() -> Any:
    """Load the local sentence embedding model lazily."""
    _configure_offline_model_loading()
    from sentence_transformers import SentenceTransformer

    model_path = _embedding_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Embedding model not found: {model_path}. "
            "Run scripts/download_embedding_model.py first."
        )
    model = SentenceTransformer(str(model_path), trust_remote_code=True)
    max_seq_length = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", DEFAULT_EMBEDDING_MAX_SEQ_LENGTH))
    model.max_seq_length = max_seq_length
    return model


@lru_cache(maxsize=1)
def _load_bm25_index() -> tuple[list[str], Any]:
    """Build a lightweight in-memory BM25 index for the local chunk corpus."""
    from rank_bm25 import BM25Okapi

    chunks = _load_chunks()
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    tokenized_corpus = [
        tokenize_for_bm25(chunk_retrieval_text(chunk))
        for chunk in chunks
    ]
    return chunk_ids, BM25Okapi(tokenized_corpus)


@lru_cache(maxsize=1)
def _load_reranker_model() -> Any:
    """Load the local multilingual cross-encoder only when reranking is used."""
    _configure_offline_model_loading()
    from sentence_transformers import CrossEncoder

    model_path = _reranker_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Reranker model not found: {model_path}. "
            "Run scripts/download_reranker_model.py first."
        )
    return CrossEncoder(
        str(model_path),
        max_length=int(
            os.getenv(
                "RERANKER_MAX_SEQ_LENGTH",
                str(DEFAULT_RERANKER_MAX_SEQ_LENGTH),
            )
        ),
        trust_remote_code=True,
        tokenizer_args={"local_files_only": True},
        automodel_args={"local_files_only": True},
    )


def _expand_query(query: str) -> str:
    """Add English aliases for common Chinese research terms."""
    expanded = [query]
    for chinese_term, english_alias in QUERY_ALIASES.items():
        if chinese_term in query:
            expanded.append(english_alias)
    return " ".join(expanded)


def _tokenize(text: str) -> list[str]:
    """Tokenize text into simple lowercase terms."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _paper_text(paper: dict[str, Any], fields: list[str]) -> str:
    values: list[str] = []
    for field in fields:
        value = paper.get(field, "")
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return " ".join(values)


def _score_paper(query: str, paper: dict[str, Any]) -> float:
    expanded_query = _expand_query(query)
    query_terms = set(_tokenize(expanded_query))
    if not query_terms:
        return 0.0

    weighted_fields = [
        (["title"], 5.0),
        (["topics"], 4.0),
        (["abstract"], 2.0),
        (["notes"], 2.0),
        (["text_preview"], 1.5),
        (["authors", "venue", "year", "source_file", "source_id"], 1.0),
    ]

    score = 0.0
    lower_query = expanded_query.lower().strip()
    for fields, weight in weighted_fields:
        text = _paper_text(paper, fields)
        lower_text = text.lower()
        field_terms = set(_tokenize(text))
        score += len(query_terms & field_terms) * weight
        if lower_query and lower_query in lower_text:
            score += weight * 2

    return score


def _score_chunk(query: str, chunk: dict[str, Any]) -> float:
    expanded_query = _expand_query(query)
    query_terms = set(_tokenize(expanded_query))
    if not query_terms:
        return 0.0

    weighted_fields = [
        (["title"], 4.0),
        (["section"], 3.0),
        (["content"], 1.5),
        (["paper_id", "source_file"], 1.0),
    ]

    score = 0.0
    lower_query = expanded_query.lower().strip()
    for fields, weight in weighted_fields:
        text = _paper_text(chunk, fields)
        lower_text = text.lower()
        field_terms = set(_tokenize(text))
        score += len(query_terms & field_terms) * weight
        if lower_query and lower_query in lower_text:
            score += weight * 3

    return score


def _format_paper_result(index: int, paper: dict[str, Any], score: float | None = None) -> str:
    topics = paper.get("topics", [])
    topic_text = ", ".join(topics) if isinstance(topics, list) else str(topics)
    score_line = f"Relevance Score: {score:.1f}\n" if score is not None else ""
    return (
        f"Result {index}:\n"
        f"Paper ID: {paper.get('paper_id', 'unknown')}\n"
        f"Title: {paper.get('title', 'Untitled')}\n"
        f"Authors: {paper.get('authors', 'Unknown')}\n"
        f"Year: {paper.get('year', 'Unknown')}\n"
        f"Venue: {paper.get('venue', 'Unknown')}\n"
        f"Topics: {topic_text}\n"
        f"{score_line}"
        f"Abstract: {paper.get('abstract', '')}\n"
        f"Notes: {paper.get('notes', '')}"
    )


def _format_chunk_result(index: int, chunk: dict[str, Any], score: float | None = None) -> str:
    content = str(chunk.get("content", ""))
    if len(content) > MAX_CHUNK_TEXT_CHARS:
        content = content[:MAX_CHUNK_TEXT_CHARS].rstrip() + "..."

    section = chunk.get("section") or "Unknown"
    score_line = f"Relevance Score: {score:.1f}\n" if score is not None else ""
    return (
        f"Chunk Result {index}:\n"
        f"Chunk ID: {chunk.get('chunk_id', 'unknown')}\n"
        f"Paper ID: {chunk.get('paper_id', 'unknown')}\n"
        f"Title: {chunk.get('title', 'Untitled')}\n"
        f"Page: {chunk.get('page', 'Unknown')}\n"
        f"Section: {section}\n"
        f"Source File: {chunk.get('source_file', 'Unknown')}\n"
        f"{score_line}"
        f"Content: {content}"
    )


def _format_vector_chunk_result(index: int, chunk: dict[str, Any], similarity: float) -> str:
    content = str(chunk.get("content", ""))
    if len(content) > MAX_CHUNK_TEXT_CHARS:
        content = content[:MAX_CHUNK_TEXT_CHARS].rstrip() + "..."

    section = chunk.get("section") or "Unknown"
    return (
        f"Vector Chunk Result {index}:\n"
        f"Chunk ID: {chunk.get('chunk_id', 'unknown')}\n"
        f"Paper ID: {chunk.get('paper_id', 'unknown')}\n"
        f"Title: {chunk.get('title', 'Untitled')}\n"
        f"Page: {chunk.get('page', 'Unknown')}\n"
        f"Section: {section}\n"
        f"Source File: {chunk.get('source_file', 'Unknown')}\n"
        f"Similarity Score: {similarity:.4f}\n"
        f"Content: {content}"
    )


def _dense_chunk_ranking(query: str, top_n: int) -> list[tuple[str, float]]:
    """Retrieve chunk candidates with normalized embedding similarity."""
    model = _load_embedding_model()
    chunk_ids, embeddings = _load_vector_index()
    query_vector = model.encode(
        [_expand_query(query)],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")[0]
    similarities = embeddings @ query_vector
    return dense_ranking(chunk_ids, similarities, top_n)


def _bm25_chunk_ranking(query: str, top_n: int) -> list[tuple[str, float]]:
    """Retrieve exact-term candidates with BM25."""
    chunk_ids, bm25 = _load_bm25_index()
    return bm25_ranking(
        query_tokens=tokenize_for_bm25(_expand_query(query)),
        chunk_ids=chunk_ids,
        bm25=bm25,
        top_n=top_n,
    )


def _single_source_hits(
    source: str,
    ranking: list[tuple[str, float]],
) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk_id,
            "rrf_score": None,
            "source_ranks": {source: rank},
            "source_scores": {source: score},
        }
        for rank, (chunk_id, score) in enumerate(ranking, 1)
    ]


def _rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score query-passage pairs with the local multilingual cross-encoder."""
    if not hits:
        return []
    model = _load_reranker_model()
    pairs = [
        [query, chunk_retrieval_text(chunks_by_id[str(hit["chunk_id"])])]
        for hit in hits
    ]
    scores = model.predict(
        pairs,
        batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "8")),
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    reranked = []
    for hit, score in zip(hits, scores, strict=True):
        updated = dict(hit)
        updated["rerank_score"] = float(score)
        reranked.append(updated)
    return sorted(
        reranked,
        key=lambda hit: (-hit["rerank_score"], -float(hit.get("rrf_score") or 0.0), hit["chunk_id"]),
    )


def retrieve_paper_evidence(
    query: str,
    *,
    top_k: int = 6,
    candidate_k: int = 20,
    mode: str = "hybrid_rerank",
) -> list[dict[str, Any]]:
    """Run dense, BM25, hybrid, or hybrid-plus-reranker retrieval."""
    supported_modes = {"dense", "bm25", "hybrid", "hybrid_rerank"}
    if mode not in supported_modes:
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    if not query.strip():
        raise ValueError("query must not be empty.")

    top_k = max(1, min(int(top_k), MAX_RETRIEVAL_CANDIDATES))
    candidate_k = max(top_k, min(int(candidate_k), MAX_RETRIEVAL_CANDIDATES))
    retrieval_pool = min(MAX_RETRIEVAL_CANDIDATES, candidate_k * 2)
    chunks_by_id = _load_chunks_by_id()

    dense_results: list[tuple[str, float]] = []
    bm25_results: list[tuple[str, float]] = []
    if mode in {"dense", "hybrid", "hybrid_rerank"}:
        dense_results = _dense_chunk_ranking(query, retrieval_pool)
    if mode in {"bm25", "hybrid", "hybrid_rerank"}:
        bm25_results = _bm25_chunk_ranking(query, retrieval_pool)

    if mode == "dense":
        ranked_hits = _single_source_hits("dense", dense_results)
    elif mode == "bm25":
        ranked_hits = _single_source_hits("bm25", bm25_results)
    else:
        ranked_hits = reciprocal_rank_fusion(
            {"dense": dense_results, "bm25": bm25_results},
            rrf_k=int(os.getenv("PAPER_RRF_K", "60")),
        )

    candidates = deduplicate_ranked_hits(
        ranked_hits,
        chunks_by_id,
        limit=candidate_k,
    )
    if mode == "hybrid_rerank":
        candidates = _rerank_hits(query, candidates, chunks_by_id)

    results: list[dict[str, Any]] = []
    for rank, hit in enumerate(candidates[:top_k], 1):
        chunk = chunks_by_id[str(hit["chunk_id"])]
        source_ranks = hit.get("source_ranks", {})
        source_scores = hit.get("source_scores", {})
        results.append(
            {
                "rank": rank,
                "mode": mode,
                "chunk_id": str(chunk.get("chunk_id")),
                "paper_id": str(chunk.get("paper_id")),
                "title": str(chunk.get("title") or "Untitled"),
                "page": chunk.get("page"),
                "section": chunk.get("section") or "Unknown",
                "source_file": chunk.get("source_file") or "Unknown",
                "content": str(chunk.get("content") or ""),
                "dense_rank": source_ranks.get("dense"),
                "dense_score": source_scores.get("dense"),
                "bm25_rank": source_ranks.get("bm25"),
                "bm25_score": source_scores.get("bm25"),
                "rrf_score": hit.get("rrf_score"),
                "rerank_score": hit.get("rerank_score"),
                "duplicate_chunk_ids": hit.get("duplicate_chunk_ids", []),
            }
        )
    return results


def _format_hybrid_evidence(result: dict[str, Any]) -> str:
    content = str(result["content"])
    if len(content) > MAX_CHUNK_TEXT_CHARS:
        content = content[:MAX_CHUNK_TEXT_CHARS].rstrip() + "..."

    score_parts = []
    if result.get("dense_score") is not None:
        score_parts.append(
            f"Dense={result['dense_score']:.4f} (rank {result['dense_rank']})"
        )
    if result.get("bm25_score") is not None:
        score_parts.append(
            f"BM25={result['bm25_score']:.4f} (rank {result['bm25_rank']})"
        )
    if result.get("rrf_score") is not None:
        score_parts.append(f"RRF={result['rrf_score']:.6f}")
    if result.get("rerank_score") is not None:
        score_parts.append(f"Reranker={result['rerank_score']:.4f}")

    duplicate_line = ""
    if result.get("duplicate_chunk_ids"):
        duplicate_line = (
            "Equivalent Chunk IDs: "
            + ", ".join(result["duplicate_chunk_ids"])
            + "\n"
        )
    return (
        f"Evidence {result['rank']}:\n"
        f"Retrieval Mode: {result['mode']}\n"
        f"Chunk ID: {result['chunk_id']}\n"
        f"Paper ID: {result['paper_id']}\n"
        f"Title: {result['title']}\n"
        f"Page: {result['page']}\n"
        f"Section: {result['section']}\n"
        f"Source File: {result['source_file']}\n"
        f"Scores: {', '.join(score_parts) or 'Unavailable'}\n"
        f"{duplicate_line}"
        f"Content: {content}"
    )


@tool
def search_paper_evidence(query: str, top_k: int = 6) -> str:
    """Retrieve page-level evidence with Dense + BM25 + RRF + optional reranking.

    Use this as the default evidence search for methods, experiments,
    definitions, limitations, comparisons, and Chinese queries over English PDFs.
    """
    top_k = max(1, min(top_k, MAX_TOP_K))
    reranker_enabled = os.getenv("RERANKER_ENABLED", "false").lower() not in {
        "0",
        "false",
        "no",
    }
    requested_mode = "hybrid_rerank" if reranker_enabled else "hybrid"
    fallback_note = ""
    try:
        results = retrieve_paper_evidence(
            query,
            top_k=top_k,
            candidate_k=max(20, top_k),
            mode=requested_mode,
        )
    except FileNotFoundError as exc:
        if requested_mode != "hybrid_rerank":
            return str(exc)
        results = retrieve_paper_evidence(
            query,
            top_k=top_k,
            candidate_k=max(20, top_k),
            mode="hybrid",
        )
        fallback_note = f"Reranker unavailable; used hybrid retrieval only. {exc}\n\n"

    if not results:
        return "No matching paper evidence was found. Try a broader query."
    return fallback_note + "\n---\n\n".join(
        _format_hybrid_evidence(result) for result in results
    )


@tool
def search_papers(query: str, top_k: int = 5) -> str:
    """Search the local research paper library by topic, method, title, or abstract.

    Use this tool when the user asks for related papers, RAG/Agent concepts,
    paper comparison candidates, surveys, methods, or research directions.
    """
    top_k = max(1, min(top_k, MAX_TOP_K))
    papers = _load_papers()

    scored = [
        (paper, _score_paper(query, paper))
        for paper in papers
    ]
    scored = [item for item in scored if item[1] > 0]
    scored.sort(key=lambda item: item[1], reverse=True)

    if not scored:
        return (
            "No matching papers were found in the local paper library. "
            "Try a broader query such as 'RAG', 'agent', 'multi-agent', or 'retrieval'."
        )

    formatted = [
        _format_paper_result(index, paper, score)
        for index, (paper, score) in enumerate(scored[:top_k], 1)
    ]
    return "\n---\n\n".join(formatted)


@tool
def search_paper_vectors(query: str, top_k: int = 6) -> str:
    """Search PDF chunks with local embedding similarity.

    Use this for semantic retrieval when the user's wording may not exactly
    match the paper text, especially for Chinese questions over English PDFs.
    """
    top_k = max(1, min(top_k, MAX_TOP_K))

    import numpy as np

    try:
        model = _load_embedding_model()
        chunk_ids, embeddings = _load_vector_index()
    except FileNotFoundError as exc:
        return str(exc)

    chunks_by_id = _load_chunks_by_id()

    query_vector = model.encode(
        [_expand_query(query)],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")[0]
    similarities = embeddings @ query_vector
    top_indices = np.argsort(similarities)[::-1][:top_k]

    formatted = []
    for rank, index in enumerate(top_indices, 1):
        chunk_id = chunk_ids[int(index)]
        chunk = chunks_by_id.get(chunk_id)
        if chunk:
            formatted.append(_format_vector_chunk_result(rank, chunk, float(similarities[int(index)])))

    if not formatted:
        return (
            "No matching vector chunks were found. "
            "Check that paper_chunks.jsonl and paper_vector_index.npz were built from the same corpus."
        )
    return "\n---\n\n".join(formatted)


@tool
def get_paper_detail(paper_id: str) -> str:
    """Get the full metadata and notes for a paper by paper_id."""
    papers = _load_papers()
    normalized_id = paper_id.strip().lower()
    for paper in papers:
        if str(paper.get("paper_id", "")).lower() == normalized_id:
            return _format_paper_result(1, paper)

    known_ids = ", ".join(str(paper.get("paper_id", "")) for paper in papers)
    return f"Paper ID '{paper_id}' was not found. Known paper IDs: {known_ids}"


@tool
def list_papers(topic: str | None = None) -> str:
    """List papers in the local library, optionally filtered by a topic keyword."""
    papers = _load_papers()
    if topic:
        topic_terms = set(_tokenize(_expand_query(topic)))
        papers = [
            paper
            for paper in papers
            if topic_terms & set(_tokenize(_paper_text(paper, ["title", "topics", "abstract", "notes"])))
        ]

    if not papers:
        return "No papers matched the requested topic."

    lines = []
    for index, paper in enumerate(papers, 1):
        topics = paper.get("topics", [])
        topic_text = ", ".join(topics) if isinstance(topics, list) else str(topics)
        lines.append(
            f"{index}. {paper.get('paper_id', 'unknown')} | "
            f"{paper.get('title', 'Untitled')} | "
            f"{paper.get('year', 'Unknown')} | "
            f"{topic_text}"
        )
    return "\n".join(lines)


@tool
def search_paper_chunks(query: str, top_k: int = 6) -> str:
    """Search page-level PDF text chunks for detailed evidence.

    Use this after or alongside `search_papers` when the user asks about
    concrete methods, experiments, definitions, limitations, or evidence from
    the paper text.
    """
    top_k = max(1, min(top_k, MAX_TOP_K))
    chunks = _load_chunks()

    scored = [
        (chunk, _score_chunk(query, chunk))
        for chunk in chunks
    ]
    scored = [item for item in scored if item[1] > 0]
    scored.sort(key=lambda item: item[1], reverse=True)

    if not scored:
        return (
            "No matching PDF chunks were found in the local paper library. "
            "Try a broader query or use search_papers to find candidate papers first."
        )

    formatted = [
        _format_chunk_result(index, chunk, score)
        for index, (chunk, score) in enumerate(scored[:top_k], 1)
    ]
    return "\n---\n\n".join(formatted)


@tool
def get_paper_chunk_context(chunk_id: str, window: int = 1) -> str:
    """Get a PDF chunk and nearby chunks from the same paper."""
    chunks = _load_chunks()
    normalized_id = chunk_id.strip().lower()
    target_index = None
    sorted_chunks = sorted(
        chunks,
        key=lambda item: (
            str(item.get("paper_id", "")),
            int(item.get("page") or 0),
            int(item.get("chunk_index") or 0),
        ),
    )

    for index, chunk in enumerate(sorted_chunks):
        if str(chunk.get("chunk_id", "")).lower() == normalized_id:
            target_index = index
            break

    if target_index is None:
        known_ids = ", ".join(str(chunk.get("chunk_id", "")) for chunk in sorted_chunks[:30])
        return f"Chunk ID '{chunk_id}' was not found. Sample known chunk IDs: {known_ids}"

    target = sorted_chunks[target_index]
    paper_id = target.get("paper_id")
    window = max(0, min(window, 3))
    nearby: list[dict[str, Any]] = []
    for index in range(max(0, target_index - window), min(len(sorted_chunks), target_index + window + 1)):
        chunk = sorted_chunks[index]
        if chunk.get("paper_id") == paper_id:
            nearby.append(chunk)

    return "\n---\n\n".join(
        _format_chunk_result(index, chunk)
        for index, chunk in enumerate(nearby, 1)
    )
