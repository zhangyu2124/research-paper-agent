"""Local research paper search tools for the paper agent."""

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER_LIBRARY_PATH = PROJECT_ROOT / "data" / "papers" / "papers.jsonl"
DEFAULT_PAPER_CHUNKS_PATH = PROJECT_ROOT / "data" / "papers" / "paper_chunks.jsonl"
DEFAULT_PAPER_VECTOR_INDEX_PATH = PROJECT_ROOT / "data" / "papers" / "paper_vector_index.npz"
DEFAULT_EMBEDDING_MODEL_PATH = PROJECT_ROOT / "models" / "gte-multilingual-base"
DEFAULT_EMBEDDING_MAX_SEQ_LENGTH = 512
MAX_TOP_K = 10
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


def _configure_huggingface_cache() -> None:
    """Keep embedding model caches inside the project by default."""
    cache_dir = PROJECT_ROOT / "models" / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir / "sentence_transformers"))


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
    _configure_huggingface_cache()
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
