"""Build a local vector index for paper PDF chunks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "papers" / "paper_chunks.jsonl"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "gte-multilingual-base"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "papers" / "paper_vector_index.npz"
DEFAULT_CACHE = PROJECT_ROOT / "models" / ".cache" / "huggingface"
DEFAULT_MAX_SEQ_LENGTH = 512


def _configure_huggingface_cache(cache_dir: Path = DEFAULT_CACHE) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir / "sentence_transformers"))


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                chunks.append(json.loads(stripped))
    return chunks


def _chunk_text(chunk: dict[str, Any]) -> str:
    title = chunk.get("title", "")
    section = chunk.get("section") or ""
    content = chunk.get("content", "")
    return f"Title: {title}\nSection: {section}\nContent: {content}"


def _model_location(model_path: Path) -> str:
    configured = os.getenv("EMBEDDING_MODEL_PATH")
    if configured:
        path = Path(configured)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        return str(path)
    return str(model_path)


def build_index(
    chunks_path: Path,
    model_path: Path,
    output: Path,
    batch_size: int,
    max_seq_length: int,
) -> None:
    chunks = _load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    model_location = _model_location(model_path)
    _configure_huggingface_cache()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_location, trust_remote_code=True)
    model.max_seq_length = max_seq_length
    texts = [_chunk_text(chunk) for chunk in chunks]
    chunk_ids = np.array([str(chunk["chunk_id"]) for chunk in chunks])

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        model_location=np.array([model_location]),
    )
    print(f"Wrote {len(chunks)} vectors to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    args = parser.parse_args()

    build_index(
        chunks_path=args.chunks,
        model_path=args.model_path,
        output=args.output,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
