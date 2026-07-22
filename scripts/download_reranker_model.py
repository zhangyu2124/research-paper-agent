"""Download the local multilingual cross-encoder reranker."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "bge-reranker-v2-m3"
DEFAULT_CACHE = PROJECT_ROOT / "models" / ".cache" / "huggingface"


def _configure_huggingface_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME",
        str(cache_dir / "sentence_transformers"),
    )


def main() -> None:
    """Download the configured reranker and keep all files in the project."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    _configure_huggingface_cache(args.cache_dir)
    from huggingface_hub import snapshot_download

    args.output.mkdir(parents=True, exist_ok=True)
    local_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(args.output),
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    )
    print(f"Downloaded {args.repo_id} to {local_path}")  # noqa: T201


if __name__ == "__main__":
    main()
