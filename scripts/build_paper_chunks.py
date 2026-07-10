"""Build page-level text chunks from local research paper PDFs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw_papers"
DEFAULT_LIBRARY = PROJECT_ROOT / "data" / "papers" / "papers.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "papers" / "paper_chunks.jsonl"


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def _slug(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:80] or "paper"


def _load_paper_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    metadata: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            source_file = str(record.get("source_file", ""))
            if source_file:
                metadata[source_file] = record
    return metadata


def _section_hint(text: str) -> str | None:
    for raw_line in text.splitlines()[:10]:
        line = _clean_text(raw_line)
        if not line or len(line) > 120:
            continue
        if re.match(r"^(abstract|introduction|related work|method|experiments?|conclusion)s?$", line, re.I):
            return line
        if re.match(r"^\d+(\.\d+)*\s+[A-Z][A-Za-z0-9 ,:()/-]{3,}$", line):
            return line
    return None


def _split_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            lower_bound = min(start + int(chunk_chars * 0.65), end)
            boundary = max(
                text.rfind("\n\n", lower_bound, end),
                text.rfind(". ", lower_bound, end),
                text.rfind(" ", lower_bound, end),
            )
            if boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(0, end - overlap_chars)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def build_chunks(
    input_dir: Path,
    library_path: Path,
    output: Path,
    chunk_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    paper_metadata = _load_paper_metadata(library_path)
    records: list[dict[str, Any]] = []

    for pdf_path in sorted(input_dir.glob("*.pdf")):
        metadata = paper_metadata.get(pdf_path.name, {})
        paper_id = str(metadata.get("paper_id") or _slug(pdf_path.stem))
        title = str(metadata.get("title") or pdf_path.stem)

        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, 1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            page_text = _clean_text(page_text)
            section = _section_hint(page_text)
            for chunk_index, content in enumerate(
                _split_text(page_text, chunk_chars, overlap_chars),
                1,
            ):
                chunk_id = f"{paper_id}_p{page_number:03d}_c{chunk_index:03d}"
                records.append(
                    {
                        "chunk_id": chunk_id,
                        "paper_id": paper_id,
                        "title": title,
                        "source_file": pdf_path.name,
                        "pdf_path": str(pdf_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "section": section,
                        "content": content,
                    }
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-chars", type=int, default=1400)
    parser.add_argument("--overlap-chars", type=int, default=220)
    args = parser.parse_args()

    records = build_chunks(
        input_dir=args.input_dir,
        library_path=args.library,
        output=args.output,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
    )
    print(f"Wrote {len(records)} PDF chunks to {args.output}")


if __name__ == "__main__":
    main()
