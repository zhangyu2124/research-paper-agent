"""Build a JSONL paper library from local PDF files.

This is the first ingestion step for the research paper agent. It extracts
paper-level metadata and a short text preview from PDFs. A later step can add
chunk-level vector indexing.
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw_papers"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "papers" / "papers.jsonl"

TOPIC_KEYWORDS = {
    "rag": ["rag", "retrieval-augmented", "retrieval augmented"],
    "retrieval": ["retrieval", "retrieve", "retriever"],
    "agent": ["agent", "agents", "agentic"],
    "multi-agent": ["multi-agent", "multi agent", "multiagent"],
    "workflow": ["workflow", "workflows"],
    "prompting": ["prompt", "prompting"],
    "long-document": ["long-document", "long document", "long-context", "long context"],
    "summarization": ["summarization", "summary", "summarisation"],
    "compression": ["compression", "compressive"],
    "planning": ["planning", "planner", "plan"],
    "reflection": ["reflection", "reflective", "self-reflection"],
    "evaluation": ["evaluation", "benchmark", "metrics"],
}

TITLE_OVERRIDES = {
    "2401.18059": "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
    "2404.14469": "SnapKV: LLM Knows What You are Looking for Before Generation",
    "2502.00977": "Context-Aware Hierarchical Merging for Long Document Summarization",
    "CoTHSSum": "CoTHSSum: Structured Long-Document Summarization via Chain-of-Thought Reasoning and Hierarchical Segmentation",
    "frai-8-1604034": "Divide and Summarize: Improve SLM Text Summarization",
}


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def _extract_pdf_text(path: Path, max_pages: int) -> tuple[str, int]:
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    parts: list[str] = []
    for page in reader.pages[:max_pages]:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return _clean_text("\n".join(parts)), page_count


def _metadata_title(path: Path) -> str | None:
    try:
        reader = PdfReader(str(path))
        title = reader.metadata.title if reader.metadata else None
    except Exception:
        title = None
    if not title:
        return None
    title = _clean_text(str(title))
    if not title or title.lower() in {"untitled", "unknown"}:
        return None
    return title


def _filename_title(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"\(?\d+\)?$", "", stem).strip()
    stem = re.sub(r"\d{4}\.\d{4,5}v\d+", "", stem).strip()
    stem = re.sub(r"\d{4}\.acl-long\.\d+v?\d*", "", stem).strip()
    stem = stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or path.stem


def _looks_like_noise(line: str) -> bool:
    lower = line.lower()
    if len(line) < 8:
        return True
    if lower.startswith(("arxiv:", "preprint", "proceedings", "copyright")):
        return True
    if "http://" in lower or "https://" in lower:
        return True
    return False


def _text_title(text: str) -> str | None:
    lines = [_clean_text(line) for line in text.splitlines()]
    lines = [line for line in lines if line and not _looks_like_noise(line)]
    if not lines:
        return None

    title_parts: list[str] = []
    for line in lines[:8]:
        if re.fullmatch(r"abstract", line, flags=re.IGNORECASE):
            break
        if "@" in line:
            break
        title_parts.append(line)
        if len(" ".join(title_parts)) > 180:
            break

    title = " ".join(title_parts).strip()
    if len(title.split()) < 3:
        return None
    return title[:220]


def _extract_abstract(text: str) -> str:
    match = re.search(r"\babstract\b", text, flags=re.IGNORECASE)
    if not match:
        return ""

    after = text[match.end() :]
    end_match = re.search(
        r"\n\s*(?:1\.?\s+)?(?:introduction|keywords|index terms)\b",
        after,
        flags=re.IGNORECASE,
    )
    abstract = after[: end_match.start()] if end_match else after[:1800]
    abstract = _clean_text(abstract)
    abstract = re.sub(r"^\W+", "", abstract)
    return abstract[:1600]


def _arxiv_id(name: str) -> str | None:
    match = re.search(r"(\d{4}\.\d{4,5})v?(\d+)?", name)
    if not match:
        return None
    version = f"v{match.group(2)}" if match.group(2) else ""
    return f"{match.group(1)}{version}"


def _acl_id(name: str) -> str | None:
    match = re.search(r"(\d{4}\.acl-long\.\d+v?\d*)", name, flags=re.IGNORECASE)
    if not match:
        return None
    return re.sub(r"v\d+$", "", match.group(1), flags=re.IGNORECASE)


def _slug(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:80] or "paper"


def _infer_topics(text: str, title: str, filename: str) -> list[str]:
    haystack = f"{title}\n{filename}\n{text}".lower()
    topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            topics.append(topic)
    return topics or ["research-paper"]


def _paper_record(path: Path, input_dir: Path, max_pages: int) -> dict[str, Any]:
    text, page_count = _extract_pdf_text(path, max_pages=max_pages)
    title = _metadata_title(path) or _text_title(text) or _filename_title(path)
    abstract = _extract_abstract(text)
    topics = _infer_topics(text[:5000], title, path.name)
    arxiv = _arxiv_id(path.name)
    acl = _acl_id(path.name)
    source_id = arxiv or acl or path.stem
    override_title = _title_override(path.name, source_id)

    return {
        "paper_id": _slug(source_id),
        "title": override_title or title,
        "authors": "Unknown",
        "year": _year_from_name(path.name),
        "venue": "Unknown",
        "topics": topics,
        "abstract": abstract or text[:900],
        "notes": (
            "Auto-generated from local PDF metadata and first pages. "
            "Review title/authors/venue before using as final bibliography data."
        ),
        "source_file": path.name,
        "pdf_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_id": source_id,
        "arxiv_id": arxiv,
        "acl_id": acl,
        "page_count": page_count,
        "text_preview": text[:1200],
    }


def _title_override(filename: str, source_id: str) -> str | None:
    for key, title in TITLE_OVERRIDES.items():
        if key in filename or key in source_id:
            return title
    return None


def _fetch_url(url: str, timeout: int = 20) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "paper-library-builder/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                return response.read().decode("utf-8", errors="replace")
        raise


def _clean_arxiv_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def fetch_arxiv_metadata(arxiv_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch title, authors, abstract, and year from arXiv."""
    clean_ids = sorted({_clean_arxiv_id(arxiv_id) for arxiv_id in arxiv_ids if arxiv_id})
    if not clean_ids:
        return {}

    query = urllib.parse.urlencode({"id_list": ",".join(clean_ids)})
    xml_text = _fetch_url(f"https://export.arxiv.org/api/query?{query}")
    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    result: dict[str, dict[str, Any]] = {}
    for entry in root.findall("atom:entry", ns):
        id_text = entry.findtext("atom:id", default="", namespaces=ns)
        arxiv_id = id_text.rstrip("/").split("/")[-1]
        arxiv_id = _clean_arxiv_id(arxiv_id)
        title = _clean_text(entry.findtext("atom:title", default="", namespaces=ns))
        summary = _clean_text(entry.findtext("atom:summary", default="", namespaces=ns))
        published = entry.findtext("atom:published", default="", namespaces=ns)
        authors = [
            _clean_text(author.findtext("atom:name", default="", namespaces=ns))
            for author in entry.findall("atom:author", ns)
        ]
        result[arxiv_id] = {
            "title": title,
            "authors": ", ".join(author for author in authors if author),
            "abstract": summary,
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "venue": "arXiv",
        }
    return result


def _parse_bib_field(bibtex: str, field: str) -> str:
    pattern = re.compile(rf"{field}\s*=\s*[{{\"](.+?)[}}\"],?\s*(?:\n|$)", re.IGNORECASE | re.DOTALL)
    match = pattern.search(bibtex)
    if not match:
        return ""
    value = match.group(1)
    value = re.sub(r"\s+", " ", value)
    value = value.replace("{", "").replace("}", "")
    return _clean_text(value)


def fetch_acl_metadata(acl_id: str) -> dict[str, Any] | None:
    """Fetch metadata from ACL Anthology BibTeX."""
    try:
        bibtex = _fetch_url(f"https://aclanthology.org/{acl_id}.bib")
    except Exception:
        return None

    title = _parse_bib_field(bibtex, "title")
    authors = _parse_bib_field(bibtex, "author").replace(" and ", ", ")
    year_text = _parse_bib_field(bibtex, "year")
    booktitle = _parse_bib_field(bibtex, "booktitle")
    return {
        "title": title,
        "authors": authors,
        "year": int(year_text) if year_text.isdigit() else None,
        "venue": booktitle or "ACL Anthology",
    }


def enrich_records(records: list[dict[str, Any]]) -> None:
    """Update records in-place with arXiv/ACL metadata when available."""
    arxiv_ids = [
        _clean_arxiv_id(record["arxiv_id"])
        for record in records
        if record.get("arxiv_id")
    ]
    try:
        arxiv_metadata = fetch_arxiv_metadata(arxiv_ids)
    except Exception as exc:
        print(f"Warning: failed to fetch arXiv metadata: {exc}")
        arxiv_metadata = {}

    for record in records:
        if record.get("arxiv_id"):
            metadata = arxiv_metadata.get(_clean_arxiv_id(record["arxiv_id"]))
            if metadata:
                for field in ["title", "authors", "abstract", "year", "venue"]:
                    if metadata.get(field):
                        record[field] = metadata[field]
                record["metadata_source"] = "arxiv"
                continue

        if record.get("acl_id"):
            metadata = fetch_acl_metadata(record["acl_id"])
            if metadata:
                for field in ["title", "authors", "year", "venue"]:
                    if metadata.get(field):
                        record[field] = metadata[field]
                record["metadata_source"] = "acl"


def _year_from_name(name: str) -> int | None:
    match = re.search(r"(20\d{2})", name)
    return int(match.group(1)) if match else None


def build_library(
    input_dir: Path,
    output: Path,
    max_pages: int,
    fetch_metadata: bool,
) -> list[dict[str, Any]]:
    records = []
    seen_ids: dict[str, int] = {}
    for path in sorted(input_dir.glob("*.pdf")):
        record = _paper_record(path, input_dir=input_dir, max_pages=max_pages)
        base_id = record["paper_id"]
        seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
        if seen_ids[base_id] > 1:
            record["paper_id"] = f"{base_id}_{seen_ids[base_id]}"
        records.append(record)

    if fetch_metadata:
        enrich_records(records)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument(
        "--fetch-metadata",
        action="store_true",
        help="Fetch arXiv/ACL metadata for recognized IDs.",
    )
    args = parser.parse_args()

    records = build_library(args.input_dir, args.output, args.max_pages, args.fetch_metadata)
    print(f"Wrote {len(records)} paper records to {args.output}")


if __name__ == "__main__":
    main()
