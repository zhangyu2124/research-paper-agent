"""Compare dense, BM25, hybrid, and reranked paper retrieval."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import (  # noqa: E402
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    unique_paper_ids,
)
from src.tools.paper_tools import retrieve_paper_evidence  # noqa: E402

DEFAULT_CASES_PATH = PROJECT_ROOT / "evaluation" / "retrieval_cases.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "evaluation" / "reports" / "latest_retrieval.json"
DEFAULT_MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=DEFAULT_MODES)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation cases must be a non-empty JSON array.")
    return cases


def _evaluate_mode(
    mode: str,
    cases: list[dict[str, Any]],
    *,
    top_k: int,
    candidate_k: int,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        hits = retrieve_paper_evidence(
            str(case["query"]),
            top_k=top_k,
            candidate_k=candidate_k,
            mode=mode,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        ranked_papers = unique_paper_ids(hits)
        groups = case["relevant_groups"]
        per_case.append(
            {
                "id": case["id"],
                "query": case["query"],
                "ranked_paper_ids": ranked_papers,
                "recall_at_5": recall_at_k(ranked_papers, groups, 5),
                "mrr_at_10": reciprocal_rank(ranked_papers, groups, 10),
                "ndcg_at_10": ndcg_at_k(ranked_papers, groups, 10),
                "latency_ms": latency_ms,
            }
        )

    return {
        "mode": mode,
        "case_count": len(per_case),
        "recall_at_5": statistics.fmean(item["recall_at_5"] for item in per_case),
        "mrr_at_10": statistics.fmean(item["mrr_at_10"] for item in per_case),
        "ndcg_at_10": statistics.fmean(item["ndcg_at_10"] for item in per_case),
        "mean_latency_ms": statistics.fmean(item["latency_ms"] for item in per_case),
        "p95_latency_ms": sorted(item["latency_ms"] for item in per_case)[
            max(0, int(len(per_case) * 0.95) - 1)
        ],
        "cases": per_case,
    }


def _print_summary(results: list[dict[str, Any]]) -> None:
    print("mode              Recall@5  MRR@10  nDCG@10  mean_ms  p95_ms")  # noqa: T201
    for result in results:
        print(  # noqa: T201
            f"{result['mode']:<17} "
            f"{result['recall_at_5']:.4f}    "
            f"{result['mrr_at_10']:.4f}   "
            f"{result['ndcg_at_10']:.4f}    "
            f"{result['mean_latency_ms']:.1f}    "
            f"{result['p95_latency_ms']:.1f}"
        )


def main() -> None:
    """Run all requested retrieval modes and save a JSON report."""
    args = _parse_args()
    cases = _load_cases(args.cases)

    if not args.no_warmup:
        warmup_query = str(cases[0]["query"])
        for mode in args.modes:
            retrieve_paper_evidence(
                warmup_query,
                top_k=min(args.top_k, 3),
                candidate_k=args.candidate_k,
                mode=mode,
            )

    results = [
        _evaluate_mode(
            mode,
            cases,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )
        for mode in args.modes
    ]
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus": {"papers": 20, "unique_titles": 18, "chunks": 1569},
        "settings": {
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "warmup": not args.no_warmup,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_summary(results)
    print(f"Report: {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
