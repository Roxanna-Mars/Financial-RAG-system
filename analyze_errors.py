"""Build a reproducible per-query error analysis table from RAGAS details."""

import argparse
import json
from pathlib import Path

import pandas as pd

from finrag.config import DATASET_CONFIGS, DATASET_DIR
from finrag.data import load_qrels
from finrag.routing import NUMERICAL, COMPARISON, MULTI_HOP, classify_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze retrieval and generation errors.")
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS), default="financebench")
    parser.add_argument("--retrieval-results", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details = pd.read_csv(args.details)
    cfg = DATASET_CONFIGS[args.dataset]
    qrels = load_qrels(DATASET_DIR / cfg.qrels_file)
    retrieval_results = {}
    if args.retrieval_results:
        retrieval_results = json.loads(args.retrieval_results.read_text(encoding="utf-8"))

    rows: list[dict] = []
    for record in details.to_dict(orient="records"):
        query_id = str(record["query_id"])
        query = str(record["user_input"])
        query_type = classify_query(query)
        error_types: list[str] = []

        faithfulness = float(record.get("faithfulness", float("nan")))
        relevancy = float(record.get("answer_relevancy", float("nan")))
        utilization = float(record.get("context_utilization", float("nan")))
        if faithfulness < 0.70:
            error_types.append("unsupported_claim")
        if relevancy < 0.65:
            error_types.append("answer_irrelevant")
        if utilization < 0.70:
            error_types.append("evidence_underused")

        relevant_ids = {
            document_id
            for document_id, relevance in qrels.get(query_id, {}).items()
            if relevance > 0
        }
        ranked_ids = list(retrieval_results.get(query_id, {}))
        relevant_rank = next(
            (rank for rank, doc_id in enumerate(ranked_ids, start=1) if doc_id in relevant_ids),
            None,
        )
        if retrieval_results and relevant_ids and relevant_rank is None:
            error_types.insert(0, "retrieval_miss")
        elif relevant_rank is not None and relevant_rank > 5:
            error_types.insert(0, "ranking_error")

        rows.append({
            "query_id": query_id,
            "query_type": query_type,
            "calculation_required": query_type in {NUMERICAL, COMPARISON, MULTI_HOP},
            "relevant_rank": relevant_rank,
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_utilization": utilization,
            "error_types": ";".join(error_types) if error_types else "none",
            "query": query,
        })

    output = pd.DataFrame(rows).sort_values(
        ["faithfulness", "answer_relevancy"], ascending=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(output["query_type"].value_counts().to_string())
    print("\nError labels")
    print(output["error_types"].value_counts().head(10).to_string())
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
