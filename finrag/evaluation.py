"""Standard retrieval metrics for ranked document results."""

import logging
import math
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalMetrics:
    """Macro-averaged retrieval metrics at a fixed rank cutoff."""

    ndcg: float
    recall: float
    precision: float
    mrr: float
    hit_rate: float
    evaluated_queries: int


def compute_retrieval_metrics(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k: int = 10,
    query_ids: Iterable[str] | None = None,
) -> RetrievalMetrics:
    """Compute standard macro metrics over all positively labeled queries.

    Each query with at least one positive relevance label participates in the
    average. Empty or missing retrieval results receive zero for every metric.
    NDCG compares the returned ranking with the ideal ranking built from the
    query's complete qrels, so missed relevant documents are penalized.

    Precision@k always uses ``k`` as its denominator. MRR@k is truncated at
    ``k``: it is zero when no relevant document appears in the top-k results.
    """
    if k < 1:
        raise ValueError("k must be positive")

    evaluation_ids = list(dict.fromkeys(query_ids if query_ids is not None else qrels))
    ndcg_scores: list[float] = []
    recall_scores: list[float] = []
    precision_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    hit_scores: list[float] = []

    for query_id in evaluation_ids:
        labels = qrels.get(query_id, {})
        relevant = {doc_id for doc_id, relevance in labels.items() if relevance > 0}
        if not relevant:
            continue

        ranked = sorted(
            results.get(query_id, {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
        ranked_ids = [doc_id for doc_id, _ in ranked]
        matched = sum(doc_id in relevant for doc_id in ranked_ids)

        recall_scores.append(matched / len(relevant))
        precision_scores.append(matched / k)
        hit_scores.append(float(matched > 0))

        first_relevant_rank = next(
            (
                rank
                for rank, doc_id in enumerate(ranked_ids, start=1)
                if doc_id in relevant
            ),
            None,
        )
        reciprocal_ranks.append(
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        )

        dcg = sum(
            max(0, labels.get(doc_id, 0)) / math.log2(rank + 1)
            for rank, doc_id in enumerate(ranked_ids, start=1)
        )
        ideal_relevances = sorted(
            (relevance for relevance in labels.values() if relevance > 0),
            reverse=True,
        )[:k]
        idcg = sum(
            relevance / math.log2(rank + 1)
            for rank, relevance in enumerate(ideal_relevances, start=1)
        )
        ndcg_scores.append(dcg / idcg if idcg else 0.0)

    if not ndcg_scores:
        logger.warning("No positively labeled queries found; check qrels alignment.")
        nan = float("nan")
        return RetrievalMetrics(nan, nan, nan, nan, nan, 0)

    count = len(ndcg_scores)
    return RetrievalMetrics(
        ndcg=sum(ndcg_scores) / count,
        recall=sum(recall_scores) / count,
        precision=sum(precision_scores) / count,
        mrr=sum(reciprocal_ranks) / count,
        hit_rate=sum(hit_scores) / count,
        evaluated_queries=count,
    )


def compute_ndcg(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k: int = 10,
) -> float:
    """Compatibility helper returning corrected NDCG@k."""
    return compute_retrieval_metrics(qrels, results, k=k).ndcg
