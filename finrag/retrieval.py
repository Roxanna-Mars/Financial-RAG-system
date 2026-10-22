"""
Retrieval pipeline — manual implementation, no broken LangChain 1.x dependencies.

Stage 1  Dense retrieval   Chroma MMR (passage) or similarity (tabular)
Stage 2  Sparse retrieval  BM25Retriever
Stage 3  RRF fusion        Reciprocal Rank Fusion, k=60
Stage 4  Best-chunk pick   BM25Okapi scores per candidate doc
Stage 5  Reranking         Cross-encoder on (query, best_chunk) pairs
"""

import logging
import re

from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi

from .chunking import ChunkResult
from .routing import route_query

logger = logging.getLogger(__name__)


# ── BM25 indexes ─────────────────────────────────────────────────────────────

_FINANCIAL_ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "capex": ("capital", "expenditure"),
    "cogs": ("cost", "goods", "sold"),
    "ebit": ("earnings", "interest", "taxes"),
    "ebitda": ("earnings", "interest", "taxes", "depreciation", "amortization"),
    "eps": ("earnings", "per", "share"),
    "opex": ("operating", "expenses"),
    "ppne": ("property", "plant", "equipment"),
    "roa": ("return", "assets"),
    "roe": ("return", "equity"),
}


def preprocess_bm25(text: str) -> list[str]:
    """Normalize English financial text into tokens used by both BM25 stages."""
    normalized = text.lower().replace("’", "'")
    normalized = re.sub(r"(?<=\w)'s\b", "", normalized)
    normalized = re.sub(r"\bpp\s*&\s*e\b", "ppne", normalized)
    normalized = normalized.replace("&", " and ")
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", normalized)

    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(_FINANCIAL_ABBREVIATIONS.get(token, ()))
    return expanded

def build_bm25_retriever(lc_docs, fetch_k: int) -> BM25Retriever:
    """LangChain BM25Retriever for Stage 2 retrieval."""
    return BM25Retriever.from_documents(
        lc_docs,
        k=fetch_k,
        preprocess_func=preprocess_bm25,
    )


def build_bm25_okapi(texts: list[str]) -> BM25Okapi:
    """Raw BM25Okapi used in Stage 4 to select the best chunk per candidate doc."""
    return BM25Okapi([preprocess_bm25(text) for text in texts])


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked doc-ID lists with Reciprocal Rank Fusion.

    RRF score for document d:
        score(d) = Σ  1 / (k + rank_r(d))   over all retrievers r

    k=60 is the standard constant — robust across datasets with no tuning.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def deduplicate_ranked_ids(doc_ids: list[str]) -> list[str]:
    """Keep the first occurrence of each document ID in a ranked list."""
    return list(dict.fromkeys(doc_ids))


# ── Optional: MultiQuery expansion ───────────────────────────────────────────

def expand_query(query: str, llm) -> list[str]:
    """
    Use an LLM to generate 3 alternative phrasings of the query.
    Returns [original] + [variants] so retrieval always includes the original.
    Falls back gracefully if the LLM fails.
    """
    prompt = (
        "Generate 3 alternative search queries for the following financial question. "
        "Output only the queries, one per line, no numbering or explanation.\n\n"
        f"Original: {query}"
    )
    try:
        response = llm.invoke(prompt)
        variants = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip() and line.strip() != query
        ][:3]
        queries = [query] + variants
        logger.debug("MultiQuery expanded to %d variants.", len(queries))
        return queries
    except Exception as exc:
        logger.warning("MultiQuery expansion failed (%s) — using original query.", exc)
        return [query]


# ── Core retrieval function ───────────────────────────────────────────────────

def retrieve_and_rerank(
    vectorstore,
    bm25_retriever: BM25Retriever,
    bm25_okapi: BM25Okapi,
    chunk_result: ChunkResult,
    query: str,
    reranker,
    dataset_type: str,
    fetch_k: int,
    llm=None,
    k: int = 10,
    rerank_top_n: int = 30,
    retrieval_mode: str = "hybrid",
    candidate_top_k: int | None = None,
) -> list[tuple[str, float]]:
    """
    Full per-query retrieval pipeline.

    Parameters
    ----------
    vectorstore   : ChromaDB store for this dataset
    bm25_retriever: LangChain BM25Retriever (Stage 2)
    bm25_okapi    : BM25Okapi index for best-chunk selection (Stage 4)
    chunk_result  : output of split_documents
    query         : raw query string
    reranker      : CrossEncoder instance, or None to use the RRF ranking
    dataset_type  : "passage" | "tabular"
    fetch_k       : candidates to fetch from each retriever
    llm           : optional LLM for MultiQuery expansion (None = disabled)
    k             : final top-k to return
    rerank_top_n  : how many unique doc IDs to pass to the cross-encoder
    candidate_top_k: optional per-retriever document limit before fusion

    Returns
    -------
    List of (corpus_id, reranker_score) sorted descending.
    """
    # ── Stage 1 — Optional MultiQuery expansion ───────────────────────────
    if retrieval_mode not in {"dense", "bm25", "hybrid", "adaptive"}:
        raise ValueError("retrieval_mode must be one of: dense, bm25, hybrid, adaptive")

    effective_mode = retrieval_mode
    use_query_expansion = llm is not None
    if retrieval_mode == "adaptive":
        decision = route_query(query)
        effective_mode = decision.retrieval_mode
        use_query_expansion = llm is not None and decision.use_multiquery
        logger.debug(
            "Adaptive route: type=%s mode=%s multiquery=%s reason=%s",
            decision.query_type,
            effective_mode,
            use_query_expansion,
            decision.reason,
        )

    queries = (
        expand_query(query, llm)
        if use_query_expansion and effective_mode != "bm25"
        else [query]
    )

    # ── Stage 2 — Dense retrieval (per query variant) ─────────────────────
    dense_ranked: list[list[str]] = []
    if effective_mode != "bm25":
        for q in queries:
            if dataset_type == "passage":
                docs = vectorstore.max_marginal_relevance_search(
                    q, k=fetch_k, fetch_k=fetch_k * 3, lambda_mult=0.7
                )
            else:
                docs = vectorstore.similarity_search(q, k=fetch_k)
            dense_ranked.append(
                deduplicate_ranked_ids([d.metadata["id"] for d in docs])
            )
            if candidate_top_k is not None:
                dense_ranked[-1] = dense_ranked[-1][:candidate_top_k]

    # ── Stage 3 — BM25 retrieval ──────────────────────────────────────────
    bm25_docs = (
        bm25_retriever.invoke(query) if effective_mode != "dense" else []
    )
    bm25_ranked = deduplicate_ranked_ids(
        [d.metadata["id"] for d in bm25_docs]
    )
    if candidate_top_k is not None:
        bm25_ranked = bm25_ranked[:candidate_top_k]

    # ── Stage 4 — RRF fusion ──────────────────────────────────────────────
    if effective_mode == "dense":
        all_ranked_lists = dense_ranked
    elif effective_mode == "bm25":
        all_ranked_lists = [bm25_ranked]
    else:
        all_ranked_lists = dense_ranked + [bm25_ranked]
    fused = rrf_fuse(all_ranked_lists, k=60)

    # Ablation path: retain hybrid recall and RRF fusion while skipping the
    # best-chunk selection and cross-encoder reranking stages.
    if reranker is None:
        return fused[:k]

    # Deduplicate and cap at rerank_top_n
    # Build a page_content map from dense results for Stage 5 fallback
    seen: dict[str, str] = {}
    for doc_list in dense_ranked:
        pass   # doc_list has IDs only; content stored below via bm25_okapi
    # Collect content from BM25 docs (always have page_content)
    for d in bm25_docs:
        seen[d.metadata["id"]] = d.page_content

    candidate_ids: list[str] = []
    for doc_id, _ in fused:
        if doc_id not in [c for c in candidate_ids]:
            candidate_ids.append(doc_id)
        if len(candidate_ids) >= rerank_top_n:
            break

    if not candidate_ids:
        return []

    # ── Stage 5 — Best chunk per doc via BM25Okapi ────────────────────────
    chunk_scores  = bm25_okapi.get_scores(preprocess_bm25(query))
    candidate_set = set(candidate_ids)
    best_chunk: dict[str, tuple[float, str]] = {}

    for chunk_text, orig_id, score in zip(
        chunk_result.texts, chunk_result.original_ids, chunk_scores
    ):
        if orig_id not in candidate_set:
            continue
        s = float(score)
        if orig_id not in best_chunk or s > best_chunk[orig_id][0]:
            best_chunk[orig_id] = (s, chunk_text)

    for oid in candidate_ids:
        if oid not in best_chunk:
            best_chunk[oid] = (0.0, seen.get(oid, ""))

    # ── Stage 6 — Cross-encoder reranking ────────────────────────────────
    pairs  = [(query, best_chunk[oid][1]) for oid in candidate_ids]
    scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)

    ranked = sorted(
        zip(candidate_ids, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:k]
