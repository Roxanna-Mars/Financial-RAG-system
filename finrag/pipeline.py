"""
End-to-end retrieval pipeline orchestrator.

Usage
-----
from finrag.pipeline import run_dataset, run_all

# Single dataset
result = run_dataset("financebench")
print(f"NDCG@10: {result.ndcg:.4f}")

# All datasets
results = run_all()
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

from .chunking import split_documents
from .config import DATASET_CONFIGS, DATASET_DIR, DatasetConfig
from .data import load_jsonl, load_qrels, make_corpus_lookup
from .evaluation import compute_retrieval_metrics
from .generation import GenerationResult, generate_answer
from .models import (
    DEEPSEEK_DEFAULT_MODEL,
    GROQ_DEFAULT_MODEL,
    OLLAMA_DEFAULT_MODEL,
    empty_cache,
    get_embedding_model,
    get_provider_llm,
    get_reranker,
)
from .retrieval import build_bm25_okapi, build_bm25_retriever, retrieve_and_rerank
from .vectorstore import get_vectorstore

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class DatasetResult:
    """Retrieval scores and run metadata for one dataset."""
    name:         str
    ndcg:         float                           # NDCG@10 (NaN if no qrels)
    results:      dict[str, dict[str, float]]     # query_id → {corpus_id: score}
    num_queries:  int
    num_chunks:   int
    elapsed_sec:  float
    errors:       list[str] = field(default_factory=list)
    recall:       float = float("nan")
    precision:    float = float("nan")
    mrr:          float = float("nan")
    hit_rate:     float = float("nan")
    evaluated_queries: int = 0


def run_dataset(
    dataset_name: str,
    *,
    use_multiquery: bool = True,
    force_rebuild: bool = False,
    top_k: int = 10,
    provider: str = "deepseek",
    model: str | None = None,
    query_ids: set[str] | None = None,
    use_reranker: bool = True,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    no_chunking: bool = False,
    retrieval_mode: str = "hybrid",
    candidate_top_k: int | None = None,
    rerank_top_n: int | None = None,
) -> DatasetResult:
    """
    Run the full retrieval pipeline on a single dataset.

    Parameters
    ----------
    dataset_name   : key from DATASET_CONFIGS (e.g. "financebench")
    use_multiquery : use LLM query expansion if the selected provider is available
    force_rebuild  : re-embed corpus even if a ChromaDB cache exists
    top_k          : documents to return per query
    query_ids      : optional query-ID subset; combined with qrels filtering
    use_reranker   : apply cross-encoder reranking after RRF fusion
    chunk_size     : optional character chunk-size override
    chunk_overlap  : optional character overlap override
    no_chunking    : keep each corpus record as one retrieval unit
    candidate_top_k: optional documents retained from each retriever before fusion
    rerank_top_n   : optional number of fused candidates passed to the reranker
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid options: {list(DATASET_CONFIGS)}"
        )

    cfg: DatasetConfig = DATASET_CONFIGS[dataset_name]
    t_start = time.time()

    logger.info("=" * 60)
    logger.info(
        "Dataset: %s  |  type=%s  |  chunk=%d",
        cfg.name, cfg.dataset_type, cfg.chunk_size,
    )
    logger.info("=" * 60)

    # ── Load ───────────────────────────────────────────────────────────────
    corpus  = load_jsonl(DATASET_DIR / cfg.corpus_file)
    queries = load_jsonl(DATASET_DIR / cfg.query_file)
    qrels   = load_qrels(DATASET_DIR / cfg.qrels_file)
    total_queries = len(queries)

    if query_ids is not None:
        queries = [q for q in queries if q["_id"] in query_ids]
        if not queries:
            raise ValueError("No queries match the requested query IDs.")

    # Retrieval metrics cannot be computed for queries without relevance
    # labels. Skip them before the expensive retrieval and reranking stages.
    if qrels:
        labeled_query_ids = {
            query_id
            for query_id, document_labels in qrels.items()
            if any(relevance > 0 for relevance in document_labels.values())
        }
        queries = [q for q in queries if q["_id"] in labeled_query_ids]
        if not queries:
            raise ValueError(
                "No selected queries have positive relevance labels in qrels."
            )

    logger.info(
        "Corpus: %d docs | Queries: %d/%d selected | Labeled: %d",
        len(corpus), len(queries), total_queries, len(qrels),
    )

    # ── Chunk ──────────────────────────────────────────────────────────────
    effective_chunk_size = chunk_size or cfg.chunk_size
    effective_chunk_overlap = (
        cfg.chunk_overlap if chunk_overlap is None else chunk_overlap
    )
    chunks = split_documents(
        corpus,
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_chunk_overlap,
        dataset_type=cfg.dataset_type,
        no_chunking=no_chunking,
    )

    if no_chunking:
        vectorstore_name = f"{cfg.name}_whole_documents"
    elif chunk_size is not None or chunk_overlap is not None:
        vectorstore_name = (
            f"{cfg.name}_chunk{effective_chunk_size}"
            f"_overlap{effective_chunk_overlap}"
        )
    else:
        vectorstore_name = cfg.name

    # ── Models (lazy singletons — loaded once, shared across datasets) ─────
    embedding_model = get_embedding_model()
    reranker        = get_reranker() if use_reranker else None
    llm             = get_provider_llm(provider, model) if use_multiquery else None

    # ── Vector store ───────────────────────────────────────────────────────
    vectorstore = get_vectorstore(
        dataset_name=vectorstore_name,
        chunk_result=chunks,
        embedding_model=embedding_model,
        force_rebuild=force_rebuild,
    )

    # ── BM25 (built once per dataset) ─────────────────────────────────────
    logger.info("Building BM25 indexes …")
    bm25_retriever = build_bm25_retriever(chunks.lc_docs, fetch_k=cfg.fetch_k)
    bm25_okapi = build_bm25_okapi(chunks.texts) if use_reranker else None

    # ── Per-query retrieval ────────────────────────────────────────────────
    effective_rerank_top_n = rerank_top_n or cfg.rerank_top_n
    logger.info(
        "Retrieving %d queries  (fetch_k=%d, rerank_top_n=%d) …",
        len(queries), cfg.fetch_k, effective_rerank_top_n,
    )

    dataset_results: dict[str, dict[str, float]] = {}
    errors: list[str] = []

    for q in tqdm(queries, desc=f"  {cfg.name}", unit="query", leave=True):
        try:
            ranked = retrieve_and_rerank(
                vectorstore=vectorstore,
                bm25_retriever=bm25_retriever,
                bm25_okapi=bm25_okapi,
                chunk_result=chunks,
                query=q["text"],
                reranker=reranker,
                dataset_type=cfg.dataset_type,
                fetch_k=cfg.fetch_k,
                llm=llm,
                k=top_k,
                rerank_top_n=effective_rerank_top_n,
                retrieval_mode=retrieval_mode,
                candidate_top_k=candidate_top_k,
            )
            dataset_results[q["_id"]] = {doc_id: float(s) for doc_id, s in ranked}
        except Exception as exc:
            logger.warning("Query %s failed: %s", q["_id"], exc)
            errors.append(f"{q['_id']}: {exc}")
            dataset_results[q["_id"]] = {}

    # ── Evaluate ──────────────────────────────────────────────────────────
    ndcg = float("nan")
    recall = precision = mrr = hit_rate = float("nan")
    evaluated_queries = 0
    if qrels:
        metrics = compute_retrieval_metrics(
            qrels,
            dataset_results,
            k=top_k,
            query_ids=(query["_id"] for query in queries),
        )
        ndcg = metrics.ndcg
        recall = metrics.recall
        precision = metrics.precision
        mrr = metrics.mrr
        hit_rate = metrics.hit_rate
        evaluated_queries = metrics.evaluated_queries
        logger.info("NDCG@%d: %.4f", top_k, ndcg)
        logger.info(
            "Recall@%d: %.4f | Precision@%d: %.4f | MRR@%d: %.4f | "
            "Hit Rate@%d: %.4f",
            top_k, recall, top_k, precision, top_k, mrr, top_k, hit_rate,
        )
        logger.info("Evaluated queries with positive labels: %d", evaluated_queries)
    else:
        logger.info("No qrels available — skipping NDCG evaluation.")

    elapsed = time.time() - t_start
    logger.info("Finished '%s' in %.1f s", cfg.name, elapsed)

    empty_cache()

    return DatasetResult(
        name=cfg.name,
        ndcg=ndcg,
        results=dataset_results,
        num_queries=len(queries),
        num_chunks=len(chunks.texts),
        elapsed_sec=elapsed,
        errors=errors,
        recall=recall,
        precision=precision,
        mrr=mrr,
        hit_rate=hit_rate,
        evaluated_queries=evaluated_queries,
    )


def run_all(
    datasets: Optional[list[str]] = None,
    *,
    provider: str = "deepseek",
    model: str | None = None,
    **kwargs,
) -> dict[str, DatasetResult]:
    """
    Run the pipeline over multiple (or all) datasets sequentially.

    Parameters
    ----------
    datasets : subset of DATASET_CONFIGS keys — defaults to all 7.
    **kwargs : forwarded to run_dataset.
    """
    names = datasets or list(DATASET_CONFIGS)
    all_results: dict[str, DatasetResult] = {}

    for name in names:
        all_results[name] = run_dataset(name, provider=provider, model=model, **kwargs)

    # ── Summary table ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 55)
    logger.info("%-16s  %8s  %8s  %10s", "Dataset", "NDCG@10", "Queries", "Time (s)")
    logger.info("-" * 55)
    for name, res in all_results.items():
        logger.info(
            "%s Recall@%d: %.4f | Precision@%d: %.4f | MRR@%d: %.4f | "
            "Hit Rate@%d: %.4f",
            name, kwargs.get("top_k", 10), res.recall,
            kwargs.get("top_k", 10), res.precision,
            kwargs.get("top_k", 10), res.mrr,
            kwargs.get("top_k", 10), res.hit_rate,
        )
        ndcg_str = f"{res.ndcg:.4f}" if res.ndcg == res.ndcg else "   N/A  "
        logger.info(
            "%-16s  %8s  %8d  %10.1f",
            name, ndcg_str, res.num_queries, res.elapsed_sec,
        )
    logger.info("=" * 55)

    return all_results


# ── Interactive single-query RAG ──────────────────────────────────────────────

def run_rag_query(
    query:          str,
    dataset_name:   str  = "financebench",
    *,
    top_k:          int  = 5,
    use_multiquery: bool = True,
    force_rebuild:  bool = False,
    provider:       str  = "deepseek",
    model:          str | None = None,
    use_reranker:   bool = True,
    chunk_size:     int | None = None,
    chunk_overlap:  int | None = None,
    no_chunking:    bool = False,
    retrieval_mode: str = "hybrid",
    candidate_top_k: int | None = None,
    rerank_top_n:   int | None = None,
) -> GenerationResult:
    """
    Full RAG for a single query: retrieve → generate → cite.

    This is the primary demo / MCP entry point. Reuses cached ChromaDB and
    model singletons so repeated calls are fast (no re-embedding).

    Parameters
    ----------
    query          : natural language financial question
    dataset_name   : which corpus to search (default: "financebench")
    top_k          : docs to retrieve and pass to the LLM (default: 5)
    use_multiquery : use LLM query expansion (requires LLM to be available)
    force_rebuild  : rebuild ChromaDB index even if cache exists
    provider       : "deepseek" | "groq" | "ollama"
    model          : model name override — defaults per provider apply if None

    Returns
    -------
    GenerationResult with .answer (str), .sources (list[CitedSource]), .model (str)

    Example
    -------
    >>> from finrag.pipeline import run_rag_query
    >>> result = run_rag_query("What was Apple's total revenue in FY2022?")
    >>> print(result.pretty())
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid options: {list(DATASET_CONFIGS)}"
        )

    cfg = DATASET_CONFIGS[dataset_name]

    # ── Load corpus + build lookup ─────────────────────────────────────────
    corpus        = load_jsonl(DATASET_DIR / cfg.corpus_file)
    corpus_lookup = make_corpus_lookup(corpus)

    # ── Chunk + models (cached after first call) ───────────────────────────
    effective_chunk_size = chunk_size or cfg.chunk_size
    effective_chunk_overlap = (
        cfg.chunk_overlap if chunk_overlap is None else chunk_overlap
    )
    chunks = split_documents(
        corpus,
        effective_chunk_size,
        effective_chunk_overlap,
        cfg.dataset_type,
        no_chunking=no_chunking,
    )
    embedding_model = get_embedding_model()
    reranker        = get_reranker() if use_reranker else None
    gen_llm         = get_provider_llm(provider, model)

    if gen_llm is None:
        raise RuntimeError(
            "LLM unavailable — for DeepSeek: set DEEPSEEK_API_KEY in .env; "
            "for Groq: set GROQ_API_KEY in .env; "
            "for Ollama: run `ollama serve` and `ollama pull <model>`."
        )

    # MultiQuery uses the same LLM as generation (avoids loading two models)
    retrieval_llm = gen_llm if use_multiquery else None

    # ── Vector store + BM25 (vector store cached on disk) ─────────────────
    if no_chunking:
        vectorstore_name = f"{cfg.name}_whole_documents"
    elif chunk_size is not None or chunk_overlap is not None:
        vectorstore_name = (
            f"{cfg.name}_chunk{effective_chunk_size}"
            f"_overlap{effective_chunk_overlap}"
        )
    else:
        vectorstore_name = cfg.name
    vectorstore = get_vectorstore(
        vectorstore_name, chunks, embedding_model, force_rebuild=force_rebuild
    )
    bm25_retriever = build_bm25_retriever(chunks.lc_docs, fetch_k=cfg.fetch_k)
    bm25_okapi = build_bm25_okapi(chunks.texts) if use_reranker else None

    # ── Retrieve ───────────────────────────────────────────────────────────
    retrieved = retrieve_and_rerank(
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        bm25_okapi=bm25_okapi,
        chunk_result=chunks,
        query=query,
        reranker=reranker,
        dataset_type=cfg.dataset_type,
        fetch_k=cfg.fetch_k,
        llm=retrieval_llm,
        k=top_k,
        rerank_top_n=rerank_top_n or cfg.rerank_top_n,
        retrieval_mode=retrieval_mode,
        candidate_top_k=candidate_top_k,
    )

    # ── Generate ───────────────────────────────────────────────────────────
    default_models = {
        "deepseek": DEEPSEEK_DEFAULT_MODEL,
        "groq": GROQ_DEFAULT_MODEL,
        "ollama": OLLAMA_DEFAULT_MODEL,
    }
    actual_model = model or default_models[provider]

    return generate_answer(
        query=query,
        retrieved=retrieved,
        corpus_lookup=corpus_lookup,
        llm=gen_llm,
        top_k=top_k,
        multiquery=retrieval_llm is not None,
        model_name=actual_model,
        use_structured_output=provider != "deepseek",
    )
