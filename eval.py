"""
FinRAG RAGAS Evaluation CLI
============================

Runs faithfulness, answer_relevancy, and context_utilization on a random
sample of queries, records the result in the regression CSV, and optionally
plots a comparison chart between configs.

Examples
--------
# Baseline eval — 20 queries, no MultiQuery expansion
python eval.py --dataset financebench --config baseline --n 20 --no-multiquery

# MultiQuery eval — same 20 queries (same seed → same sample)
python eval.py --dataset financebench --config multiquery_k10 --n 20

# Just compare recorded configs (no new eval)
python eval.py --compare --dataset financebench --chart results/ragas_comparison.png

# Show historical trend for one config
python eval.py --trend --dataset financebench --config baseline

Flags
-----
--dataset        DATASET_CONFIGS key (default: financebench)
--config         Run label saved to CSV (default: baseline)
--n              Queries to evaluate (default: 20, ~5-8 min with Groq)
--seed           Random seed for sampling (default: 42 — reproducible)
--top-k          Docs retrieved per query (default: 5)
--no-multiquery  Disable LLM query expansion
--rebuild        Force re-embed corpus (ignore ChromaDB cache)
--chart          PNG path for comparison bar chart
--compare        Print config comparison table (no new eval)
--trend          Print trend for --config (no new eval)
--log-level      Logging verbosity (default: INFO)
--csv            Override regression CSV path
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eval",
        description="Run RAGAS evaluation and track regression.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Target ────────────────────────────────────────────────────────────
    from finrag.config import DATASET_CONFIGS  # imported here to avoid top-level cost
    p.add_argument(
        "--dataset", "-d",
        default="financebench",
        choices=list(DATASET_CONFIGS),
        metavar="NAME",
        help=f"Dataset to evaluate. Choices: {', '.join(DATASET_CONFIGS)}  (default: financebench)",
    )
    p.add_argument(
        "--config", "-c",
        default="baseline",
        metavar="LABEL",
        help="Config label saved to regression CSV (default: baseline)",
    )

    # ── Eval options ──────────────────────────────────────────────────────
    p.add_argument("--n", type=int, default=20, metavar="N",
                   help="Number of queries to evaluate (default: 20)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for query sampling (default: 42)")
    p.add_argument("--top-k", type=int, default=5, metavar="K",
                   help="Docs retrieved per query (default: 5)")
    p.add_argument("--chunk-size", type=int, default=None, metavar="CHARS",
                   help="Override the dataset chunk size in characters")
    p.add_argument("--chunk-overlap", type=int, default=None, metavar="CHARS",
                   help="Override the chunk overlap in characters")
    p.add_argument("--no-chunking", action="store_true", default=False,
                   help="Keep each corpus record as one retrieval unit")
    p.add_argument("--no-multiquery", action="store_true", default=False,
                   help="Disable LLM query expansion (faster, no extra LLM calls)")
    p.add_argument("--no-rerank", action="store_true", default=False,
                   help="Disable BGE cross-encoder reranking and use RRF ranking")
    p.add_argument("--retrieval-mode", choices=["dense", "bm25", "hybrid", "adaptive"],
                   default="hybrid", help="Retriever ablation mode")
    p.add_argument("--candidate-top-k", type=int, default=None, metavar="K",
                   help="Keep top K unique documents from each retriever before fusion")
    p.add_argument("--rerank-top-n", type=int, default=None, metavar="N",
                   help="Override the number of fused candidates passed to reranking")
    p.add_argument("--rebuild", action="store_true", default=False,
                   help="Force re-embed corpus even if ChromaDB cache exists")
    p.add_argument(
        "--provider",
        choices=["deepseek", "groq", "ollama"],
        default="deepseek",
        help="LLM provider for generation and RAGAS judge (default: deepseek)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        help=(
            "Model name override. "
            "Defaults: deepseek=deepseek-flash, "
            "groq=llama-3.3-70b-versatile, ollama=llama3.1:8b. "
            "Example: --provider ollama --model mistral:7b"
        ),
    )

    # ── Reporting ─────────────────────────────────────────────────────────
    p.add_argument("--chart", type=Path, default=None, metavar="PATH",
                   help="Save comparison bar chart to this PNG path")
    p.add_argument("--compare", action="store_true", default=False,
                   help="Print config comparison table (no new evaluation)")
    p.add_argument("--trend", action="store_true", default=False,
                   help="Print metric trend for --config (no new evaluation)")
    p.add_argument("--csv", type=Path, default=None, metavar="PATH",
                   help="Override regression CSV path")
    p.add_argument("--details", type=Path, default=None, metavar="PATH",
                   help="Save per-query RAGAS scores to CSV")
    p.add_argument("--generation-cache", type=Path, default=None, metavar="PATH",
                   help="Checkpoint retrieval results and generated answers as JSON")

    # ── General ───────────────────────────────────────────────────────────
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"],
                   help="Logging verbosity (default: INFO)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy third-party loggers
    for lib in ("httpx", "httpcore", "urllib3", "chromadb",
                "sentence_transformers", "huggingface_hub"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)

    # ── Regression tracker ─────────────────────────────────────────────────
    from finrag.regression import RegressionTracker
    tracker = RegressionTracker(csv_path=args.csv) if args.csv else RegressionTracker()

    # ── Report-only modes ─────────────────────────────────────────────────
    if args.compare:
        tracker.compare_configs(
            dataset=args.dataset,
            output_path=args.chart,
            show=False,
        )
        return

    if args.trend:
        tracker.trend(
            dataset=args.dataset,
            config_name=args.config,
            output_path=args.chart,
            show=False,
        )
        return

    # ── Full evaluation run ───────────────────────────────────────────────
    from finrag.config import DATASET_CONFIGS
    from finrag.data import load_jsonl, load_qrels, make_corpus_lookup
    from finrag.chunking import split_documents
    from finrag.models import (
        DEEPSEEK_DEFAULT_MODEL,
        GROQ_DEFAULT_MODEL,
        OLLAMA_DEFAULT_MODEL,
        empty_cache,
        get_embedding_model,
        get_provider_eval_llm,
        get_provider_llm,
        get_reranker,
    )
    from finrag.retrieval import build_bm25_okapi, build_bm25_retriever, retrieve_and_rerank
    from finrag.vectorstore import get_vectorstore
    from finrag.generation import generate_answer

    cfg = DATASET_CONFIGS[args.dataset]
    use_multiquery = not args.no_multiquery

    logger.info("=" * 60)
    logger.info(
        "RAGAS Eval  |  dataset=%s  config=%s  n=%d  multiquery=%s",
        args.dataset, args.config, args.n, use_multiquery,
    )
    logger.info("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    from finrag.config import DATASET_DIR
    corpus  = load_jsonl(DATASET_DIR / cfg.corpus_file)
    queries = load_jsonl(DATASET_DIR / cfg.query_file)
    corpus_lookup = make_corpus_lookup(corpus)

    logger.info("Corpus: %d docs | Queries: %d", len(corpus), len(queries))

    # ── Build pipeline components ──────────────────────────────────────────
    effective_chunk_size = args.chunk_size or cfg.chunk_size
    effective_chunk_overlap = (
        cfg.chunk_overlap if args.chunk_overlap is None else args.chunk_overlap
    )
    chunks = split_documents(
        corpus,
        effective_chunk_size,
        effective_chunk_overlap,
        cfg.dataset_type,
        no_chunking=args.no_chunking,
    )
    embedding_model = get_embedding_model()
    reranker        = get_reranker() if not args.no_rerank else None

    gen_llm = get_provider_llm(args.provider, args.model)
    if gen_llm is None:
        logger.error(
            "LLM unavailable — for DeepSeek: set DEEPSEEK_API_KEY in .env; "
            "for Groq: set GROQ_API_KEY in .env; "
            "for Ollama: run `ollama serve` and `ollama pull <model>`."
        )
        sys.exit(1)

    eval_llm = get_provider_eval_llm(args.provider, args.model)
    if eval_llm is None:
        logger.error("RAGAS judge LLM unavailable.")
        sys.exit(1)

    # MultiQuery reuses gen_llm — avoids loading two separate models
    retrieval_llm = gen_llm if use_multiquery else None

    default_models = {
        "deepseek": DEEPSEEK_DEFAULT_MODEL,
        "groq": GROQ_DEFAULT_MODEL,
        "ollama": OLLAMA_DEFAULT_MODEL,
    }
    actual_model = args.model or default_models[args.provider]

    logger.info(
        "Provider: %s  |  model: %s  |  multiquery: %s",
        args.provider, actual_model, use_multiquery,
    )

    if args.no_chunking:
        vectorstore_name = f"{cfg.name}_whole_documents"
    elif args.chunk_size is not None or args.chunk_overlap is not None:
        vectorstore_name = (
            f"{cfg.name}_chunk{effective_chunk_size}"
            f"_overlap{effective_chunk_overlap}"
        )
    else:
        vectorstore_name = cfg.name
    vectorstore = get_vectorstore(
        vectorstore_name,
        chunks,
        embedding_model,
        force_rebuild=args.rebuild,
    )
    bm25_retriever = build_bm25_retriever(chunks.lc_docs, fetch_k=cfg.fetch_k)
    bm25_okapi = build_bm25_okapi(chunks.texts) if not args.no_rerank else None

    # ── Sample queries ─────────────────────────────────────────────────────
    import random
    random.seed(args.seed)

    eligible = queries
    sample   = (
        random.sample(eligible, min(args.n, len(eligible)))
        if len(eligible) > args.n
        else eligible
    )

    logger.info("Retrieving + generating for %d sampled queries …", len(sample))

    retrieved_map: dict[str, list[tuple[str, float]]] = {}
    answers_map:   dict[str, str]                     = {}

    cache_path = args.generation_cache or (
        Path("results") / f"generation_{args.dataset}_{args.config}.json"
    )
    generation_cache: dict[str, dict] = {}
    if cache_path.exists():
        generation_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        logger.info(
            "Loaded %d cached generations from %s",
            len(generation_cache),
            cache_path,
        )

    from tqdm import tqdm

    for q in tqdm(sample, desc="  retrieve+gen", unit="query"):
        qid = q["_id"]
        cached = generation_cache.get(qid)
        if cached is not None:
            retrieved_map[qid] = [tuple(item) for item in cached["retrieved"]]
            answers_map[qid] = cached["answer"]
            continue
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
                llm=retrieval_llm,
                k=args.top_k,
                rerank_top_n=args.rerank_top_n or cfg.rerank_top_n,
                retrieval_mode=args.retrieval_mode,
                candidate_top_k=args.candidate_top_k,
            )
            retrieved_map[qid] = ranked

            gen_result = generate_answer(
                query=q["text"],
                retrieved=ranked,
                corpus_lookup=corpus_lookup,
                llm=gen_llm,
                top_k=args.top_k,
                multiquery=retrieval_llm is not None,
                model_name=actual_model,
                use_structured_output=args.provider != "deepseek",
            )
            if gen_result.error is None:
                answers_map[qid] = gen_result.answer
                generation_cache[qid] = {
                    "retrieved": ranked,
                    "answer": gen_result.answer,
                }
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
                temporary_cache.write_text(
                    json.dumps(generation_cache, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temporary_cache.replace(cache_path)
            else:
                logger.warning("Generation failed for %s: %s", qid, gen_result.error)

        except Exception as exc:
            logger.warning("Query %s failed: %s", qid, exc)

    empty_cache()

    # ── RAGAS evaluation ───────────────────────────────────────────────────
    # Import here — compat patch is applied inside ragas_eval module
    from finrag.ragas_eval import run_ragas_eval

    ragas_result = run_ragas_eval(
        queries=sample,
        retrieved_map=retrieved_map,
        answers_map=answers_map,
        corpus_lookup=corpus_lookup,
        llm=eval_llm,
        embeddings=embedding_model,
        dataset_name=args.dataset,
        config_name=args.config,
        model_name=f"{args.provider}/{actual_model}",  # e.g. "groq/llama-3.3-70b-versatile"
        n_samples=args.n,
        sample_seed=args.seed,
        context_top_k=args.top_k,
    )

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + ragas_result.summary())

    details_path = args.details or (
        Path("results") / f"ragas_{args.dataset}_{args.config}_details.csv"
    )
    details_path.parent.mkdir(parents=True, exist_ok=True)
    ragas_result.per_query_df.to_csv(details_path, index=False)
    logger.info("Per-query RAGAS details saved to %s", details_path)

    # ── Record in regression CSV ───────────────────────────────────────────
    tracker.record(ragas_result)

    # ── Optional comparison chart ──────────────────────────────────────────
    if args.chart:
        tracker.compare_configs(
            dataset=args.dataset,
            output_path=args.chart,
            show=False,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
