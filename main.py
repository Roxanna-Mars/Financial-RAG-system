"""
把各种参数接收进来，然后决定调用哪个 pipeline。
FinRAG CLI

Two modes
─────────
1. Retrieval benchmark  — evaluate NDCG@10 over a full dataset
2. Interactive RAG      — ask a question, get a grounded answer with citations

Examples
--------
# Ask a single question (full RAG — retrieve + generate)
python main.py --dataset financebench --query "What was PepsiCo's net revenue in FY2022?"

# Run the retrieval benchmark on one dataset
python main.py --dataset financebench

# Run benchmark on all 7 datasets
python main.py --all

# Force re-embed corpus (ignore ChromaDB cache)
python main.py --dataset financebench --rebuild

# Disable MultiQuery (no Groq needed, faster)
python main.py --dataset financebench --no-multiquery
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from finrag.config import DATASET_CONFIGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="finrag",
        description="Financial RAG — hybrid retrieval + grounded generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Target dataset ─────────────────────────────────────────────────────
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS),
        metavar="NAME",
        help=f"Dataset to use. Choices: {', '.join(DATASET_CONFIGS)}",
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Run retrieval benchmark over all 7 datasets sequentially.",
    )

    # ── RAG query (interactive mode) ───────────────────────────────────────
    parser.add_argument(
        "--query", "-q",
        type=str,
        metavar="QUESTION",
        help="Ask a financial question. Enables full RAG (retrieve + generate).",
    )

    # ── Retrieval options ──────────────────────────────────────────────────
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        metavar="K",
        help="Documents to retrieve per query (default: 10; RAG mode default: 5).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="CHARS",
        help="Override the dataset chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        metavar="CHARS",
        help="Override the chunk overlap in characters.",
    )
    parser.add_argument(
        "--no-chunking",
        action="store_true",
        default=False,
        help="Keep each corpus record as one retrieval unit.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="Re-embed corpus even if ChromaDB cache exists.",
    )
    parser.add_argument(
        "--no-multiquery",
        action="store_true",
        default=False,
        help="Disable MultiQuery LLM expansion (faster, no extra LLM calls).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        default=False,
        help="Disable BGE cross-encoder reranking and use the RRF ranking.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "bm25", "hybrid", "adaptive"],
        default="hybrid",
        help="Retriever ablation mode (default: hybrid).",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=None,
        metavar="K",
        help="Keep the top K unique documents from each retriever before fusion.",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Override how many fused candidates are passed to the reranker.",
    )
    parser.add_argument(
        "--provider",
        choices=["deepseek", "groq", "ollama"],
        default="deepseek",
        metavar="PROVIDER",
        help="LLM provider: 'deepseek' (default), 'groq', or 'ollama'.",
    )
    parser.add_argument(
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

    # ── Output ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--save-results",
        type=Path,
        metavar="PATH",
        help="Save retrieval results as JSON to this path.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity (default: INFO).",
    )

    return parser.parse_args()


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

    use_multiquery = not args.no_multiquery

    # ── Mode 1: interactive RAG (--query provided) ─────────────────────────
    if args.query:
        if args.all:
            print("--query cannot be combined with --all. Pick one dataset.")
            sys.exit(1)

        from finrag.pipeline import run_rag_query

        top_k = args.top_k if args.top_k != 10 else 5   # sensible default for RAG

        print(f"\nDataset  : {args.dataset}")
        print(f"Provider : {args.provider}" + (f"  model={args.model}" if args.model else ""))
        print(f"Query    : {args.query}\n")

        result = run_rag_query(
            query=args.query,
            dataset_name=args.dataset,
            top_k=top_k,
            use_multiquery=use_multiquery,
            force_rebuild=args.rebuild,
            provider=args.provider,
            model=args.model,
            use_reranker=not args.no_rerank,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            no_chunking=args.no_chunking,
            retrieval_mode=args.retrieval_mode,
            candidate_top_k=args.candidate_top_k,
            rerank_top_n=args.rerank_top_n,
        )
        print(result.pretty())
        return

    # ── Mode 2: retrieval benchmark ────────────────────────────────────────
    from finrag.pipeline import run_all, run_dataset

    if args.all:
        all_results = run_all(
            use_multiquery=use_multiquery,
            force_rebuild=args.rebuild,
            top_k=args.top_k,
            provider=args.provider,
            model=args.model,
            use_reranker=not args.no_rerank,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            no_chunking=args.no_chunking,
            retrieval_mode=args.retrieval_mode,
            candidate_top_k=args.candidate_top_k,
            rerank_top_n=args.rerank_top_n,
        )
        if args.save_results:
            _save_json(
                {n: r.results for n, r in all_results.items()},
                args.save_results,
            )
    else:
        result = run_dataset(
            args.dataset,
            use_multiquery=use_multiquery,
            force_rebuild=args.rebuild,
            top_k=args.top_k,
            provider=args.provider,
            model=args.model,
            use_reranker=not args.no_rerank,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            no_chunking=args.no_chunking,
            retrieval_mode=args.retrieval_mode,
            candidate_top_k=args.candidate_top_k,
            rerank_top_n=args.rerank_top_n,
        )
        ndcg_str = f"{result.ndcg:.4f}" if result.ndcg == result.ndcg else "N/A"
        print(f"\n{'='*44}")
        print(f"  Dataset  : {result.name}")
        print(f"  NDCG@{args.top_k}  : {ndcg_str}")
        recall_str = f"{result.recall:.4f}" if result.recall == result.recall else "N/A"
        precision_str = (
            f"{result.precision:.4f}"
            if result.precision == result.precision else "N/A"
        )
        mrr_str = f"{result.mrr:.4f}" if result.mrr == result.mrr else "N/A"
        hit_str = f"{result.hit_rate:.4f}" if result.hit_rate == result.hit_rate else "N/A"
        print(f"  Recall@{args.top_k}: {recall_str}")
        print(f"  Precision@{args.top_k}: {precision_str}")
        print(f"  MRR@{args.top_k}: {mrr_str}")
        print(f"  Hit Rate@{args.top_k}: {hit_str}")
        print(f"  Evaluated: {result.evaluated_queries} labeled queries")
        print(f"  Queries  : {result.num_queries}")
        print(f"  Chunks   : {result.num_chunks}")
        print(f"  Time     : {result.elapsed_sec:.1f}s")
        if result.errors:
            print(f"  Errors   : {len(result.errors)}")
        print(f"{'='*44}\n")

        if args.save_results:
            _save_json(result.results, args.save_results)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    logging.getLogger(__name__).info("Results saved → %s", path)


if __name__ == "__main__":
    main()
