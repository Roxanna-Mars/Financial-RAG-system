"""
FinRAG — Financial Retrieval-Augmented Generation pipeline.

Modules
-------
config      Dataset configurations and project paths.
data        JSONL / TSV loaders.
chunking    Dataset-aware document splitting.
models      Embedding model, reranker, and LLM (lazy singletons, MPS-aware).
vectorstore ChromaDB build / load helpers.
retrieval   BM25, EnsembleRetriever, retrieve-and-rerank.
evaluation  NDCG@10 and coverage metrics.
pipeline    End-to-end orchestrator.
"""
