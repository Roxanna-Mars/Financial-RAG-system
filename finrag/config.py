"""
Project paths and per-dataset retrieval configuration.
管理项目路径和数据集参数：
统一项目中各数据集的文件位置
统一每个数据集的检索与重排参数
让后续训练/评测代码能通过 DATASET_CONFIGS["financebench"] 的方式直接获取配置，而不需要硬编码到各个脚本里
"""

from dataclasses import dataclass
from pathlib import Path

# ── Project-level paths ──────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent   # repo root
DATASET_DIR = BASE_DIR / "Dataset"
CHROMA_DIR  = BASE_DIR / "chroma_stores"               # gitignored


# ── Per-dataset configuration ────────────────────────────────────────────────
@dataclass(frozen=True)
class DatasetConfig:
    name: str
    corpus_file: str          # relative to DATASET_DIR
    query_file: str
    qrels_file: str
    chunk_size: int
    chunk_overlap: int
    dataset_type: str         # "passage" | "tabular"
    fetch_k: int              # 候选召回数量 candidates fetched before reranking
    rerank_top_n: int         # reranker 最终保留的候选数 candidates passed to cross-encoder


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    # ── Passage retrieval ────────────────────────────────────────────────────
    "financebench": DatasetConfig(
        name="financebench",
        corpus_file="financebench_corpus.jsonl/corpus.jsonl",
        query_file="financebench_queries.jsonl/queries.jsonl",
        qrels_file="FinanceBench_qrels.tsv",
        chunk_size=512,
        chunk_overlap=64,
        dataset_type="passage",
        fetch_k=75,
        rerank_top_n=30,
    ),
    "finder": DatasetConfig(
        name="finder",
        corpus_file="finder_corpus.jsonl/corpus.jsonl",
        query_file="finder_queries.jsonl/queries.jsonl",
        qrels_file="FinDER_qrels.tsv",
        chunk_size=512,
        chunk_overlap=64,
        dataset_type="passage",
        fetch_k=75,
        rerank_top_n=30,
    ),
    "finqabench": DatasetConfig(
        name="finqabench",
        corpus_file="finqabench_corpus.jsonl/corpus.jsonl",
        query_file="finqabench_queries.jsonl/queries.jsonl",
        qrels_file="FinQABench_qrels.tsv",
        chunk_size=1024,
        chunk_overlap=128,
        dataset_type="passage",
        fetch_k=75,
        rerank_top_n=30,
    ),
    # ── Tabular + text ───────────────────────────────────────────────────────
    "finqa": DatasetConfig(
        name="finqa",
        corpus_file="finqa_corpus.jsonl/corpus.jsonl",
        query_file="finqa_queries.jsonl/queries.jsonl",
        qrels_file="FinQA_qrels.tsv",
        chunk_size=1024,
        chunk_overlap=128,
        dataset_type="tabular",
        fetch_k=40,
        rerank_top_n=20,
    ),
    "tatqa": DatasetConfig(
        name="tatqa",
        corpus_file="tatqa_corpus.jsonl/corpus.jsonl",
        query_file="tatqa_queries.jsonl/queries.jsonl",
        qrels_file="TATQA_qrels.tsv",
        chunk_size=1024,
        chunk_overlap=128,
        dataset_type="tabular",
        fetch_k=50,
        rerank_top_n=25,
    ),
    "convfinqa": DatasetConfig(
        name="convfinqa",
        corpus_file="convfinqa_corpus.jsonl/corpus.jsonl",
        query_file="convfinqa_queries.jsonl/queries.jsonl",
        qrels_file="ConvFinQA_qrels.tsv",
        chunk_size=1024,
        chunk_overlap=128,
        dataset_type="tabular",
        fetch_k=50,
        rerank_top_n=25,
    ),
    "multiheirtt": DatasetConfig(
        name="multiheirtt",
        corpus_file="multiheirtt_corpus.jsonl/corpus.jsonl",
        query_file="multiheirtt_queries.jsonl/queries.jsonl",
        qrels_file="MultiHeirtt_qrels.tsv",
        chunk_size=1024,
        chunk_overlap=128,
        dataset_type="tabular",
        fetch_k=40,
        rerank_top_n=20,
    ),
}
