"""
ChromaDB vector store — build from scratch or load from disk cache.
"""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from tqdm import tqdm

from .chunking import ChunkResult
from .config import CHROMA_DIR

logger = logging.getLogger(__name__)

_BATCH_SIZE = 64


def get_vectorstore(
    dataset_name: str,
    chunk_result: ChunkResult,
    embedding_model,
    *,
    force_rebuild: bool = False,
) -> Chroma:
    """
    Return a ChromaDB vectorstore for *dataset_name*.

    If a persist directory already exists on disk (and force_rebuild is False),
    the existing index is loaded — skipping re-embedding entirely.
    Otherwise the corpus is embedded and persisted in CHROMA_DIR/dataset_name.

    Parameters
    ----------
    dataset_name    : e.g. "financebench"
    chunk_result    : output of chunking.split_documents
    embedding_model : HuggingFaceEmbeddings singleton
    force_rebuild   : delete and rebuild even if cache exists
    """
    persist_dir = CHROMA_DIR / dataset_name

    if persist_dir.exists() and not force_rebuild:
        logger.info(
            "Loading cached vector store for '%s' from %s",
            dataset_name, persist_dir,
        )
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embedding_model,
        )

    logger.info(
        "Building vector store for '%s' (%d chunks) …",
        dataset_name, len(chunk_result.texts),
    )
    persist_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_model,
    )

    texts   = chunk_result.texts
    ids     = chunk_result.chroma_ids
    orig    = chunk_result.original_ids
    batches = range(0, len(texts), _BATCH_SIZE)

    for i in tqdm(batches, desc=f"  Indexing {dataset_name}", unit="batch"):
        vectorstore.add_texts(
            texts=texts[i : i + _BATCH_SIZE],
            metadatas=[{"id": oid} for oid in orig[i : i + _BATCH_SIZE]],
            ids=ids[i : i + _BATCH_SIZE],
        )

    logger.info("Vector store for '%s' built and persisted.", dataset_name)
    return vectorstore
