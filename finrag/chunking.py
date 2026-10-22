"""
Dataset-aware document splitting.

Passage datasets  → split on sentence boundaries (". ")
Tabular datasets  → split on pipe "|" to preserve table-row integrity
"""

import logging
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    texts: list[str]          # chunk text strings
    chroma_ids: list[str]     # unique IDs for ChromaDB
    original_ids: list[str]   # original corpus doc IDs (aligned with texts)
    lc_docs: list[Document]   # LangChain Documents (for BM25Retriever)


def split_documents(
    corpus: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    dataset_type: str = "passage",
    no_chunking: bool = False,
) -> ChunkResult:
    """
    Chunk a corpus into fixed-size pieces using dataset-appropriate separators.

    Parameters
    ----------
    corpus        : raw records from load_jsonl (each has _id, title, text)
    chunk_size    : max characters per chunk (≈ tokens × 4)
    chunk_overlap : overlap between consecutive chunks
    dataset_type  : "passage" or "tabular"
    """
    separators = (
        ["\n\n", "\n", "|", " ", ""]
        if dataset_type == "tabular"
        else ["\n\n", "\n", ". ", " ", ""]
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    texts        : list[str]      = []
    chroma_ids   : list[str]      = []
    original_ids : list[str]      = []
    lc_docs      : list[Document] = []

    for doc_idx, doc in enumerate(corpus):
        full_text = (doc.get("title", "") + " " + doc["text"]).strip()
        chunks = [full_text] if no_chunking else splitter.split_text(full_text)

        for chunk_idx, chunk in enumerate(chunks):
            cid = f"{doc_idx}_{doc['_id']}_chunk{chunk_idx}"
            texts.append(chunk)
            chroma_ids.append(cid)
            original_ids.append(doc["_id"])
            lc_docs.append(
                Document(page_content=chunk, metadata={"id": doc["_id"]})
            )

    logger.info(
        "Chunked %d docs → %d chunks (size=%d, overlap=%d, type=%s)",
        len(corpus),
        len(texts),
        chunk_size,
        chunk_overlap,
        dataset_type,
    )
    return ChunkResult(
        texts=texts,
        chroma_ids=chroma_ids,
        original_ids=original_ids,
        lc_docs=lc_docs,
    )
