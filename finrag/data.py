"""
Data loaders for corpus JSONL files and qrels TSV files.
负责把 JSONL 语料和 qrels 评估文件转成程序能直接使用的结构。
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def make_corpus_lookup(corpus: list[dict]) -> dict[str, dict]:
    """
    Index a corpus list by _id for O(1) doc lookup during generation.

    Returns
    -------
    dict mapping corpus_id -> full document dict (has 'title' and 'text')
    """
    lookup = {doc["_id"]: doc for doc in corpus}
    logger.debug("Built corpus lookup for %d documents.", len(lookup))
    return lookup


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONLines file into a list of dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.debug("Loaded %d records from %s", len(records), path.name)
    return records


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """
    Load a qrels TSV file.

    Returns
    -------
    dict mapping query_id -> {corpus_id: relevance_score}
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Qrels file not found: %s — evaluation will be skipped", path)
        return {}

    df = pd.read_csv(path, sep="\t")
    qrels = (
        df.groupby("query_id")
        .apply(
            lambda x: dict(zip(x["corpus_id"], x["score"])),
            include_groups=False,
        )
        .to_dict()
    )
    logger.debug(
        "Loaded qrels for %d queries from %s", len(qrels), path.name
    )
    return qrels
