"""
RAGAS evaluation harness — faithfulness, answer relevancy, context utilization.

Uses the official ragas 0.4.x library with:
  Judge LLM  : selected DeepSeek, Groq, or Ollama provider
  Embeddings : pipeline embedding model (reused locally for answer_relevancy)

Metrics (all reference-free — no ground-truth answers required):
  Faithfulness       : fraction of answer claims grounded in retrieved context
  AnswerRelevancy    : cosine similarity of back-generated questions → original query
  ContextUtilization : whether retrieved contexts are actually used in the response

Note: ContextPrecision (ragas 0.4.x) requires a ground-truth 'reference' column
which our benchmark datasets don't provide as free text. ContextUtilization is the
correct reference-free equivalent — it scores how well the answer leverages context.

The compat module MUST be imported before ragas to patch the missing
langchain_community.chat_models.vertexai module (removed in lc-community 1.x).
"""

# ── Compatibility patch — must come before any ragas import ──────────────────
from . import compat as _compat  # side-effect import: patches sys.modules before ragas loads

import logging
import random
import warnings
from dataclasses import dataclass, field

import pandas as pd
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

# ragas 0.4.x emits DeprecationWarning for this import path — safe to suppress;
# the v0.4 LangchainLLMWrapper-compatible API lives here, not in .collections
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from ragas.metrics import (
        AnswerRelevancy,
        ContextUtilization,  # reference-free; replaces ContextPrecision (needs 'reference')
        Faithfulness,
    )

logger = logging.getLogger(__name__)

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RAGASResult:
    """Aggregate RAGAS scores for one evaluation run."""
    dataset:               str
    config_name:           str
    model:                 str       # "provider/model-name" e.g. "groq/llama-3.3-70b-versatile"
    n_samples:             int
    faithfulness:          float    # [0,1]  fraction of claims grounded in context
    answer_relevancy:      float    # [0,1]  cosine similarity of back-questions → query
    context_utilization:   float    # [0,1]  fraction of retrieved context used in answer
    per_query_df:          pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def summary(self) -> str:
        lines = [
            f"{'─'*52}",
            f"  Dataset               : {self.dataset}",
            f"  Config                : {self.config_name}",
            f"  Model                 : {self.model}",
            f"  Samples               : {self.n_samples}",
            f"{'─'*52}",
            f"  Faithfulness          : {self.faithfulness:.4f}",
            f"  Answer Relevancy      : {self.answer_relevancy:.4f}",
            f"  Context Utilization   : {self.context_utilization:.4f}",
            f"{'─'*52}",
        ]
        return "\n".join(lines)


# ── Metric builders ───────────────────────────────────────────────────────────

def build_ragas_metrics(
    llm,
    embeddings,
) -> tuple[Faithfulness, AnswerRelevancy, ContextUtilization]:
    """
    Instantiate RAGAS metrics with the selected judge LLM and local embeddings.

    The LLM is wrapped in LangchainLLMWrapper so ragas can call it.
    AnswerRelevancy also needs an embedding model to compute cosine similarity.
    All three metrics are reference-free (no ground-truth answers required).
    """
    ragas_llm = LangchainLLMWrapper(llm)
    logger.info("Reusing pipeline embedding model for answer relevancy.")
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    return (
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextUtilization(llm=ragas_llm),
    )


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_eval_dataset(
    sample_queries:  list[dict],
    retrieved_map:   dict[str, list[tuple[str, float]]],
    answers_map:     dict[str, str],
    corpus_lookup:   dict[str, dict],
    max_ctx_chars:   int = 1_500,
    context_top_k:   int = 5,
) -> tuple[EvaluationDataset, list[str]]:
    """
    Convert pipeline outputs into a ragas EvaluationDataset.

    Each sample contains:
      user_input          : the original query text
      retrieved_contexts  : list of retrieved doc texts (one string per doc)
      response            : the LLM-generated answer

    Returns (EvaluationDataset, aligned_query_ids).
    """
    from .generation import format_context_document

    samples:   list[SingleTurnSample] = []
    query_ids: list[str]              = []

    for q in sample_queries:
        qid = q["_id"]

        if qid not in retrieved_map or qid not in answers_map:
            continue

        answer = answers_map[qid]
        if not answer or not answer.strip():
            continue

        contexts: list[str] = []
        for doc_id, _ in retrieved_map[qid][:context_top_k]:
            doc = corpus_lookup.get(doc_id)
            if doc is None:
                continue
            contexts.append(format_context_document(doc_id, doc, max_ctx_chars))

        if not contexts:
            continue

        samples.append(SingleTurnSample(
            user_input=q["text"],
            retrieved_contexts=contexts,
            response=answer,
        ))
        query_ids.append(qid)

    if not samples:
        raise ValueError(
            "No valid samples built — ensure retrieved_map and answers_map "
            "contain matching query IDs with non-empty answers."
        )

    logger.info("Built EvaluationDataset with %d samples.", len(samples))
    return EvaluationDataset(samples=samples), query_ids


# ── Main evaluation entry point ───────────────────────────────────────────────

def run_ragas_eval(
    queries:        list[dict],
    retrieved_map:  dict[str, list[tuple[str, float]]],
    answers_map:    dict[str, str],
    corpus_lookup:  dict[str, dict],
    llm,
    embeddings,
    *,
    dataset_name:   str = "financebench",
    config_name:    str = "baseline",
    model_name:     str = "unknown",   # "provider/model" — shown in summary + CSV
    n_samples:      int = 20,
    sample_seed:    int = 42,
    context_top_k:  int = 5,
) -> RAGASResult:
    """
    Run RAGAS evaluation on a random sample of queries.

    Parameters
    ----------
    queries        : full query list (dicts with _id + text)
    retrieved_map  : query_id → [(corpus_id, score)]  from retrieve_and_rerank
    answers_map    : query_id → answer string          from generate_answer
    corpus_lookup  : {corpus_id: doc}                  from make_corpus_lookup
    llm            : ChatGroq instance (judge LLM — rate-limited)
    dataset_name   : label for regression tracking
    config_name    : config label, e.g. "baseline_k5" or "multiquery_k10"
    n_samples      : queries to evaluate (default 20 — ~3-5 min with Groq)
    sample_seed    : random seed for reproducible sampling

    Returns
    -------
    RAGASResult with per-metric mean scores + full per_query_df DataFrame.
    """
    # Sample queries that have both retrieval + generation outputs
    eligible = [
        q for q in queries
        if q["_id"] in retrieved_map
        and q["_id"] in answers_map
        and answers_map.get(q["_id"], "").strip()
    ]

    if not eligible:
        raise ValueError("No eligible queries — run retrieval + generation first.")

    random.seed(sample_seed)
    sample = (
        random.sample(eligible, min(n_samples, len(eligible)))
        if len(eligible) > n_samples
        else eligible
    )

    logger.info(
        "RAGAS evaluation  |  dataset=%s  config=%s  samples=%d/%d",
        dataset_name, config_name, len(sample), len(queries),
    )

    # Build dataset
    eval_dataset, aligned_query_ids = build_eval_dataset(
        sample_queries=sample,
        retrieved_map=retrieved_map,
        answers_map=answers_map,
        corpus_lookup=corpus_lookup,
        context_top_k=context_top_k,
    )

    # Wire metrics to judge LLM
    faithfulness_m, answer_relevancy_m, context_utilization_m = build_ragas_metrics(
        llm,
        embeddings,
    )

    logger.info(
        "Calling ragas.evaluate()  model=%s  samples=%d …",
        model_name, len(eval_dataset.samples),
    )

    # RunConfig increases per-job timeout (default 60 s is too short when the
    # rate limiter queues 24 concurrent RAGAS jobs behind Groq's 30 req/min cap)
    from ragas import RunConfig
    run_cfg = RunConfig(
        timeout=600,       # seconds per async job (rate limiting)
        max_retries=5,     # ragas-level retries before marking a job as failed
        max_wait=120,      # max backoff wait between retries
    )

    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness_m, answer_relevancy_m, context_utilization_m],
        run_config=run_cfg,
    )

    df = result.to_pandas()
    df.insert(0, "query_id", aligned_query_ids)

    # Log per-metric distributions
    for col in ["faithfulness", "answer_relevancy", "context_utilization"]:
        if col in df.columns:
            logger.info(
                "  %s: mean=%.4f  median=%.4f  std=%.4f",
                col, df[col].mean(), df[col].median(), df[col].std(),
            )

    return RAGASResult(
        dataset=dataset_name,
        config_name=config_name,
        model=model_name,
        n_samples=len(df),
        faithfulness=float(df["faithfulness"].mean()),
        answer_relevancy=float(df["answer_relevancy"].mean()),
        context_utilization=float(df["context_utilization"].mean()),
        per_query_df=df,
    )
