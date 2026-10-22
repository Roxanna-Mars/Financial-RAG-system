"""
Generation layer — grounded answer synthesis with source citation.

Pipeline
--------
1. Format top-k retrieved docs as numbered context blocks
2. Invoke the LLM (Groq or Ollama) with a financial-analyst prompt
3. Parse structured output: answer text + list of cited doc IDs
4. Return GenerationResult with full citation metadata

Structured output uses Pydantic + LangChain's .with_structured_output(),
with a plain-text fallback if JSON mode fails (common with smaller Ollama models).
"""

import logging
import re
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM = """\
You are a precise financial analyst assistant with expertise in SEC filings, \
earnings reports, and financial disclosures.

Rules:
- Answer using ONLY the document excerpts provided. Do not use outside knowledge.
- Be specific: quote exact figures, percentages, and fiscal periods from the context.
- For numerical questions, show your reasoning step-by-step.
- Cite every document you use with its ID in square brackets, e.g. [dd2af2336].
- If the context does not contain enough information, respond with exactly: \
"Insufficient information in the provided documents."
"""

_HUMAN = """\
Document excerpts:
{context}

---
Question: {query}
"""

# ── Pydantic schema for structured output ────────────────────────────────────

class _StructuredAnswer(BaseModel):
    answer: str = Field(
        description=(
            "The complete answer to the financial question. "
            "Cite document IDs inline using [doc_id] notation."
        )
    )
    cited_ids: list[str] = Field(
        description="Exact document IDs (e.g. 'dd2af2336') referenced in the answer.",
        default_factory=list,
    )


# ── Public data classes ───────────────────────────────────────────────────────

@dataclass
class CitedSource:
    """A single retrieved document used in the answer."""
    corpus_id:  str
    title:      str
    score:      float          # reranker score
    excerpt:    str            # first 400 chars of doc text


@dataclass
class GenerationResult:
    """Full RAG output: answer + grounded citations."""
    query:       str
    answer:      str
    sources:     list[CitedSource]
    model:       str  = "unknown"   # set at call time from the actual LLM used
    multiquery:  bool = False        # whether MultiQuery was used for retrieval
    error:       str | None = None   # populated only when generation itself fails

    def pretty(self) -> str:
        """Human-readable string for CLI / notebook display."""
        lines = [
            "─" * 60,
            f"Query   : {self.query}",
            "─" * 60,
            f"\nAnswer\n{self.answer}\n",
        ]
        if self.sources:
            lines.append("Sources")
            for i, src in enumerate(self.sources, 1):
                title = f" — {src.title}" if src.title else ""
                lines.append(f"  [{i}] {src.corpus_id}{title}")
                lines.append(f"       score={src.score:.3f}")
        lines.append("─" * 60)
        return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def format_context_document(
    doc_id: str,
    doc: dict,
    max_chars: int = 1_500,
) -> str:
    """Format one corpus document exactly as it is presented to the LLM."""
    title = doc.get("title", "")
    text = doc.get("text", "")
    text_trimmed = text[:max_chars]
    if len(text) > max_chars:
        text_trimmed += " …[truncated]"
    header = f"[{doc_id}]" + (f" {title}" if title else "")
    return f"{header}\n{text_trimmed}"


def _build_context(
    retrieved:      list[tuple[str, float]],
    corpus_lookup:  dict[str, dict],
    max_chars:      int = 1_500,
) -> tuple[str, list[CitedSource]]:
    """
    Format retrieved docs as a numbered context block.

    Returns (context_string, ordered_sources).
    """
    blocks:  list[str]         = []
    sources: list[CitedSource] = []

    for doc_id, score in retrieved:
        doc = corpus_lookup.get(doc_id)
        if doc is None:
            logger.debug("Doc %s not found in corpus lookup — skipping.", doc_id)
            continue

        title = doc.get("title", "")
        text  = doc.get("text", "")
        blocks.append(format_context_document(doc_id, doc, max_chars))

        sources.append(CitedSource(
            corpus_id=doc_id,
            title=title,
            score=score,
            excerpt=text[:400],
        ))

    return "\n\n---\n\n".join(blocks), sources


def _extract_ids_from_text(text: str, valid_ids: set[str]) -> list[str]:
    """
    Fallback: extract [doc_id] citations directly from answer text.
    Used when structured output doesn't return cited_ids.
    """
    found = re.findall(r"\[([a-zA-Z0-9_]+)\]", text)
    return [fid for fid in found if fid in valid_ids]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_answer(
    query:          str,
    retrieved:      list[tuple[str, float]],
    corpus_lookup:  dict[str, dict],
    llm,
    *,
    top_k:          int  = 5,
    max_chars:      int  = 1_500,
    multiquery:     bool = False,
    model_name:     str  = "unknown",
    use_structured_output: bool = True,
) -> GenerationResult:
    """
    Synthesise a grounded answer from retrieved documents.

    Parameters
    ----------
    query          : the user's original question
    retrieved      : [(corpus_id, reranker_score), …] from retrieve_and_rerank
    corpus_lookup  : {corpus_id: doc_dict} — built by make_corpus_lookup()
    llm            : ChatGroq instance from models.get_llm()
    top_k          : how many retrieved docs to include in context (default 5)
    max_chars      : max characters per document in the context block
    multiquery     : metadata flag — was MultiQuery used during retrieval?

    Returns
    -------
    GenerationResult with answer text and cited sources.
    """
    if not retrieved:
        return GenerationResult(
            query=query,
            answer="No relevant documents were retrieved for this query.",
            sources=[],
            model=model_name,
            multiquery=multiquery,
        )

    # Use top_k docs for generation (fewer = tighter context, less hallucination)
    top_retrieved = retrieved[:top_k]
    context, sources = _build_context(top_retrieved, corpus_lookup, max_chars=max_chars)

    if not sources:
        return GenerationResult(
            query=query,
            answer="Retrieved documents could not be loaded from corpus.",
            sources=[],
            model=model_name,
            multiquery=multiquery,
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human",  _HUMAN),
    ])

    valid_ids = {s.corpus_id for s in sources}

    # Some OpenAI-compatible providers do not implement response_format.
    # Call them once in plain-text mode instead of paying for a failed request
    # before the fallback request.
    if not use_structured_output:
        try:
            chain = prompt | llm
            response = chain.invoke({"context": context, "query": query})
            answer = response.content
            cited_set = set(_extract_ids_from_text(answer, valid_ids))
            cited_sources = [s for s in sources if s.corpus_id in cited_set]
            return GenerationResult(
                query=query,
                answer=answer,
                sources=cited_sources,
                model=model_name,
                multiquery=multiquery,
            )
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            return GenerationResult(
                query=query,
                answer=f"Generation error: {exc}",
                sources=sources,
                model=model_name,
                multiquery=multiquery,
                error=str(exc),
            )

    # ── Attempt 1: structured output (answer + cited_ids) ─────────────────
    try:
        chain  = prompt | llm.with_structured_output(_StructuredAnswer)
        result = chain.invoke({"context": context, "query": query})

        # Keep only sources that were actually cited
        cited_set = set(result.cited_ids) & valid_ids
        if not cited_set:
            # LLM may have omitted cited_ids but still cited in text
            cited_set = set(_extract_ids_from_text(result.answer, valid_ids))

        cited_sources = [s for s in sources if s.corpus_id in cited_set]

        logger.info(
            "Generated answer  |  cited %d / %d sources  |  answer_len=%d",
            len(cited_sources), len(sources), len(result.answer),
        )

        return GenerationResult(
            query=query,
            answer=result.answer,
            sources=cited_sources,
            model=model_name,
            multiquery=multiquery,
        )

    except Exception as exc:
        logger.warning(
            "Structured output failed (%s) — falling back to plain text.", exc
        )

    # ── Attempt 2: plain-text fallback ────────────────────────────────────
    try:
        chain    = prompt | llm
        response = chain.invoke({"context": context, "query": query})
        answer   = response.content

        cited_set     = set(_extract_ids_from_text(answer, valid_ids))
        cited_sources = [s for s in sources if s.corpus_id in cited_set]

        return GenerationResult(
            query=query,
            answer=answer,
            sources=cited_sources,
            model=model_name,
            multiquery=multiquery,
        )

    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        return GenerationResult(
            query=query,
            answer=f"Generation error: {exc}",
            sources=sources,
            model=model_name,
            multiquery=multiquery,
            error=str(exc),
        )
