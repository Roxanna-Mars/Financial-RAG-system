"""
Model singletons — embedding model, cross-encoder reranker, and LLM.

All models are loaded lazily on first access and cached for reuse across
datasets. Device detection prioritises: CUDA → MPS (Apple Silicon) → CPU.

LLM providers
-------------
groq   : Groq cloud API — llama-3.3-70b-versatile (free tier, 100 K tokens/day)
deepseek: DeepSeek OpenAI-compatible API — deepseek-flash default
ollama : Local Ollama server — llama3.1:8b default (no limits, no internet)

Use get_provider_llm(provider, model) anywhere in the pipeline without caring
which backend is active.  get_provider_eval_llm() gives a judge-suitable variant
(rate-limited for Groq, larger context budget for both).
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ── Default model names per provider ─────────────────────────────────────────
# GROQ_MODEL env var lets you switch models via .env without touching code:
#   GROQ_MODEL=llama-3.3-70b-versatile   ← best quality
#   GROQ_MODEL=llama-3.1-8b-instant      ← faster, higher TPM limit
GROQ_DEFAULT_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEEPSEEK_DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
OLLAMA_DEFAULT_MODEL = "llama3.1:8b"


# ── Device detection ─────────────────────────────────────────────────────────

def get_device() -> str:
    """Return the best available compute device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_cache() -> None:
    """Release GPU / MPS memory cache."""
    device = get_device()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


# ── Lazy singletons ───────────────────────────────────────────────────────────
_embedding_model = None
_reranker        = None
_llm             = None   # Groq singleton only


def get_embedding_model():
    """
    Load FinLang/finance-embeddings-investopedia (finance-domain finetuned).
    Cached after first call.
    """
    global _embedding_model
    if _embedding_model is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        device = get_device()
        logger.info("Loading embedding model on device=%s …", device)
        _embedding_model = HuggingFaceEmbeddings(
            model_name="FinLang/finance-embeddings-investopedia",
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
        )
        logger.info("Embedding model ready.")
    return _embedding_model


def get_reranker():
    """
    Load BAAI/bge-reranker-v2-m3 cross-encoder (2.27 GB).
    Cached after first call.
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        device = get_device()
        logger.info("Loading reranker on device=%s …", device)
        _reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device=device,
            max_length=512,
        )
        logger.info("Reranker ready.")
    return _reranker


# ── Groq (cloud) ──────────────────────────────────────────────────────────────

def get_llm() -> Optional[object]:
    """
    Load Groq LLM (default model, singleton).
    Returns None if GROQ_API_KEY is absent — pipeline continues without MultiQuery.
    """
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning(
                "GROQ_API_KEY not set — MultiQuery expansion disabled. "
                "Add it to .env to enable."
            )
            return None

        from langchain_groq import ChatGroq

        logger.info("Loading Groq LLM (%s) …", GROQ_DEFAULT_MODEL)
        _llm = ChatGroq(
            model=GROQ_DEFAULT_MODEL,
            temperature=0,
            max_tokens=256,
        )
        logger.info("LLM ready.")
    return _llm


def get_eval_llm(model_name: str | None = None) -> Optional[object]:
    """
    Rate-limited Groq LLM for RAGAS evaluation (judge calls).

    RAGAS issues one LLM call per metric per sample — 20 samples × 3 metrics
    = 60 calls.  Groq free tier allows 30 req/min, so we cap at 0.4 req/s.

    Returns None if GROQ_API_KEY is absent.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — RAGAS evaluation requires an LLM.")
        return None

    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_groq import ChatGroq

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.4,   # 24 req/min — safe under 30 req/min free-tier cap
        check_every_n_seconds=0.1,
        max_bucket_size=5,
    )

    selected_model = model_name or GROQ_DEFAULT_MODEL
    logger.info(
        "Loading rate-limited Groq eval LLM (%s, 0.4 req/s) …",
        selected_model,
    )
    return ChatGroq(
        model=selected_model,
        temperature=0,
        max_tokens=1024,
        rate_limiter=rate_limiter,
    )


# ── Ollama (local) ────────────────────────────────────────────────────────────

def get_ollama_llm(model_name: str = OLLAMA_DEFAULT_MODEL) -> object:
    """
    Load a local Ollama LLM.

    Requires Ollama to be running:
        ollama serve
        ollama pull llama3.1:8b   # or any other model

    OLLAMA_BASE_URL env var overrides the default http://localhost:11434.
    No singleton — ChatOllama is cheap to construct (no network call until
    the first .invoke()).
    """
    from langchain_ollama import ChatOllama

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    logger.info("Loading Ollama LLM (%s) @ %s …", model_name, base_url)
    return ChatOllama(
        model=model_name,
        temperature=0,
        num_predict=256,   # ChatOllama uses num_predict, not max_tokens
        base_url=base_url,
    )


def get_deepseek_llm(
    model_name: str = DEEPSEEK_DEFAULT_MODEL,
    *,
    max_tokens: int = 256,
) -> Optional[object]:
    """Create a DeepSeek chat model through its OpenAI-compatible endpoint."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning(
            "DEEPSEEK_API_KEY not set — add it to .env to enable DeepSeek."
        )
        return None

    from langchain_openai import ChatOpenAI

    logger.info("Loading DeepSeek LLM (%s) …", model_name)
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0,
        max_tokens=max_tokens,
    )


# ── Unified provider dispatchers ──────────────────────────────────────────────

def get_provider_llm(
    provider: str = "deepseek",
    model: str | None = None,
) -> Optional[object]:
    """
    Return the right LLM for the given provider.

    Parameters
    ----------
    provider : "deepseek" | "groq" | "ollama"
    model    : override the default model name for the chosen provider.
               Defaults: deepseek=deepseek-flash,
               groq=llama-3.3-70b-versatile, ollama=llama3.1:8b

    Returns None only for Groq when GROQ_API_KEY is absent.
    """
    if provider == "ollama":
        return get_ollama_llm(model or OLLAMA_DEFAULT_MODEL)

    if provider == "deepseek":
        return get_deepseek_llm(model or DEEPSEEK_DEFAULT_MODEL)

    if provider == "groq":
        if model and model != GROQ_DEFAULT_MODEL:
            # Non-default model — bypass singleton, create a fresh instance
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY not set — Groq LLM disabled.")
                return None
            from langchain_groq import ChatGroq
            logger.info("Loading Groq LLM (%s) …", model)
            return ChatGroq(model=model, temperature=0, max_tokens=256)
        return get_llm()   # default model — use singleton

    raise ValueError(
        f"Unknown provider '{provider}'. Choose 'deepseek', 'groq', or 'ollama'."
    )


def get_provider_eval_llm(
    provider: str = "deepseek",
    model: str | None = None,
) -> Optional[object]:
    """
    Return a judge-quality LLM for RAGAS evaluation.

    Groq  : rate-limited at 0.4 req/s, max_tokens=1024
    Ollama: no rate limiting needed, num_predict=1024
    """
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        m = model or OLLAMA_DEFAULT_MODEL
        logger.info("Loading Ollama eval LLM (%s) — no rate limiting …", m)
        return ChatOllama(
            model=m,
            temperature=0,
            num_predict=1024,
            base_url=base_url,
        )

    if provider == "deepseek":
        return get_deepseek_llm(
            model or DEEPSEEK_DEFAULT_MODEL,
            max_tokens=1024,
        )

    if provider == "groq":
        return get_eval_llm(model)   # rate-limited Groq path

    raise ValueError(
        f"Unknown provider '{provider}'. Choose 'deepseek', 'groq', or 'ollama'."
    )
