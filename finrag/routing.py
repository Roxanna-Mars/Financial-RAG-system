"""Explainable query classification and adaptive retrieval routing."""

from dataclasses import dataclass
import re


FACT_LOOKUP = "fact_lookup"
TERM_LOOKUP = "term_lookup"
NUMERICAL = "numerical"
COMPARISON = "comparison"
MULTI_HOP = "multi_hop"


_NUMERICAL_PATTERNS = (
    r"\bcalculate\b",
    r"\bcompute\b",
    r"\bratio\b",
    r"\bpercentage\b",
    r"\bpercent\b",
    r"\baverage\b",
    r"\bmargin\b",
    r"\bgrowth rate\b",
    r"\bround(?:ed)?\b",
    r"\bdivided by\b",
    r"\bper share\b",
)
_COMPARISON_PATTERNS = (
    r"\bcompare\b",
    r"\bchange from\b",
    r"\bincrease(?:d)?\b",
    r"\bdecrease(?:d)?\b",
    r"\bdifference\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\byear[- ]over[- ]year\b",
)
_TERM_PATTERNS = (
    r"\bwhat (?:does|is|are) .{0,30} mean\b",
    r"\bdefine\b",
    r"\bdefinition\b",
    r"\bppne\b",
    r"\bpp\s*&\s*e\b",
    r"\bebitda\b",
    r"\bebit\b",
    r"\bcapex\b",
    r"\bopex\b",
    r"\bcogs\b",
    r"\bro[ae]\b",
)


@dataclass(frozen=True)
class RoutingDecision:
    """Retrieval path selected for one query."""

    query_type: str
    retrieval_mode: str
    use_multiquery: bool
    reason: str


def classify_query(query: str) -> str:
    """Classify a financial question using deterministic, auditable rules."""
    normalized = " ".join(query.lower().split())
    fiscal_years = set(re.findall(r"\b(?:19|20)\d{2}\b", normalized))
    has_numerical = any(re.search(pattern, normalized) for pattern in _NUMERICAL_PATTERNS)
    has_comparison = any(re.search(pattern, normalized) for pattern in _COMPARISON_PATTERNS)
    has_term = any(re.search(pattern, normalized) for pattern in _TERM_PATTERNS)

    if (
        (len(fiscal_years) >= 2 and has_numerical)
        or "based on" in normalized and normalized.count(" and ") >= 2
        or "multi-step" in normalized
    ):
        return MULTI_HOP
    if has_comparison or len(fiscal_years) >= 2:
        return COMPARISON
    if has_numerical:
        return NUMERICAL
    if has_term:
        return TERM_LOOKUP
    return FACT_LOOKUP


def route_query(query: str) -> RoutingDecision:
    """Choose the cheapest retrieval path suitable for the query type."""
    query_type = classify_query(query)
    if query_type == FACT_LOOKUP:
        return RoutingDecision(
            query_type=query_type,
            retrieval_mode="dense",
            use_multiquery=False,
            reason="semantic fact lookup; BM25 added little coverage in the baseline",
        )
    if query_type == TERM_LOOKUP:
        return RoutingDecision(
            query_type=query_type,
            retrieval_mode="hybrid",
            use_multiquery=False,
            reason="financial terminology benefits from exact and semantic matching",
        )
    if query_type == COMPARISON:
        return RoutingDecision(
            query_type=query_type,
            retrieval_mode="hybrid",
            use_multiquery=False,
            reason="comparison queries use exact years and metrics; expansion hurt ranking",
        )
    return RoutingDecision(
        query_type=query_type,
        retrieval_mode="hybrid",
        use_multiquery=True,
        reason="complex query benefits from expanded semantic retrieval",
    )
