"""
Advanced Query Analyzer – multi-level query analysis that detects
query type, complexity, temporal sensitivity, and chooses the optimal
retrieval strategy.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from pydantic import BaseModel, Field
from groq import Groq

from config.settings import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)


# ======================================================================
# Models
# ======================================================================

class QueryStrategy(BaseModel):
    """Strategy output from the query analyzer."""
    query_type: str = Field(
        ...,
        description="One of: simple_fact, comparison, procedural, "
                    "analytical, multi_part, time_sensitive, location_based",
    )
    search_depth: str = Field(
        default="medium",
        description="shallow (3 sources), medium (8), or deep (15+)",
    )
    force_web_search: bool = Field(default=False)
    temporal_boost: bool = Field(
        default=False,
        description="Prioritize recent information",
    )
    sub_questions: List[str] = Field(
        default_factory=list,
        description="Decomposed sub-questions for multi-part queries",
    )
    expected_answer_format: str = Field(
        default="paragraph",
        description="paragraph, bullet_list, table, comparison, step_by_step",
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Key entities extracted from the query",
    )
    top_k_override: Optional[int] = Field(
        default=None,
        description="Override for number of chunks to retrieve",
    )


# ======================================================================
# Temporal keyword detection
# ======================================================================

_TEMPORAL_KEYWORDS = {
    "latest", "recent", "new", "newest", "current", "today",
    "2024", "2025", "2026", "now", "updated", "this year",
    "this month", "this week", "just", "breaking",
}

_COMPARISON_PATTERNS = [
    r"\bvs\.?\b", r"\bversus\b", r"\bcompare\b", r"\bcomparison\b",
    r"\bbetter\b", r"\bworse\b", r"\bdifference\b", r"\bor\b",
]

_LOCATION_KEYWORDS = {
    "where", "location", "find", "buy", "purchase", "dealer",
    "store", "shop", "nearby", "closest", "dubai", "uae",
    "abu dhabi", "saudi", "middle east", "address",
}

_PROCEDURAL_KEYWORDS = {
    "how to", "how do", "steps", "guide", "tutorial",
    "install", "setup", "configure", "process", "procedure",
}


def _contains_temporal(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _TEMPORAL_KEYWORDS)


def _is_comparison(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pat, lower) for pat in _COMPARISON_PATTERNS)


def _is_location_based(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _LOCATION_KEYWORDS)


def _is_procedural(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _PROCEDURAL_KEYWORDS)


def _is_multi_part(text: str) -> bool:
    lower = text.lower()
    and_count = lower.count(" and ")
    comma_count = text.count(",")
    question_count = text.count("?")
    return (and_count >= 2) or (comma_count >= 2) or (question_count >= 2)


# ======================================================================
# LLM-powered analysis for complex cases
# ======================================================================

_SYSTEM_PROMPT = """\
You are a query analysis assistant. Analyze the user query and return
a JSON strategy. Be precise.

Return valid JSON:
{
  "query_type": "simple_fact|comparison|procedural|analytical|multi_part|time_sensitive|location_based",
  "search_depth": "shallow|medium|deep",
  "force_web_search": true/false,
  "temporal_boost": true/false,
  "sub_questions": ["sub-question 1", ...],
  "expected_answer_format": "paragraph|bullet_list|table|comparison|step_by_step",
  "entities": ["entity1", "entity2"]
}

Rules:
- simple_fact: "What is X?" → shallow depth
- comparison: "X vs Y" → medium depth, table format
- procedural: "How to X?" → medium depth, step_by_step format
- analytical: "Why X?" → deep depth, paragraph
- multi_part: Multiple sub-questions → deep depth, decompose into sub_questions
- time_sensitive: Recent events → force_web_search=true, temporal_boost=true
- location_based: "Where to buy X?" → force_web_search=true for current data
"""


def analyze_query(
    query: str,
    conversation_history: Optional[List[dict]] = None,
) -> QueryStrategy:
    """Analyze *query* and return an optimal retrieval strategy."""

    # ---- Fast-path heuristics (skip LLM for obvious cases) ----
    if _contains_temporal(query):
        strategy_type = "time_sensitive"
        force_web = True
        temporal = True
        depth = "deep"
        fmt = "paragraph"
    elif _is_comparison(query):
        strategy_type = "comparison"
        force_web = False
        temporal = False
        depth = "medium"
        fmt = "comparison"
    elif _is_location_based(query):
        strategy_type = "location_based"
        force_web = True
        temporal = True
        depth = "medium"
        fmt = "bullet_list"
    elif _is_procedural(query):
        strategy_type = "procedural"
        force_web = False
        temporal = False
        depth = "medium"
        fmt = "step_by_step"
    elif _is_multi_part(query):
        strategy_type = "multi_part"
        force_web = False
        temporal = False
        depth = "deep"
        fmt = "bullet_list"
    else:
        # Fall through to LLM analysis for ambiguous queries
        return _llm_analyze(query, conversation_history)

    # Compute top_k override
    top_k_map = {"shallow": 4, "medium": 8, "deep": 16}
    top_k = top_k_map.get(depth, 8)

    result = QueryStrategy(
        query_type=strategy_type,
        search_depth=depth,
        force_web_search=force_web,
        temporal_boost=temporal,
        sub_questions=[],
        expected_answer_format=fmt,
        entities=[],
        top_k_override=top_k,
    )
    logger.info("Query strategy (heuristic): %s", result.query_type)
    return result


def _llm_analyze(
    query: str,
    conversation_history: Optional[List[dict]] = None,
) -> QueryStrategy:
    """Use the LLM for nuanced query analysis."""
    client = Groq(api_key=GROQ_API_KEY)

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if conversation_history:
        for msg in conversation_history[-4:]:
            messages.append(msg)
    messages.append({"role": "user", "content": query})

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)

        depth = data.get("search_depth", "medium")
        top_k_map = {"shallow": 4, "medium": 8, "deep": 16}

        result = QueryStrategy(
            query_type=data.get("query_type", "simple_fact"),
            search_depth=depth,
            force_web_search=data.get("force_web_search", False),
            temporal_boost=data.get("temporal_boost", False),
            sub_questions=data.get("sub_questions", []),
            expected_answer_format=data.get("expected_answer_format", "paragraph"),
            entities=data.get("entities", []),
            top_k_override=top_k_map.get(depth, 8),
        )
        logger.info("Query strategy (LLM): %s", result.query_type)
        return result
    except Exception as exc:
        logger.error("Query analysis failed: %s", exc)
        return QueryStrategy(
            query_type="simple_fact",
            search_depth="medium",
            force_web_search=False,
            temporal_boost=False,
            expected_answer_format="paragraph",
            top_k_override=8,
        )
