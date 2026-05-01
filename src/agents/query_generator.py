"""
Query generator agent.

Uses LangChain's with_structured_output() instead of hard-prompting.

Converts a natural-language question into 1-3 optimised search queries
for the vector database. Returns a guaranteed-valid QueryGeneration
Pydantic object — no json.loads, no parsing risk.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config.settings import GROQ_API_KEY, LLM_MODEL
from src.agents.models import QueryGeneration

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────
# Schema enforcement is handled by LangChain — no JSON instructions needed.

_SYSTEM_PROMPT = """\
You are a search-query optimisation assistant. Given a user question,
produce 1-3 highly targeted search queries that will retrieve the most
relevant chunks from a vector database.

Rules:
1. Remove filler words (the, a, an, is, are, do, does).
2. Keep key nouns, verbs, and domain-specific terms.
3. Include synonyms if helpful.
4. Each query should be 3-6 words.
5. Generate 1 query for simple questions, 2-3 for complex ones.
6. If the user asks a VAGUE question like "what is the pdf about" or
   "summarize the document", generate BROAD queries that will match
   the main topic of any document. For example:
   - "overview summary main topic"
   - "product description specifications"
   - "company services features"
   This ensures the vector search retrieves the most representative
   chunks from the knowledge base.
"""

# ── Structured LLM ──────────────────────────────────────────────────────────
_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=0.1,
    max_tokens=300,
)
_structured_llm = _llm.with_structured_output(QueryGeneration)


# ── Public function ──────────────────────────────────────────────────────────

def generate_queries(
    question: str,
    conversation_history: Optional[List[dict]] = None,
) -> QueryGeneration:
    """Return optimised vector search queries for *question*.

    Returns a guaranteed valid QueryGeneration Pydantic object.
    """
    messages = [SystemMessage(content=_SYSTEM_PROMPT)]

    if conversation_history:
        for msg in conversation_history[-4:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))

    messages.append(HumanMessage(content=question))

    try:
        result: QueryGeneration = _structured_llm.invoke(messages)
        logger.info("Generated %d queries: %s", len(result.queries), result.queries)
        return result
    except Exception as exc:
        logger.error("Query generation failed: %s – falling back to raw question", exc)
        return QueryGeneration(
            queries=[question],
            primary_entities=[],
            query_type="factual",
        )
