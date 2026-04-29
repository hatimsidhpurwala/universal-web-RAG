"""
Query generator agent – converts a natural-language question into
1-3 optimised search queries for the vector database.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from groq import Groq

from config.settings import GROQ_API_KEY, LLM_MODEL
from src.agents.models import QueryGeneration

logger = logging.getLogger(__name__)

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

Respond with valid JSON matching this schema:
{
  "queries": ["query1", "query2"],
  "primary_entities": ["entity1", "entity2"],
  "query_type": "factual" | "comparative" | "explanatory" | "procedural"
}

Example:
  Input: "How much does your security camera system cost?"
  Output: {
    "queries": ["security camera system pricing cost", "installation fees charges"],
    "primary_entities": ["security camera", "cost", "pricing"],
    "query_type": "factual"
  }

  Input: "What is the uploaded PDF about?"
  Output: {
    "queries": ["product overview description", "main features specifications", "company services summary"],
    "primary_entities": ["product", "overview", "summary"],
    "query_type": "explanatory"
  }
"""


def generate_queries(
    question: str,
    conversation_history: Optional[List[dict]] = None,
) -> QueryGeneration:
    """Return optimised search queries for *question*."""
    client = Groq(api_key=GROQ_API_KEY)

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if conversation_history:
        for msg in conversation_history[-4:]:
            messages.append(msg)
    messages.append({"role": "user", "content": question})

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        result = QueryGeneration(**data)
        logger.info("Generated %d queries: %s", len(result.queries), result.queries)
        return result
    except Exception as exc:
        logger.error("Query generation failed: %s – falling back to raw question", exc)
        return QueryGeneration(
            queries=[question],
            primary_entities=[],
            query_type="factual",
        )
