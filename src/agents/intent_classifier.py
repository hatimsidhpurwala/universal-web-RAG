"""
Intent classification agent – decides how to route a user's message.
Uses Groq LLM with structured JSON output.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from groq import Groq

from config.settings import GROQ_API_KEY, LLM_MODEL
from src.agents.models import IntentClassification

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an intent-classification assistant. Given a user message and
optional conversation history, classify the user's intent into exactly
one of the following categories:

  • greeting        – Hi, hello, how are you, hey
  • gratitude       – Thank you, thanks, appreciate it
  • general_knowledge – Questions answerable without any company data
  • retrieval_needed – Questions that require searching a knowledge base
                       (company products, services, specs, pricing, etc.)
  • clarification   – User asking for more explanation of a previous answer
  • farewell        – Goodbye, see you later, bye

Rules:
1. If the user asks about any company, product, service, document content,
   uploaded data, or scraped website → ALWAYS classify as "retrieval_needed".
2. If the message is a conversational pleasantry → NOT "retrieval_needed".
3. Set confidence to 0.9+ for clear cases, 0.6–0.8 for ambiguous ones.

Respond with valid JSON matching this schema:
{
  "intent": "<one of the categories>",
  "confidence": <float 0-1>,
  "reasoning": "<brief explanation>",
  "needs_retrieval": <bool>
}
"""


def classify_intent(
    question: str,
    conversation_history: Optional[List[dict]] = None,
) -> IntentClassification:
    """Classify the *question* and return a structured intent result."""
    client = Groq(api_key=GROQ_API_KEY)

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Inject recent history for context
    if conversation_history:
        for msg in conversation_history[-6:]:
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
        result = IntentClassification(**data)
        logger.info(
            "Intent: %s (confidence=%.2f, retrieval=%s)",
            result.intent,
            result.confidence,
            result.needs_retrieval,
        )
        return result
    except Exception as exc:
        logger.error("Intent classification failed: %s – defaulting to retrieval", exc)
        return IntentClassification(
            intent="retrieval_needed",
            confidence=0.5,
            reasoning=f"Classification error: {exc}",
            needs_retrieval=True,
        )
