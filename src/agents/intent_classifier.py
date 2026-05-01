"""
Intent classification agent.

Uses LangChain's with_structured_output() instead of hard-prompting.

Before (hard prompting):
  - System prompt contained the raw JSON schema as text
  - LLM returned a plain string → json.loads() → manual Pydantic validation
  - Risk: LLM could forget the format, use wrong field names, wrong types

After (with_structured_output):
  - Pydantic model IS the schema (single source of truth)
  - LangChain converts it to a tool-calling contract
  - Groq is FORCED to return that exact structure
  - We get back a typed Pydantic object — no parsing, no risk of crashes
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config.settings import GROQ_API_KEY, LLM_MODEL
from src.agents.models import IntentClassification

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────
# Note: NO "Respond with valid JSON" instruction needed anymore.
# LangChain enforces the schema via tool-calling at the API level.

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
"""

# ── Structured LLM (built once, reused) ────────────────────────────────────
# with_structured_output reads IntentClassification's fields + descriptions
# and builds a tool-calling contract automatically.
_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=0.1,
    max_tokens=300,
)
_structured_llm = _llm.with_structured_output(IntentClassification)


# ── Public function ──────────────────────────────────────────────────────────

def classify_intent(
    question: str,
    conversation_history: Optional[List[dict]] = None,
) -> IntentClassification:
    """Classify *question* and return a structured IntentClassification.

    Returns a guaranteed valid Pydantic object — no json.loads, no
    manual field mapping, no crash risk from malformed LLM output.
    """
    messages = [SystemMessage(content=_SYSTEM_PROMPT)]

    # Inject recent history for context
    if conversation_history:
        for msg in conversation_history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            # Skip assistant messages to keep token count low

    messages.append(HumanMessage(content=question))

    try:
        result: IntentClassification = _structured_llm.invoke(messages)
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
