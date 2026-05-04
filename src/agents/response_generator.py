"""
Response generator agent.

Uses LangChain's with_structured_output() instead of hard-prompting.

Synthesises a final answer from retrieved context chunks. Returns a
guaranteed-valid ResponseGeneration Pydantic object — no json.loads,
no parsing risk, no "Respond with valid JSON" in the prompt.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config.settings import GROQ_API_KEY, LLM_MODEL
from src.agents.models import ResponseGeneration

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────
# Schema enforcement is done via tool-calling — no JSON block needed here.

_SYSTEM_PROMPT = """\
You are a knowledgeable AI assistant. Generate detailed, specific, and
actionable answers to the user's question.

PRIORITY RULES:
1. Read ALL context chunks carefully — extract EVERY relevant detail.
2. PRIMARY source: use context chunks. Be specific — include phone numbers,
   emails, addresses, URLs, specifications, names.
3. SUPPLEMENTATION: if the context is THIN (< 3 useful chunks) or the user
   asks about contacts/locations/distributors/pricing and context lacks specifics,
   SUPPLEMENT with your own knowledge to give a useful answer. Clearly note
   "Based on general knowledge:" when doing so.
4. NEVER say only "visit their website" without providing the actual website URL
   and as much specific info as you know. Always be actionable.
5. For CONTACT / WHERE TO BUY questions:
   - List all emails, phone numbers, addresses found in context
   - List official website URL
   - List country-specific office info if known
   - Suggest specific pages on the website (e.g. "their Contact page at
     saltosystems.com/contact" or "Use the Partner Finder tool")
6. For DOCUMENT questions ("what is this about", "summarize"):
   - Give a COMPREHENSIVE overview of ALL content in the chunks
   - Use bullet points for specs, features, models
   - Confidence: 0.85-0.95 (the doc IS the answer)
7. Use bullet points and clear section headers for multi-part answers.
8. Set confidence:
   • 0.9–1.0  – Direct match in context, complete answer
   • 0.7–0.8  – Good answer, some gaps filled with general knowledge
   • 0.5–0.6  – Partial — context limited, supplemented significantly
   • < 0.5    – Context irrelevant, answer from general knowledge only
9. Suggest 2 specific follow-up questions the user might find useful.
10. ALWAYS give the most complete, helpful answer possible. Never be vague.
"""

# ── Structured LLM ──────────────────────────────────────────────────────────
_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=0.3,
    max_tokens=2000,
)
_structured_llm = _llm.with_structured_output(ResponseGeneration)


# ── Public function ──────────────────────────────────────────────────────────

def generate_response(
    question: str,
    context_chunks: List[dict],
    conversation_history: Optional[List[dict]] = None,
) -> ResponseGeneration:
    """Generate a response for *question* grounded in *context_chunks*.

    Returns a guaranteed valid ResponseGeneration Pydantic object.
    """
    # Build the context block from retrieved chunks
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source_url", "unknown")
        score = chunk.get("score", 0)
        context_parts.append(
            f"--- Chunk {i} (source: {source}, relevance: {score:.2f}) ---\n"
            f"{chunk['text']}\n"
        )
    context_block = (
        "\n".join(context_parts) if context_parts else "(no context available)"
    )

    messages = [SystemMessage(content=_SYSTEM_PROMPT)]

    if conversation_history:
        for msg in conversation_history[-6:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))

    user_content = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    )
    messages.append(HumanMessage(content=user_content))

    try:
        result: ResponseGeneration = _structured_llm.invoke(messages)
        logger.info(
            "Response generated (confidence=%.2f, sources=%d)",
            result.confidence,
            len(result.sources_used),
        )
        return result
    except Exception as exc:
        logger.error("Response generation failed: %s", exc)
        return ResponseGeneration(
            answer=f"I encountered an error generating the response: {exc}",
            confidence=0.0,
            sources_used=[],
            follow_up_suggestions=None,
        )
