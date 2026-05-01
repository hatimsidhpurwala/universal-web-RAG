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
You are a helpful AI assistant. Generate a clear, accurate answer to the
user's question using ONLY the provided context chunks.

Rules:
1. Read ALL context chunks carefully before answering.
2. Answer using ONLY the information in the provided context.
3. Be specific and detailed – include numbers, names, specifications.
4. Combine information from multiple chunks when useful.
5. NEVER say "according to page X" or cite chunk indexes.
6. Use bullet points for lists and key details.
7. If the user asks a BROAD question like "what is this about" or
   "summarize the document" or "what is the pdf about", provide a
   comprehensive overview of ALL the information in the context chunks.
   This should be treated as a HIGH-CONFIDENCE answer (0.8-0.9) because
   the context itself IS the answer.
8. Set your confidence based on how well the context answers the question:
   • 0.9–1.0  – Perfect, direct match
   • 0.7–0.8  – Good match with some uncertainty
   • 0.5–0.6  – Partial answer only
   • < 0.5    – Information not found in context
9. Suggest 1-2 relevant follow-up questions the user might ask.
10. If the context does NOT contain the answer, say so honestly and
    set confidence below 0.5.
11. IMPORTANT: If context chunks are provided (not empty), ALWAYS try to
    provide a useful answer. "No information available" should ONLY be
    used when the context is truly empty or completely irrelevant.
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
