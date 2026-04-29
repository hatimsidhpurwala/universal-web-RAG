"""
Multi-Step Reasoning Agent – transparent chain-of-thought reasoning
that shows step-by-step analysis before delivering a final answer.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from groq import Groq

from config.settings import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)


# ======================================================================
# Models
# ======================================================================

class ReasoningStep(BaseModel):
    step_name: str
    content: str
    confidence: float = 0.0


class ReasoningChain(BaseModel):
    steps: List[ReasoningStep] = Field(default_factory=list)
    final_answer: str = ""
    confidence_breakdown: Dict[str, float] = Field(default_factory=dict)
    gaps_identified: List[str] = Field(default_factory=list)
    needs_web_search: bool = False


# ======================================================================
# System prompt
# ======================================================================

_SYSTEM_PROMPT = """\
You are a reasoning assistant. Given a user question and context chunks,
perform transparent step-by-step reasoning before delivering an answer.

Return valid JSON:
{
  "steps": [
    {"step_name": "Knowledge Analysis", "content": "...", "confidence": 0.8},
    {"step_name": "Gap Identification", "content": "...", "confidence": 0.7},
    {"step_name": "Synthesis", "content": "...", "confidence": 0.85}
  ],
  "final_answer": "The complete answer...",
  "confidence_breakdown": {
    "from_knowledge_base": 0.6,
    "from_web_search": 0.2,
    "inferred": 0.2
  },
  "gaps_identified": ["missing pricing info", "no dealer data"],
  "needs_web_search": false
}

Steps to follow:
1. Knowledge Analysis: What information do we have from the context?
2. Gap Identification: What's missing to fully answer the question?
3. Inference: What can we logically infer from available data?
4. Synthesis: Combine all findings into a coherent answer.
5. Confidence Assessment: How confident is each part?

Be thorough but concise in each step.
"""


def reason_through(
    question: str,
    context_chunks: List[dict],
    conversation_history: Optional[List[dict]] = None,
) -> ReasoningChain:
    """Perform chain-of-thought reasoning over *context_chunks*."""
    client = Groq(api_key=GROQ_API_KEY)

    # Build context
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source_url", "unknown")
        score = chunk.get("score", 0)
        context_parts.append(
            f"--- Chunk {i} (source: {source}, score: {score:.2f}) ---\n"
            f"{chunk['text']}\n"
        )
    context_block = "\n".join(context_parts) if context_parts else "(no context)"

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if conversation_history:
        for msg in conversation_history[-4:]:
            messages.append(msg)

    messages.append({
        "role": "user",
        "content": f"Context:\n{context_block}\n\nQuestion: {question}",
    })

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)

        steps = [
            ReasoningStep(**s) for s in data.get("steps", [])
        ]

        chain = ReasoningChain(
            steps=steps,
            final_answer=data.get("final_answer", ""),
            confidence_breakdown=data.get("confidence_breakdown", {}),
            gaps_identified=data.get("gaps_identified", []),
            needs_web_search=data.get("needs_web_search", False),
        )
        logger.info(
            "Reasoning chain: %d steps, gaps=%d, web_needed=%s",
            len(steps), len(chain.gaps_identified), chain.needs_web_search,
        )
        return chain
    except Exception as exc:
        logger.error("Reasoning failed: %s", exc)
        return ReasoningChain(
            steps=[ReasoningStep(
                step_name="Error",
                content=f"Reasoning failed: {exc}",
                confidence=0.0,
            )],
            final_answer="I encountered an error during reasoning.",
            needs_web_search=False,
        )
