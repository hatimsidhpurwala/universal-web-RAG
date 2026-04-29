"""
Interactive Clarification Agent – detects ambiguous queries and
generates smart clarifying questions before searching.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from groq import Groq

from config.settings import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a clarification assistant. Analyze the user's query and
determine if it is ambiguous or missing important context.

Return JSON:
{
  "needs_clarification": true/false,
  "ambiguity_level": "none|low|medium|high",
  "reason": "Why clarification is needed",
  "clarifying_questions": [
    {
      "question": "The clarifying question",
      "options": ["option1", "option2", "option3"],
      "why": "Why this matters for the answer",
      "type": "single_choice|multi_choice|open_text"
    }
  ],
  "assumed_context": "What we can reasonably assume if user skips"
}

Rules:
- Only flag as needing clarification if the ambiguity would
  significantly change the answer.
- Maximum 2-3 clarifying questions (don't overwhelm the user).
- Always provide reasonable defaults/assumptions.
- If the query is clear enough, set needs_clarification=false.

Common ambiguities:
- "How much?" → For how many units? Which region?
- "Is it compatible?" → Compatible with what system?
- "it", "this", "that" → What does the pronoun refer to?
- Product name alone → Inquiry, pricing, specs, or availability?
"""


class ClarificationAgent:
    """Detects ambiguous queries and generates clarifying questions."""

    def check_if_clarification_needed(
        self,
        query: str,
        conversation_history: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Analyze *query* for ambiguity.

        Returns
        -------
        dict
            ``needs_clarification``, ``clarifying_questions``,
            ``assumed_context``, ``ambiguity_level``.
        """
        # Short simple queries are usually clear
        word_count = len(query.split())
        if word_count <= 3:
            # Very short → might be ambiguous
            pass
        elif word_count > 15:
            # Long queries tend to be self-explanatory
            return {
                "needs_clarification": False,
                "ambiguity_level": "none",
                "clarifying_questions": [],
                "assumed_context": "",
            }

        # Check if conversation history provides enough context
        if conversation_history and len(conversation_history) >= 4:
            # Mid-conversation follow-ups usually have implicit context
            has_pronouns = any(
                p in query.lower()
                for p in ["it", "this", "that", "they", "them", "their"]
            )
            if not has_pronouns:
                return {
                    "needs_clarification": False,
                    "ambiguity_level": "none",
                    "clarifying_questions": [],
                    "assumed_context": "",
                }

        return self._llm_check(query, conversation_history)

    def _llm_check(
        self,
        query: str,
        conversation_history: Optional[List[dict]],
    ) -> Dict[str, Any]:
        client = Groq(api_key=GROQ_API_KEY)

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if conversation_history:
            for msg in conversation_history[-4:]:
                messages.append(msg)
        messages.append({"role": "user", "content": query})

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            result = {
                "needs_clarification": data.get("needs_clarification", False),
                "ambiguity_level": data.get("ambiguity_level", "none"),
                "clarifying_questions": data.get("clarifying_questions", []),
                "assumed_context": data.get("assumed_context", ""),
                "reason": data.get("reason", ""),
            }
            logger.info(
                "Clarification check: needed=%s, level=%s",
                result["needs_clarification"],
                result["ambiguity_level"],
            )
            return result
        except Exception as exc:
            logger.error("Clarification check failed: %s", exc)
            return {
                "needs_clarification": False,
                "ambiguity_level": "none",
                "clarifying_questions": [],
                "assumed_context": "",
            }
