"""
Smart Conversation Branching – tracks conversation threads, detects
topic switches, handles follow-ups with proper context, and manages
multi-topic conversations.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from groq import Groq

from config.settings import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

_ANALYSIS_PROMPT = """\
You are a conversation state analyzer. Given the conversation history
and a new message, determine the conversational state.

Return JSON:
{
  "state": "new_topic|followup|clarification|comparison|topic_switch|correction",
  "current_topic": "the main topic being discussed",
  "previous_topic": "topic before switch (if topic_switch)",
  "context_needed": "what context from history is needed",
  "enhanced_query": "the user's message rewritten with full context",
  "topic_threads": [
    {"topic": "topic name", "status": "active|paused", "summary": "brief summary"}
  ]
}

States:
- new_topic: First message or completely new subject
- followup: Continuing the same topic ("What about pricing?")
- clarification: Asking to explain something ("What do you mean?")
- comparison: Comparing with something ("How does that compare to X?")
- topic_switch: Changing to a different topic ("Now tell me about Y")
- correction: User correcting the system ("No, I meant X not Y")

IMPORTANT: For "enhanced_query", rewrite the message to be self-contained.
Example: If topic is "SALTO keycards" and user says "How much?",
enhanced_query = "How much do SALTO keycards cost?"
"""


class ConversationState:
    """Tracks conversation state and enhances queries with context."""

    def __init__(self):
        self.topic_threads: List[Dict[str, Any]] = []
        self.current_topic: str = ""
        self.paused_topics: List[str] = []

    def analyze_message(
        self,
        message: str,
        conversation_history: List[dict],
    ) -> Dict[str, Any]:
        """Analyze *message* in context of *conversation_history*.

        Returns
        -------
        dict
            ``state``, ``enhanced_query``, ``current_topic``,
            ``topic_threads``.
        """
        # Short history – no context needed
        if not conversation_history or len(conversation_history) < 2:
            self.current_topic = message[:100]
            return {
                "state": "new_topic",
                "enhanced_query": message,
                "current_topic": message[:100],
                "topic_threads": [],
                "is_correction": False,
            }

        # Check for correction patterns
        lower = message.lower().strip()
        correction_signals = [
            lower.startswith("no,"),
            lower.startswith("actually"),
            lower.startswith("that's wrong"),
            lower.startswith("incorrect"),
            "i meant" in lower,
            "not what i asked" in lower,
        ]
        if any(correction_signals):
            return {
                "state": "correction",
                "enhanced_query": message,
                "current_topic": self.current_topic,
                "topic_threads": self.topic_threads,
                "is_correction": True,
            }

        # Use LLM for nuanced analysis
        return self._llm_analyze(message, conversation_history)

    def _llm_analyze(
        self,
        message: str,
        conversation_history: List[dict],
    ) -> Dict[str, Any]:
        client = Groq(api_key=GROQ_API_KEY)

        # Build compact history
        history_text = ""
        for msg in conversation_history[-8:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:200]
            history_text += f"{role}: {content}\n"

        messages = [
            {"role": "system", "content": _ANALYSIS_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"New message: {message}"
                ),
            },
        ]

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)

            # Update internal state
            state = data.get("state", "followup")
            new_topic = data.get("current_topic", self.current_topic)

            if state == "topic_switch":
                if self.current_topic:
                    self.paused_topics.append(self.current_topic)
                self.current_topic = new_topic
            else:
                self.current_topic = new_topic

            # Update threads
            if data.get("topic_threads"):
                self.topic_threads = data["topic_threads"]

            result = {
                "state": state,
                "enhanced_query": data.get("enhanced_query", message),
                "current_topic": self.current_topic,
                "previous_topic": data.get("previous_topic", ""),
                "topic_threads": self.topic_threads,
                "is_correction": state == "correction",
            }

            logger.info(
                "Conversation state: %s, topic: %s, enhanced: %s",
                state,
                self.current_topic[:50],
                result["enhanced_query"][:80],
            )
            return result

        except Exception as exc:
            logger.error("Conversation analysis failed: %s", exc)
            return {
                "state": "followup",
                "enhanced_query": message,
                "current_topic": self.current_topic,
                "topic_threads": [],
                "is_correction": False,
            }
