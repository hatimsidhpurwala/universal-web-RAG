"""
Real-Time Memory & Learning – persistent memory system that learns
from corrections, tracks user preferences, and personalizes responses.

Uses a JSON file for persistence so data survives restarts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

_MEMORY_FILE = DATA_DIR / "memory.json"
_CORRECTIONS_FILE = DATA_DIR / "corrections.json"


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


class AdaptiveMemory:
    """Persistent memory that learns from every interaction."""

    def __init__(self) -> None:
        self._memory = _load_json(_MEMORY_FILE)
        self._corrections = _load_json(_CORRECTIONS_FILE)

        # Ensure structure
        self._memory.setdefault("user_profiles", {})
        self._memory.setdefault("global_knowledge", {})
        self._memory.setdefault("conversation_summaries", {})
        self._memory.setdefault("frequently_asked", {})
        self._corrections.setdefault("log", [])

    # ------------------------------------------------------------------
    # Correction learning
    # ------------------------------------------------------------------

    def learn_from_correction(
        self,
        query: str,
        wrong_answer: str,
        correct_answer: str,
        user_id: str = "default",
    ) -> None:
        """Store a user correction so the system learns."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "query": query,
            "wrong_answer": wrong_answer[:500],
            "correct_answer": correct_answer,
        }
        self._corrections["log"].append(entry)
        _save_json(_CORRECTIONS_FILE, self._corrections)

        # Also store as global knowledge
        key = query.lower().strip()[:100]
        self._memory["global_knowledge"][key] = {
            "answer": correct_answer,
            "source": "user_correction",
            "timestamp": datetime.now().isoformat(),
        }
        self._save()
        logger.info("Learned correction for: %s", key[:60])

    def get_correction(self, query: str) -> Optional[str]:
        """Check if we have a known correction for *query*."""
        key = query.lower().strip()[:100]
        entry = self._memory["global_knowledge"].get(key)
        if entry and entry.get("source") == "user_correction":
            return entry["answer"]
        return None

    # ------------------------------------------------------------------
    # User preference tracking
    # ------------------------------------------------------------------

    def update_user_profile(
        self,
        user_id: str,
        interaction: Dict[str, Any],
    ) -> None:
        """Update user preferences based on an interaction."""
        profiles = self._memory["user_profiles"]
        if user_id not in profiles:
            profiles[user_id] = {
                "answer_style": "balanced",
                "preferred_sources": [],
                "topics_of_interest": [],
                "interaction_count": 0,
                "created_at": datetime.now().isoformat(),
            }

        profile = profiles[user_id]
        profile["interaction_count"] = profile.get("interaction_count", 0) + 1

        # Track style preference
        if interaction.get("asked_for_more_detail"):
            profile["answer_style"] = "detailed"
        elif interaction.get("asked_for_shorter"):
            profile["answer_style"] = "concise"

        # Track topics
        topic = interaction.get("topic", "")
        if topic and topic not in profile.get("topics_of_interest", []):
            profile.setdefault("topics_of_interest", []).append(topic)
            profile["topics_of_interest"] = profile["topics_of_interest"][-20:]

        # Track thumbs up/down
        if interaction.get("thumbs_up"):
            profile["answer_style_confirmed"] = True
        if interaction.get("upvoted_source"):
            src = interaction["upvoted_source"]
            if src not in profile.get("preferred_sources", []):
                profile.setdefault("preferred_sources", []).append(src)

        self._save()

    def get_user_profile(self, user_id: str) -> dict:
        return self._memory["user_profiles"].get(user_id, {})

    def personalize_response(self, user_id: str, answer: str) -> str:
        """Adapt *answer* to the user's learned preferences."""
        profile = self.get_user_profile(user_id)
        if not profile:
            return answer

        style = profile.get("answer_style", "balanced")

        if style == "concise" and len(answer) > 300:
            # Trim to key points
            lines = answer.split("\n")
            key_lines = [l for l in lines if l.strip()][:5]
            return "\n".join(key_lines)
        elif style == "detailed":
            # Append follow-up encouragement
            answer += (
                "\n\n💡 *Would you like me to go deeper into any of "
                "these points?*"
            )
        return answer

    # ------------------------------------------------------------------
    # Frequently asked tracking
    # ------------------------------------------------------------------

    def track_query(self, query: str) -> None:
        """Increment the frequency counter for *query*."""
        key = query.lower().strip()[:100]
        faq = self._memory["frequently_asked"]
        faq[key] = faq.get(key, 0) + 1
        self._save()

    def get_top_queries(self, n: int = 10) -> List[dict]:
        """Return the *n* most frequently asked queries."""
        faq = self._memory["frequently_asked"]
        sorted_q = sorted(faq.items(), key=lambda x: x[1], reverse=True)
        return [{"query": q, "count": c} for q, c in sorted_q[:n]]

    # ------------------------------------------------------------------
    # Conversation summaries
    # ------------------------------------------------------------------

    def store_conversation_summary(
        self, conversation_id: str, summary: str
    ) -> None:
        self._memory["conversation_summaries"][conversation_id] = {
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }
        # Keep only last 100
        summaries = self._memory["conversation_summaries"]
        if len(summaries) > 100:
            oldest_keys = sorted(
                summaries, key=lambda k: summaries[k]["timestamp"]
            )[: len(summaries) - 100]
            for k in oldest_keys:
                del summaries[k]
        self._save()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save(self) -> None:
        _save_json(_MEMORY_FILE, self._memory)

    def get_stats(self) -> dict:
        return {
            "user_profiles": len(self._memory["user_profiles"]),
            "corrections": len(self._corrections["log"]),
            "global_knowledge": len(self._memory["global_knowledge"]),
            "conversations": len(self._memory["conversation_summaries"]),
            "faq_entries": len(self._memory["frequently_asked"]),
        }
