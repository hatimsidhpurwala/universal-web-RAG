"""
Self-Improvement Engine – analyzes past interactions, identifies
patterns of failure, slow responses, and low confidence, and
applies improvements automatically.

Stores analytics in a JSON file and can be run as a daily job
or triggered manually via /improve command.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

_ANALYTICS_FILE = DATA_DIR / "analytics.json"


def _load_analytics() -> dict:
    if _ANALYTICS_FILE.exists():
        try:
            return json.loads(_ANALYTICS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_analytics(data: dict) -> None:
    _ANALYTICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ANALYTICS_FILE.write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )


class SelfImprovementEngine:
    """Tracks interaction metrics and identifies improvement opportunities."""

    def __init__(self) -> None:
        self._data = _load_analytics()
        self._data.setdefault("interactions", [])
        self._data.setdefault("improvements_applied", [])
        self._data.setdefault("performance_history", [])

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_interaction(
        self,
        query: str,
        answer: str,
        confidence: float,
        response_time_ms: int,
        web_search_used: bool,
        sources: List[str],
        user_feedback: str = "",  # "positive", "negative", ""
    ) -> None:
        """Record a single interaction for analysis."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:200],
            "answer_length": len(answer),
            "confidence": confidence,
            "response_time_ms": response_time_ms,
            "web_search_used": web_search_used,
            "sources_count": len(sources),
            "user_feedback": user_feedback,
        }
        self._data["interactions"].append(entry)

        # Keep only last 1000 interactions
        if len(self._data["interactions"]) > 1000:
            self._data["interactions"] = self._data["interactions"][-1000:]

        _save_analytics(self._data)

    def record_feedback(
        self, query: str, feedback: str
    ) -> None:
        """Record user feedback (positive/negative) for a query."""
        # Find the most recent matching interaction
        for entry in reversed(self._data["interactions"]):
            if entry["query"][:100] == query[:100]:
                entry["user_feedback"] = feedback
                break
        _save_analytics(self._data)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze all recorded interactions and generate insights.

        Returns
        -------
        dict
            ``summary``, ``common_failures``, ``slow_queries``,
            ``low_confidence_topics``, ``improvements``.
        """
        interactions = self._data["interactions"]
        if not interactions:
            return {"summary": "No interactions recorded yet."}

        # Basic stats
        confidences = [i["confidence"] for i in interactions]
        times = [i["response_time_ms"] for i in interactions]
        avg_confidence = sum(confidences) / len(confidences)
        avg_time = sum(times) / len(times)
        web_search_rate = (
            sum(1 for i in interactions if i["web_search_used"]) / len(interactions)
        )

        # Failures (confidence < 0.3)
        failures = [
            i for i in interactions if i["confidence"] < 0.3
        ]

        # Slow queries (> 10s)
        slow = [i for i in interactions if i["response_time_ms"] > 10000]

        # Low confidence (0.3-0.6)
        low_conf = [
            i for i in interactions if 0.3 <= i["confidence"] < 0.6
        ]

        # Negative feedback
        negative = [i for i in interactions if i.get("user_feedback") == "negative"]

        # Generate improvement suggestions
        improvements = []

        if len(failures) > len(interactions) * 0.1:
            improvements.append({
                "type": "index_more_data",
                "priority": "high",
                "description": (
                    f"{len(failures)} queries ({len(failures)/len(interactions):.0%}) "
                    "failed. Consider indexing more documents."
                ),
                "sample_queries": [f["query"] for f in failures[:5]],
            })

        if avg_time > 5000:
            improvements.append({
                "type": "optimize_speed",
                "priority": "medium",
                "description": (
                    f"Average response time is {avg_time/1000:.1f}s. "
                    "Consider caching frequent queries."
                ),
            })

        if web_search_rate > 0.5:
            improvements.append({
                "type": "expand_knowledge_base",
                "priority": "high",
                "description": (
                    f"Web search triggered in {web_search_rate:.0%} of queries. "
                    "Index more content to reduce reliance on web search."
                ),
            })

        if negative:
            improvements.append({
                "type": "review_negative_feedback",
                "priority": "high",
                "description": (
                    f"{len(negative)} interactions received negative feedback."
                ),
                "sample_queries": [n["query"] for n in negative[:5]],
            })

        report = {
            "summary": {
                "total_interactions": len(interactions),
                "avg_confidence": round(avg_confidence, 3),
                "avg_response_time_ms": round(avg_time),
                "web_search_rate": round(web_search_rate, 3),
                "failure_rate": round(len(failures) / len(interactions), 3),
                "negative_feedback_count": len(negative),
            },
            "common_failures": [f["query"] for f in failures[:10]],
            "slow_queries": [s["query"] for s in slow[:10]],
            "low_confidence_topics": [l["query"] for l in low_conf[:10]],
            "improvements": improvements,
        }

        # Store performance snapshot
        self._data["performance_history"].append({
            "timestamp": datetime.now().isoformat(),
            "summary": report["summary"],
        })
        if len(self._data["performance_history"]) > 90:
            self._data["performance_history"] = self._data["performance_history"][-90:]

        _save_analytics(self._data)
        logger.info("Performance analysis: %s", report["summary"])
        return report

    def get_performance_trend(self) -> List[dict]:
        """Return historical performance snapshots."""
        return self._data.get("performance_history", [])
