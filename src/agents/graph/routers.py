"""
Router functions for the LangGraph agent pipeline.

A router is a plain function that receives the current AgentState
and returns a string key that LangGraph uses to decide which node
to execute next.  No LLM calls, no side-effects – pure decision logic.
"""

from __future__ import annotations

import logging

from config.settings import WEB_SEARCH_CONFIDENCE_THRESHOLD
from src.agents.graph.state import AgentState

logger = logging.getLogger(__name__)

# Keywords that indicate the question is about an uploaded document.
# We never send document-specific questions to the web search node.
_DOC_KEYWORDS = (
    "pdf", "document", "uploaded", "file", "datasheet",
    "the data", "this data", "above data", "attached",
)


def route_after_intent(state: AgentState) -> str:
    """Decide what to do after intent classification.

    Returns
    -------
    "generate_queries"  – question needs the retrieval pipeline
    "direct_response"   – simple greeting / gratitude / farewell
    """
    if state.get("needs_retrieval"):
        return "generate_queries"
    return "direct_response"


def route_after_response(state: AgentState) -> str:
    """Decide what to do after the first response is generated.

    Returns
    -------
    "web_search"  – confidence too low, fetch more information
    "end"         – answer is good enough, stop here
    """
    confidence = state.get("confidence", 1.0)
    already_searched = state.get("web_search_performed", False)

    # Never search twice in the same turn
    if already_searched:
        return "end"

    # Query strategy can force a web search (e.g. time-sensitive query)
    strategy = state.get("query_strategy", {})
    if strategy.get("force_web_search"):
        return "web_search"

    # Document questions should be answered from the local KB only
    question = state.get("question", "").lower()
    if any(kw in question for kw in _DOC_KEYWORDS):
        logger.info(
            "Skipping web search – document question (confidence=%.2f)", confidence
        )
        return "end"

    # Fall back to web search when confidence is below the threshold
    if confidence < WEB_SEARCH_CONFIDENCE_THRESHOLD:
        return "web_search"

    return "end"
