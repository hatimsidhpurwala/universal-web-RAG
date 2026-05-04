"""
Node functions for the LangGraph agent pipeline.

Each node is a plain function:
  - receives the current AgentState
  - does one unit of work (call an agent, search, retrieve …)
  - returns a dict with ONLY the keys it wants to update in the state

Nodes never call each other directly – LangGraph wires them together
via edges defined in builder.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.agents.direct_responder import direct_response
from src.agents.intent_classifier import classify_intent
from src.agents.query_generator import generate_queries
from src.agents.response_generator import generate_response
from src.agents.retriever import retrieve_chunks
from src.agents.web_searcher import search_and_scrape
from src.agents.graph.state import AgentState

if TYPE_CHECKING:
    from src.database.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Node: classify_intent
# ──────────────────────────────────────────────────────────────────────

def node_classify_intent(state: AgentState) -> dict:
    """Classify the user's intent and decide whether retrieval is needed."""
    result = classify_intent(
        state["question"],
        state.get("conversation_history"),
    )
    features = list(state.get("enhanced_features_used", []))
    features.append("intent_classification")
    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "needs_retrieval": result.needs_retrieval,
        "enhanced_features_used": features,
    }


# ──────────────────────────────────────────────────────────────────────
# Node: direct_response
# ──────────────────────────────────────────────────────────────────────

def node_direct_response(state: AgentState) -> dict:
    """Handle greetings, farewells, and gratitude without retrieval."""
    answer = direct_response(
        state["question"],
        state.get("intent", "greeting"),
        state.get("conversation_history"),
    )
    return {
        "final_answer": answer,
        "confidence": 1.0,
        "sources": [],
        "follow_up_suggestions": [],
    }


# ──────────────────────────────────────────────────────────────────────
# Node: generate_queries
# ──────────────────────────────────────────────────────────────────────

def node_generate_queries(state: AgentState) -> dict:
    """Convert the user question into 1–3 optimised vector search queries."""
    result = generate_queries(
        state["question"],
        state.get("conversation_history"),
    )
    features = list(state.get("enhanced_features_used", []))
    features.append("query_optimization")
    return {
        "generated_queries": result.queries,
        "enhanced_features_used": features,
    }


# ──────────────────────────────────────────────────────────────────────
# Node: retrieve_chunks   (needs vector_store – injected at runtime)
# ──────────────────────────────────────────────────────────────────────

def make_node_retrieve_chunks(vector_store: "VectorStore"):
    """Factory: returns a node function bound to *vector_store*."""

    def node_retrieve_chunks(state: AgentState) -> dict:
        """Search the vector store and return the top-k relevant chunks.

        When ``source_prefix`` is set in state (e.g. ``"pdf_"``), restricts
        the search to chunks from that source type only, falling back to
        global search if no results are found.
        """
        strategy = state.get("query_strategy", {})
        top_k = strategy.get("top_k_override")

        kwargs: dict = {}
        if top_k:
            kwargs["final_top_k"] = top_k

        # Source filter set by the agent when question is about an uploaded doc
        source_prefix = state.get("source_prefix")
        if source_prefix:
            kwargs["source_prefix"] = source_prefix

        chunks = retrieve_chunks(
            state.get("generated_queries", [state["question"]]),
            vector_store,
            **kwargs,
        )
        return {"retrieved_chunks": chunks}

    return node_retrieve_chunks


# ──────────────────────────────────────────────────────────────────────
# Node: generate_response
# ──────────────────────────────────────────────────────────────────────

def node_generate_response(state: AgentState) -> dict:
    """Generate a first-pass answer from retrieved chunks."""
    result = generate_response(
        state["question"],
        state.get("retrieved_chunks", []),
        state.get("conversation_history"),
    )
    return {
        "final_answer": result.answer,
        "confidence": result.confidence,
        "sources": [s.source_url for s in result.sources_used],
        "follow_up_suggestions": result.follow_up_suggestions or [],
    }


# ──────────────────────────────────────────────────────────────────────
# Node: web_search   (needs vector_store – injected at runtime)
# ──────────────────────────────────────────────────────────────────────

def make_node_web_search(vector_store: "VectorStore"):
    """Factory: returns a node function bound to *vector_store*."""

    def node_web_search(state: AgentState) -> dict:
        """Scrape the web and index fresh content into the vector store.

        Skipped automatically when ``source_prefix`` is set in state —
        there is no point web-searching for content from a local document.
        """
        if state.get("source_prefix"):
            logger.info("Web search skipped – question is about a local document")
            features = list(state.get("enhanced_features_used", []))
            return {
                "web_search_performed": False,
                "research_info": {},
                "enhanced_features_used": features,
            }

        logger.info(
            "Web search triggered (confidence=%.2f)", state.get("confidence", 0)
        )
        try:
            info = search_and_scrape(state["question"], vector_store)
        except Exception as exc:
            logger.error("Web search failed: %s", exc)
            info = {"sites_indexed": [], "total_chunks": 0}

        features = list(state.get("enhanced_features_used", []))
        features.append("web_search")
        return {
            "web_search_performed": True,
            "research_info": info,
            "enhanced_features_used": features,
        }

    return node_web_search


# ──────────────────────────────────────────────────────────────────────
# Node: re_retrieve   (needs vector_store – injected at runtime)
# ──────────────────────────────────────────────────────────────────────

def make_node_re_retrieve(vector_store: "VectorStore"):
    """Factory: returns a node function bound to *vector_store*."""

    def node_re_retrieve(state: AgentState) -> dict:
        """Re-search the vector store after web content has been indexed."""
        queries = state.get("generated_queries", [state["question"]])
        try:
            chunks = retrieve_chunks(queries, vector_store)
        except Exception as exc:
            logger.error("Re-retrieval failed: %s", exc)
            chunks = state.get("retrieved_chunks", [])
        return {"retrieved_chunks": chunks}

    return node_re_retrieve


# ──────────────────────────────────────────────────────────────────────
# Node: re_generate_response
# ──────────────────────────────────────────────────────────────────────

def node_re_generate_response(state: AgentState) -> dict:
    """Re-generate the answer using newly retrieved (post-web-search) chunks."""
    result = generate_response(
        state["question"],
        state.get("retrieved_chunks", []),
        state.get("conversation_history"),
    )
    return {
        "final_answer": result.answer,
        "confidence": result.confidence,
        "sources": [s.source_url for s in result.sources_used],
        "follow_up_suggestions": result.follow_up_suggestions or [],
    }
