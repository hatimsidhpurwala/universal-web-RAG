"""
Graph builder for the LangGraph agent pipeline.

This module's single job is to wire nodes and edges together and
return a compiled LangGraph.  It knows nothing about business logic –
that lives in nodes.py and routers.py.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.graph.state import AgentState
from src.agents.graph.routers import route_after_intent, route_after_response
from src.agents.graph import nodes as N
from src.database.vector_store import VectorStore


def build_graph(vector_store: VectorStore):
    """Construct and compile the full agent graph.

    Parameters
    ----------
    vector_store : VectorStore
        Injected into the nodes that need to talk to Qdrant.

    Returns
    -------
    CompiledGraph
        Ready to call with `graph.invoke(initial_state)`.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ─────────────────────────────────────────────────
    graph.add_node("classify_intent",      N.node_classify_intent)
    graph.add_node("direct_response",      N.node_direct_response)
    graph.add_node("generate_queries",     N.node_generate_queries)
    graph.add_node("retrieve_chunks",      N.make_node_retrieve_chunks(vector_store))
    graph.add_node("generate_response",    N.node_generate_response)
    graph.add_node("web_search",           N.make_node_web_search(vector_store))
    graph.add_node("re_retrieve",          N.make_node_re_retrieve(vector_store))
    graph.add_node("re_generate_response", N.node_re_generate_response)

    # ── Entry point ────────────────────────────────────────────────────
    graph.set_entry_point("classify_intent")

    # ── Conditional edge: after intent classification ──────────────────
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "direct_response": "direct_response",
            "generate_queries": "generate_queries",
        },
    )

    # ── Linear edges: retrieval pipeline ──────────────────────────────
    graph.add_edge("direct_response",   END)
    graph.add_edge("generate_queries",  "retrieve_chunks")
    graph.add_edge("retrieve_chunks",   "generate_response")

    # ── Conditional edge: after first response ─────────────────────────
    graph.add_conditional_edges(
        "generate_response",
        route_after_response,
        {
            "web_search": "web_search",
            "end":        END,
        },
    )

    # ── Linear edges: web-search fallback loop ─────────────────────────
    graph.add_edge("web_search",           "re_retrieve")
    graph.add_edge("re_retrieve",          "re_generate_response")
    graph.add_edge("re_generate_response", END)

    return graph.compile()
