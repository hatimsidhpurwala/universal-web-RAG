"""
src/agents/agent_graph.py  – backward-compatibility shim

This file now delegates everything to the refactored graph package.
The original 571-line monolith has been split into:

    src/agents/graph/
    ├── state.py    – AgentState TypedDict
    ├── nodes.py    – 8 node functions
    ├── routers.py  – 2 routing functions
    ├── builder.py  – graph wiring & compilation
    └── agent.py    – RAGAgent public API

Import from either location – both work:
    from src.agents.agent_graph import RAGAgent   # old path (still works)
    from src.agents.graph import RAGAgent          # new preferred path
"""

from src.agents.graph.agent import RAGAgent        # noqa: F401
from src.agents.graph.state import AgentState      # noqa: F401

__all__ = ["RAGAgent", "AgentState"]
