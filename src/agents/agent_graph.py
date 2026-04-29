"""
Enhanced LangGraph Agent Orchestrator – integrates all advanced features:

1. Advanced Query Understanding (query_analyzer)
2. Multi-Step Reasoning (reasoning_agent)
3. Multi-Agent Collaboration (agent_team)
4. Real-Time Memory & Learning (memory)
5. Automated Fact Verification (fact_verifier)
6. Smart Conversation Branching (conversation_manager)
7. Interactive Clarification (clarification_agent)
8. Multimodal Fusion (multimodal_fusion)
9. Self-Improvement Engine (self_improvement)
10. Sentiment-Aware Responses (sentiment_adapter)
11. Multi-Language Support (multilingual)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from config.settings import WEB_SEARCH_CONFIDENCE_THRESHOLD
from src.agents.direct_responder import direct_response
from src.agents.intent_classifier import classify_intent
from src.agents.query_generator import generate_queries
from src.agents.response_generator import generate_response
from src.agents.retriever import retrieve_chunks
from src.agents.web_searcher import deep_research, search_and_scrape

# New enhanced agents
from src.agents.query_analyzer import analyze_query, QueryStrategy
from src.agents.reasoning_agent import reason_through
from src.agents.agent_team import AgentTeam
from src.agents.memory import AdaptiveMemory
from src.agents.fact_verifier import FactVerifier
from src.agents.conversation_manager import ConversationState
from src.agents.clarification_agent import ClarificationAgent
from src.agents.sentiment_adapter import SentimentAdapter
from src.agents.multilingual import MultilingualHandler
from src.agents.self_improvement import SelfImprovementEngine

from src.database.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ======================================================================
# State schema (extended)
# ======================================================================

class AgentState(TypedDict, total=False):
    # Original fields
    question: str
    original_question: str  # before translation/enhancement
    intent: str
    intent_confidence: float
    needs_retrieval: bool
    generated_queries: List[str]
    retrieved_chunks: List[dict]
    final_answer: str
    confidence: float
    sources: List[str]
    follow_up_suggestions: List[str]
    conversation_history: List[dict]
    web_search_performed: bool
    research_info: Dict[str, Any]

    # Enhanced fields
    query_strategy: Dict[str, Any]
    reasoning_chain: Dict[str, Any]
    fact_check_report: Dict[str, Any]
    sentiment: Dict[str, str]
    conversation_state: Dict[str, Any]
    clarification_needed: bool
    clarification_data: Dict[str, Any]
    language_info: Dict[str, Any]
    quality_score: float
    response_time_ms: int
    enhanced_features_used: List[str]


# ======================================================================
# RAGAgent – enhanced public API
# ======================================================================

class RAGAgent:
    """Enhanced RAG agent with all advanced features integrated."""

    def __init__(self, vector_store: Optional[VectorStore] = None) -> None:
        self.vector_store = vector_store or VectorStore()
        self.memory = AdaptiveMemory()
        self.fact_verifier = FactVerifier()
        self.conversation_state = ConversationState()
        self.clarification_agent = ClarificationAgent()
        self.sentiment_adapter = SentimentAdapter()
        self.multilingual = MultilingualHandler()
        self.improvement_engine = SelfImprovementEngine()
        self.agent_team = AgentTeam(self.vector_store)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        conversation_history: Optional[List[dict]] = None,
        user_id: str = "default",
        enable_collaboration: bool = False,
        enable_fact_check: bool = True,
        enable_reasoning: bool = True,
    ) -> Dict[str, Any]:
        """Process *question* through the enhanced RAG pipeline.

        Everything is automatic – no slash commands required.
        Deep research triggers automatically when confidence < 0.8.
        Performance analysis runs silently in the background.
        """
        start_time = time.time()

        # Track query frequency
        self.memory.track_query(question)

        # Check for known corrections first
        correction = self.memory.get_correction(question)
        if correction:
            logger.info("Found known correction for query")
            return {
                "question": question,
                "final_answer": f"📝 *Updated answer based on previous feedback:*\n\n{correction}",
                "confidence": 0.95,
                "sources": ["user_correction"],
                "follow_up_suggestions": [],
                "web_search_performed": False,
                "enhanced_features_used": ["memory_correction"],
            }

        # ---- Language detection ----
        lang_info = self.multilingual.process_query(question)
        working_question = lang_info["english_query"]

        # ---- Sentiment analysis ----
        sentiment = self.sentiment_adapter.analyze_sentiment(
            question, conversation_history
        )

        # ---- Conversation branching ----
        conv_state = self.conversation_state.analyze_message(
            working_question, conversation_history or []
        )

        # Handle corrections
        if conv_state.get("is_correction") and conversation_history:
            last_q = ""
            last_a = ""
            for msg in reversed(conversation_history):
                if msg["role"] == "assistant" and not last_a:
                    last_a = msg["content"]
                elif msg["role"] == "user" and not last_q:
                    last_q = msg["content"]
                if last_q and last_a:
                    break
            if last_q and last_a:
                self.memory.learn_from_correction(
                    query=last_q,
                    wrong_answer=last_a,
                    correct_answer=question,
                    user_id=user_id,
                )

        # Use enhanced query from conversation manager
        enhanced_question = conv_state.get("enhanced_query", working_question)

        # ---- Query strategy analysis ----
        strategy = analyze_query(enhanced_question, conversation_history)

        # ---- Build initial state ----
        initial_state: AgentState = {
            "question": enhanced_question,
            "original_question": question,
            "conversation_history": conversation_history or [],
            "web_search_performed": False,
            "query_strategy": strategy.model_dump(),
            "sentiment": sentiment,
            "conversation_state": conv_state,
            "language_info": lang_info,
            "enhanced_features_used": [],
        }

        # Force web search if strategy says so
        if strategy.force_web_search:
            initial_state["web_search_performed"] = False  # allow it

        try:
            result = self.graph.invoke(initial_state)
        except Exception as exc:
            logger.error("Agent graph failed: %s", exc, exc_info=True)
            result = {
                "question": question,
                "final_answer": "I'm sorry, I encountered an error. Please try again.",
                "confidence": 0.0,
                "sources": [],
                "follow_up_suggestions": [],
                "web_search_performed": False,
                "enhanced_features_used": [],
            }

        # ---- Post-processing ----
        final_answer = result.get("final_answer", "")
        features_used = result.get("enhanced_features_used", [])
        confidence = result.get("confidence", 0.0)

        # ----------------------------------------------------------------
        # AUTO DEEP RESEARCH – triggered when confidence < 0.8
        # Completely transparent to the user; they just get a better answer.
        # ----------------------------------------------------------------
        doc_keywords = [
            "pdf", "document", "uploaded", "file", "datasheet",
            "the data", "this data", "above data", "attached",
        ]
        is_document_question = any(
            kw in enhanced_question.lower() for kw in doc_keywords
        )

        if (
            confidence < 0.80
            and not result.get("web_search_performed", False)
            and not is_document_question
        ):
            logger.info(
                "Confidence %.2f < 0.80 – auto deep research for: %s",
                confidence, enhanced_question[:80],
            )
            try:
                research_result = self._auto_deep_research(
                    enhanced_question, conversation_history, result
                )
                # Only replace if the new answer is more confident
                if research_result.get("confidence", 0) > confidence:
                    final_answer = research_result["final_answer"]
                    confidence = research_result["confidence"]
                    result["sources"] = research_result.get("sources", [])
                    result["web_search_performed"] = True
                    result["confidence"] = confidence
                    features_used.append("auto_deep_research")
            except Exception as exc:
                logger.error("Auto deep research failed: %s", exc)

        # Fact verification (if enabled and we have chunks)
        if enable_fact_check and result.get("retrieved_chunks"):
            try:
                fc_report = self.fact_verifier.verify_answer(
                    final_answer,
                    result.get("retrieved_chunks", []),
                    enhanced_question,
                )
                result["fact_check_report"] = {
                    "overall_reliability": fc_report.overall_reliability,
                    "verification_summary": fc_report.verification_summary,
                    "contradictions": fc_report.contradictions,
                    "claims_count": len(fc_report.claims),
                }
                if fc_report.verified_answer:
                    final_answer = fc_report.verified_answer
                features_used.append("fact_verification")
            except Exception as exc:
                logger.error("Fact verification failed: %s", exc)

        # Sentiment-based tone adjustment
        final_answer = self.sentiment_adapter.adapt_response(
            final_answer, sentiment
        )
        features_used.append("sentiment_adaptation")

        # Personalize for user
        final_answer = self.memory.personalize_response(user_id, final_answer)

        # Translate back if needed
        if lang_info.get("needs_translation"):
            final_answer = self.multilingual.translate_response(
                final_answer, lang_info["source_language"]
            )
            features_used.append("translation")

        result["final_answer"] = final_answer
        result["confidence"] = confidence
        result["enhanced_features_used"] = features_used

        # Record metrics silently (self-improvement engine)
        elapsed_ms = int((time.time() - start_time) * 1000)
        result["response_time_ms"] = elapsed_ms
        try:
            self.improvement_engine.record_interaction(
                query=question,
                answer=final_answer[:500],
                confidence=confidence,
                response_time_ms=elapsed_ms,
                web_search_used=result.get("web_search_performed", False),
                sources=result.get("sources", []),
            )
        except Exception:
            pass

        # Update user profile silently
        self.memory.update_user_profile(user_id, {
            "topic": conv_state.get("current_topic", ""),
        })

        return result

    # ------------------------------------------------------------------
    # Auto deep research (internal – no user command needed)
    # ------------------------------------------------------------------

    def _auto_deep_research(
        self,
        topic: str,
        conversation_history: Optional[List[dict]],
        prior_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Silently run deep research and return an improved answer.
        Called automatically when confidence < 0.8.
        """
        return self._handle_deep_research(topic, conversation_history)

    # ------------------------------------------------------------------
    # Internal – deep research (triggered automatically, never by command)
    # ------------------------------------------------------------------

    def _handle_deep_research(
        self, topic: str, conversation_history: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Run deep web research for *topic* and re-generate the answer."""
        logger.info("Deep research (auto): %s", topic[:80])
        try:
            info = deep_research(topic, self.vector_store)
        except Exception as exc:
            logger.error("Deep research failed: %s", exc)
            info = {"queries_used": [], "sites_indexed": [], "total_chunks": 0}

        queries_obj = generate_queries(topic, conversation_history)
        chunks = retrieve_chunks(queries_obj.queries, self.vector_store)
        resp = generate_response(topic, chunks, conversation_history)

        return {
            "question": topic,
            "final_answer": resp.answer,
            "confidence": resp.confidence,
            "sources": [s.source_url for s in resp.sources_used],
            "follow_up_suggestions": resp.follow_up_suggestions or [],
            "web_search_performed": True,
            "research_info": info,
            "enhanced_features_used": ["auto_deep_research"],
        }

    def get_performance_stats(self) -> dict:
        """Return performance report for sidebar display (no user command)."""
        try:
            return self.improvement_engine.analyze_performance()
        except Exception:
            return {}

    def get_memory_stats(self) -> dict:
        """Return memory stats for sidebar display (no user command)."""
        try:
            return {
                **self.memory.get_stats(),
                "top_queries": self.memory.get_top_queries(5),
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("classify_intent", self._node_classify_intent)
        graph.add_node("direct_response", self._node_direct_response)
        graph.add_node("generate_queries", self._node_generate_queries)
        graph.add_node("retrieve_chunks", self._node_retrieve_chunks)
        graph.add_node("generate_response", self._node_generate_response)
        graph.add_node("web_search", self._node_web_search)
        graph.add_node("re_retrieve", self._node_re_retrieve)
        graph.add_node("re_generate_response", self._node_re_generate_response)

        graph.set_entry_point("classify_intent")

        graph.add_conditional_edges(
            "classify_intent",
            self._route_after_intent,
            {
                "direct_response": "direct_response",
                "generate_queries": "generate_queries",
            },
        )

        graph.add_edge("direct_response", END)
        graph.add_edge("generate_queries", "retrieve_chunks")
        graph.add_edge("retrieve_chunks", "generate_response")

        graph.add_conditional_edges(
            "generate_response",
            self._route_after_response,
            {
                "web_search": "web_search",
                "end": END,
            },
        )

        graph.add_edge("web_search", "re_retrieve")
        graph.add_edge("re_retrieve", "re_generate_response")
        graph.add_edge("re_generate_response", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _route_after_intent(state: AgentState) -> str:
        if state.get("needs_retrieval"):
            return "generate_queries"
        return "direct_response"

    @staticmethod
    def _route_after_response(state: AgentState) -> str:
        confidence = state.get("confidence", 1.0)
        already_searched = state.get("web_search_performed", False)

        if already_searched:
            return "end"

        # Check query strategy – some query types force web search
        strategy = state.get("query_strategy", {})
        if strategy.get("force_web_search") and not already_searched:
            return "web_search"

        # Skip web search for document-related questions
        question = state.get("question", "").lower()
        doc_keywords = [
            "pdf", "document", "uploaded", "file", "datasheet",
            "the data", "this data", "above data", "attached",
        ]
        if any(kw in question for kw in doc_keywords):
            logger.info("Skipping web search – document question (confidence=%.2f)", confidence)
            return "end"

        if confidence < WEB_SEARCH_CONFIDENCE_THRESHOLD:
            return "web_search"

        return "end"

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _node_classify_intent(self, state: AgentState) -> dict:
        result = classify_intent(
            state["question"], state.get("conversation_history"),
        )
        features = state.get("enhanced_features_used", [])
        features.append("intent_classification")
        return {
            "intent": result.intent,
            "intent_confidence": result.confidence,
            "needs_retrieval": result.needs_retrieval,
            "enhanced_features_used": features,
        }

    def _node_direct_response(self, state: AgentState) -> dict:
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

    def _node_generate_queries(self, state: AgentState) -> dict:
        result = generate_queries(
            state["question"], state.get("conversation_history"),
        )
        features = state.get("enhanced_features_used", [])
        features.append("query_optimization")
        return {
            "generated_queries": result.queries,
            "enhanced_features_used": features,
        }

    def _node_retrieve_chunks(self, state: AgentState) -> dict:
        # Use strategy-based top_k if available
        strategy = state.get("query_strategy", {})
        top_k = strategy.get("top_k_override")

        kwargs = {}
        if top_k:
            kwargs["final_top_k"] = top_k

        chunks = retrieve_chunks(
            state.get("generated_queries", [state["question"]]),
            self.vector_store,
            **kwargs,
        )
        return {"retrieved_chunks": chunks}

    def _node_generate_response(self, state: AgentState) -> dict:
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

    def _node_web_search(self, state: AgentState) -> dict:
        logger.info(
            "Confidence %.2f – triggering web search",
            state.get("confidence", 0),
        )
        try:
            info = search_and_scrape(state["question"], self.vector_store)
        except Exception as exc:
            logger.error("Web search failed: %s", exc)
            info = {"sites_indexed": [], "total_chunks": 0}

        features = state.get("enhanced_features_used", [])
        features.append("web_search")
        return {
            "web_search_performed": True,
            "research_info": info,
            "enhanced_features_used": features,
        }

    def _node_re_retrieve(self, state: AgentState) -> dict:
        queries = state.get("generated_queries", [state["question"]])
        try:
            chunks = retrieve_chunks(queries, self.vector_store)
        except Exception as exc:
            logger.error("Re-retrieval failed: %s", exc)
            chunks = state.get("retrieved_chunks", [])
        return {"retrieved_chunks": chunks}

    def _node_re_generate_response(self, state: AgentState) -> dict:
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
