"""
RAGAgent – public API for the LangGraph-based RAG pipeline.

This module's single job is to expose the .ask() method that the
Streamlit app (and any other caller) uses.  It orchestrates:

  1. Pre-processing  – language, sentiment, conversation context,
                       query strategy, memory checks
  2. Graph execution – delegates to builder.build_graph()
  3. Post-processing – auto deep research, fact-check, sentiment
                       adaptation, personalisation, translation,
                       metrics recording
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from src.agents.graph.builder import build_graph
from src.agents.graph.state import AgentState

from src.agents.query_analyzer import analyze_query
from src.agents.memory import AdaptiveMemory
from src.agents.fact_verifier import FactVerifier
from src.agents.conversation_manager import ConversationState
from src.agents.clarification_agent import ClarificationAgent
from src.agents.sentiment_adapter import SentimentAdapter
from src.agents.multilingual import MultilingualHandler
from src.agents.self_improvement import SelfImprovementEngine
from src.agents.agent_team import AgentTeam

from src.agents.query_generator import generate_queries
from src.agents.response_generator import generate_response
from src.agents.retriever import retrieve_chunks
from src.agents.web_searcher import deep_research

from src.database.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGAgent:
    """High-level agent that exposes a single `.ask()` method.

    Internally it builds and runs a LangGraph state machine, then
    applies post-processing (fact-check, sentiment, memory, translation).
    """

    def __init__(self, vector_store: Optional[VectorStore] = None) -> None:
        self.vector_store = vector_store or VectorStore()

        # Sub-agents (all stateless helpers or persistent stores)
        self.memory = AdaptiveMemory()
        self.fact_verifier = FactVerifier()
        self.conversation_state = ConversationState()
        self.clarification_agent = ClarificationAgent()
        self.sentiment_adapter = SentimentAdapter()
        self.multilingual = MultilingualHandler()
        self.improvement_engine = SelfImprovementEngine()
        self.agent_team = AgentTeam(self.vector_store)

        # Compile the graph once at startup
        self.graph = build_graph(self.vector_store)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        conversation_history: Optional[List[dict]] = None,
        user_id: str = "default",
        enable_fact_check: bool = True,
    ) -> Dict[str, Any]:
        """Process *question* through the full enhanced RAG pipeline.

        Everything is automatic – deep research triggers when confidence
        is below 0.8, performance analysis runs silently in the
        background, and all answers are personalised per user.

        Parameters
        ----------
        question : str
            Raw user input (any language).
        conversation_history : list[dict] | None
            Previous ``{"role": …, "content": …}`` turns.
        user_id : str
            Identifier for per-user preference personalisation.
        enable_fact_check : bool
            Set False to skip the fact-verification step (faster).

        Returns
        -------
        dict
            final_answer, confidence, sources, web_search_performed,
            fact_check_report, enhanced_features_used, response_time_ms …
        """
        start_time = time.time()

        # ── 1. Memory: return known correction immediately ──────────────
        self.memory.track_query(question)
        correction = self.memory.get_correction(question)
        if correction:
            logger.info("Returning known correction for: %s", question[:60])
            return {
                "question": question,
                "final_answer": f"📝 *Updated answer based on previous feedback:*\n\n{correction}",
                "confidence": 0.95,
                "sources": ["user_correction"],
                "follow_up_suggestions": [],
                "web_search_performed": False,
                "enhanced_features_used": ["memory_correction"],
                "response_time_ms": 0,
            }

        # ── 2. Pre-processing ───────────────────────────────────────────
        lang_info = self.multilingual.process_query(question)
        working_question = lang_info["english_query"]

        sentiment = self.sentiment_adapter.analyze_sentiment(
            question, conversation_history
        )

        conv_state = self.conversation_state.analyze_message(
            working_question, conversation_history or []
        )

        # Learn from user corrections ("Actually, the answer is …")
        if conv_state.get("is_correction") and conversation_history:
            last_q, last_a = "", ""
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

        enhanced_question = conv_state.get("enhanced_query", working_question)
        strategy = analyze_query(enhanced_question, conversation_history)

        # ── 3. Build initial state and run graph ────────────────────────
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

        try:
            result = self.graph.invoke(initial_state)
        except Exception as exc:
            logger.error("Graph invocation failed: %s", exc, exc_info=True)
            result = {
                "question": question,
                "final_answer": "I'm sorry, I encountered an error. Please try again.",
                "confidence": 0.0,
                "sources": [],
                "follow_up_suggestions": [],
                "web_search_performed": False,
                "enhanced_features_used": [],
            }

        # ── 4. Post-processing ──────────────────────────────────────────
        final_answer = result.get("final_answer", "")
        features_used = list(result.get("enhanced_features_used", []))
        confidence = result.get("confidence", 0.0)

        # 4a. Auto deep research when confidence < 0.80
        doc_keywords = (
            "pdf", "document", "uploaded", "file", "datasheet",
            "the data", "this data", "above data", "attached",
        )
        is_document_q = any(kw in enhanced_question.lower() for kw in doc_keywords)

        if (
            confidence < 0.80
            and not result.get("web_search_performed", False)
            and not is_document_q
        ):
            logger.info(
                "Confidence %.2f < 0.80 – auto deep research: %s",
                confidence, enhanced_question[:80],
            )
            try:
                research = self._deep_research(enhanced_question, conversation_history)
                if research.get("confidence", 0) > confidence:
                    final_answer = research["final_answer"]
                    confidence = research["confidence"]
                    result["sources"] = research.get("sources", [])
                    result["web_search_performed"] = True
                    result["confidence"] = confidence
                    features_used.append("auto_deep_research")
            except Exception as exc:
                logger.error("Auto deep research failed: %s", exc)

        # 4b. Fact verification
        if enable_fact_check and result.get("retrieved_chunks"):
            try:
                fc = self.fact_verifier.verify_answer(
                    final_answer,
                    result.get("retrieved_chunks", []),
                    enhanced_question,
                )
                result["fact_check_report"] = {
                    "overall_reliability": fc.overall_reliability,
                    "verification_summary": fc.verification_summary,
                    "contradictions": fc.contradictions,
                    "claims_count": len(fc.claims),
                }
                if fc.verified_answer:
                    final_answer = fc.verified_answer
                features_used.append("fact_verification")
            except Exception as exc:
                logger.error("Fact verification failed: %s", exc)

        # 4c. Sentiment-based tone adaptation
        final_answer = self.sentiment_adapter.adapt_response(final_answer, sentiment)
        features_used.append("sentiment_adaptation")

        # 4d. Per-user personalisation
        final_answer = self.memory.personalize_response(user_id, final_answer)

        # 4e. Translate answer back to user's language
        if lang_info.get("needs_translation"):
            final_answer = self.multilingual.translate_response(
                final_answer, lang_info["source_language"]
            )
            features_used.append("translation")

        # ── 5. Finalise result ──────────────────────────────────────────
        result["final_answer"] = final_answer
        result["confidence"] = confidence
        result["enhanced_features_used"] = features_used

        elapsed_ms = int((time.time() - start_time) * 1000)
        result["response_time_ms"] = elapsed_ms

        # Record silently for self-improvement engine
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

        self.memory.update_user_profile(user_id, {
            "topic": conv_state.get("current_topic", ""),
        })

        return result

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _deep_research(
        self,
        topic: str,
        conversation_history: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Run deep web research and return a fresh answer.
        Called automatically (never by the user).
        """
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

    # ──────────────────────────────────────────────────────────────────
    # Sidebar helpers (called by Streamlit, not the graph)
    # ──────────────────────────────────────────────────────────────────

    def get_performance_stats(self) -> dict:
        """Return performance report for sidebar display."""
        try:
            return self.improvement_engine.analyze_performance()
        except Exception:
            return {}

    def get_memory_stats(self) -> dict:
        """Return memory stats for sidebar display."""
        try:
            return {
                **self.memory.get_stats(),
                "top_queries": self.memory.get_top_queries(5),
            }
        except Exception:
            return {}
