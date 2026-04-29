"""
Multi-Agent Collaboration System – a team of specialized agents
(researcher, analyst, fact-checker, synthesizer, critic) that work
together to produce high-quality answers.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from groq import Groq

from config.settings import GROQ_API_KEY, LLM_MODEL
from src.agents.retriever import retrieve_chunks
from src.agents.web_searcher import search_and_scrape
from src.core.embedder import embed_query
from src.database.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ======================================================================
# Individual Agent Prompts
# ======================================================================

_RESEARCHER_PROMPT = """\
You are a Research Agent. Your job is to thoroughly investigate a topic
using the provided context chunks. Extract ALL relevant facts, figures,
specifications, and details.

Return JSON:
{
  "findings": ["finding1", "finding2", ...],
  "key_facts": {"fact_name": "fact_value", ...},
  "sources_quality": "high|medium|low",
  "coverage_assessment": "How well do the sources cover the topic"
}
"""

_ANALYST_PROMPT = """\
You are an Analysis Agent. Given research findings, perform deep analysis:
identify patterns, make comparisons, draw insights, and highlight
implications.

Return JSON:
{
  "analysis": "Detailed analysis text...",
  "key_insights": ["insight1", "insight2"],
  "comparisons": [{"item_a": "...", "item_b": "...", "comparison": "..."}],
  "data_points": {"metric": "value", ...}
}
"""

_FACT_CHECKER_PROMPT = """\
You are a Fact-Checking Agent. Given claims and evidence, verify each
claim. Flag contradictions, unverified claims, and potential inaccuracies.

Return JSON:
{
  "verified_claims": [
    {"claim": "...", "status": "verified|unverified|contradicted", "evidence": "...", "confidence": 0.9}
  ],
  "contradictions": ["contradiction description"],
  "reliability_score": 0.85
}
"""

_SYNTHESIZER_PROMPT = """\
You are a Synthesis Agent. Combine research findings, analysis, and
fact-check results into a comprehensive, well-structured answer.

Rules:
1. Lead with the most important information
2. Use bullet points for key details
3. Include verified facts with confidence markers
4. Flag any unverified claims
5. Provide a balanced view if comparing things
6. Be specific – include numbers, names, specifications

Return JSON:
{
  "synthesized_answer": "The comprehensive answer...",
  "key_points": ["point1", "point2"],
  "confidence": 0.85,
  "caveats": ["caveat1"]
}
"""

_CRITIC_PROMPT = """\
You are a Quality Critic Agent. Review the draft answer for:
1. Completeness: Does it fully answer the question?
2. Accuracy: Are facts correctly stated?
3. Clarity: Is it easy to understand?
4. Bias: Is it balanced and objective?
5. Sources: Are claims properly supported?

Return JSON:
{
  "quality_score": 0.85,
  "completeness": 0.9,
  "accuracy": 0.85,
  "clarity": 0.9,
  "bias_check": "No bias detected" or "Potential bias: ...",
  "improvements": ["improvement suggestion 1", ...],
  "improved_answer": "The improved final answer...",
  "passed_review": true
}
"""


# ======================================================================
# Agent Team
# ======================================================================

class AgentTeam:
    """Orchestra of specialized agents working together."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()

    def collaborate(
        self,
        query: str,
        context_chunks: List[dict],
        conversation_history: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Run the full multi-agent collaboration pipeline.

        Returns
        -------
        dict
            ``final_answer``, ``confidence``, ``reasoning_trace``,
            ``fact_check_report``, ``quality_score``.
        """
        trace: Dict[str, Any] = {}

        # ---- Phase 1: Research ----
        research = self._run_agent(
            "Researcher",
            _RESEARCHER_PROMPT,
            query,
            context_chunks,
            conversation_history,
        )
        trace["research"] = research

        # ---- Phase 2: Analysis ----
        analysis_context = (
            f"Research findings: {json.dumps(research)}\n\n"
            f"Original question: {query}"
        )
        analysis = self._run_agent(
            "Analyst",
            _ANALYST_PROMPT,
            analysis_context,
            [],
            conversation_history,
        )
        trace["analysis"] = analysis

        # ---- Phase 3: Fact Check ----
        fc_context = (
            f"Claims to verify from research:\n{json.dumps(research)}\n\n"
            f"Analysis:\n{json.dumps(analysis)}"
        )
        fact_check = self._run_agent(
            "Fact Checker",
            _FACT_CHECKER_PROMPT,
            fc_context,
            context_chunks,
            conversation_history,
        )
        trace["fact_check"] = fact_check

        # ---- Phase 4: Synthesis ----
        synth_context = (
            f"Original question: {query}\n\n"
            f"Research: {json.dumps(research)}\n\n"
            f"Analysis: {json.dumps(analysis)}\n\n"
            f"Fact-check: {json.dumps(fact_check)}"
        )
        synthesis = self._run_agent(
            "Synthesizer",
            _SYNTHESIZER_PROMPT,
            synth_context,
            [],
            conversation_history,
        )
        trace["synthesis"] = synthesis
        draft_answer = synthesis.get("synthesized_answer", "")

        # ---- Phase 5: Critic Review ----
        critic_context = (
            f"Original question: {query}\n\n"
            f"Draft answer: {draft_answer}\n\n"
            f"Fact-check reliability: {fact_check.get('reliability_score', 0)}"
        )
        critique = self._run_agent(
            "Critic",
            _CRITIC_PROMPT,
            critic_context,
            [],
            conversation_history,
        )
        trace["critique"] = critique

        # Use the improved answer if the critic provided one
        final_answer = critique.get("improved_answer", draft_answer)
        quality_score = critique.get("quality_score", 0.7)
        passed = critique.get("passed_review", True)

        if not passed and quality_score < 0.6:
            # If quality is too low, use synthesized answer directly
            final_answer = draft_answer

        logger.info(
            "Agent team completed: quality=%.2f, reliability=%.2f",
            quality_score,
            fact_check.get("reliability_score", 0),
        )

        return {
            "final_answer": final_answer,
            "confidence": synthesis.get("confidence", 0.7),
            "quality_score": quality_score,
            "fact_check_report": fact_check,
            "reasoning_trace": trace,
            "key_points": synthesis.get("key_points", []),
            "caveats": synthesis.get("caveats", []),
            "improvements": critique.get("improvements", []),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_agent(
        self,
        agent_name: str,
        system_prompt: str,
        user_content: str,
        context_chunks: List[dict],
        conversation_history: Optional[List[dict]],
    ) -> dict:
        """Run a single agent and return parsed JSON output."""
        client = Groq(api_key=GROQ_API_KEY)

        # Build context if chunks provided
        if context_chunks:
            ctx_parts = []
            for i, c in enumerate(context_chunks[:8], 1):
                ctx_parts.append(f"[Chunk {i}] {c['text']}")
            ctx_block = "\n".join(ctx_parts)
            user_content = f"Context:\n{ctx_block}\n\n{user_content}"

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            for msg in conversation_history[-4:]:
                messages.append(msg)
        messages.append({"role": "user", "content": user_content})

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            logger.info("%s agent completed", agent_name)
            return data
        except Exception as exc:
            logger.error("%s agent failed: %s", agent_name, exc)
            return {"error": str(exc)}
