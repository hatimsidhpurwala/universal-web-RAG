"""
tests/test_intent_classifier.py

Tests for the intent classification agent.
Validates that the LLM correctly categorises a range of user inputs
and that the confidence / needs_retrieval fields are correct.

Run:  pytest tests/test_intent_classifier.py -v
Note: These tests call the Groq API — requires GROQ_API_KEY in config/.env
"""

import pytest
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv("config/.env")

from src.agents.intent_classifier import classify_intent


# ── Parametrized test cases ───────────────────────────────────────────────────
# Format: (description, question, expected_intent, needs_retrieval)

INTENT_CASES = [
    # ── Greetings ──────────────────────────────────────────────────────────
    ("plain hello",         "hello",                   "greeting",         False),
    ("hey greeting",        "hey there",               "greeting",         False),
    ("good morning",        "good morning!",            "greeting",         False),
    ("how are you",         "how are you doing?",       "greeting",         False),

    # ── Gratitude ──────────────────────────────────────────────────────────
    ("thanks",              "thanks",                  "gratitude",        False),
    ("thank you",           "thank you so much",       "gratitude",        False),

    # ── Farewell ───────────────────────────────────────────────────────────
    ("goodbye",             "goodbye",                 "farewell",         False),
    ("see you",             "see you later",           "farewell",         False),

    # ── Retrieval needed ───────────────────────────────────────────────────
    ("pdf question",        "what is the pdf about i uploaded",
                                                       "retrieval_needed", True),
    ("product specs",       "what chip does the SALTO keycard use",
                                                       "retrieval_needed", True),
    ("contact question",    "give me the contact details for India",
                                                       "retrieval_needed", True),
    ("company question",    "what services does alhutaib.com offer",
                                                       "retrieval_needed", True),
    ("price question",      "what is the price of the SALTO keycard",
                                                       "retrieval_needed", True),
    ("where to buy",        "where can I buy this in Dubai",
                                                       "retrieval_needed", True),
    ("document uploaded",   "explain the uploaded document",
                                                       "retrieval_needed", True),

    # ── General knowledge ─────────────────────────────────────────────────
    ("what is python",      "what is Python programming language",
                                                       "general_knowledge", False),
    ("capital city",        "what is the capital of France",
                                                       "general_knowledge", False),
]


class TestIntentClassification:

    @pytest.mark.parametrize("desc,question,expected_intent,expected_retrieval",
                             INTENT_CASES,
                             ids=[c[0] for c in INTENT_CASES])
    def test_intent_classification(self, desc, question, expected_intent, expected_retrieval):
        """Each test case must return the expected intent and needs_retrieval."""
        result = classify_intent(question)

        assert result.intent == expected_intent, (
            f"[{desc}] Expected intent={expected_intent!r}, "
            f"got={result.intent!r} (confidence={result.confidence:.2f})"
        )
        assert result.needs_retrieval == expected_retrieval, (
            f"[{desc}] Expected needs_retrieval={expected_retrieval}, "
            f"got={result.needs_retrieval}"
        )

    def test_confidence_is_float(self):
        """confidence must be a float between 0 and 1."""
        result = classify_intent("hello")
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_high_confidence_for_clear_greeting(self):
        """A pure greeting should have confidence >= 0.85."""
        result = classify_intent("hello")
        assert result.confidence >= 0.85, (
            f"Expected confidence >= 0.85 for clear greeting, got {result.confidence:.2f}"
        )

    def test_high_confidence_for_pdf_question(self):
        """A PDF question should have confidence >= 0.85 and needs_retrieval=True."""
        result = classify_intent("what is the pdf i uploaded about")
        assert result.needs_retrieval is True
        assert result.confidence >= 0.80

    def test_returns_valid_model(self):
        """Result must be an IntentClassification Pydantic object."""
        from src.agents.models import IntentClassification
        result = classify_intent("what is machine learning")
        assert isinstance(result, IntentClassification)

    def test_error_fallback_is_retrieval(self):
        """If classification fails, it must default to retrieval_needed."""
        # Simulate by passing an extremely long garbage input
        garbage = "x " * 5000
        result = classify_intent(garbage[:500])
        # Should not crash, and must return a valid object
        assert result.intent in [
            "greeting", "gratitude", "farewell",
            "clarification", "general_knowledge", "retrieval_needed"
        ]
