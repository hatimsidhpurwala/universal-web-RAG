"""
Tests for agent models – validates Pydantic schemas parse correctly.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.models import (
    IntentClassification,
    QueryGeneration,
    ResponseGeneration,
    SourceReference,
)


def test_intent_classification_valid():
    data = {
        "intent": "retrieval_needed",
        "confidence": 0.95,
        "reasoning": "User asks about product specs",
        "needs_retrieval": True,
    }
    ic = IntentClassification(**data)
    assert ic.intent == "retrieval_needed"
    assert ic.confidence == 0.95
    assert ic.needs_retrieval is True


def test_query_generation_valid():
    data = {
        "queries": ["security camera pricing", "installation cost"],
        "primary_entities": ["security camera", "pricing"],
        "query_type": "factual",
    }
    qg = QueryGeneration(**data)
    assert len(qg.queries) == 2
    assert qg.query_type == "factual"


def test_response_generation_valid():
    data = {
        "answer": "The camera costs $299.",
        "confidence": 0.9,
        "sources_used": [
            {"source_url": "https://example.com", "site_name": "example", "relevance_score": 0.92}
        ],
        "follow_up_suggestions": ["What about installation?"],
    }
    rg = ResponseGeneration(**data)
    assert rg.confidence == 0.9
    assert len(rg.sources_used) == 1
    assert isinstance(rg.sources_used[0], SourceReference)


def test_intent_classification_rejects_invalid_intent():
    import pydantic

    try:
        IntentClassification(
            intent="invalid_category",
            confidence=0.5,
            reasoning="test",
            needs_retrieval=False,
        )
        assert False, "Should have raised validation error"
    except pydantic.ValidationError:
        pass


if __name__ == "__main__":
    test_intent_classification_valid()
    print("✅ test_intent_classification_valid passed")

    test_query_generation_valid()
    print("✅ test_query_generation_valid passed")

    test_response_generation_valid()
    print("✅ test_response_generation_valid passed")

    test_intent_classification_rejects_invalid_intent()
    print("✅ test_intent_classification_rejects_invalid_intent passed")

    print("\n🎉 All model tests passed!")
