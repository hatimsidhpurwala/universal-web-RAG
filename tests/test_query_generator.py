"""
tests/test_query_generator.py

Tests for the query generation agent.
Validates query count, format, diversity, and type-specific generation.

Run:  pytest tests/test_query_generator.py -v
"""

import pytest
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv("config/.env")

from src.agents.query_generator import generate_queries


class TestQueryGeneratorBasics:

    def test_returns_at_least_two_queries(self):
        """Must generate at least 2 queries (new minimum)."""
        result = generate_queries("what chip does the SALTO keycard use")
        assert len(result.queries) >= 2, (
            f"Expected >= 2 queries, got {len(result.queries)}: {result.queries}"
        )

    def test_returns_at_most_four_queries(self):
        """Must not generate more than 4 queries."""
        result = generate_queries("give me full contact details and specs of SALTO")
        assert len(result.queries) <= 4, (
            f"Expected <= 4 queries, got {len(result.queries)}: {result.queries}"
        )

    def test_no_empty_queries(self):
        """No query should be empty or whitespace."""
        result = generate_queries("contact details for SALTO in India")
        for q in result.queries:
            assert q.strip(), f"Found empty query in: {result.queries}"

    def test_no_duplicate_queries(self):
        """All queries must be semantically distinct (no identical strings)."""
        result = generate_queries("SALTO keycard technical specifications")
        lower = [q.lower().strip() for q in result.queries]
        assert len(lower) == len(set(lower)), (
            f"Duplicate queries found: {result.queries}"
        )

    def test_queries_are_short(self):
        """Each query should be no longer than 10 words."""
        result = generate_queries("what is the price of SALTO keycards")
        for q in result.queries:
            word_count = len(q.split())
            assert word_count <= 10, (
                f"Query too long ({word_count} words): {q!r}"
            )

    def test_queries_are_not_full_sentences(self):
        """Queries should not end with a question mark."""
        result = generate_queries("explain the SALTO keycard features")
        for q in result.queries:
            assert not q.strip().endswith("?"), (
                f"Query should not be a question: {q!r}"
            )


class TestQueryGeneratorTypes:

    def test_contact_question_generates_location_queries(self):
        """Contact questions should include location-relevant keywords."""
        result = generate_queries("contact details for SALTO in India")
        combined = " ".join(result.queries).lower()
        # Should have India or contact or distributor
        has_relevant = any(
            kw in combined
            for kw in ["india", "contact", "distributor", "address", "office"]
        )
        assert has_relevant, (
            f"Contact question should produce location queries. Got: {result.queries}"
        )

    def test_pdf_question_generates_broad_queries(self):
        """Vague document questions should produce broad content queries."""
        result = generate_queries("what is the pdf about")
        combined = " ".join(result.queries).lower()
        has_content = any(
            kw in combined
            for kw in ["overview", "specification", "summary", "feature",
                       "description", "product", "document", "details"]
        )
        assert has_content, (
            f"PDF question should produce broad queries. Got: {result.queries}"
        )

    def test_technical_question_has_product_terms(self):
        """Technical questions should keep product/spec terms."""
        result = generate_queries("what is the chip security certification of SALTO keycard")
        combined = " ".join(result.queries).lower()
        has_tech = any(
            kw in combined
            for kw in ["salto", "chip", "security", "certification", "keycard", "mifare"]
        )
        assert has_tech, (
            f"Technical question should retain product terms. Got: {result.queries}"
        )


class TestQueryGeneratorModel:

    def test_returns_valid_pydantic_model(self):
        """Result must be a QueryGeneration Pydantic object."""
        from src.agents.models import QueryGeneration
        result = generate_queries("SALTO keycard specs")
        assert isinstance(result, QueryGeneration)

    def test_query_type_is_string(self):
        """query_type field must be a non-empty string."""
        result = generate_queries("SALTO price list")
        assert isinstance(result.query_type, str)
        assert result.query_type.strip()

    def test_fallback_on_empty_question(self):
        """Empty question should not crash — returns the raw question as query."""
        result = generate_queries("a")
        assert len(result.queries) >= 1
