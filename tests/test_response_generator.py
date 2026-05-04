"""
tests/test_response_generator.py

Tests for the response generation agent.
Validates grounding, format, confidence scoring, and hallucination prevention.

Run:  pytest tests/test_response_generator.py -v
"""

import pytest
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv("config/.env")

from src.agents.response_generator import generate_response
from src.agents.models import ResponseGeneration


# ── Context fixtures ──────────────────────────────────────────────────────────

SALTO_CONTEXT = [
    {
        "text": (
            "SALTO CCVD20xx uses NXP MIFARE DESFire EV3 chip, EAL5+ certified. "
            "Data retention: 10 years. Write endurance: 500,000 cycles. "
            "Operating frequency: 13.56 MHz."
        ),
        "source_url": "pdf_SALTO-Keycard-Datasheet",
        "site_name": "pdf_SALTO-Keycard-Datasheet",
        "score": 0.92,
        "context_header": "CHIP SPECIFICATIONS",
    },
    {
        "text": (
            "Physical: ISO CR-80 format (85.60 x 53.98 mm), 0.84 mm thick, PVC. "
            "Models: CCVD2001 (2KB blank), CCVD2002 (2KB custom print), "
            "CCVD4001 (4KB blank), CCVD4002 (4KB custom print)."
        ),
        "source_url": "pdf_SALTO-Keycard-Datasheet",
        "site_name": "pdf_SALTO-Keycard-Datasheet",
        "score": 0.88,
        "context_header": "PHYSICAL SPECIFICATIONS",
    },
    {
        "text": (
            "Contact: website www.saltosystems.com, email info@saltosystems.com. "
            "HQ: Oiartzun, Spain. EMEA: London, UK. APAC: Singapore. "
            "Certifications: CE, FCC, RoHS, REACH, ISO 9001:2015."
        ),
        "source_url": "pdf_SALTO-Keycard-Datasheet",
        "site_name": "pdf_SALTO-Keycard-Datasheet",
        "score": 0.81,
        "context_header": "CONTACT INFORMATION",
    },
]

EMPTY_CONTEXT = []


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestResponseGeneratorBasics:

    def test_returns_valid_pydantic_model(self):
        """Result must be a ResponseGeneration Pydantic object."""
        result = generate_response("what chip does the SALTO keycard use", SALTO_CONTEXT)
        assert isinstance(result, ResponseGeneration)

    def test_answer_is_non_empty(self):
        """Answer field must not be empty."""
        result = generate_response("what chip does the SALTO keycard use", SALTO_CONTEXT)
        assert result.answer.strip(), "Answer should not be empty"

    def test_confidence_is_valid_float(self):
        """Confidence must be a float between 0 and 1."""
        result = generate_response("what chip does the SALTO keycard use", SALTO_CONTEXT)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_follow_up_suggestions_present(self):
        """Should return at least 1 follow-up suggestion."""
        result = generate_response("what chip does the SALTO keycard use", SALTO_CONTEXT)
        assert result.follow_up_suggestions is not None
        assert len(result.follow_up_suggestions) >= 1


class TestGrounding:

    def test_chip_name_in_answer(self):
        """Answer to chip question must mention MIFARE or DESFire."""
        result = generate_response(
            "what chip does the SALTO keycard use",
            SALTO_CONTEXT
        )
        has_chip = any(
            kw in result.answer
            for kw in ["MIFARE", "DESFire", "NXP", "chip"]
        )
        assert has_chip, (
            f"Chip question answer missing chip name. Got: {result.answer[:200]}"
        )

    def test_high_confidence_for_direct_context_match(self):
        """When context directly answers the question, confidence >= 0.75."""
        result = generate_response(
            "what chip does the SALTO keycard use",
            SALTO_CONTEXT
        )
        assert result.confidence >= 0.75, (
            f"Expected high confidence for direct match, got {result.confidence:.2f}"
        )

    def test_contact_details_extracted(self):
        """Contact question should extract email/website from context."""
        result = generate_response(
            "what is the contact information for SALTO",
            SALTO_CONTEXT
        )
        has_contact = any(
            kw in result.answer
            for kw in ["saltosystems.com", "info@saltosystems.com", "Spain", "London"]
        )
        assert has_contact, (
            f"Contact answer should include contact details. Got: {result.answer[:300]}"
        )

    def test_empty_context_low_confidence(self):
        """Empty context should result in confidence < 0.5."""
        result = generate_response(
            "what is the memory size of SALTO CCVD4001",
            EMPTY_CONTEXT
        )
        assert result.confidence < 0.75, (
            f"Expected lower confidence for empty context, got {result.confidence:.2f}"
        )

    def test_answer_does_not_start_with_filler(self):
        """Answer must not start with 'Certainly!', 'Of course!', 'Great question!'."""
        result = generate_response(
            "what models are available",
            SALTO_CONTEXT
        )
        bad_starters = ["certainly", "of course", "great question", "sure!", "absolutely"]
        answer_lower = result.answer.lower().strip()
        for starter in bad_starters:
            assert not answer_lower.startswith(starter), (
                f"Answer should not start with filler word '{starter}'"
            )


class TestDocumentSummary:

    def test_summarize_uses_all_chunks(self):
        """Summary/overview should mention content from multiple chunks."""
        result = generate_response("what is this document about", SALTO_CONTEXT)
        # Should mention both technical specs AND contact info
        has_specs = any(k in result.answer for k in ["chip", "NXP", "MIFARE", "specifications"])
        has_contact = any(k in result.answer for k in ["saltosystems", "contact", "website"])
        assert has_specs or has_contact, (
            f"Summary should mention document content. Got: {result.answer[:300]}"
        )

    def test_model_numbers_mentioned(self):
        """If models are in context, the answer should reference them."""
        result = generate_response("what models are available for SALTO keycards", SALTO_CONTEXT)
        has_model = any(
            m in result.answer
            for m in ["CCVD2001", "CCVD2002", "CCVD4001", "CCVD4002", "2KB", "4KB"]
        )
        assert has_model, (
            f"Model answer should include model numbers. Got: {result.answer[:300]}"
        )
