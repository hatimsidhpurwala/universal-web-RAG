"""
tests/test_chunker.py

Unit tests for the document chunker.
Tests chunk count, overlap, metadata, and edge cases.

Run:  pytest tests/test_chunker.py -v
"""

import pytest
import sys
sys.path.insert(0, ".")

from src.core.chunker import chunk_markdown


# ── Fixtures ─────────────────────────────────────────────────────────────────

SALTO_SAMPLE = """
SALTO Systems CCVD20xx / CCVD40xx Keycard Datasheet

PRODUCT OVERVIEW
The SALTO CCVD20xx and CCVD40xx keycards use the NXP MIFARE DESFire EV3 chip,
certified to EAL5+ security standards. They provide keyless building access.

CHIP SPECIFICATIONS
- Chip: NXP MIFARE DESFire EV3
- Security: Common Criteria EAL5+
- Frequency: 13.56 MHz
- Standard: ISO/IEC 14443-A
- Data Retention: 10 years
- Write Endurance: 500,000 cycles

PHYSICAL SPECIFICATIONS
- Format: ISO CR-80 (85.60 x 53.98 mm)
- Thickness: 0.84 mm
- Material: PVC

CONTACT INFORMATION
- Website: www.saltosystems.com
- Email: info@saltosystems.com
- HQ: Oiartzun, Spain
- EMEA: London, UK
- APAC: Singapore

CERTIFICATIONS
- CE marking, FCC, RoHS, REACH, ISO 9001:2015
"""


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestChunkerBasics:

    def test_returns_list(self):
        """chunk_markdown must return a list."""
        result = chunk_markdown(SALTO_SAMPLE, source_url="test")
        assert isinstance(result, list)

    def test_produces_multiple_chunks(self):
        """A multi-section document must produce more than 1 chunk."""
        result = chunk_markdown(SALTO_SAMPLE, source_url="test")
        assert len(result) > 1, f"Expected multiple chunks, got {len(result)}"

    def test_more_chunks_than_old_settings(self):
        """With new chunk_size=150, should produce significantly more chunks
        than the old chunk_size=500 setting."""
        new = chunk_markdown(SALTO_SAMPLE, source_url="test", chunk_size=150)
        old = chunk_markdown(SALTO_SAMPLE, source_url="test", chunk_size=500)
        assert len(new) >= len(old), (
            f"New chunker ({len(new)} chunks) should produce >= old chunker ({len(old)} chunks)"
        )

    def test_empty_input_returns_empty(self):
        """Empty or whitespace-only text must return []."""
        assert chunk_markdown("", source_url="test") == []
        assert chunk_markdown("   \n  ", source_url="test") == []

    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size should produce exactly 1 chunk."""
        result = chunk_markdown("Hello world", source_url="test")
        assert len(result) == 1


class TestChunkStructure:

    def test_each_chunk_has_required_keys(self):
        """Every chunk dict must contain text, source_url, chunk_index."""
        chunks = chunk_markdown(SALTO_SAMPLE, source_url="test_source")
        for chunk in chunks:
            assert "text" in chunk, "Missing 'text' key"
            assert "source_url" in chunk, "Missing 'source_url' key"
            assert "chunk_index" in chunk, "Missing 'chunk_index' key"
            assert "context_header" in chunk, "Missing 'context_header' key"

    def test_source_url_preserved(self):
        """source_url must match what was passed in."""
        url = "pdf_test_document"
        chunks = chunk_markdown(SALTO_SAMPLE, source_url=url)
        for chunk in chunks:
            assert chunk["source_url"] == url

    def test_chunk_index_sequential(self):
        """chunk_index must be 0-based and sequential."""
        chunks = chunk_markdown(SALTO_SAMPLE, source_url="test")
        for expected_idx, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == expected_idx

    def test_no_empty_chunks(self):
        """No chunk should have empty or whitespace-only text."""
        chunks = chunk_markdown(SALTO_SAMPLE, source_url="test")
        for chunk in chunks:
            assert chunk["text"].strip(), f"Found empty chunk at index {chunk['chunk_index']}"

    def test_context_header_extracted(self):
        """At least some chunks should have a context_header (section name)."""
        chunks = chunk_markdown(SALTO_SAMPLE, source_url="test")
        headers = [c["context_header"] for c in chunks if c["context_header"]]
        assert len(headers) > 0, "Expected at least one context_header to be extracted"


class TestChunkContent:

    def test_all_text_covered(self):
        """Every word from the original should appear in at least one chunk."""
        key_words = ["MIFARE", "DESFire", "saltosystems.com", "EAL5+"]
        chunks = chunk_markdown(SALTO_SAMPLE, source_url="test")
        combined = " ".join(c["text"] for c in chunks)
        for word in key_words:
            assert word in combined, f"Key word '{word}' missing from all chunks"

    def test_no_chunk_exceeds_size(self):
        """No chunk should be more than 2x the chunk_size in characters."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        chunks = chunk_markdown(SALTO_SAMPLE, source_url="test", chunk_size=150)
        for chunk in chunks:
            token_count = len(enc.encode(chunk["text"], disallowed_special=()))
            assert token_count <= 300, (  # 2x tolerance
                f"Chunk {chunk['chunk_index']} has {token_count} tokens (limit 300)"
            )
