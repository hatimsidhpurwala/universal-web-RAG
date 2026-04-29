"""
Tests for the core processing pipeline: scraper, cleaner, chunker, embedder.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# Cleaner tests
# =====================================================================

def test_normalize_text():
    from src.core.cleaner import normalize_text

    raw = "  Hello   world  \n\n\n\n\ntest  "
    result = normalize_text(raw)
    assert "   " not in result, "Multiple spaces should be collapsed"
    assert "\n\n\n" not in result, "3+ newlines should be collapsed to 2"
    assert result == "Hello world\n\ntest"


def test_deduplicate_chunks():
    from src.core.cleaner import deduplicate_chunks

    chunks = [
        {"text": "The quick brown fox jumps over the lazy dog"},
        {"text": "The quick brown fox jumps over the lazy dog"},  # exact dup
        {"text": "Something completely different here"},
    ]
    result = deduplicate_chunks(chunks)
    assert len(result) == 2, f"Expected 2 unique chunks, got {len(result)}"


# =====================================================================
# Chunker tests
# =====================================================================

def test_chunk_markdown_produces_chunks():
    from src.core.chunker import chunk_markdown

    text = "Hello world. " * 200  # ~200 words
    chunks = chunk_markdown(text, source_url="test://doc")
    assert len(chunks) > 0, "Should produce at least one chunk"
    assert all("text" in c for c in chunks)
    assert all("source_url" in c for c in chunks)
    assert all("chunk_index" in c for c in chunks)


def test_chunk_markdown_empty_input():
    from src.core.chunker import chunk_markdown

    chunks = chunk_markdown("", source_url="test://empty")
    assert chunks == [], "Empty input should return empty list"


# =====================================================================
# Scraper tests (offline – no network)
# =====================================================================

def test_scrape_returns_none_on_bad_url():
    from src.core.scraper import scrape_website

    result = scrape_website("http://this-domain-does-not-exist-xyz.invalid")
    assert result is None


# =====================================================================
# Run
# =====================================================================

if __name__ == "__main__":
    test_normalize_text()
    print("✅ test_normalize_text passed")

    test_deduplicate_chunks()
    print("✅ test_deduplicate_chunks passed")

    test_chunk_markdown_produces_chunks()
    print("✅ test_chunk_markdown_produces_chunks passed")

    test_chunk_markdown_empty_input()
    print("✅ test_chunk_markdown_empty_input passed")

    test_scrape_returns_none_on_bad_url()
    print("✅ test_scrape_returns_none_on_bad_url passed")

    print("\n🎉 All core tests passed!")
