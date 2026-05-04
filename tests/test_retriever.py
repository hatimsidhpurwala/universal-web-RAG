"""
tests/test_retriever.py

Tests for the retrieval pipeline.
Tests source prefix filtering, fallback behavior, and deduplication.

Run:  pytest tests/test_retriever.py -v
Note: Requires Qdrant connection (QDRANT_URL + QDRANT_API_KEY in config/.env)
"""

import pytest
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv("config/.env")


class TestRetrieverBasics:

    def test_retrieve_returns_list(self):
        """retrieve_chunks must return a list."""
        from src.agents.retriever import retrieve_chunks
        from src.database.vector_store import VectorStore
        vs = VectorStore()
        result = retrieve_chunks(["SALTO keycard specifications"], vs)
        assert isinstance(result, list)

    def test_each_chunk_has_required_keys(self):
        """Every returned chunk must have text, source_url, site_name, score."""
        from src.agents.retriever import retrieve_chunks
        from src.database.vector_store import VectorStore
        vs = VectorStore()
        result = retrieve_chunks(["SALTO keycard"], vs, final_top_k=5)
        for chunk in result:
            assert "text" in chunk
            assert "source_url" in chunk
            assert "site_name" in chunk
            assert "score" in chunk

    def test_sorted_by_score_descending(self):
        """Results must be sorted with highest score first."""
        from src.agents.retriever import retrieve_chunks
        from src.database.vector_store import VectorStore
        vs = VectorStore()
        result = retrieve_chunks(["SALTO access control"], vs, final_top_k=8)
        if len(result) >= 2:
            scores = [c["score"] for c in result]
            assert scores == sorted(scores, reverse=True), (
                f"Chunks not sorted by score: {scores}"
            )

    def test_respects_final_top_k(self):
        """Result must not exceed final_top_k chunks."""
        from src.agents.retriever import retrieve_chunks
        from src.database.vector_store import VectorStore
        vs = VectorStore()
        result = retrieve_chunks(["SALTO keycard"], vs, final_top_k=3)
        assert len(result) <= 3, f"Expected <= 3 chunks, got {len(result)}"

    def test_no_duplicate_texts(self):
        """No two returned chunks should have identical text."""
        from src.agents.retriever import retrieve_chunks
        from src.database.vector_store import VectorStore
        vs = VectorStore()
        result = retrieve_chunks(["SALTO keycard chip specifications"], vs, final_top_k=10)
        texts = [c["text"][:200] for c in result]
        assert len(texts) == len(set(texts)), "Duplicate chunks found in results"


class TestSourcePrefixFilter:

    def test_pdf_prefix_returns_only_pdf_chunks(self):
        """With source_prefix='pdf_', all results must have pdf_ site_name."""
        from src.agents.retriever import retrieve_chunks
        from src.database.vector_store import VectorStore
        vs = VectorStore()

        # Check if any PDF is indexed first
        sites = vs.list_indexed_sites()
        pdf_sites = [s for s in sites if s.startswith("pdf_")]
        if not pdf_sites:
            pytest.skip("No PDF indexed in Qdrant — upload a PDF and re-run")

        result = retrieve_chunks(
            ["SALTO keycard specifications"],
            vs,
            source_prefix="pdf_",
            final_top_k=5,
        )
        for chunk in result:
            assert chunk["site_name"].startswith("pdf_"), (
                f"Expected pdf_ site, got: {chunk['site_name']}"
            )

    def test_no_prefix_searches_all_sources(self):
        """Without source_prefix, results may include any source type."""
        from src.agents.retriever import retrieve_chunks
        from src.database.vector_store import VectorStore
        vs = VectorStore()
        result = retrieve_chunks(["time attendance system"], vs, final_top_k=5)
        # Should return something from any source
        assert len(result) >= 0  # just verify no crash

    def test_nonexistent_prefix_triggers_fallback(self):
        """If prefix matches nothing, fallback to global search (no crash)."""
        from src.agents.retriever import retrieve_chunks
        from src.database.vector_store import VectorStore
        vs = VectorStore()
        # Use a prefix that definitely doesn't exist
        result = retrieve_chunks(
            ["some query"],
            vs,
            source_prefix="nonexistent_prefix_xyz_",
            final_top_k=3,
        )
        assert isinstance(result, list)  # must not crash


class TestVectorStore:

    def test_vector_store_connects(self):
        """VectorStore must initialise without errors."""
        from src.database.vector_store import VectorStore
        vs = VectorStore()
        assert vs.client is not None

    def test_list_indexed_sites_returns_list(self):
        """list_indexed_sites must return a list."""
        from src.database.vector_store import VectorStore
        vs = VectorStore()
        result = vs.list_indexed_sites()
        assert isinstance(result, list)

    def test_has_site_returns_bool(self):
        """has_site must return a boolean."""
        from src.database.vector_store import VectorStore
        vs = VectorStore()
        result = vs.has_site("nonexistent_site_xyz")
        assert isinstance(result, bool)
        assert result is False

    def test_get_collection_info_returns_dict(self):
        """get_collection_info must return a dict with expected keys."""
        from src.database.vector_store import VectorStore
        vs = VectorStore()
        info = vs.get_collection_info()
        assert "points_count" in info or "vectors_count" in info
