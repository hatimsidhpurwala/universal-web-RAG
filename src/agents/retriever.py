"""
Retriever agent – takes generated queries, embeds them, searches
Qdrant, deduplicates, and returns the top-k most relevant chunks.
"""

from __future__ import annotations

import logging
from typing import List

from src.core.embedder import embed_query
from src.database.vector_store import VectorStore
from config.settings import TOP_K_PER_QUERY, TOP_K_RESULTS, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


def retrieve_chunks(
    queries: List[str],
    vector_store: VectorStore,
    *,
    top_k_per_query: int = TOP_K_PER_QUERY,
    final_top_k: int = TOP_K_RESULTS,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> List[dict]:
    """Retrieve and deduplicate the most relevant chunks for *queries*.

    Parameters
    ----------
    queries : list[str]
        Optimised search queries from the query generator.
    vector_store : VectorStore
        The Qdrant wrapper instance.
    top_k_per_query : int
        Number of results to fetch per query.
    final_top_k : int
        Maximum number of chunks to return after deduplication.
    similarity_threshold : float
        Minimum score to accept a chunk.

    Returns
    -------
    list[dict]
        Sorted by descending relevance score.
    """
    all_results: list[dict] = []
    seen_texts: set[str] = set()

    # Use a more generous per-query limit to get broader coverage
    effective_top_k = max(top_k_per_query, 5)

    for query in queries:
        query_vector = embed_query(query)
        results = vector_store.search_chunks(
            query_vector=query_vector,
            top_k=effective_top_k,
            score_threshold=similarity_threshold,
        )

        for chunk in results:
            text_key = chunk["text"][:200]  # first 200 chars as dedup key
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_results.append(chunk)

    # Fallback: if no results found with threshold, retry without it
    if not all_results:
        logger.info("No results with threshold %.2f – retrying without threshold", similarity_threshold)
        for query in queries:
            query_vector = embed_query(query)
            results = vector_store.search_chunks(
                query_vector=query_vector,
                top_k=effective_top_k,
                score_threshold=None,  # no threshold
            )
            for chunk in results:
                text_key = chunk["text"][:200]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    all_results.append(chunk)

    # Sort by score descending and trim
    all_results.sort(key=lambda c: c.get("score", 0), reverse=True)
    top_chunks = all_results[:final_top_k]

    logger.info(
        "Retriever: %d unique chunks from %d queries → returning top %d",
        len(all_results),
        len(queries),
        len(top_chunks),
    )
    return top_chunks
