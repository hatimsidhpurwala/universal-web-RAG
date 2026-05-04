"""
Web searcher agent – DuckDuckGo search, scrape-and-index, and
deep-research capabilities.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse

from duckduckgo_search import DDGS
from groq import Groq

from config.settings import (
    DEEP_RESEARCH_NUM_QUERIES,
    GROQ_API_KEY,
    LLM_MODEL,
    SCRAPE_DELAY_SECONDS,
    WEB_SEARCH_MAX_RESULTS,
)
from src.core.chunker import chunk_markdown
from src.core.cleaner import deduplicate_chunks, normalize_text
from src.core.embedder import embed_chunks
from src.core.scraper import scrape_website
from src.database.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ======================================================================
# Basic search
# ======================================================================

def search(
    query: str,
    max_results: int = WEB_SEARCH_MAX_RESULTS,
) -> List[dict]:
    """Search DuckDuckGo and return a list of result dicts.

    Each dict has keys: ``title``, ``url``, ``snippet``.
    """
    raw: list = []          # ← always defined; avoids UnboundLocalError
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        logger.info("DuckDuckGo returned %d results for '%s'", len(raw), query)
    except Exception as exc:
        logger.error("DuckDuckGo search failed: %s", exc)

    # Safe to use raw here — it is always a list (empty on failure)
    return [
        {
            "title": r.get("title", ""),
            "url":   r.get("href", r.get("link", "")),
            "snippet": r.get("body", ""),
        }
        for r in raw
    ]


# ======================================================================
# Search + scrape + index
# ======================================================================

def search_and_scrape(
    query: str,
    vector_store: VectorStore,
    max_results: int = WEB_SEARCH_MAX_RESULTS,
) -> Dict:
    """Search the web, scrape top results, and index them.

    Returns
    -------
    dict
        ``sites_indexed`` (list of site labels),
        ``total_chunks`` (int).
    """
    results = search(query, max_results=max_results)
    sites_indexed: list[str] = []
    total_chunks = 0

    for result in results:
        url = result.get("url", "")
        if not url:
            continue

        markdown = scrape_website(url, delay=SCRAPE_DELAY_SECONDS)
        if not markdown:
            continue

        markdown = normalize_text(markdown)
        domain = urlparse(url).netloc.replace("www.", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        site_name = f"web_{domain}_{timestamp}"

        chunks = chunk_markdown(markdown, source_url=url)
        chunks = deduplicate_chunks(chunks)
        chunks = embed_chunks(chunks, show_progress=False)

        stored = vector_store.store_chunks_for_site(chunks, site_name)
        total_chunks += stored
        sites_indexed.append(site_name)

        logger.info("Indexed %d chunks from %s", stored, url)

    return {"sites_indexed": sites_indexed, "total_chunks": total_chunks}


# ======================================================================
# Deep research
# ======================================================================

def _generate_research_queries(topic: str) -> List[str]:
    """Use the LLM to produce multiple search queries for *topic*."""
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate exactly 3 distinct web-search queries to "
                    "comprehensively research the given topic. Return JSON: "
                    '{"queries": ["q1", "q2", "q3"]}'
                ),
            },
            {"role": "user", "content": topic},
        ],
        temperature=0.3,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    queries = data.get("queries", [topic])
    logger.info("Deep research queries: %s", queries)
    return queries[:DEEP_RESEARCH_NUM_QUERIES]


def deep_research(
    topic: str,
    vector_store: VectorStore,
    num_queries: int = DEEP_RESEARCH_NUM_QUERIES,
) -> Dict:
    """Perform deep research: generate multiple queries, scrape, and index.

    Returns
    -------
    dict
        ``queries_used``, ``sites_indexed``, ``total_chunks``.
    """
    queries = _generate_research_queries(topic)
    all_sites: list[str] = []
    total_chunks = 0

    for query in queries:
        info = search_and_scrape(query, vector_store, max_results=2)
        all_sites.extend(info["sites_indexed"])
        total_chunks += info["total_chunks"]

    logger.info(
        "Deep research complete: %d queries, %d sites, %d chunks",
        len(queries),
        len(all_sites),
        total_chunks,
    )
    return {
        "queries_used": queries,
        "sites_indexed": all_sites,
        "total_chunks": total_chunks,
    }
