"""
Document chunking – deep, overlapping splits with rich metadata.

Strategy
--------
- CHUNK_SIZE = 150 tokens  (was 500) → more, smaller, precise chunks
- CHUNK_OVERLAP = 30 tokens          → smooth context continuity
- Separator hierarchy: headers → paragraphs → sentences → words → chars
- Each chunk carries source_url, chunk_index, and a context_header
  (the nearest heading above it) so the LLM knows which section it came from.

Why smaller chunks?
  A 2000-token PDF with CHUNK_SIZE=500 → 4 chunks (one per page).
  With CHUNK_SIZE=150 → ~14 chunks (one per section/paragraph).
  Smaller chunks = more granular retrieval = more specific answers.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# Reuse the same encoder instance across calls
_encoding = tiktoken.get_encoding("cl100k_base")


def _tiktoken_len(text: str) -> int:
    """Return the number of tokens in *text* using cl100k_base."""
    return len(_encoding.encode(text, disallowed_special=()))


def _extract_heading(text: str) -> Optional[str]:
    """Return the first Markdown/ALL-CAPS heading found in *text*, or None."""
    for line in text.splitlines():
        line = line.strip()
        # Markdown heading
        if line.startswith("#"):
            return line.lstrip("#").strip()
        # ALL-CAPS short line (common in PDFs)
        if line.isupper() and 3 < len(line) < 80:
            return line
    return None


def chunk_markdown(
    text: str,
    source_url: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[dict]:
    """Split *text* into deep, overlapping chunks.

    Parameters
    ----------
    text : str
        Full document text (Markdown, plain, or PDF-extracted).
    source_url : str
        URL or identifier stored as metadata on every chunk.
    chunk_size : int
        Target tokens per chunk. Default 150 (≈ 1-2 paragraphs).
    chunk_overlap : int
        Overlap tokens between adjacent chunks. Default 30.

    Returns
    -------
    list[dict]
        Each dict has ``text``, ``source_url``, ``chunk_index``,
        and ``context_header`` (nearest section heading, if any).
    """
    if not text or not text.strip():
        return []

    # ── Splitter hierarchy ──────────────────────────────────────────
    # Ordered from coarsest to finest so the splitter respects document
    # structure: section breaks → paragraphs → sentences → words → chars
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_tiktoken_len,
        separators=[
            "\n\n\n",     # section gap (3 blank lines)
            "\n\n",        # paragraph break
            "\n",          # single line break
            ". ",          # sentence end
            "? ",          # question end
            "! ",          # exclamation end
            "; ",          # semi-colon
            ", ",          # comma
            " ",           # word boundary
            "",            # character (last resort)
        ],
    )

    raw_chunks = splitter.split_text(text)

    # ── Enrich each chunk with a context header ─────────────────────
    chunks: List[dict] = []
    current_heading: Optional[str] = None

    for idx, chunk in enumerate(raw_chunks):
        if not chunk.strip():
            continue

        # Update current heading if this chunk starts a new section
        heading = _extract_heading(chunk)
        if heading:
            current_heading = heading

        chunks.append(
            {
                "text": chunk,
                "source_url": source_url,
                "chunk_index": idx,
                "context_header": current_heading or "",
            }
        )

    logger.info(
        "Deep chunked %d chars → %d chunks (size=%d, overlap=%d, source=%s)",
        len(text),
        len(chunks),
        chunk_size,
        chunk_overlap,
        source_url,
    )
    return chunks
