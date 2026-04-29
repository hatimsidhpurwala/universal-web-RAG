"""
Document chunking using RecursiveCharacterTextSplitter with tiktoken
token counting.
"""

from __future__ import annotations

import logging
from typing import List

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# Reuse the same encoder instance across calls
_encoding = tiktoken.get_encoding("cl100k_base")


def _tiktoken_len(text: str) -> int:
    """Return the number of tokens in *text* using cl100k_base."""
    return len(_encoding.encode(text, disallowed_special=()))


def chunk_markdown(
    text: str,
    source_url: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[dict]:
    """Split *text* into overlapping chunks of roughly *chunk_size* tokens.

    Parameters
    ----------
    text : str
        The full document text (Markdown or plain).
    source_url : str
        URL or identifier for the source (stored as metadata).
    chunk_size : int
        Target number of tokens per chunk.
    chunk_overlap : int
        Number of overlapping tokens between adjacent chunks.

    Returns
    -------
    list[dict]
        Each dict has keys ``text``, ``source_url``, and ``chunk_index``.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_tiktoken_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    chunks = [
        {
            "text": chunk,
            "source_url": source_url,
            "chunk_index": idx,
        }
        for idx, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]

    logger.info(
        "Chunked %d characters into %d chunks (source=%s)",
        len(text),
        len(chunks),
        source_url,
    )
    return chunks
