"""
Batch indexing tool – scan local Markdown / PDF files and index
them into Qdrant.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR, MD_FILES_DIR
from src.core.chunker import chunk_markdown
from src.core.cleaner import deduplicate_chunks, normalize_text
from src.core.embedder import embed_chunks
from src.database.vector_store import VectorStore

logger = logging.getLogger(__name__)


def index_markdown_files(
    directory: Path | None = None,
    vector_store: VectorStore | None = None,
) -> dict:
    """Index all ``.md`` files in *directory* into the vector store.

    Returns
    -------
    dict
        ``files_indexed`` (int), ``total_chunks`` (int).
    """
    directory = directory or MD_FILES_DIR
    vs = vector_store or VectorStore()

    md_files = sorted(directory.glob("*.md"))
    if not md_files:
        logger.warning("No .md files found in %s", directory)
        return {"files_indexed": 0, "total_chunks": 0}

    total_chunks = 0
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8", errors="replace")
        text = normalize_text(text)

        source_name = f"file_{md_file.stem}"
        chunks = chunk_markdown(text, source_url=str(md_file))
        chunks = deduplicate_chunks(chunks)
        chunks = embed_chunks(chunks, show_progress=True)

        stored = vs.store_chunks_for_site(chunks, source_name)
        total_chunks += stored
        logger.info("Indexed %s → %d chunks", md_file.name, stored)

    return {"files_indexed": len(md_files), "total_chunks": total_chunks}


def index_pdf_file(
    pdf_path: Path,
    vector_store: VectorStore | None = None,
) -> dict:
    """Index a single PDF into the vector store.

    Returns
    -------
    dict
        ``site_name``, ``chunks_stored``.
    """
    from pypdf import PdfReader

    vs = vector_store or VectorStore()

    reader = PdfReader(str(pdf_path))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n\n".join(pages_text)
    full_text = normalize_text(full_text)

    site_name = f"pdf_{pdf_path.stem}"
    chunks = chunk_markdown(full_text, source_url=str(pdf_path))
    chunks = deduplicate_chunks(chunks)
    chunks = embed_chunks(chunks, show_progress=True)

    stored = vs.store_chunks_for_site(chunks, site_name)
    logger.info("PDF '%s' → %d chunks indexed", pdf_path.name, stored)

    return {"site_name": site_name, "chunks_stored": stored}


# ======================================================================
# CLI entry-point
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    result = index_markdown_files()
    print(f"\n✅ Indexed {result['files_indexed']} files → {result['total_chunks']} chunks")
