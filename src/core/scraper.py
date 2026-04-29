"""
Web scraping module – fetches a URL's HTML, strips boilerplate,
and converts to clean Markdown using BeautifulSoup + html2text.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from typing import Callable, Dict, List, Optional
from urllib.parse import urljoin, urldefrag, urlparse

import html2text
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Rotate through common user-agent strings to reduce blocking
_USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.4 Safari/605.1.15"
    ),
]

_ua_index = 0


def _next_user_agent() -> str:
    global _ua_index
    ua = _USER_AGENTS[_ua_index % len(_USER_AGENTS)]
    _ua_index += 1
    return ua


def scrape_website(
    url: str,
    *,
    timeout: int = 15,
    delay: float = 0.0,
) -> Optional[str]:
    """Fetch *url*, strip boilerplate, return clean Markdown text.

    Parameters
    ----------
    url : str
        The page to scrape.
    timeout : int
        HTTP request timeout in seconds.
    delay : float
        Seconds to wait before making the request (rate-limiting).

    Returns
    -------
    str | None
        Clean Markdown content, or ``None`` on failure.
    """
    if delay > 0:
        time.sleep(delay)

    headers = {"User-Agent": _next_user_agent()}

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to fetch %s: %s", url, exc)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove non-content elements
    for tag in soup.find_all(
        ["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]
    ):
        tag.decompose()

    # Convert to Markdown
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = True
    converter.ignore_emphasis = False
    converter.body_width = 0  # no line-wrapping
    converter.skip_internal_links = True

    markdown = converter.handle(str(soup))

    # Light cleanup – collapse excessive blank lines
    lines = markdown.splitlines()
    cleaned: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append("")
        else:
            blank_count = 0
            cleaned.append(line)

    result = "\n".join(cleaned).strip()
    if not result:
        logger.warning("Scrape returned empty content for %s", url)
        return None

    logger.info(
        "Scraped %s – %d characters of Markdown",
        urlparse(url).netloc,
        len(result),
    )
    return result


# ======================================================================
# Internal link discovery
# ======================================================================

# File extensions to skip (static assets, documents, etc.)
_SKIP_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".pdf", ".zip", ".rar", ".tar", ".gz", ".7z",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".flv",
    ".css", ".js", ".json", ".xml", ".rss", ".atom",
    ".woff", ".woff2", ".ttf", ".eot",
})


def extract_internal_links(
    html: str,
    base_url: str,
    base_domain: str,
) -> List[str]:
    """Parse *html* and return de-duplicated internal links on *base_domain*.

    Normalises URLs, strips fragments, and filters out static assets.
    """
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()

        # Skip javascript:, mailto:, tel:, etc.
        if re.match(r"^(javascript|mailto|tel|ftp):", href, re.I):
            continue

        # Resolve relative URLs
        full_url = urljoin(base_url, href)
        # Strip fragment
        full_url, _ = urldefrag(full_url)
        # Strip trailing slash for consistency
        full_url = full_url.rstrip("/")

        parsed = urlparse(full_url)

        # Must be http(s) and same domain
        if parsed.scheme not in ("http", "https"):
            continue
        link_domain = parsed.netloc.lower().replace("www.", "")
        if link_domain != base_domain:
            continue

        # Skip static asset extensions
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in _SKIP_EXTENSIONS):
            continue

        links.append(full_url)

    return list(dict.fromkeys(links))  # de-dup, preserve order


# ======================================================================
# Deep (recursive) site scraping
# ======================================================================

def deep_scrape_website(
    start_url: str,
    *,
    max_pages: int = 50,
    delay: float = 0.5,
    timeout: int = 15,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, str]:
    """BFS-crawl *start_url* and all same-domain internal pages.

    Parameters
    ----------
    start_url : str
        The entry-point URL (e.g. ``https://example.com``).
    max_pages : int
        Maximum number of pages to scrape (safety cap).
    delay : float
        Seconds to wait between requests (politeness).
    timeout : int
        HTTP timeout per request.
    on_progress : callable | None
        Optional ``(current_count, queue_size, current_url)`` callback
        for live UI updates.

    Returns
    -------
    dict[str, str]
        Mapping of URL → clean Markdown content for every successfully
        scraped page.
    """
    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc.lower().replace("www.", "")

    visited: set[str] = set()
    queue: deque[str] = deque()
    results: dict[str, str] = {}

    # Normalise start URL
    normalised_start = start_url.rstrip("/")
    queue.append(normalised_start)
    visited.add(normalised_start)

    headers = {"User-Agent": _next_user_agent()}

    while queue and len(results) < max_pages:
        url = queue.popleft()

        if on_progress:
            on_progress(len(results) + 1, len(queue), url)

        if delay > 0:
            time.sleep(delay)

        # Fetch raw HTML
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Deep-scrape: failed to fetch %s: %s", url, exc)
            continue

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            continue

        raw_html = resp.text

        # Discover internal links before stripping nav/footer
        new_links = extract_internal_links(raw_html, url, base_domain)
        for link in new_links:
            normalised = link.rstrip("/")
            if normalised not in visited:
                visited.add(normalised)
                queue.append(normalised)

        # Convert to Markdown (reuse same logic as scrape_website)
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup.find_all(
            ["script", "style", "nav", "footer", "header", "aside",
             "noscript", "iframe"]
        ):
            tag.decompose()

        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True
        converter.ignore_emphasis = False
        converter.body_width = 0
        converter.skip_internal_links = True

        markdown = converter.handle(str(soup))

        # Collapse blank lines
        lines = markdown.splitlines()
        cleaned: list[str] = []
        blank_count = 0
        for line in lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 2:
                    cleaned.append("")
            else:
                blank_count = 0
                cleaned.append(line)

        page_md = "\n".join(cleaned).strip()
        if page_md and len(page_md) > 50:  # skip near-empty pages
            results[url] = page_md

        logger.info(
            "Deep-scrape: [%d/%d] %s (%d chars, %d queued)",
            len(results), max_pages, url, len(page_md), len(queue),
        )

    logger.info(
        "Deep-scrape complete for %s: %d pages scraped, %d URLs visited",
        base_domain, len(results), len(visited),
    )
    return results
