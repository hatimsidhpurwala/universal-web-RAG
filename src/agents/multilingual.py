"""
Multi-Language Support – detects the user's language, translates to
English for processing, and translates the answer back.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from groq import Groq

from config.settings import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

# Languages we can detect / handle
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ar": "Arabic",
    "hi": "Hindi",
    "ur": "Urdu",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "tr": "Turkish",
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "tl": "Tagalog",
}


def detect_language(text: str) -> str:
    """Detect the language of *text*. Returns ISO 639-1 code.

    Uses a fast heuristic first (ASCII-only → English), then falls
    back to the LLM for non-ASCII scripts.
    """
    # Fast check: if all ASCII printable, assume English
    if all(ord(c) < 128 for c in text.replace("\n", "").replace("\r", "")):
        return "en"

    client = Groq(api_key=GROQ_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        'Detect the language of the user\'s text. '
                        'Return JSON: {"language_code": "en", "language_name": "English"} '
                        f'Supported codes: {", ".join(SUPPORTED_LANGUAGES.keys())}'
                    ),
                },
                {"role": "user", "content": text[:200]},
            ],
            temperature=0.0,
            max_tokens=50,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        code = data.get("language_code", "en")
        logger.info("Detected language: %s (%s)", code, data.get("language_name", ""))
        return code
    except Exception as exc:
        logger.error("Language detection failed: %s", exc)
        return "en"


def translate(
    text: str,
    target_language: str,
    source_language: Optional[str] = None,
) -> str:
    """Translate *text* to *target_language*.

    Parameters
    ----------
    text : str
        The text to translate.
    target_language : str
        ISO 639-1 target language code.
    source_language : str | None
        Source language code (auto-detected if None).
    """
    if target_language == "en" and source_language == "en":
        return text
    if target_language == (source_language or ""):
        return text

    target_name = SUPPORTED_LANGUAGES.get(target_language, target_language)
    source_name = (
        SUPPORTED_LANGUAGES.get(source_language, source_language)
        if source_language
        else "auto-detected"
    )

    client = Groq(api_key=GROQ_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Translate the following text from {source_name} to "
                        f"{target_name}. Return ONLY the translation, no "
                        "explanations or commentary."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        translated = resp.choices[0].message.content.strip()
        logger.info("Translated %s → %s (%d chars)", source_name, target_name, len(translated))
        return translated
    except Exception as exc:
        logger.error("Translation failed: %s", exc)
        return text


class MultilingualHandler:
    """Wrapper that transparently handles non-English queries."""

    def process_query(self, query: str) -> dict:
        """Detect language, translate to English if needed.

        Returns
        -------
        dict
            ``original_query``, ``english_query``, ``source_language``,
            ``needs_translation``.
        """
        lang = detect_language(query)
        needs_translation = lang != "en"

        if needs_translation:
            english_query = translate(query, target_language="en", source_language=lang)
        else:
            english_query = query

        return {
            "original_query": query,
            "english_query": english_query,
            "source_language": lang,
            "language_name": SUPPORTED_LANGUAGES.get(lang, lang),
            "needs_translation": needs_translation,
        }

    def translate_response(
        self,
        answer: str,
        target_language: str,
    ) -> str:
        """Translate *answer* back to the user's language."""
        if target_language == "en":
            return answer
        return translate(answer, target_language=target_language, source_language="en")
