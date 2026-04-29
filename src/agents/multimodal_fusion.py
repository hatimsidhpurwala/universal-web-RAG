"""
Multimodal Fusion – unified understanding across text, voice, image,
and document inputs. Instead of processing each modality separately,
this module fuses context from all modalities into a single enhanced
query with rich context.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from groq import Groq

from config.settings import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

_FUSION_PROMPT = """\
You are a multimodal context fusion agent. You receive information
extracted from multiple input modalities (text, image OCR, PDF content,
voice transcription) and must create a UNIFIED understanding.

Return JSON:
{
  "unified_query": "The user's actual intent combining all modalities",
  "enhanced_context": "Rich context paragraph combining all inputs",
  "detected_product": "If a product is identified across modalities",
  "key_specs": ["spec1", "spec2"],
  "user_intent": "What the user is trying to accomplish",
  "implicit_context": "Context clues inferred from the combination"
}

Example:
  Image OCR: "SALTO MIFARE DESFire EV2"
  PDF content: "Operating frequency: 13.56MHz, Memory: 4KB"
  Voice: "Is this compatible with our system?"
  
  → unified_query: "Is the SALTO MIFARE DESFire EV2 keycard (13.56MHz, 4KB) compatible with existing access control systems?"
  → implicit_context: "User has physical access to the card, has specs, is evaluating compatibility with their current infrastructure"
"""


class MultimodalFusion:
    """Fuse information from multiple input modalities into a unified
    understanding."""

    def fuse_inputs(
        self,
        text_input: str = "",
        image_ocr: str = "",
        pdf_text: str = "",
        voice_transcription: str = "",
        scraped_content: str = "",
    ) -> Dict[str, Any]:
        """Combine all input modalities into a unified context.

        Returns
        -------
        dict
            ``unified_query``, ``enhanced_context``, ``detected_product``,
            ``key_specs``, ``user_intent``, ``implicit_context``.
        """
        # Collect available modalities
        modalities: Dict[str, str] = {}
        if text_input.strip():
            modalities["text_input"] = text_input.strip()
        if image_ocr.strip():
            modalities["image_ocr"] = image_ocr.strip()[:500]
        if pdf_text.strip():
            modalities["pdf_content"] = pdf_text.strip()[:1000]
        if voice_transcription.strip():
            modalities["voice_transcription"] = voice_transcription.strip()
        if scraped_content.strip():
            modalities["web_content"] = scraped_content.strip()[:500]

        # If only one modality, no fusion needed
        if len(modalities) <= 1:
            single_text = next(iter(modalities.values()), text_input)
            return {
                "unified_query": single_text,
                "enhanced_context": "",
                "detected_product": "",
                "key_specs": [],
                "user_intent": "standard_query",
                "implicit_context": "",
                "modalities_used": list(modalities.keys()),
            }

        # Multiple modalities – use LLM fusion
        return self._llm_fuse(modalities)

    def _llm_fuse(self, modalities: Dict[str, str]) -> Dict[str, Any]:
        client = Groq(api_key=GROQ_API_KEY)

        # Build modality description
        parts = []
        for modality, content in modalities.items():
            parts.append(f"[{modality}]\n{content}\n")
        modality_block = "\n".join(parts)

        messages = [
            {"role": "system", "content": _FUSION_PROMPT},
            {
                "role": "user",
                "content": f"Inputs from multiple modalities:\n\n{modality_block}",
            },
        ]

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            data["modalities_used"] = list(modalities.keys())
            logger.info(
                "Multimodal fusion: %d modalities → %s",
                len(modalities),
                data.get("user_intent", "unknown"),
            )
            return data
        except Exception as exc:
            logger.error("Multimodal fusion failed: %s", exc)
            combined = " ".join(modalities.values())
            return {
                "unified_query": combined,
                "enhanced_context": "",
                "detected_product": "",
                "key_specs": [],
                "user_intent": "standard_query",
                "implicit_context": "",
                "modalities_used": list(modalities.keys()),
            }
