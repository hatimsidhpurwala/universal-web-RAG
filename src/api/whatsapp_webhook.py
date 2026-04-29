"""
WhatsApp webhook – FastAPI application that receives messages from
Twilio, routes them through the RAG agent, and returns TwiML responses.
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, Form, Response
from fastapi.responses import JSONResponse

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.agent_graph import RAGAgent
from src.utils.assets import APP_DESCRIPTION, APP_NAME, APP_VERSION

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ======================================================================
# Application setup
# ======================================================================

app = FastAPI(title=APP_NAME, version=APP_VERSION, description=APP_DESCRIPTION)

# Lazy-initialised agent
_agent: RAGAgent | None = None


def _get_agent() -> RAGAgent:
    global _agent
    if _agent is None:
        _agent = RAGAgent()
    return _agent


# In-memory session store: phone → list of message dicts
_user_sessions: Dict[str, List[dict]] = defaultdict(list)
_MAX_HISTORY = 10  # keep last 10 messages (5 exchanges)


# ======================================================================
# Endpoints
# ======================================================================

@app.get("/", tags=["info"])
async def root():
    """API information."""
    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "description": APP_DESCRIPTION,
        "endpoints": {
            "/webhook": "POST – Twilio WhatsApp webhook",
            "/health": "GET – Health check",
        },
    }


@app.get("/health", tags=["info"])
async def health():
    """Simple health check."""
    return {"status": "ok"}


@app.post("/webhook", tags=["whatsapp"])
async def whatsapp_webhook(
    From: str = Form(""),
    Body: str = Form(""),
):
    """Receive a WhatsApp message from Twilio, process it, and return a
    TwiML response.

    Twilio sends form-encoded data with (among others):
    - ``From`` – the sender's WhatsApp number
    - ``Body`` – the message text
    """
    user_number = From.replace("whatsapp:", "").strip()
    message = Body.strip()

    logger.info("WhatsApp message from %s: %s", user_number, message[:80])

    if not message:
        return _twiml_response("I didn't receive a message. Please try again.")

    # Load conversation history
    history = _user_sessions[user_number]

    try:
        agent = _get_agent()
        result = agent.ask(message, conversation_history=history)
        answer = result.get("final_answer", "Sorry, I couldn't process that.")

        # Append sources if available
        sources = result.get("sources", [])
        if sources:
            answer += "\n\n📚 Sources:\n" + "\n".join(f"• {s}" for s in sources[:3])

        if result.get("web_search_performed"):
            answer = "🌐 (web research performed)\n\n" + answer

    except Exception as exc:
        logger.error("Error processing WhatsApp message: %s", exc, exc_info=True)
        answer = "I ran into an error processing your message. Please try again."

    # Update history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    _user_sessions[user_number] = history[-_MAX_HISTORY:]

    return _twiml_response(answer)


# ======================================================================
# Helpers
# ======================================================================

def _twiml_response(body: str) -> Response:
    """Return a Twilio-compatible TwiML XML response."""
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Message>{_escape_xml(body)}</Message>"
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")


def _escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
