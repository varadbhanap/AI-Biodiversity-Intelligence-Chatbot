"""
FastAPI backend for the live demo. Thin HTTP wrapper around the same
BiodiversityAssistant orchestrator the MCP server uses, so the demo and
the MCP tools are guaranteed to behave identically.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.orchestrator import BiodiversityAssistant

app = FastAPI(
    title="Darukaa.Earth Biodiversity Intelligence API",
    description="AI environmental scientist: knowledge-grounded, multi-metric "
    "biodiversity recommendations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = BiodiversityAssistant()


class ChatRequest(BaseModel):
    session_id: str
    message: Optional[str] = None
    structured_input: Optional[dict[str, Any]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def list_metrics() -> dict:
    """List every metric the system understands, with thresholds and citations."""
    return assistant.kb.metrics


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    """
    Main conversational endpoint. Accepts free text, structured JSON, and/or
    geo-coordinates in the same call. Supports multi-turn use via session_id.
    """
    return assistant.handle_message(
        session_id=req.session_id,
        text=req.message,
        structured_input=req.structured_input,
        latitude=req.latitude,
        longitude=req.longitude,
    )


@app.get("/session/{session_id}")
def get_session(session_id: str) -> dict:
    session = assistant.get_session(session_id)
    return {
        "known": session.known,
        "missing": session.missing_slots(),
        "history": [{"role": t.role, "content": t.content} for t in session.history],
    }


# Serve the simple chat frontend at /
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
