"""
Orchestrator: the single entry point both the MCP server and the FastAPI
demo call into. Keeping this logic in one place (rather than duplicating it
between the two interfaces) is what makes the MCP server and the web demo
provably run the same reasoning, not two different implementations.
"""

from __future__ import annotations

from typing import Any, Optional

from src.conversation.session import ConversationSession, SessionStore
from src.knowledge.geo_data import enrich_from_coordinates
from src.knowledge.loader import KnowledgeBase, load_knowledge_base
from src.knowledge.vector_store import KnowledgeVectorStore
from src.reasoning.engine import ReasoningEngine
from src.recommendation.generator import RecommendationGenerator


class BiodiversityAssistant:
    def __init__(self):
        self.kb: KnowledgeBase = load_knowledge_base()
        self.reasoning_engine = ReasoningEngine(self.kb)
        self.vector_store = KnowledgeVectorStore()
        self.vector_store.build()
        self.generator = RecommendationGenerator(
            self.kb, self.reasoning_engine, self.vector_store
        )
        self.sessions = SessionStore()

    def handle_message(
        self,
        session_id: str,
        text: Optional[str] = None,
        structured_input: Optional[dict[str, Any]] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> dict[str, Any]:
        session = self.sessions.get_or_create(session_id)

        if text:
            session.record("user", text)
            session.update_from_text(text)

        if structured_input:
            session.update_from_json(structured_input)

        geo_note = None
        if latitude is not None and longitude is not None:
            enrichment = enrich_from_coordinates(latitude, longitude)
            session.update_from_json({"latitude": latitude, "longitude": longitude})
            session.apply_geo_enrichment(enrichment)
            geo_note = enrichment.source_notes

        if not session.ready_to_reason():
            question = session.next_clarifying_question()
            session.record("system", question or "Could you share more detail about the land?")
            return {
                "type": "clarifying_question",
                "question": question,
                "known_so_far": dict(session.known),
                "missing": session.missing_slots(),
                "geo_note": geo_note,
            }

        summary = self.reasoning_engine.multi_metric_summary(session.known)
        recommendations = self.generator.generate(session.known)

        response = {
            "type": "assessment",
            "input_summary": dict(session.known),
            "limiting_metrics": [
                {
                    "metric": lm.metric,
                    "value": lm.value,
                    "severity": lm.severity,
                    "num_downstream_effects": len(lm.downstream_effects),
                }
                for lm in summary["limiting_metrics"]
            ],
            "cross_metric_links": summary["cross_metric_links"],
            "num_variables_considered": summary["num_variables_considered"],
            "recommendations": [r.to_dict() for r in recommendations],
            "still_missing": session.missing_slots(),
            "geo_note": geo_note,
        }
        session.record("system", f"Generated {len(recommendations)} recommendation(s).")
        return response

    def get_session(self, session_id: str) -> ConversationSession:
        return self.sessions.get_or_create(session_id)
