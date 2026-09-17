"""
MCP server for the Darukaa.Earth biodiversity intelligence system.

Follows the same architecture pattern as EIOS (Enterprise Intelligence
Operating System): an MCP server exposing a small set of composable tools
over a shared reasoning core, so any MCP-compatible client (Claude
Desktop, a custom agent, or the FastAPI demo in this repo) can drive the
same system.

Run standalone with:
    python -m src.mcp_server.server

Or point an MCP client's config at this module.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from src.orchestrator import BiodiversityAssistant

mcp = FastMCP("darukaa-biodiversity-intelligence")

# Single shared assistant instance for the lifetime of the server process.
# Loads the knowledge base, builds the causal graph, and indexes the vector
# store once at startup rather than per-call.
_assistant: Optional[BiodiversityAssistant] = None


def _get_assistant() -> BiodiversityAssistant:
    global _assistant
    if _assistant is None:
        _assistant = BiodiversityAssistant()
    return _assistant


@mcp.tool()
def query_knowledge_base(metric_name: str) -> str:
    """
    Look up a single environmental metric's definition, healthy thresholds,
    what it affects downstream, and its scientific reference.
    """
    kb = _get_assistant().kb
    metric = kb.get_metric(metric_name)
    if not metric:
        available = ", ".join(kb.metric_names())
        return json.dumps({"error": f"Unknown metric '{metric_name}'. Available: {available}"})
    return json.dumps(metric, indent=2)


@mcp.tool()
def retrieve_evidence(query: str, top_k: int = 3) -> str:
    """
    Retrieve the most relevant passages from the indexed scientific
    knowledge corpus (RAG) for a free-text query, e.g. 'agroforestry soil
    moisture' or 'pollution impact on pollinators'.
    """
    hits = _get_assistant().vector_store.query(query, top_k=top_k)
    return json.dumps(hits, indent=2)


@mcp.tool()
def reason_multi_metric(observed_metrics_json: str) -> str:
    """
    Run multi-metric causal reasoning over a JSON object of observed
    environmental metrics, e.g.
    '{"soil_organic_carbon": 0.3, "rainfall": "low", "land_use_type": "monoculture"}'.
    Returns which metrics are limiting, their severity, and how they causally
    connect to other metrics (e.g. soil health -> biodiversity).
    """
    assistant = _get_assistant()
    try:
        observed = json.loads(observed_metrics_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid JSON: {exc}"})

    summary = assistant.reasoning_engine.multi_metric_summary(observed)
    result = {
        "limiting_metrics": [
            {
                "metric": lm.metric,
                "value": lm.value,
                "severity": lm.severity,
                "downstream_effects": lm.downstream_effects[:5],
            }
            for lm in summary["limiting_metrics"]
        ],
        "cross_metric_links": summary["cross_metric_links"],
        "num_variables_considered": summary["num_variables_considered"],
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def generate_recommendation(observed_metrics_json: str, max_recommendations: int = 3) -> str:
    """
    Generate evidence-backed, structured recommendations for a plot of land
    given its observed environmental metrics (JSON object). Each
    recommendation includes what to do, why it works, which metrics improve,
    a quantified estimate, time horizon, confidence, and a citation.
    """
    assistant = _get_assistant()
    try:
        observed = json.loads(observed_metrics_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid JSON: {exc}"})

    recommendations = assistant.generator.generate(observed, max_recommendations)
    return json.dumps([r.to_dict() for r in recommendations], indent=2)


@mcp.tool()
def chat(session_id: str, message: str) -> str:
    """
    Conversational entry point with memory. Pass a stable session_id across
    turns; the system tracks what it already knows about this session's
    land and will ask a clarifying question if critical metrics are still
    missing, or return a full assessment once enough is known.
    """
    assistant = _get_assistant()
    result = assistant.handle_message(session_id=session_id, text=message)
    return json.dumps(result, indent=2)


@mcp.tool()
def chat_with_coordinates(
    session_id: str, message: str, latitude: float, longitude: float
) -> str:
    """
    Same as chat(), but also enriches the session with live geo-coordinate
    data (soil organic carbon and pH from ISRIC SoilGrids, observed species
    counts from GBIF) for the given latitude/longitude. Falls back
    gracefully to manual input if those services are unreachable.
    """
    assistant = _get_assistant()
    result = assistant.handle_message(
        session_id=session_id, text=message, latitude=latitude, longitude=longitude
    )
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    mcp.run()
