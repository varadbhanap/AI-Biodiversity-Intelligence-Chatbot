"""
Recommendation generator.

Combines three things into each recommendation, matching the brief's
mandatory output format exactly:
  - what to do            -> intervention name
  - why it works           -> mechanism + causal graph path
  - which metric improves  -> primary_effect + secondary_effects
  - reference               -> citation from the interventions catalog
  - impacted metrics, time horizon, confidence -> output_quality section

It also pulls supporting evidence from the RAG vector store so the
explanation is grounded in retrieved text, not just the interventions
JSON, satisfying "clearly show how knowledge is retrieved and used."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.knowledge.loader import KnowledgeBase
from src.knowledge.vector_store import KnowledgeVectorStore
from src.reasoning.engine import LimitingMetric, ReasoningEngine


@dataclass
class Recommendation:
    action: str
    mechanism: str
    impacted_metrics: list[str]
    primary_effect: dict[str, Any]
    secondary_effects: list[dict[str, Any]]
    time_horizon: str
    confidence: str
    reference: str
    causal_explanation: list[dict[str, Any]] = field(default_factory=list)
    supporting_evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation": self.action,
            "why_it_works": self.mechanism,
            "impacted_metrics": self.impacted_metrics,
            "primary_effect": self.primary_effect,
            "secondary_effects": self.secondary_effects,
            "time_horizon": self.time_horizon,
            "confidence": self.confidence,
            "reference": self.reference,
            "causal_explanation": self.causal_explanation,
            "supporting_evidence": self.supporting_evidence,
        }


def _intervention_applies(intervention: dict, observed: dict[str, Any]) -> bool:
    condition = intervention.get("applicable_when", {})
    for metric, rule in condition.items():
        if metric not in observed:
            return False
        value = observed[metric]
        if isinstance(rule, dict):
            if "max" in rule and not (isinstance(value, (int, float)) and value <= rule["max"]):
                return False
            if "min" in rule and not (isinstance(value, (int, float)) and value >= rule["min"]):
                return False
        elif isinstance(rule, list):
            if value not in rule:
                return False
    return True


class RecommendationGenerator:
    def __init__(
        self,
        kb: KnowledgeBase,
        reasoning_engine: ReasoningEngine,
        vector_store: Optional[KnowledgeVectorStore] = None,
    ):
        self.kb = kb
        self.engine = reasoning_engine
        self.vector_store = vector_store

    def _matching_interventions(self, observed: dict[str, Any]) -> list[dict]:
        return [iv for iv in self.kb.interventions if _intervention_applies(iv, observed)]

    def _causal_explanation_for(self, intervention: dict) -> list[dict]:
        primary_metric = intervention["primary_effect"]["metric"]
        drivers = self.engine.graph.upstream_drivers(primary_metric)
        downstream = self.engine.graph.downstream_effects(primary_metric, max_hops=2)
        return {
            "primary_metric": primary_metric,
            "upstream_drivers": drivers,
            "downstream_effects": downstream[:4],
        }

    def _retrieve_evidence(self, query: str, top_k: int = 2) -> list[dict]:
        if not self.vector_store:
            return []
        try:
            return self.vector_store.query(query, top_k=top_k)
        except Exception:  # noqa: BLE001 - evidence retrieval is best-effort
            return []

    def generate(
        self, observed: dict[str, Any], max_recommendations: int = 3
    ) -> list[Recommendation]:
        matches = self._matching_interventions(observed)

        # Rank by how many metrics the observed limiting set overlaps with
        # this intervention's effect set - i.e. prioritize interventions
        # that address the *most limiting* metrics for this specific input.
        limiting = self.engine.identify_limiting_metrics(observed)
        limiting_names = {m.metric for m in limiting}

        def score(intervention: dict) -> int:
            affected = {intervention["primary_effect"]["metric"]} | {
                e["metric"] for e in intervention.get("secondary_effects", [])
            }
            return len(affected & limiting_names)

        matches.sort(key=score, reverse=True)

        recommendations = []
        for intervention in matches[:max_recommendations]:
            causal = self._causal_explanation_for(intervention)
            impacted = [intervention["primary_effect"]["metric"]] + [
                e["metric"] for e in intervention.get("secondary_effects", [])
            ]
            evidence = self._retrieve_evidence(intervention["name"])
            recommendations.append(
                Recommendation(
                    action=intervention["name"],
                    mechanism=intervention["mechanism"],
                    impacted_metrics=impacted,
                    primary_effect=intervention["primary_effect"],
                    secondary_effects=intervention.get("secondary_effects", []),
                    time_horizon=intervention["time_horizon"],
                    confidence=intervention["confidence"],
                    reference=intervention["reference"],
                    causal_explanation=[causal],
                    supporting_evidence=evidence,
                )
            )

        return recommendations
