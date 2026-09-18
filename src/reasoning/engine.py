"""
Multi-metric reasoning engine.

Given a partial or complete snapshot of a plot's environmental metrics,
this identifies which metrics are limiting (below their healthy threshold
per the knowledge base), then uses the causal graph to work out which
limiting metric has the widest downstream effect - i.e. which single
intervention would improve the most other metrics at once. This is the
"connect soil health <-> biodiversity <-> water availability" requirement,
implemented as actual graph traversal rather than a prompt instruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.knowledge.loader import KnowledgeBase
from src.reasoning.causal_graph import CausalGraph


@dataclass
class LimitingMetric:
    metric: str
    value: Any
    severity: str  # "critical" | "low" | "borderline"
    downstream_effects: list[dict] = field(default_factory=list)


def _classify_severity(metric_def: dict, value: Any) -> str | None:
    # Categorical metrics (land use, pollution level, etc.) declare their
    # own risk mapping directly, since "critical/low" thresholds don't make
    # sense for a string value.
    risk_categories = metric_def.get("risk_categories")
    if risk_categories and isinstance(value, str):
        return risk_categories.get(value.lower())

    thresholds = metric_def.get("thresholds")
    if not thresholds or not isinstance(value, (int, float)):
        return None

    direction = metric_def.get("direction", "lower_is_worse")

    if direction == "range":
        # e.g. soil pH: stress at both extremes, healthy in the middle band.
        low_stress = thresholds.get("acidic_stress")
        opt_low = thresholds.get("optimal_low")
        opt_high = thresholds.get("optimal_high")
        high_stress = thresholds.get("alkaline_stress")
        if low_stress is not None and value <= low_stress:
            return "critical"
        if high_stress is not None and value >= high_stress:
            return "critical"
        if opt_low is not None and value < opt_low:
            return "low"
        if opt_high is not None and value > opt_high:
            return "low"
        return None

    if direction == "higher_is_worse":
        # e.g. deforestation rate: larger values are worse, not smaller.
        if "critical_high" in thresholds and value >= thresholds["critical_high"]:
            return "critical"
        if "concerning" in thresholds and value >= thresholds["concerning"]:
            return "low"
        return None

    # Default: lower_is_worse (e.g. soil organic carbon, soil moisture).
    if "critical_low" in thresholds and value <= thresholds["critical_low"]:
        return "critical"
    if "low" in thresholds and value <= thresholds["low"]:
        return "low"
    if "healthy" in thresholds and value < thresholds["healthy"]:
        return "borderline"
    return None


class ReasoningEngine:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.graph = CausalGraph(kb)

    def identify_limiting_metrics(self, observed: dict[str, Any]) -> list[LimitingMetric]:
        limiting: list[LimitingMetric] = []
        for name, value in observed.items():
            metric_def = self.kb.get_metric(name)
            if not metric_def:
                continue
            severity = _classify_severity(metric_def, value)
            if severity is None:
                continue
            downstream = self.graph.downstream_effects(name, max_hops=3)
            limiting.append(
                LimitingMetric(
                    metric=name,
                    value=value,
                    severity=severity,
                    downstream_effects=downstream,
                )
            )

        severity_rank = {"critical": 0, "low": 1, "borderline": 2}
        limiting.sort(
            key=lambda m: (
                severity_rank.get(m.severity, 3),
                -len(m.downstream_effects),
            )
        )
        return limiting

    def explain_link(self, metric_a: str, metric_b: str) -> dict[str, Any]:
        """
        Explicit multi-metric connection explainer, e.g.
        "soil health <-> biodiversity" or "water availability <-> species survival".
        """
        connected = self.graph.connects(metric_a, metric_b)
        if not connected:
            return {
                "connected": False,
                "explanation": f"No causal path found between {metric_a} and {metric_b} "
                f"within the current knowledge graph.",
            }
        effects = self.graph.downstream_effects(metric_a, max_hops=4)
        path = next((e for e in effects if e["target"] == metric_b), None)
        return {
            "connected": True,
            "path": path["path"] if path else None,
            "path_detail": path["path_detail"] if path else None,
        }

    def multi_metric_summary(self, observed: dict[str, Any]) -> dict[str, Any]:
        """
        Produces the cross-cutting narrative required by the brief: how
        soil health, water availability, and land use interact for this
        specific input, not just isolated single-variable findings.
        """
        limiting = self.identify_limiting_metrics(observed)
        cross_links = []
        metrics_present = list(observed.keys())
        for i, m1 in enumerate(metrics_present):
            for m2 in metrics_present[i + 1 :]:
                if self.graph.connects(m1, m2, max_hops=3) or self.graph.connects(
                    m2, m1, max_hops=3
                ):
                    cross_links.append((m1, m2))
        return {
            "limiting_metrics": limiting,
            "cross_metric_links": cross_links,
            "num_variables_considered": len(metrics_present),
        }
