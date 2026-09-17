"""
Causal graph over environmental metrics, backed by networkx.

Every edge is hand-curated from a cited source (see data/causal_graph.json)
rather than learned, so any reasoning trace produced by this module can be
walked back to a specific paper or report. This is what lets the
recommendation layer say *why* fixing soil organic carbon also improves
pollinator diversity three hops downstream, instead of just listing
unconnected facts.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from src.knowledge.loader import KnowledgeBase


class CausalGraph:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.graph = nx.DiGraph()
        for edge in kb.graph_edges:
            self.graph.add_edge(
                edge["from"],
                edge["to"],
                relation=edge["relation"],
                strength=edge.get("strength", "moderate"),
                reference=edge.get("reference"),
            )

    def downstream_effects(self, metric: str, max_hops: int = 3) -> list[dict[str, Any]]:
        """
        Return every metric reachable from `metric` within max_hops,
        along with the causal path taken to reach it. Used to explain why
        improving one metric has knock-on effects elsewhere in the system.
        """
        if metric not in self.graph:
            return []

        effects = []
        for target in self.graph.nodes:
            if target == metric:
                continue
            try:
                path = nx.shortest_path(self.graph, source=metric, target=target)
            except nx.NetworkXNoPath:
                continue
            if len(path) - 1 > max_hops:
                continue
            edges_on_path = []
            for a, b in zip(path[:-1], path[1:]):
                data = self.graph.get_edge_data(a, b)
                edges_on_path.append(
                    {
                        "from": a,
                        "to": b,
                        "relation": data["relation"],
                        "strength": data["strength"],
                        "reference": data["reference"],
                    }
                )
            effects.append(
                {
                    "target": target,
                    "hops": len(path) - 1,
                    "path": path,
                    "path_detail": edges_on_path,
                }
            )
        # Shortest / strongest causal chains first.
        effects.sort(key=lambda e: e["hops"])
        return effects

    def upstream_drivers(self, metric: str) -> list[dict[str, Any]]:
        """Return the direct predecessors of a metric - what drives it."""
        if metric not in self.graph:
            return []
        drivers = []
        for pred in self.graph.predecessors(metric):
            data = self.graph.get_edge_data(pred, metric)
            drivers.append(
                {
                    "driver": pred,
                    "relation": data["relation"],
                    "strength": data["strength"],
                    "reference": data["reference"],
                }
            )
        return drivers

    def connects(self, metric_a: str, metric_b: str, max_hops: int = 3) -> bool:
        if metric_a not in self.graph or metric_b not in self.graph:
            return False
        try:
            length = nx.shortest_path_length(self.graph, metric_a, metric_b)
            return length <= max_hops
        except nx.NetworkXNoPath:
            return False
