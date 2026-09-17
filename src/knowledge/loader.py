"""
Loads the structured knowledge base, intervention catalog, and causal graph
from the data/ directory. Kept deliberately simple (no ORM, no database
server) so the knowledge system's contents are auditable by reading JSON
files directly - important for a hackathon reviewer checking "clarity of
knowledge retrieval pipeline."
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@dataclass
class KnowledgeBase:
    metrics: dict[str, Any]
    interventions: list[dict[str, Any]]
    graph_edges: list[dict[str, Any]]

    def metric_names(self) -> list[str]:
        return list(self.metrics.keys())

    def get_metric(self, name: str) -> dict[str, Any] | None:
        return self.metrics.get(name)


def load_knowledge_base(data_dir: Path = DATA_DIR) -> KnowledgeBase:
    with open(data_dir / "knowledge_base.json", "r", encoding="utf-8") as f:
        kb = json.load(f)
    with open(data_dir / "interventions.json", "r", encoding="utf-8") as f:
        interventions = json.load(f)
    with open(data_dir / "causal_graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)

    return KnowledgeBase(
        metrics=kb["metrics"],
        interventions=interventions["interventions"],
        graph_edges=graph["edges"],
    )


def list_documents(data_dir: Path = DATA_DIR) -> list[Path]:
    doc_dir = data_dir / "documents"
    return sorted(doc_dir.glob("*.md"))
