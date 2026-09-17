from src.knowledge.loader import load_knowledge_base
from src.reasoning.causal_graph import CausalGraph
from src.reasoning.engine import ReasoningEngine


def test_load_knowledge_base():
    kb = load_knowledge_base()
    assert "soil_organic_carbon" in kb.metrics
    assert len(kb.interventions) >= 5
    assert len(kb.graph_edges) >= 10


def test_causal_graph_downstream_effects():
    kb = load_knowledge_base()
    graph = CausalGraph(kb)
    effects = graph.downstream_effects("soil_organic_carbon", max_hops=3)
    targets = {e["target"] for e in effects}
    # soil_organic_carbon -> microbial_diversity -> species_richness
    assert "species_richness" in targets


def test_causal_graph_connects_soil_to_biodiversity():
    kb = load_knowledge_base()
    graph = CausalGraph(kb)
    assert graph.connects("soil_organic_carbon", "species_richness", max_hops=4)


def test_reasoning_engine_identifies_limiting_metrics():
    kb = load_knowledge_base()
    engine = ReasoningEngine(kb)
    observed = {
        "soil_organic_carbon": 0.3,
        "rainfall": "low",
        "land_use_type": "monoculture",
    }
    limiting = engine.identify_limiting_metrics(observed)
    limiting_names = {m.metric for m in limiting}
    assert "soil_organic_carbon" in limiting_names
    critical = [m for m in limiting if m.metric == "soil_organic_carbon"][0]
    assert critical.severity == "critical"
    assert len(critical.downstream_effects) > 0


def test_multi_metric_summary_finds_cross_links():
    kb = load_knowledge_base()
    engine = ReasoningEngine(kb)
    observed = {
        "soil_organic_carbon": 0.3,
        "rainfall": "low",
        "land_use_type": "monoculture",
    }
    summary = engine.multi_metric_summary(observed)
    assert summary["num_variables_considered"] == 3
    assert isinstance(summary["cross_metric_links"], list)
