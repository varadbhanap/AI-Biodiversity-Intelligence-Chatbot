from src.knowledge.loader import load_knowledge_base
from src.knowledge.vector_store import KnowledgeVectorStore
from src.reasoning.engine import ReasoningEngine
from src.recommendation.generator import RecommendationGenerator


def _build_generator():
    kb = load_knowledge_base()
    engine = ReasoningEngine(kb)
    store = KnowledgeVectorStore()
    store.build()
    return RecommendationGenerator(kb, engine, store)


def test_recommendation_matches_example_use_case():
    """
    Reproduces the exact example from the brief:
    SOC 0.3%, low rainfall, monoculture wheat, semi-arid ->
    expects agroforestry / intercropping style recommendation with
    quantified impact and a citation, not a generic answer.
    """
    generator = _build_generator()
    observed = {
        "soil_organic_carbon": 0.3,
        "rainfall": "low",
        "land_use_type": "monoculture",
    }
    recs = generator.generate(observed)
    assert len(recs) > 0

    action_names = [r.action.lower() for r in recs]
    assert any("cover crop" in a or "agroforestry" in a or "tillage" in a for a in action_names)

    for r in recs:
        d = r.to_dict()
        assert d["recommendation"]
        assert d["why_it_works"]
        assert d["reference"]
        assert d["time_horizon"] in ("short_term", "medium_term", "long_term")
        assert d["confidence"] in ("high", "medium", "low")
        assert "metric" in d["primary_effect"]
        assert "estimate" in d["primary_effect"]


def test_recommendations_are_not_generic():
    """Guards against the brief's explicit 'not acceptable' example."""
    generator = _build_generator()
    observed = {"soil_organic_carbon": 0.4, "land_use_type": "monoculture"}
    recs = generator.generate(observed)
    for r in recs:
        assert r.action.strip().lower() != "use sustainable practices"
        assert len(r.mechanism) > 20


def test_vector_store_retrieves_relevant_evidence():
    store = KnowledgeVectorStore()
    store.build()
    hits = store.query("agroforestry soil moisture biodiversity", top_k=2)
    assert len(hits) > 0
    assert any("agroforestry" in h["source"] for h in hits)
