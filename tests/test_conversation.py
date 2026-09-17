from src.conversation.session import ConversationSession, SessionStore


def test_slot_filling_from_text():
    session = ConversationSession(session_id="test-1")
    session.update_from_text("SOC is 0.3, pH 5.2, moderate rainfall, monoculture wheat field")
    assert session.known.get("soil_organic_carbon") == 0.3
    assert session.known.get("soil_ph") == 5.2
    assert session.known.get("rainfall") == "moderate"
    assert session.known.get("land_use_type") == "monoculture"


def test_missing_slots_and_clarifying_question():
    session = ConversationSession(session_id="test-2")
    session.update_from_text("rainfall is low")
    missing = session.missing_slots()
    assert "soil_organic_carbon" in missing
    question = session.next_clarifying_question()
    assert question is not None


def test_shared_option_words_dont_cross_contaminate_slots():
    """
    Regression test: 'Rainfall: low' must not also set pollution_load=low
    just because 'low' is a shared option word across categorical slots.
    Context-scoped matching should keep each value tied to its own slot.
    """
    session = ConversationSession(session_id="test-cross")
    session.update_from_text(
        "Soil organic carbon: 0.3%, Rainfall: low, Crop: monoculture wheat, Region: semi-arid"
    )
    assert session.known.get("rainfall") == "low"
    assert "pollution_load" not in session.known
    assert session.known.get("land_use_type") == "monoculture"
    assert session.known.get("soil_organic_carbon") == 0.3


def test_pollution_context_still_matches_when_present():
    session = ConversationSession(session_id="test-pollution")
    session.update_from_text("There is high pollution and heavy agrochemical runoff here")
    assert session.known.get("pollution_load") == "high"


def test_structured_json_input():
    session = ConversationSession(session_id="test-3")
    updated = session.update_from_json(
        {"soil_organic_carbon": 0.3, "rainfall": "low", "land_use_type": "monoculture"}
    )
    assert set(updated) == {"soil_organic_carbon", "rainfall", "land_use_type"}
    assert session.ready_to_reason()


def test_session_store_persists_across_calls():
    store = SessionStore()
    s1 = store.get_or_create("abc")
    s1.known["soil_ph"] = 6.5
    s2 = store.get_or_create("abc")
    assert s2.known["soil_ph"] == 6.5
