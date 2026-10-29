from agent_factory.sim.scenario import VarSpec
from agent_factory.sim.state import WorldState


def make_state() -> WorldState:
    return WorldState({
        "blue": {"strength": VarSpec(value=100, min=0, max=100), "ready": VarSpec(value=False)},
        "red": {"strength": VarSpec(value=100, min=0, max=100)},
        "world": {"weather": VarSpec(value="clear", choices=["clear", "fog"]), "note": VarSpec(value="")},
    })


def test_numbers_are_clamped_with_a_note():
    state = make_state()
    change = state.increment("blue.strength", -130, "battle")
    assert state.get("blue.strength") == 0
    assert change.before == 100 and change.after == 0
    assert "clamped" in change.note


def test_numbers_accept_numeric_text_and_stay_tidy():
    state = make_state()
    state.assign("blue.strength", "72.0", "gm")
    assert state.get("blue.strength") == 72 and isinstance(state.get("blue.strength"), int)
    state.increment("blue.strength", 0.3333, "gm")
    assert state.get("blue.strength") == 72.33


def test_choices_match_case_insensitively_and_reject_others():
    state = make_state()
    assert state.assign("world.weather", "FOG", "event").after == "fog"
    change = state.assign("world.weather", "sandstorm", "gm")
    assert state.get("world.weather") == "fog"
    assert "not one of" in change.note and not change.changed


def test_unknown_variables_and_bad_types_are_ignored():
    state = make_state()
    assert "unknown variable" in state.assign("blue.position", "hill", "gm").note
    assert "only numbers" in state.increment("world.weather", 1, "gm").note
    assert "not a number" in state.assign("blue.strength", "lots", "gm").note
    assert state.get("blue.strength") == 100


def test_booleans_parse_words():
    state = make_state()
    assert state.assign("blue.ready", "yes", "gm").after is True


def test_copies_are_independent():
    state = make_state()
    snapshot = state.copy()
    state.increment("blue.strength", -10, "battle")
    assert snapshot.get("blue.strength") == 100


def test_actors_do_not_see_each_others_variables():
    state = make_state()
    visible = state.visible_to("blue", ["blue", "red"], "own")
    assert set(visible) == {"blue", "world"}
    assert set(state.visible_to("blue", ["blue", "red"], "full")) == {"blue", "red", "world"}
