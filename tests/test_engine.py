import json

import pytest
from conftest import FakeLLM

from agent_factory.llm.privacy import PrivacyError, PrivacyMode
from agent_factory.sim.engine import Simulation, resolve_models
from agent_factory.sim.report import write_run
from agent_factory.sim.scenario import Scenario, ScenarioError

LOCAL = PrivacyMode.LOCAL


def odds(*actors, p=0.6):
    return [{"actor": a, "p_success": p, "rationale": "test"} for a in actors]


def verdict(*actors, score=7):
    return {"summary": "Done.", "actors": [
        {"actor": a, "summary": "ok", "scores": [{"objective": "o", "score": score, "evidence": "e"}]}
        for a in actors]}


def tank_llm(blue_loss=-10, p=0.6, proposed=None) -> FakeLLM:
    return FakeLLM(
        Action=lambda c: {"intent": f"{c.agent} pushes", "orders": ["move"], "rationale": "test"},
        AssessmentWithEvent=lambda c: {"odds": odds("blue", "red", p=p), "proposed_event": proposed},
        Resolution=lambda c: {"narrative": "Fighting.", "changes": [
            {"path": "blue.strength", "op": "add", "value": blue_loss, "reason": "losses"},
            {"path": "red.strength", "op": "add", "value": -15, "reason": "losses"},
            {"path": "world.crossroads", "op": "set", "value": "Contested", "reason": "fighting"}],
            "observations": [{"actor": "blue", "text": "Saw tanks."}, {"actor": "nobody", "text": "x"}]},
        Verdict=lambda c: verdict("blue", "red"),
    )


def quiet(scenario: Scenario) -> Scenario:
    """The same scenario with no base-rate events, so steps without actions stay quiet."""
    scenario = scenario.model_copy(deep=True)
    scenario.nature.events = []
    return scenario


def simulate(scenario, llm, seed=7, **kwargs):
    return Simulation(scenario, llm, resolve_models(scenario, env={}), seed=seed, privacy=LOCAL,
                      run_id="test", **kwargs).run()


def test_tank_steps_call_actors_then_the_game_master(tank):
    llm = tank_llm()
    result = simulate(quiet(tank), llm, steps=3)
    assert [n.step for n in result.nodes] == [1, 2, 3]
    assert all(n.model_calls == 4 for n in result.nodes)  # two actors, assess, resolve
    assert [c.agent for c in llm.calls[:4]] == ["actor.blue", "actor.red", "gm.assess", "gm.resolve"]
    final = result.nodes[-1].state
    assert final["blue"]["strength"] == 70 and final["red"]["strength"] == 55
    assert final["red"]["supply"] == 60 - 3 * 8 and final["world"]["crossroads"] == "contested"
    assert result.nodes[0].observations == {"blue": "Saw tanks."}
    assert result.scores == {"blue": 7.0, "red": 7.0}
    assert result.ended_by == "stopped after hour 3 of 6 (step limit)"


def test_actors_see_their_private_information_but_not_the_others(tank):
    llm = tank_llm()
    simulate(quiet(tank), llm, steps=1)
    blue_prompt = llm.by_agent("actor.blue")[0].user
    assert "anti-tank guided missile companies" in blue_prompt
    assert "fuel for about three hours" not in blue_prompt
    assert "red.strength" not in blue_prompt and "blue.strength" in blue_prompt
    gm_prompt = llm.by_agent("gm.assess")[0].user
    assert "anti-tank guided missile companies" in gm_prompt and "fuel for about three hours" in gm_prompt


def test_same_seed_same_run_and_different_seed_different_dice(tank):
    first = simulate(quiet(tank), tank_llm(), seed=1, steps=3)
    again = simulate(quiet(tank), tank_llm(), seed=1, steps=3)
    other = simulate(quiet(tank), tank_llm(), seed=2, steps=3)
    assert [n.id for n in first.nodes] == [n.id for n in again.nodes]
    assert [r.roll for n in first.nodes for r in n.rolls] == [r.roll for n in again.nodes for r in n.rolls]
    assert [r.roll for n in first.nodes for r in n.rolls] != [r.roll for n in other.nodes for r in n.rolls]


def test_end_condition_stops_the_run_and_clamps(tank):
    result = simulate(quiet(tank), tank_llm(blue_loss=-60))
    assert len(result.nodes) == 2
    assert result.ended_by == "end condition met at the end of hour 2 of 6: blue.strength <= 20"
    clamped = [c for c in result.nodes[1].gm_changes if c.path == "blue.strength"][0]
    assert clamped.after == 0 and "clamped" in clamped.note


def test_open_events_are_rolled_not_declared(tank):
    certain = {"name": "Bridge collapses", "description": "The river bridge falls.", "p": 1.0}
    impossible = {**certain, "p": 0.0}
    happened = simulate(quiet(tank), tank_llm(proposed=certain), steps=1).nodes[0]
    assert [(e.name, e.source) for e in happened.events] == [("Bridge collapses", "open")]
    skipped = simulate(quiet(tank), tank_llm(proposed=impossible), steps=1).nodes[0]
    assert skipped.events == [] and skipped.proposed_event.name == "Bridge collapses"


def test_listed_events_proposed_again_are_not_rolled_twice(tank):
    scenario = tank.model_copy(deep=True)
    scenario.nature.events = [e.model_copy(update={"p": 0.0}) for e in tank.nature.events]
    duplicate = {"name": "fog rolls in", "description": "Fog.", "p": 1.0}
    node = simulate(scenario, tank_llm(proposed=duplicate), steps=1).nodes[0]
    assert node.events == [] and node.proposed_event.name == "fog rolls in"


def test_listed_events_apply_their_effects_in_code(tank):
    scenario = quiet(tank)
    scenario.nature.events = [e.model_copy(update={"p": 1.0}) for e in tank.nature.events
                              if e.name == "Fog rolls in"]
    result = simulate(scenario, tank_llm(), steps=2)
    assert [e.name for e in result.nodes[0].events] == ["Fog rolls in"]
    assert result.nodes[0].state["world"]["weather"] == "fog"
    assert result.nodes[1].events == []  # its condition (clear weather) no longer holds


def amp_llm(changes) -> FakeLLM:
    def assess(call):
        return {"odds": odds("me", p=1.0) if "Intent:" in call.user else [], "proposed_event": None}

    return FakeLLM(AssessmentWithEvent=assess,
                   Resolution=lambda c: {"narrative": "Bought it.", "changes": changes},
                   Verdict=lambda c: verdict("me", score=6))


def test_policy_scenario_only_calls_models_when_something_happens(amp):
    llm = amp_llm([{"path": "me.savings", "op": "add", "value": -1500, "reason": "amp"},
                   {"path": "me.amp", "op": "set", "value": "new", "reason": "bought"},
                   {"path": "me.emergency_months", "op": "set", "value": 99, "reason": "x"}])
    result = simulate(quiet(amp), llm, option=0)
    calls = [n.model_calls for n in result.nodes]
    assert calls == [2, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]  # step 1 acts; review every 3 months
    assert [n.quiet for n in result.nodes] == [False] + [True] * 11
    final = result.nodes[-1].state["me"]
    assert final["savings"] == 4000 + 12 * 250 - 1500
    assert final["emergency_months"] == round(final["savings"] / 1100, 1)
    ignored = [c for c in result.nodes[0].gm_changes if c.path == "me.emergency_months"][0]
    assert "computed by the scenario" in ignored.note
    assert result.info.option == "Buy a new amplifier now for 1500 euros"
    assert result.scores == {"me": 6.0}


def test_policy_option_with_steps_acts_when_planned(amp):
    llm = amp_llm([])
    result = simulate(quiet(amp), llm, option=2)
    assert [n.step for n in result.nodes if n.actions] == [1, 7]


def test_unknown_option_is_an_error(amp):
    with pytest.raises(ScenarioError, match="option 5"):
        Simulation(amp, amp_llm([]), resolve_models(amp, env={}), seed=1, privacy=LOCAL,
                   run_id="t", option=5)


def test_simulation_refuses_cloud_models_in_local_mode(tank):
    models = resolve_models(tank, gm_model="openrouter/qwen/qwen3-235b-a22b-2507", env={})
    with pytest.raises(PrivacyError):
        Simulation(tank, tank_llm(), models, seed=1, privacy=LOCAL, run_id="t")


def test_model_resolution_order(tank):
    env = {"AF_MODEL": "ollama/env-actor", "AF_GM_MODEL": "ollama/env-gm"}
    models = resolve_models(tank, env=env)
    assert models.actors["blue"] == "ollama/gemma3:4b"  # the scenario's default beats the environment
    assert models.game_master == "ollama/gpt-oss:20b"
    models = resolve_models(tank, model="ollama/cli", env=env)
    assert set(models.actors.values()) == {"ollama/cli"}


def test_run_is_written_as_trace_and_report(tank, tmp_path):
    result = simulate(quiet(tank), tank_llm(), steps=2)
    trace, report = write_run(result, tank, tmp_path / "run")
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert [r["type"] for r in records] == ["run", "node", "node", "end"]
    assert records[1]["parent_id"] == records[0]["root_id"]
    text = report.read_text(encoding="utf-8")
    assert "# Armoured pincer at the Hill 112 crossroads" in text
    assert "| blue | o | 7 | e |" in text and "### hour 2 of 6" in text
    assert "> Privately, blue: Saw tanks." in text
