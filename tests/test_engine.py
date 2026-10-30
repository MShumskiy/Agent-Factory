import json
import random

import pytest
from conftest import FakeLLM

from janus.llm.privacy import PrivacyError, PrivacyMode
from janus.sim.engine import Simulation, resolve_models
from janus.sim.report import write_run
from janus.sim.scenario import Scenario, ScenarioError
from janus.sim.world import Dice

LOCAL = PrivacyMode.LOCAL
ORDERS = ["advance", "fire"]


def odds(*actors, p=0.6, orders=2):
    return [{"actor": a, "order": i, "p_success": p, "rationale": "test"}
            for a in actors for i in range(1, orders + 1)]


def verdict(*actors, score=7):
    return {"summary": "Done.", "actors": [
        {"actor": a, "summary": "ok", "scores": [{"objective": "o", "score": score, "evidence": "e"}]}
        for a in actors]}


def tank_llm(blue_loss=-10, p=0.6, proposed=None, discoveries=(), odds_list=None) -> FakeLLM:
    return FakeLLM(
        Action=lambda c: {"intent": f"{c.agent} pushes", "orders": ORDERS, "rationale": "test"},
        Assessment=lambda c: {"odds": odds_list if odds_list is not None else odds("blue", "red", p=p),
                              "proposed_event": proposed, "discoveries": list(discoveries)},
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


def simulation(scenario, llm, **kwargs) -> Simulation:
    return Simulation(scenario, llm, resolve_models(scenario, env={}), privacy=LOCAL, run_id="test", **kwargs)


def simulate(scenario, llm, seed=7, option=0, **kwargs):
    return simulation(scenario, llm, option=option).run(seed=seed, **kwargs)


def test_tank_steps_call_actors_then_the_game_master(tank):
    llm = tank_llm()
    result = simulate(quiet(tank), llm, steps=3)
    assert [n.step for n in result.nodes] == [1, 2, 3]
    assert all(n.model_calls == 4 for n in result.nodes)  # two actors, assess, resolve
    assert [c.agent for c in llm.calls[:4]] == ["actor.blue", "actor.red", "gm.assess", "gm.resolve"]
    assert all(len(n.rolls) == 4 for n in result.nodes)  # one roll per order
    final = result.nodes[-1].state
    assert final["blue"]["strength"] == 70 and final["red"]["strength"] == 55
    assert final["red"]["supply"] == 60 - 3 * 8 and final["world"]["crossroads"] == "contested"
    assert result.nodes[0].observations == {"blue": "Saw tanks."}
    # every tank objective has a graded metric, so the judge's scores are not used
    assert result.nodes[-1].scores == {
        "blue": {"Hold the crossroads until the reinforcements arrive": 5.0, "Keep combat strength above 50": 10.0},
        "red": {"Take the crossroads by the end of hour 4": 5.0, "Lose no more than 30 combat strength": 5.0}}
    assert result.scores == {"blue": 7.5, "red": 5.0}
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


def test_each_order_is_rolled_and_reported_to_the_game_master(tank):
    certain_then_impossible = [{"actor": "blue", "order": 1, "p_success": 1.0, "rationale": "easy"},
                               {"actor": "blue", "order": 2, "p_success": 0.0, "rationale": "hopeless"},
                               {"actor": "red", "order": 1, "p_success": 0.7, "rationale": "fine"}]
    llm = tank_llm(odds_list=certain_then_impossible)
    node = simulation(quiet(tank), llm).step([], Dice(random.Random(1), most_likely=True))
    blue = node.rolls_for("blue")
    # odds come from the fixed scale: 1.0 and 0.0 become its ends, 0.7 its nearest step
    assert [(r.text, r.p, r.success) for r in blue] == [("advance", 0.95, True), ("fire", 0.05, False)]
    red = node.rolls_for("red")
    assert [r.p for r in red] == [0.65, 0.65]  # order 2 had no estimate: the actor's other estimate is used
    resolve = llm.by_agent("gm.resolve")[0].user
    assert "1. [WORKED, p=0.95] advance" in resolve and "2. [DID NOT WORK, p=0.05] fire" in resolve


def test_same_seed_same_run_and_different_seed_different_dice(tank):
    first = simulate(quiet(tank), tank_llm(), seed=1, steps=3)
    again = simulate(quiet(tank), tank_llm(), seed=1, steps=3)
    other = simulate(quiet(tank), tank_llm(), seed=2, steps=3)
    assert [n.id for n in first.nodes] == [n.id for n in again.nodes]
    rolls = [[r.roll for n in run.nodes for r in n.rolls] for run in (first, again, other)]
    assert rolls[0] == rolls[1] != rolls[2]


def test_end_condition_stops_the_run_and_clamps(tank):
    result = simulate(quiet(tank), tank_llm(blue_loss=-60))
    assert len(result.nodes) == 2
    assert result.nodes[-1].terminal == "end: blue.strength <= 20"
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
    assert not any(label.startswith("open:") for label, _ in node.signature)


def test_listed_events_apply_their_effects_in_code(tank):
    scenario = quiet(tank)
    scenario.nature.events = [e.model_copy(update={"p": 1.0}) for e in tank.nature.events
                              if e.name == "Fog rolls in"]
    result = simulate(scenario, tank_llm(), steps=2)
    assert [e.name for e in result.nodes[0].events] == ["Fog rolls in"]
    assert result.nodes[0].state["world"]["weather"] == "fog"
    assert result.nodes[1].events == []  # its condition (clear weather) no longer holds


def test_private_information_only_comes_out_through_a_discovery_roll(tank):
    found = {"by": "red", "of": "blue", "what": "anti-tank companies on the reverse slope", "p": 1.0}
    llm = tank_llm(discoveries=[found, {**found, "by": "blue", "of": "red", "what": "fuel", "p": 0.0},
                                {**found, "by": "red", "of": "red", "what": "own plans"}])
    node = simulation(quiet(tank), llm).step([], Dice(random.Random(1), most_likely=True))
    # discoveries use the rare scale: 1.0 becomes its top step (0.5, which the most likely outcome takes)
    assert [(d.by, d.of, d.p, d.success) for d in node.discoveries] == [
        ("red", "blue", 0.5, True), ("blue", "red", 0.01, False)]
    resolve = llm.by_agent("gm.resolve")[0].user
    assert "red discovers blue's: anti-tank companies on the reverse slope" in resolve
    assert "blue does not discover red's: fuel" in resolve
    assert "Never reveal an actor's private information" in resolve


def amp_llm(changes, p=1.0) -> FakeLLM:
    def assess(call):
        return {"odds": odds("me", p=p, orders=1) if "Intent:" in call.user else [], "proposed_event": None}

    return FakeLLM(Assessment=assess,
                   Resolution=lambda c: {"narrative": "Bought it.", "changes": changes},
                   Verdict=lambda c: verdict("me", score=6))


BUY_NEW = [{"path": "me.savings", "op": "add", "value": -1500, "reason": "amp"},
           {"path": "me.amp", "op": "set", "value": "new", "reason": "bought"},
           {"path": "me.emergency_months", "op": "set", "value": 99, "reason": "x"}]


def test_policy_scenario_only_calls_models_when_something_happens(amp):
    llm = amp_llm(BUY_NEW)
    result = simulate(quiet(amp), llm, option=0, judge=False)
    calls = [n.model_calls for n in result.nodes]
    assert calls == [2, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]  # step 1 acts; review every 3 months
    assert [n.quiet for n in result.nodes] == [False] + [True] * 11
    final = result.nodes[-1].state["me"]
    assert final["savings"] == 4000 + 12 * 250 - 1500
    assert final["emergency_months"] == round(final["savings"] / 1100, 1)
    ignored = [c for c in result.nodes[0].gm_changes if c.path == "me.emergency_months"][0]
    assert "computed by the scenario" in ignored.note
    assert result.info.option == "Buy a new amplifier now for 1500 euros"
    # metrics: fund 5 months (10), an amp but no extra playing (5), stress 0 (10)
    assert result.scores == {"me": round((10 + 5 + 10) / 3, 2)}
    assert result.nodes[-1].terminal == "horizon" and result.verdict is None


def test_failed_planned_orders_are_retried(amp):
    result = simulate(quiet(amp), amp_llm([], p=0.0), option=1, judge=False)
    acting = [(n.step, n.actions["me"].intent) for n in result.nodes if n.actions]
    assert [step for step, _ in acting] == [1, 2, 3]  # the order, then two retries
    assert acting[1][1] == "Try again: Buy a used amplifier now for 800 euros"


def test_policy_option_with_steps_acts_when_planned(amp):
    result = simulate(quiet(amp), amp_llm([]), option=2, judge=False)
    assert [n.step for n in result.nodes if n.actions] == [1, 7]


def test_unknown_option_is_an_error(amp):
    with pytest.raises(ScenarioError, match="option 5"):
        simulation(amp, amp_llm([]), option=5)


def test_simulation_refuses_cloud_models_in_local_mode(tank):
    models = resolve_models(tank, gm_model="openrouter/qwen/qwen3-235b-a22b-2507", env={})
    with pytest.raises(PrivacyError):
        Simulation(tank, tank_llm(), models, privacy=LOCAL, run_id="t")


def test_model_resolution_order(tank):
    env = {"JANUS_MODEL": "ollama/env-actor", "JANUS_GM_MODEL": "ollama/env-gm"}
    models = resolve_models(tank, env=env)
    assert models.actors["blue"] == "ollama/gemma3:4b"  # the scenario's default beats the environment
    assert models.game_master == "ollama/gpt-oss:20b"
    models = resolve_models(tank, model="ollama/cli", env=env)
    assert set(models.actors.values()) == {"ollama/cli"}


def test_dice_can_force_the_most_likely_outcome():
    sampled, likely = Dice(random.Random(1)), Dice(random.Random(1), most_likely=True)
    for dice in (sampled, likely):
        dice.roll("a", 0.8)
        dice.roll("b", 0.3)
    assert likely.signature == [("a", True), ("b", False)]
    assert likely.probability == pytest.approx(0.8 * 0.7)
    assert sampled.probability == pytest.approx(
        (0.8 if sampled.signature[0][1] else 0.2) * (0.3 if sampled.signature[1][1] else 0.7))


def test_run_is_written_as_trace_and_report(tank, tmp_path):
    result = simulate(quiet(tank), tank_llm(), steps=2)
    trace, report = write_run(result, tank, tmp_path / "run")
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert [r["type"] for r in records] == ["run", "node", "node", "end"]
    assert records[1]["parent_id"] == records[0]["root_id"]
    text = report.read_text(encoding="utf-8")
    assert "# Armoured pincer at the Hill 112 crossroads" in text
    assert "| blue | Keep combat strength above 50 | 10 | metric `clamp((blue.strength - 20) / 30 * 10, 0, 10)` | |" in text
    assert "### hour 2 of 6" in text and "orders worked" in text
    assert "> Privately, blue: Saw tanks." in text
