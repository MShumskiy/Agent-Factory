import textwrap

import pytest

from agent_factory.llm.privacy import PrivacyMode
from agent_factory.sim.scenario import ScenarioError, load_scenario

MINIMAL = """
name: t
setting: A test.
clock: {unit: day, horizon: 3}
actors:
  me: {kind: llm, brief: tester}
state:
  me:
    money: {value: 10, min: 0}
    mood: {value: ok, choices: [ok, bad]}
"""


def write(tmp_path, text, name="s.yaml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_repo_scenarios_load(tank, amp):
    assert tank.privacy is PrivacyMode.LOCAL and not tank.sensitive
    assert set(tank.actors) == {"blue", "red"}
    assert amp.sensitive and amp.loaded_profile.actor == "me"
    assert "sound technician" in amp.about("me") and amp.about("other") == ""
    assert [o.label for o in amp.actors["me"].options][2] == "Save first, then buy a used amplifier"
    assert amp.actors["me"].options[2].order_at(7).startswith("Buy a used")
    assert amp.actors["me"].options[0].order_at(2) is None


def test_shorthand_values(tmp_path):
    scenario = load_scenario(write(tmp_path, MINIMAL + "  world:\n    day_off: false\n"))
    assert scenario.state["world"]["day_off"].value is False


def test_profile_overrides_values_and_keeps_bounds(tmp_path):
    write(tmp_path, "sensitive: true\nactor: me\nabout: secret\nstate:\n  me: {money: 99}\n", "p.yaml")
    scenario = load_scenario(write(tmp_path, MINIMAL + f"profile: {(tmp_path / 'p.yaml').as_posix()}\n"))
    assert scenario.state["me"]["money"].value == 99 and scenario.state["me"]["money"].min == 0
    assert scenario.sensitive


def test_profile_values_are_checked(tmp_path):
    write(tmp_path, "state:\n  me: {mood: furious}\n", "p.yaml")
    with pytest.raises(ScenarioError, match="choices"):
        load_scenario(write(tmp_path, MINIMAL + f"profile: {(tmp_path / 'p.yaml').as_posix()}\n"))


@pytest.mark.parametrize("extra, message", [
    ("nature:\n  rules:\n    me.money: me.monye + 1\n", "unknown variable"),
    ("nature:\n  rules:\n    me.cash: me.money + 1\n", "unknown variable me.cash"),
    ("nature:\n  events:\n    - {name: e, p: 0.5, when: \"exec('x')\"}\n", "only these functions"),
    ("end_when: ['me.money <']\n", "invalid expression"),
    ("nature:\n  events:\n    - {name: e, p: 1.5}\n", "less than or equal to 1"),
    ("nature:\n  events:\n    - {name: e, probability: 0.5, p: 0.1}\n", "Extra inputs are not permitted"),
    ("  world:\n    cost: {value: 1, description: rent, bills}\n", "Extra inputs are not permitted"),
])
def test_mistakes_are_reported_at_load(tmp_path, extra, message):
    with pytest.raises(ScenarioError, match=message):
        load_scenario(write(tmp_path, MINIMAL + extra))


def test_policy_actors_need_options(tmp_path):
    text = MINIMAL.replace("{kind: llm, brief: tester}", "{kind: policy}")
    with pytest.raises(ScenarioError, match="at least one option"):
        load_scenario(write(tmp_path, text))
