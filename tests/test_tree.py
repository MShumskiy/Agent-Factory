import pytest
from conftest import FakeLLM
from test_engine import BUY_NEW, amp_llm, quiet, simulation, tank_llm

from janus.llm.privacy import PrivacyMode
from janus.sim.engine import resolve_models
from janus.sim.tree import Explorer, ExploreSettings, TreeStore, distribution, outcome, tree_file


def explorer(scenario, llm, store, **settings) -> Explorer:
    return Explorer(scenario, llm, resolve_models(scenario, env={}), privacy=PrivacyMode.LOCAL, store=store,
                    settings=ExploreSettings(**settings), seed=3, run_id="explore-test")


@pytest.fixture
def store(tmp_path):
    with TreeStore(tmp_path / "tree.sqlite") as s:
        yield s


def odds_by_text(p_new: float, p_used: float) -> FakeLLM:
    """The game master finds buying a new amp likelier to work than a used one."""
    def assess(call):
        if "Intent:" not in call.user:
            return {"odds": []}
        p = p_new if "new amplifier" in call.user else p_used
        return {"odds": [{"actor": "me", "order": 1, "p_success": p, "rationale": "t"}]}

    return FakeLLM(Assessment=assess, Resolution=lambda c: {"narrative": "Done.", "changes": BUY_NEW})


def test_each_option_gets_a_root_and_chance_branches(amp, store):
    roots = explorer(quiet(amp), odds_by_text(0.9, 0.6), store, depth=1, branches=4, rollouts=0, budget=100).explore()
    assert list(roots.values()) == [o.label for o in amp.actors["me"].options]
    new_root = next(r for r, label in roots.items() if label.startswith("Buy a new"))
    children = store.children(new_root)
    # the purchase works or not: two distinct outcomes, however many branches were tried; 0.9 is on the
    # odds scale's nearest step, 0.95
    assert sorted((c.signature[0][1], round(c.p_branch, 2)) for c in children) == [(False, 0.05), (True, 0.95)]
    assert sum(c.p_branch for c in children) == pytest.approx(1.0)


def test_quiet_steps_are_one_branch_and_rollouts_reach_the_end(amp, store):
    explorer(quiet(amp), amp_llm(BUY_NEW), store, depth=2, branches=3, rollouts=1, budget=100).explore([0])
    root = store.roots()[0][0]
    worked = next(c for c in store.children(root) if c.signature == [("order:me:1", True)])
    assert worked.p_branch == pytest.approx(0.95)
    second = store.children(worked.id)
    assert len(second) == 1 and second[0].signature == []  # a quiet month has no chance at all
    leaves = distribution(store, root)
    assert {leaf.terminal for _, _, leaf in leaves} == {"horizon"}
    result = outcome(store, root)
    assert result.coverage == pytest.approx(1.0) and result.unfinished == 0


def test_projections_carry_quiet_branches_to_the_end_for_free(amp, store):
    llm = amp_llm(BUY_NEW)
    explorer(quiet(amp), llm, store, depth=1, branches=1, rollouts=0, budget=3).explore([0])
    assert len(llm.calls) == 2  # month 1: assess and resolve; months 2-12 cost nothing
    root = store.roots()[0][0]
    result = outcome(store, root)
    # one branch explored (the purchase working, 0.95), carried to the end: it covers 95%
    assert result.unfinished == 0 and result.coverage == pytest.approx(0.95)
    leaf = distribution(store, root)[0][2]
    assert leaf.step == 12 and leaf.terminal == "horizon"


def test_a_branch_the_model_cannot_produce_is_skipped(amp, store):
    from janus.llm.client import LLMError

    def resolve(call):
        if "Buy a used" in call.user:
            raise LLMError("actor: LLMP returned 502")
        return {"narrative": "Done.", "changes": BUY_NEW}

    llm = FakeLLM(Assessment=odds_by_text(0.9, 0.9).handlers["Assessment"], Resolution=resolve)
    exp = explorer(quiet(amp), llm, store, depth=1, branches=2, budget=50)
    roots = exp.explore([0, 1])
    assert exp.failures and all("502" in f for f in exp.failures)
    by_label = {label: root for root, label in roots.items()}
    assert store.children(by_label["Buy a new amplifier now for 1500 euros"])
    assert not store.children(by_label["Buy a used amplifier now for 800 euros"])


def test_exploring_again_reuses_the_tree(amp, store):
    settings = dict(depth=2, branches=2, rollouts=1, budget=100)
    explorer(quiet(amp), odds_by_text(0.9, 0.6), store, **settings).explore()
    before = store.count()
    explorer(quiet(amp), odds_by_text(0.9, 0.6), store, **settings).explore()
    assert store.count() == before


def test_budget_stops_expansion_and_leaves_are_marked_unfinished(tank, store):
    llm = tank_llm()
    explorer(quiet(tank), llm, store, depth=3, branches=2, rollouts=0, budget=10).explore()
    assert len(llm.calls) <= 10 + 4  # the step in progress when the budget ran out may finish
    result = outcome(store, store.roots()[0][0])
    assert result.leaves >= 1 and result.unfinished == result.leaves
    assert 0 < result.coverage <= 1


def test_option_comparison_reflects_the_odds(amp, store):
    roots = explorer(quiet(amp), odds_by_text(0.95, 0.3), store, depth=1, branches=4, rollouts=1, budget=200).explore([0, 1])
    by_label = {label: outcome(store, root) for root, label in roots.items()}
    new = by_label["Buy a new amplifier now for 1500 euros"]
    used = by_label["Buy a used amplifier now for 800 euros"]
    assert new.leaves >= 2 and used.leaves >= 2
    assert sum(w for w, _, _ in distribution(store, next(iter(roots)))) == pytest.approx(1.0)
    assert new.expected["me"] != used.expected["me"]


def test_candidates_are_weighed_equally_or_by_the_worst_case(tank, store):
    blue_calls = []

    def action(call):  # blue's first version is bold, its second careful; red is always careful
        if call.agent == "actor.blue":
            blue_calls.append(call)
        bold = call.agent == "actor.blue" and len(blue_calls) == 1
        return {"intent": "bold" if bold else "careful", "orders": ["go"], "rationale": "t"}

    def resolve(call):
        loss = -70 if "## blue\nIntent: bold" in call.user else -5
        return {"narrative": "x", "changes": [{"path": "blue.strength", "op": "add", "value": loss, "reason": "r"}]}

    llm = FakeLLM(Action=action, Resolution=resolve,
                  Assessment=lambda c: {"odds": [{"actor": a, "order": 1, "p_success": 1.0, "rationale": "t"}
                                                 for a in ("blue", "red")]})
    explorer(quiet(tank), llm, store, depth=1, branches=1, candidates=2, rollouts=0, budget=50).explore()
    root = store.roots()[0][0]
    candidates = {c.candidate for c in store.children(root)}
    assert candidates == {0, 1}
    expected = outcome(store, root)
    worst = outcome(store, root, opponents="worst", focal="blue")
    assert worst.expected["blue"] < expected.expected["blue"]


def test_a_run_lands_in_the_same_tree(amp, store, tmp_path):
    sim = simulation(quiet(amp), amp_llm(BUY_NEW), option=0)
    path = sim.run(seed=5, judge=False).nodes
    store.add_root(sim.root_id, sim.option_label)
    for node in path:
        store.add(node, sim.root_id, "run")
    before = store.count()
    explorer(quiet(amp), amp_llm(BUY_NEW), store, depth=1, branches=1, rollouts=1, budget=100).explore([0])
    assert store.count() == before  # the explored path is the run's path: nothing new


def test_tree_file_changes_with_models(amp, tmp_path):
    models = resolve_models(amp, env={})
    other = resolve_models(amp, gm_model="ollama/gemma3:12b", env={})
    assert tree_file(tmp_path, amp, models) != tree_file(tmp_path, amp, other)
    assert tree_file(tmp_path, amp, models).parent == tmp_path / "guitar_amp_purchase"
