import pytest

from janus.sim.engine import ENGINE_VERSION, EARLIER_ENGINE_VERSIONS, resolve_models
from janus.sim.models import OrderOdds, ProposedEvent
from janus.sim.sensitivity import Change, default_focal, sensitivity, standard_changes
from janus.sim.trace import Node
from janus.sim.tree import TreeStore, find_tree, outcome, scenario_scorer, tree_file
from janus.sim.world import weighted_score


def node(id, parent, step, p=1.0, score=None, chances=(), signature=(), terminal="horizon"):
    return Node(id=id, parent_id=parent, step=step, label=f"step {step}", quiet=True, p_branch=p,
                chances=list(chances), signature=list(signature), state={"me": {}}, terminal=terminal,
                scores={"me": {"o": score}} if score is not None else {})


@pytest.fixture
def store(tmp_path):
    with TreeStore(tmp_path / "tree.sqlite") as s:
        yield s


def storm(id, parent, hit, score):
    p = 0.1 if hit else 0.9
    return node(id, parent, 1, p, score, [("event:Storm", 0.1)], [("event:Storm", hit)])


def test_odds_snap_to_fixed_scales():
    assert OrderOdds(actor="a", order=1, p_success=0.85, rationale="r").p_success == 0.8
    assert OrderOdds(actor="a", order=1, p_success="70%", rationale="r").p_success == 0.65
    assert OrderOdds(actor="a", order=1, p_success=1.0, rationale="r").p_success == 0.95
    assert ProposedEvent(name="x", description="d", p=0.03).p == 0.02  # the rare scale keeps small steps apart
    assert ProposedEvent(name="x", description="d", p=0.9).p == 0.5


def test_scoring_is_not_part_of_a_trees_identity(amp):
    changed = amp.model_copy(deep=True)
    changed.actors["me"].objectives[0].metric = "me.emergency_months >= 3"
    changed.actors["me"].objectives[1].weight = 3
    assert changed.fingerprint() == amp.fingerprint()
    renamed = amp.model_copy(deep=True)
    renamed.actors["me"].objectives[0].text = "Something else"  # actors read the text: a new tree
    assert renamed.fingerprint() != amp.fingerprint()


def test_weights_set_how_much_each_objective_counts(amp):
    actor = amp.actors["me"].model_copy(deep=True)
    actor.objectives[0].weight = 2
    texts = [o.text for o in actor.objectives]
    assert weighted_score(actor, {texts[0]: 10, texts[1]: 4, texts[2]: 4}) == pytest.approx((20 + 4 + 4) / 4)
    assert weighted_score(actor, {}) is None


def test_scores_follow_the_current_metrics(amp):
    state = {"me": {**{k: v.value for k, v in amp.state["me"].items()}, "emergency_months": 1.5}}
    leaf = Node(id="x", parent_id="r", step=12, label="month 12", quiet=True, state=state)
    graded = scenario_scorer(amp)(leaf)["me"]
    strict = amp.model_copy(deep=True)
    strict.actors["me"].objectives[0].metric = "me.emergency_months >= 3"
    assert scenario_scorer(strict)(leaf)["me"] < graded  # 1.5 months: half marks graded, none pass/fail


def test_older_engines_trees_are_still_found(amp, tmp_path):
    models = resolve_models(amp, env={})
    assert find_tree(tmp_path, amp, models) is None
    old = tree_file(tmp_path, amp, models, EARLIER_ENGINE_VERSIONS[-1])
    old.parent.mkdir(parents=True)
    TreeStore(old).close()
    assert find_tree(tmp_path, amp, models) == (old, EARLIER_ENGINE_VERSIONS[-1])
    TreeStore(tree_file(tmp_path, amp, models)).close()
    assert find_tree(tmp_path, amp, models)[1] == ENGINE_VERSION


def test_continuations_keep_the_rarity_of_what_happens_in_them(store):
    store.add_root("R", "only")
    store.add(node("A", "R", 1, terminal=None), "R", "explore")
    store.add(node("P1", "A", 2, 0.9, terminal=None), "R", "projection")
    store.add(node("P2", "P1", 3, 0.9, score=8), "R", "projection")
    store.add(node("Q1", "A", 2, 0.9, terminal=None), "R", "rollout")
    store.add(node("Q2", "Q1", 3, 0.01, score=0), "R", "rollout")  # a rare disaster in a sampled rollout
    result = outcome(store, "R", focal="me")
    assert result.expected["me"] == pytest.approx(8 * 0.81 / (0.81 + 0.009), abs=0.01)  # not 4.0
    assert result.coverage == pytest.approx(1.0) and result.leaves == 2


def test_changes_reweight_branches_and_flag_a_new_best_option(store):
    store.add_root("A", "Go out")
    store.add(storm("A1", "A", True, 0), "A", "explore")
    store.add(storm("A0", "A", False, 9), "A", "explore")
    store.add_root("B", "Stay in")
    store.add(node("B1", "B", 1, 1.0, 7.5), "B", "explore")
    tree = store.view()
    roots = dict(tree.roots())
    changes = standard_changes(tree, policy_actors=set())
    assert [c.label for c in changes] == ["Storm twice as likely", "Storm half as likely"]
    baseline, rows = sensitivity(tree, roots, "me", lambda n: {"me": n.scores["me"]["o"]}, changes)
    assert baseline.best == "Go out" and baseline.expected["Go out"] == pytest.approx(8.1)
    assert baseline.margin == pytest.approx(0.6)
    twice, half = rows
    assert twice.expected["Go out"] == pytest.approx(7.2) and twice.best == "Stay in" and twice.changes_best
    assert half.expected["Go out"] == pytest.approx(8.55) and not half.changes_best


def test_change_arithmetic():
    rate = Change("x", "event:Storm", "rate", 2.0)
    assert rate.apply("event:Storm", 0.1) == pytest.approx(0.2)
    assert rate.apply("event:Storm", 0.7) == 0.99  # capped below certainty
    assert rate.apply("event:Fog", 0.1) == 0.1 and rate.apply("event:Storm", 1.0) == 1.0
    odds = Change("y", "order:me:", "odds", 2.0)
    assert odds.apply("order:me:1", 0.5) == pytest.approx(2 / 3)
    assert odds.weigher()(node("n", "r", 1, 0.25, chances=[("order:me:1", 0.5)], signature=[("order:me:1", False)])) \
        == pytest.approx(1 / 3)


def test_default_focal_prefers_the_actor_with_a_plan(amp, tank):
    assert default_focal(amp) == "me"
    assert default_focal(tank) == "blue"
