"""What would change the answer: the options compared again under changed assumptions, by re-weighting
the tree you already have. No model is called.

Every roll in a tree is stored with its label and probability (`Node.chances`) and its outcome
(`Node.signature`), so each branch's probability can be recomputed for any change to those
probabilities: an event made twice as likely, one actor's orders made to work less often, surprises made
more frequent. The branches stay the same; their weights change.

Limits, reported rather than hidden: branches that were never explored stay unexplored, so a change that
makes an unexplored outcome likely lowers the coverage under that change; and branches carried to the
end (projections) followed the most likely course under the original probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

from janus.sim.scenario import Scenario
from janus.sim.trace import Node
from janus.sim.tree import Scorer, TreeView, Weigher, outcome, stored_p


@dataclass(frozen=True)
class Change:
    label: str  # what it means, in words
    prefix: str  # the rolls it applies to: "event:Car repair", "order:me:", "open:", "discovery:"
    kind: str  # "rate" multiplies the probability, "odds" multiplies the odds
    factor: float

    def apply(self, label: str, p: float) -> float:
        if not label.startswith(self.prefix) or not 0 < p < 1:
            return p
        if self.kind == "rate":
            return min(0.99, p * self.factor)
        return p * self.factor / (1 - p + p * self.factor)

    def weigher(self) -> Weigher:
        def p_branch(node: Node) -> float:
            if not node.chances:
                return node.p_branch
            result = 1.0
            for (label, p), (_, hit) in zip(node.chances, node.signature):
                q = self.apply(label, p)
                result *= q if hit else 1 - q
            return result

        return p_branch


@dataclass
class Comparison:
    change: str
    expected: dict[str, float | None]  # option label -> the focal actor's expected score
    coverage: dict[str, float]
    best: str | None
    margin: float | None  # how far the best option is ahead of the next one
    changes_best: bool = False


def default_focal(scenario: Scenario) -> str | None:
    """Whose objectives decide: the actor following a plan if there is one, else the first scored actor."""
    scored = [n for n, a in scenario.actors.items() if any(o.metric for o in a.objectives)]
    policy = [n for n in scored if scenario.actors[n].kind == "policy"]
    return (policy or scored or [None])[0]


def standard_changes(tree: TreeView, policy_actors: set[str]) -> list[Change]:
    """For every kind of roll the tree contains: listed events at half and double their rate, each actor's
    orders working less and more often (odds halved and doubled), unlisted surprises and discoveries."""
    labels = {label for node in tree.nodes() for label, _ in node.chances}
    events = sorted({label.split(":", 1)[1] for label in labels if label.startswith("event:")})
    actors = sorted({label.split(":")[1] for label in labels if label.startswith("order:")})
    changes = []
    for name in events:
        changes += [Change(f"{name} twice as likely", f"event:{name}", "rate", 2.0),
                    Change(f"{name} half as likely", f"event:{name}", "rate", 0.5)]
    for actor in actors:
        whose = "the plan" if actor in policy_actors else f"{actor}'s orders"
        changes += [Change(f"{whose} working less often", f"order:{actor}:", "odds", 0.5),
                    Change(f"{whose} working more often", f"order:{actor}:", "odds", 2.0)]
    if any(label.startswith("open:") for label in labels):
        changes += [Change("unlisted surprises twice as likely", "open:", "rate", 2.0)]
    if any(label.startswith("discovery:") for label in labels):
        changes += [Change("secrets discovered more easily", "discovery:", "odds", 2.0)]
    return changes


def compare(tree: TreeView, roots: dict[str, str], focal: str, score: Scorer, p: Weigher = stored_p,
            change: str = "as explored", opponents: str = "expected") -> Comparison:
    expected, coverage = {}, {}
    for root_id, label in roots.items():
        result = outcome(tree, root_id, opponents, focal, score, p)
        expected[label] = result.expected.get(focal)
        coverage[label] = result.coverage
    ranked = sorted((value for value in expected.values() if value is not None), reverse=True)
    best = max((label for label, v in expected.items() if v is not None), key=expected.get, default=None)
    margin = round(ranked[0] - ranked[1], 2) if len(ranked) > 1 else None
    return Comparison(change=change, expected=expected, coverage=coverage, best=best, margin=margin)


def sensitivity(tree: TreeView, roots: dict[str, str], focal: str, score: Scorer, changes: list[Change],
                opponents: str = "expected") -> tuple[Comparison, list[Comparison]]:
    """The comparison as explored, then under each change, marking those where the best option differs."""
    baseline = compare(tree, roots, focal, score, opponents=opponents)
    rows = []
    for change in changes:
        row = compare(tree, roots, focal, score, change.weigher(), change.label, opponents)
        row.changes_best = len(roots) > 1 and row.best != baseline.best
        rows.append(row)
    return baseline, rows
