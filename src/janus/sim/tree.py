"""Ramification trees: every run and branch of a scenario, merged where they share a beginning.

A tree is one SQLite file per scenario fingerprint (the scenario, its profile, the models and the engine
version), so changing any of them starts a new tree instead of mixing branches that aren't comparable.
Each policy option has its own root; nodes hang off it by parent id.

The explorer grows a tree best-first: it always expands the most probable unexpanded node, trying the
most likely chance outcome and then the next most likely ones (one uncertain outcome flipped at a time),
down to a depth. Beyond it, every branch is carried to the end along its most likely course (a
projection: no surprises and no game-master reviews, so it costs nothing when nobody acts), and
optionally by sampled rollouts. It stops at a budget of model requests. The roll-up then weighs every
leaf by its probability (normalised over the branches explored) and compares the options.

Explored branches are chance outcomes and are weighed by probability. Projections and rollouts are
estimates of everything below the node they continue, so they share its weight equally, and they are
ignored wherever the node also has explored branches.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import random
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from itertools import count
from pathlib import Path
from statistics import mean

from janus.llm.client import LLM, LLMError
from janus.llm.privacy import PrivacyMode
from janus.sim.engine import ENGINE_VERSION, RoleModels, Simulation, _CountingLLM
from janus.sim.scenario import Scenario
from janus.sim.trace import Node
from janus.sim.world import Dice

SCHEMA = """
CREATE TABLE IF NOT EXISTS roots (id TEXT PRIMARY KEY, option TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    parent_id TEXT NOT NULL,
    root_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    candidate INTEGER NOT NULL,
    signature TEXT NOT NULL,
    p_branch REAL NOT NULL,
    terminal TEXT,
    origin TEXT NOT NULL,          -- explore, rollout or run: how the node was first reached
    data TEXT NOT NULL,            -- the full Node as JSON
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS nodes_by_parent ON nodes (parent_id, candidate, signature);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT NOT NULL, kind TEXT NOT NULL, root_id TEXT, leaf_id TEXT, seed INTEGER,
    settings TEXT, created_at TEXT NOT NULL
);
"""


def tree_file(out: Path, scenario: Scenario, models: RoleModels) -> Path:
    key = json.dumps([scenario.fingerprint(), models.as_dict(), ENGINE_VERSION], sort_keys=True)
    return out / scenario.name / f"tree-{hashlib.sha256(key.encode()).hexdigest()[:10]}.sqlite"


def _signature(signature: list) -> str:
    return json.dumps([[label, bool(hit)] for label, hit in signature])


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class TreeStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")  # the UI reads while an exploration writes
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> TreeStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add_root(self, root_id: str, option: str | None) -> None:
        self.db.execute("INSERT OR IGNORE INTO roots VALUES (?, ?, ?)", (root_id, option, _now()))
        self.db.commit()

    def add(self, node: Node, root_id: str, origin: str) -> bool:
        """Store a node; False if it was already there (another run or branch reached it first)."""
        cursor = self.db.execute(
            "INSERT OR IGNORE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (node.id, node.parent_id, root_id, node.step, node.candidate, _signature(node.signature),
             node.p_branch, node.terminal, origin, node.model_dump_json(), _now()))
        self.db.commit()
        return cursor.rowcount == 1

    def record_run(self, run_id: str, kind: str, root_id: str | None, leaf_id: str | None,
                   seed: int | None, settings: dict | None = None) -> None:
        self.db.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (run_id, kind, root_id, leaf_id, seed, json.dumps(settings or {}), _now()))
        self.db.commit()

    def node(self, node_id: str) -> Node | None:
        row = self.db.execute("SELECT data FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return Node.model_validate_json(row[0]) if row else None

    def children(self, parent_id: str) -> list[Node]:
        return [node for node, _ in self.children_with_origin(parent_id)]

    def children_with_origin(self, parent_id: str) -> list[tuple[Node, str]]:
        rows = self.db.execute("SELECT data, origin FROM nodes WHERE parent_id = ? ORDER BY p_branch DESC, id",
                               (parent_id,)).fetchall()
        return [(Node.model_validate_json(data), origin) for data, origin in rows]

    def child_with(self, parent_id: str, candidate: int, signature: list) -> Node | None:
        row = self.db.execute(
            "SELECT data FROM nodes WHERE parent_id = ? AND candidate = ? AND signature = ?",
            (parent_id, candidate, _signature(signature))).fetchone()
        return Node.model_validate_json(row[0]) if row else None

    def path(self, node_id: str) -> list[Node]:
        """The nodes from the root's first step down to this node."""
        nodes = []
        current = self.node(node_id)
        while current is not None:
            nodes.append(current)
            current = self.node(current.parent_id)
        return nodes[::-1]

    def roots(self) -> list[tuple[str, str | None]]:
        return self.db.execute("SELECT id, option FROM roots ORDER BY rowid").fetchall()  # the order they were added in

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]


# ── Roll-up ───────────────────────────────────────────────────────────────────


@dataclass
class Outcome:
    """How one option (or any node) turns out, over the leaves explored below it."""

    expected: dict[str, float]  # probability-weighted mean objective score per actor
    worst10: dict[str, float]  # 10th percentile of the score per actor
    p_end: float  # probability of ending early through an end condition
    coverage: float  # share of the probability mass the explored leaves actually stand for
    leaves: int
    unfinished: int  # leaves that stop before the end (the budget ran out): scored as they stand


CONTINUATIONS = ("projection", "rollout")


def leaf_score(node: Node) -> dict[str, float]:
    return {actor: mean(scores.values()) for actor, scores in node.scores.items() if scores}


def distribution(store: TreeStore, parent_id: str, opponents: str = "expected",
                 focal: str | None = None) -> list[tuple[float, float, Node]]:
    """Leaves below `parent_id` as (normalised weight, raw probability, node).

    Children are grouped by candidate (versions of the decisions). Within a group, chance branches are
    weighted by their probability, normalised over the branches explored. Across groups, the weight is
    shared equally (`opponents="expected"`), or all of it goes to the group worst for `focal` ("worst").
    A node with only projections or rollouts below it shares its weight equally among them: each one
    stands for everything below the node, so it does not shrink the probability covered.
    """
    rows = store.children_with_origin(parent_id)
    if not rows:
        return []
    explored = [child for child, origin in rows if origin not in CONTINUATIONS]
    if not explored:
        share = 1.0 / len(rows)
        leaves = []
        for child, _ in rows:
            below = distribution(store, child.id, opponents, focal) or [(1.0, 1.0, child)]
            leaves += [(w * share, raw * share, leaf) for w, raw, leaf in below]
        return leaves
    groups: dict[int, list[Node]] = {}
    for child in explored:
        groups.setdefault(child.candidate, []).append(child)
    per_group = []
    for members in groups.values():
        total = sum(c.p_branch for c in members) or 1.0
        leaves = []
        for child in members:
            below = distribution(store, child.id, opponents, focal)
            if below:
                leaves += [(w * child.p_branch / total, raw * child.p_branch, leaf) for w, raw, leaf in below]
            else:
                leaves.append((child.p_branch / total, child.p_branch, child))
        per_group.append(leaves)
    if opponents == "worst" and focal is not None and len(per_group) > 1:
        def focal_expected(leaves):
            return sum(w * leaf_score(leaf).get(focal, 0.0) for w, _, leaf in leaves)
        return min(per_group, key=focal_expected)
    share = 1.0 / len(per_group)
    return [(w * share, raw * share, leaf) for leaves in per_group for w, raw, leaf in leaves]


def outcome(store: TreeStore, parent_id: str, opponents: str = "expected", focal: str | None = None) -> Outcome:
    leaves = distribution(store, parent_id, opponents, focal)
    actors = sorted({a for _, _, leaf in leaves for a in leaf_score(leaf)})
    expected, worst10 = {}, {}
    for actor in actors:
        scored = sorted((leaf_score(leaf)[actor], w) for w, _, leaf in leaves if actor in leaf_score(leaf))
        weight = sum(w for _, w in scored) or 1.0
        expected[actor] = round(sum(s * w for s, w in scored) / weight, 2)
        cumulative = 0.0
        for score, w in scored:
            cumulative += w / weight
            if cumulative >= 0.1:
                worst10[actor] = round(score, 2)
                break
    return Outcome(
        expected=expected,
        worst10=worst10,
        p_end=round(sum(w for w, _, leaf in leaves if (leaf.terminal or "").startswith("end:")), 3),
        coverage=round(sum(raw for _, raw, _ in leaves), 3),
        leaves=len(leaves),
        unfinished=sum(1 for _, _, leaf in leaves if leaf.terminal is None),
    )


# ── Exploration ───────────────────────────────────────────────────────────────


@dataclass
class ExploreSettings:
    depth: int = 2  # steps expanded with branching; beyond it, branches are carried to the end
    branches: int = 3  # chance outcomes tried per node: the most likely, then the next most likely
    candidates: int = 1  # versions of the LLM actors' decisions per node
    rollouts: int = 0  # sampled continuations per leaf, after its most likely one (the projection)
    budget: int = 60  # model requests (cached replies are free)
    continuation_share: float = 0.4  # part of the budget kept for projections and rollouts

    def as_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class _Frontier:
    heap: list = field(default_factory=list)
    order: count = field(default_factory=count)

    def push(self, mass: float, option: int, node_id: str) -> None:
        heapq.heappush(self.heap, (-mass, next(self.order), option, node_id))

    def pop(self) -> tuple[float, int, str]:
        mass, _, option, node_id = heapq.heappop(self.heap)
        return -mass, option, node_id


class Explorer:
    def __init__(self, scenario: Scenario, llm: LLM, models: RoleModels, *, privacy: PrivacyMode,
                 store: TreeStore, settings: ExploreSettings, seed: int, run_id: str,
                 on_node: Callable[[Node, str], None] | None = None):
        self.scenario = scenario
        self.store = store
        self.settings = settings
        self.seed = seed
        self.run_id = run_id
        self.on_node = on_node
        self.base_llm = llm
        self.llm = _CountingLLM(llm)
        self.models = models
        self.privacy = privacy
        self.failures: list[str] = []  # steps a model could not produce; their branches are skipped

    @property
    def spent(self) -> int:
        """Model requests so far; replies served from the cache cost nothing."""
        return getattr(self.base_llm, "requests", self.llm.calls)

    def options(self) -> list[int]:
        counts = [len(a.options) for a in self.scenario.actors.values() if a.kind == "policy"]
        return list(range(max(counts))) if counts else [0]

    def explore(self, options: list[int] | None = None) -> dict[str, str]:
        """Grow the tree for each option; returns {root id: option label}."""
        settings = self.settings
        sims = {i: Simulation(self.scenario, self.llm, self.models, privacy=self.privacy,
                              run_id=self.run_id, option=i) for i in (options or self.options())}
        roots = {}
        frontier = _Frontier()
        for i, sim in sims.items():
            self.store.add_root(sim.root_id, sim.option_label)
            roots[sim.root_id] = sim.option_label or self.scenario.title or self.scenario.name
            frontier.push(1.0, i, sim.root_id)

        expand_budget = settings.budget * (1 - settings.continuation_share)
        leaves: list[tuple[float, int, str]] = []
        seen: set[str] = set()  # children pushed in this session; existing ones from earlier sessions count too
        while frontier.heap and self.spent < expand_budget:
            mass, i, node_id = frontier.pop()
            sim = sims[i]
            path = [] if node_id == sim.root_id else self.store.path(node_id)
            if path and path[-1].terminal:
                continue
            if len(path) >= settings.depth:
                leaves.append((mass, i, node_id))
                continue
            parent = path[-1].id if path else sim.root_id
            for candidate in range(settings.candidates):
                # The most likely outcome first, then the next most likely ones: the same step with one
                # uncertain outcome flipped, most uncertain first. Amounts drawn in code stay the same.
                flips: list[frozenset[str]] = [frozenset()]
                while flips and self.spent < expand_budget:
                    flip = flips.pop(0)
                    dice = Dice(random.Random(f"{self.seed}/{parent}/{candidate}"), most_likely=True, flip=flip)
                    try:
                        child = sim.step(path, dice, candidate=candidate, existing=self.store.child_with)
                    except LLMError as e:
                        self.failures.append(f"{node_id}: {e}")
                        continue
                    self._keep(child, sim.root_id, "explore")
                    if child.id not in seen:
                        seen.add(child.id)
                        frontier.push(mass * child.p_branch / settings.candidates, i, child.id)
                    if not flip:
                        uncertain = sorted((abs(p - 0.5), label) for label, p in child.chances if 0 < p < 1)
                        flips = [frozenset({label}) for _, label in uncertain[: settings.branches - 1]]
        # Nodes still on the frontier when the expansion budget ran out are leaves too.
        leaves += [frontier.pop() for _ in range(len(frontier.heap))]

        # Carry every leaf to the end: along its most likely course first, then sampled rollouts. Neither
        # asks for the game master's periodic reviews, so a stretch where nobody acts costs nothing.
        ordered = [leaf for leaf in sorted(leaves, reverse=True) if leaf[2] != sims[leaf[1]].root_id]
        for kind, repeats in (("projection", 1), ("rollout", settings.rollouts)):
            for _, i, node_id in ordered:
                sim = sims[i]
                for r in range(repeats):
                    path = self.store.path(node_id)
                    while not path[-1].terminal and self.spent < settings.budget:
                        parent = path[-1]
                        dice = Dice(random.Random(f"{self.seed}/{kind}/{r}/{parent.id}/{parent.step + 1}"),
                                    most_likely=kind == "projection")
                        try:
                            child = sim.step(path, dice, existing=self.store.child_with, reviews=False)
                        except LLMError as e:
                            self.failures.append(f"{parent.id}: {e}")
                            break
                        self._keep(child, sim.root_id, kind)
                        path.append(child)
        self.store.record_run(self.run_id, "explore", None, None, self.seed, settings.as_dict())
        return roots

    def _keep(self, node: Node, root_id: str, origin: str) -> bool:
        new = self.store.add(node, root_id, origin)
        if new and self.on_node:
            self.on_node(node, origin)
        return new
