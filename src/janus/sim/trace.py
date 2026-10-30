"""What a step produces, and what a run produces.

A node is the world after one step, reached from its parent through one set of chance outcomes. Its id
is derived from the parent id and what happened in the step (the decisions, the chance outcomes and the
resulting state), not from the wording of narratives, so runs and branches that reach the same situation
by the same path share nodes. That is what the ramification tree is built on.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field

from janus.sim.models import Action, ProposedEvent, Verdict
from janus.sim.scenario import Scalar
from janus.sim.state import AppliedChange
from janus.sim.world import FiredEvent


class Roll(BaseModel):
    actor: str
    order: int  # 1-based position in the actor's orders
    text: str
    p: float
    roll: float | None  # None when the branch took the most likely outcome instead of rolling
    success: bool
    rationale: str


class DiscoveryRoll(BaseModel):
    by: str
    of: str
    what: str
    p: float
    roll: float | None
    success: bool


class Node(BaseModel):
    id: str
    parent_id: str
    step: int
    label: str
    quiet: bool
    candidate: int = 0  # which version of the actors' decisions this branch follows
    p_branch: float = 1.0  # probability of this step's chance outcomes, given the parent and decisions
    most_likely: bool = False  # every chance outcome took its most likely value
    signature: list[tuple[str, bool]] = Field(default_factory=list)
    chances: list[tuple[str, float]] = Field(default_factory=list)  # every roll's label and probability
    rule_changes: list[AppliedChange] = Field(default_factory=list)
    events: list[FiredEvent] = Field(default_factory=list)
    actions: dict[str, Action] = Field(default_factory=dict)
    rolls: list[Roll] = Field(default_factory=list)
    proposed_event: ProposedEvent | None = None
    discoveries: list[DiscoveryRoll] = Field(default_factory=list)
    narrative: str = ""
    observations: dict[str, str] = Field(default_factory=dict)
    gm_changes: list[AppliedChange] = Field(default_factory=list)
    derived_changes: list[AppliedChange] = Field(default_factory=list)
    state: dict[str, dict[str, Scalar]]
    scores: dict[str, dict[str, float]] = Field(default_factory=dict)  # metric objectives, per actor
    terminal: str | None = None  # why the simulation ends here, if it does
    model_calls: int = 0

    def rolls_for(self, actor: str) -> list[Roll]:
        return [r for r in self.rolls if r.actor == actor]

    def worked(self, actor: str) -> tuple[int, int]:
        """(orders that worked, orders rolled) for an actor."""
        rolls = self.rolls_for(actor)
        return sum(r.success for r in rolls), len(rolls)


class RunInfo(BaseModel):
    run_id: str
    scenario: str
    title: str
    seed: int
    option: str | None = None
    privacy: str
    models: dict[str, str]
    started_at: str
    root_id: str


class RunResult(BaseModel):
    info: RunInfo
    initial_state: dict[str, dict[str, Scalar]]
    nodes: list[Node]
    ended_by: str | None = None
    verdict: Verdict | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    model_requests: int = 0
    cache_hits: int = 0


def node_id(parent_id: str, step: int, content: object) -> str:
    payload = json.dumps([parent_id, step, content], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
