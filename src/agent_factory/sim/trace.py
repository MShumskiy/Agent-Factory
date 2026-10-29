"""What a run produces: one node per step, plus the run's settings and the judge's verdict.

Node ids are derived from the parent id and what happened in the step (events, actions, dice
outcomes, resulting state), not from the wording of narratives, so two runs that reach the same
situation by the same path share ids. Ramification trees (M2) are built on that.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field

from agent_factory.sim.models import Action, ProposedEvent, Verdict
from agent_factory.sim.scenario import Scalar
from agent_factory.sim.state import AppliedChange
from agent_factory.sim.world import FiredEvent


class Roll(BaseModel):
    actor: str
    p: float
    roll: float
    success: bool
    rationale: str


class Node(BaseModel):
    id: str
    parent_id: str
    step: int
    label: str
    quiet: bool
    rule_changes: list[AppliedChange] = Field(default_factory=list)
    events: list[FiredEvent] = Field(default_factory=list)
    actions: dict[str, Action] = Field(default_factory=dict)
    rolls: list[Roll] = Field(default_factory=list)
    proposed_event: ProposedEvent | None = None
    narrative: str = ""
    observations: dict[str, str] = Field(default_factory=dict)
    gm_changes: list[AppliedChange] = Field(default_factory=list)
    derived_changes: list[AppliedChange] = Field(default_factory=list)
    state: dict[str, dict[str, Scalar]]
    model_calls: int = 0

    def roll_for(self, actor: str) -> Roll | None:
        return next((r for r in self.rolls if r.actor == actor), None)


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
