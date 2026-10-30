"""Actors: parties with goals that act each step.

An `LLMActor` decides with a model from what it can see. A `PolicyActor` follows a fixed option from
the scenario, so you can compare plans without a model choosing for you; a planned order that fails is
tried again on the following steps, up to the option's `retries`.

Actors keep no state of their own: everything they need comes from the path in their view, so the same
actor object can decide at any node of a tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from janus.llm.client import LLM
from janus.sim.models import Action
from janus.sim.prompts import ACTOR_SYSTEM, actor_prompt
from janus.sim.scenario import PolicyOption, Scenario
from janus.sim.trace import Node
from janus.sim.world import FiredEvent


@dataclass
class ActorView:
    """What an actor is shown in a step; the engine applies visibility before building it."""

    step: int
    label: str
    visible_state: str
    history: str
    events: list[FiredEvent]
    seed: int
    path: list[Node] = field(default_factory=list)  # the steps so far, for actors that follow a plan


class Actor(Protocol):
    name: str

    def decide(self, view: ActorView) -> Action | None: ...


class PolicyActor:
    def __init__(self, name: str, option: PolicyOption):
        self.name = name
        self.option = option

    def decide(self, view: ActorView) -> Action | None:
        order = self.option.order_at(view.step)
        if order is not None:
            return Action(intent=order, orders=[order], rationale=f"Fixed plan: {self.option.label}")
        if (retry := self._retry(view.path)) is not None:
            return Action(intent=f"Try again: {retry}", orders=[retry],
                          rationale=f"The planned order did not work; the plan allows {self.option.retries} retries.")
        return None

    def _retry(self, path: list[Node]) -> str | None:
        """The last order, if it failed and fewer than `retries` retries of it have been made."""
        failures = 0
        order = None
        for node in reversed(path):
            rolls = node.rolls_for(self.name)
            if not rolls or rolls[0].success or (order is not None and rolls[0].text != order):
                break
            order = rolls[0].text
            failures += 1
        return order if order is not None and failures <= self.option.retries else None


class LLMActor:
    def __init__(self, name: str, scenario: Scenario, llm: LLM, model: str, temperature: float = 0.8):
        self.name = name
        self.scenario = scenario
        self.llm = llm
        self.model = model
        self.temperature = temperature

    def decide(self, view: ActorView) -> Action:
        return self.llm.structured(
            Action,
            model=self.model,
            system=ACTOR_SYSTEM.format(name=self.name, unit=self.scenario.clock.unit),
            user=actor_prompt(self.scenario, self.name, view.visible_state, view.history, view.label, view.events),
            agent=f"actor.{self.name}",
            temperature=self.temperature,
            seed=view.seed,
        )
