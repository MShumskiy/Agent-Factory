"""Actors: parties with goals that act each step.

An `LLMActor` decides with a model from what it can see. A `PolicyActor` follows a fixed option
from the scenario, so you can compare plans without a model choosing for you.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent_factory.llm.client import LLM
from agent_factory.sim.models import Action
from agent_factory.sim.prompts import ACTOR_SYSTEM, actor_prompt
from agent_factory.sim.scenario import PolicyOption, Scenario
from agent_factory.sim.world import FiredEvent


@dataclass
class ActorView:
    """What an actor is shown in a step; the engine applies visibility before building it."""

    step: int
    label: str
    visible_state: str
    history: str
    events: list[FiredEvent]
    seed: int


class Actor(Protocol):
    name: str

    def decide(self, view: ActorView) -> Action | None: ...


class PolicyActor:
    def __init__(self, name: str, option: PolicyOption):
        self.name = name
        self.option = option

    def decide(self, view: ActorView) -> Action | None:
        order = self.option.order_at(view.step)
        if order is None:
            return None
        return Action(intent=order, orders=[order], rationale=f"Fixed plan: {self.option.label}")


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
