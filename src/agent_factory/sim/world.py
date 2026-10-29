"""Nature's mechanics, all applied in code: rules, derived values, base-rate events, end conditions."""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, Field

from agent_factory.sim import expr
from agent_factory.sim.scenario import ClockSpec, EventSpec
from agent_factory.sim.state import AppliedChange, WorldState


class FiredEvent(BaseModel):
    name: str
    description: str = ""
    p: float
    roll: float
    source: Literal["table", "open"] = "table"
    changes: list[AppliedChange] = Field(default_factory=list)


def clock_label(clock: ClockSpec, step: int) -> str:
    return f"{clock.unit} {step} of {clock.horizon}"


def apply_formulas(formulas: dict[str, str], state: WorldState, rng: random.Random, cause: str) -> list[AppliedChange]:
    """Rules and derived values: each target gets its expression's value, in the order written."""
    return [
        state.assign(target, expr.evaluate(source, state.values, rng), cause)
        for target, source in formulas.items()
    ]


def roll_events(events: list[EventSpec], state: WorldState, rng: random.Random) -> list[FiredEvent]:
    """Roll every eligible event once and apply the effects of those that fire."""
    fired = []
    for event in events:
        if event.when and not expr.evaluate(event.when, state.values, rng):
            continue
        roll = rng.random()
        if roll >= event.p:
            continue
        changes = []
        for target, effect in event.effects.items():
            value = expr.evaluate(effect, state.values, rng) if isinstance(effect, str) else effect
            changes.append(state.assign(target, value, f"event: {event.name}"))
        fired.append(FiredEvent(name=event.name, description=event.description, p=event.p,
                                roll=roll, changes=changes))
    return fired


def end_reason(conditions: list[str], state: WorldState, rng: random.Random) -> str | None:
    return next((c for c in conditions if expr.evaluate(c, state.values, rng)), None)
