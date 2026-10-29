"""Nature's mechanics, all applied in code: chance, rules, derived values, base-rate events, end
conditions, and the objectives that can be scored from the state."""

from __future__ import annotations

import math
import random
from typing import Literal

from pydantic import BaseModel, Field

from janus.sim import expr
from janus.sim.scenario import ActorSpec, ClockSpec, EventSpec
from janus.sim.state import AppliedChange, WorldState


class Dice:
    """The chance outcomes of one step. Each is sampled, or forced to its most likely value (or, for the
    labels in `flip`, to its less likely one), and logged, so the step's exact probability and its
    signature (which outcomes happened) are known."""

    def __init__(self, rng: random.Random, most_likely: bool = False, flip: frozenset[str] = frozenset()):
        self.rng = rng
        self.most_likely = most_likely
        self.flip = flip
        self.log: list[tuple[str, float, bool]] = []

    def roll(self, label: str, p: float) -> tuple[bool, float | None]:
        """Whether the outcome happens, and the value rolled (None when forced)."""
        value = self.rng.random()
        if label in self.flip:
            hit, shown = p < 0.5, None
        elif self.most_likely:
            hit, shown = p >= 0.5, None
        else:
            hit, shown = value < p, value
        self.log.append((label, p, hit))
        return hit, shown

    @property
    def probability(self) -> float:
        return math.prod(p if hit else 1 - p for _, p, hit in self.log)

    @property
    def signature(self) -> list[tuple[str, bool]]:
        return [(label, hit) for label, _, hit in self.log]


class FiredEvent(BaseModel):
    name: str
    description: str = ""
    p: float
    roll: float | None
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


def roll_events(events: list[EventSpec], state: WorldState, dice: Dice) -> list[FiredEvent]:
    """Roll every eligible event once and apply the effects of those that fire."""
    fired = []
    for event in events:
        if event.when and not expr.evaluate(event.when, state.values, dice.rng):
            continue
        hit, roll = dice.roll(f"event:{event.name}", event.p)
        if not hit:
            continue
        changes = []
        for target, effect in event.effects.items():
            value = expr.evaluate(effect, state.values, dice.rng) if isinstance(effect, str) else effect
            changes.append(state.assign(target, value, f"event: {event.name}"))
        fired.append(FiredEvent(name=event.name, description=event.description, p=event.p,
                                roll=roll, changes=changes))
    return fired


def end_reason(conditions: list[str], state: WorldState, rng: random.Random) -> str | None:
    return next((c for c in conditions if expr.evaluate(c, state.values, rng)), None)


def metric_scores(actors: dict[str, ActorSpec], state: WorldState) -> dict[str, dict[str, float]]:
    """Scores (0-10) of the objectives that have a metric, per actor and objective."""
    rng = random.Random(0)  # metrics should not be random; this keeps them from consuming the step's dice
    scores: dict[str, dict[str, float]] = {}
    for name, actor in actors.items():
        for objective in actor.objectives:
            if objective.metric:
                value = expr.evaluate(objective.metric, state.values, rng)
                scores.setdefault(name, {})[objective.text] = _to_score(value, objective.metric)
    return scores


def _to_score(value: object, metric: str) -> float:
    if isinstance(value, bool):
        return 10.0 if value else 0.0
    if isinstance(value, (int, float)):
        return round(min(10.0, max(0.0, float(value))), 2)
    raise expr.ExprError(f"metric {metric!r} must give true/false or a number, not {value!r}")
