"""What the models are asked to return, as pydantic models.

These become JSON schemas in `response_format`. They are kept flat and small because local models
follow simple schemas far more reliably, and parsing is lenient where small models tend to slip
(probabilities given as percentages, scores given as floats).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, BeforeValidator, Field, create_model


def _probability(value: object) -> object:
    if isinstance(value, str):
        value = value.strip().rstrip("%")
        try:
            value = float(value)
        except ValueError:
            return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if 1 < value <= 100:
            value = value / 100
        return min(1.0, max(0.0, float(value)))
    return value


def _score(value: object) -> object:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return min(10, max(0, round(value)))
    return value


Probability = Annotated[float, BeforeValidator(_probability)]
Score = Annotated[int, BeforeValidator(_score)]

# The game master's probabilities come from fixed scales, so two estimates that only differ by noise
# (0.85 against 0.80) become the same step. The schema lists the steps; anything else is snapped.
ODDS_SCALE = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)  # almost certainly not ... almost certain
RARE_SCALE = (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5)  # for surprises and discoveries, mostly rare


def _snapper(scale: tuple[float, ...]):
    def snap(p: float) -> float:
        logit = lambda x: math.log(x / (1 - x))  # nearest on the odds scale, so 0.01 and 0.02 stay apart
        clamped = min(0.999, max(0.001, p))
        return min(scale, key=lambda step: abs(logit(step) - logit(clamped)))
    return snap


Odds = Annotated[Probability, AfterValidator(_snapper(ODDS_SCALE))]
RareChance = Annotated[Probability, AfterValidator(_snapper(RARE_SCALE))]


class Action(BaseModel):
    intent: str = Field(description="One sentence: what you are trying to achieve in this step")
    orders: list[str] = Field(description="The concrete things you do in this step, 1 to 5 items")
    rationale: str = Field(description="Why, in one or two sentences")


class OrderOdds(BaseModel):
    actor: str = Field(description="Name of the actor who gave the order")
    order: int = Field(description="Number of the order, as listed under the actor's action")
    p_success: Odds = Field(description="Probability that this order achieves what it intends, one of the scale's steps",
                            json_schema_extra={"enum": list(ODDS_SCALE)})
    rationale: str = Field(description="Why this probability, in one sentence")


class ProposedEvent(BaseModel):
    name: str = Field(description="Short name of the event")
    description: str = Field(description="What would happen, in one or two sentences")
    p: RareChance = Field(description="Probability that it happens during this step, one of the scale's steps",
                          json_schema_extra={"enum": list(RARE_SCALE)})


class Discovery(BaseModel):
    by: str = Field(description="The actor who could discover it")
    of: str = Field(description="The actor whose private information it is")
    what: str = Field(description="The private fact that could be uncovered, in a few words")
    p: RareChance = Field(description="Probability that it is uncovered during this step, one of the scale's steps",
                          json_schema_extra={"enum": list(RARE_SCALE)})


class Assessment(BaseModel):
    odds: list[OrderOdds] = Field(description="One entry per order of every acting actor")


@lru_cache(maxsize=None)
def assessment_schema(open_events: bool, secrets: bool) -> type[Assessment]:
    """The assessment schema with only the parts a scenario uses, to keep it small for local models."""
    fields: dict = {}
    if open_events:
        fields["proposed_event"] = (ProposedEvent | None, Field(
            default=None, description="One plausible event not listed under Nature, or null"))
    if secrets:
        fields["discoveries"] = (list[Discovery], Field(
            default_factory=list, description="Private information an order could uncover this step, if any"))
    return create_model("Assessment", __base__=Assessment, **fields) if fields else Assessment


class StateChange(BaseModel):
    path: str = Field(description="The variable to change, written entity.variable")
    op: Literal["set", "add"] = Field(description="'add' adds value to a number, 'set' replaces the value")
    value: float | bool | str = Field(description="The amount to add or the new value")
    reason: str = Field(description="Why, in a few words")


class PrivateNote(BaseModel):
    actor: str = Field(description="Name of the actor")
    text: str = Field(description="What only this actor learns from this step")


class Resolution(BaseModel):
    narrative: str = Field(description="What happened in this step as everyone sees it, 2 to 5 sentences")
    changes: list[StateChange] = Field(description="Consequences, as changes to the declared variables")
    observations: list[PrivateNote] = Field(
        default_factory=list, description="What each actor privately learns, if anything"
    )


class ObjectiveScore(BaseModel):
    objective: str = Field(description="The objective, as written")
    score: Score = Field(description="0 = completely failed, 10 = fully achieved")
    evidence: str = Field(description="What in the history or final state supports the score")


class ActorVerdict(BaseModel):
    actor: str = Field(description="Name of the actor")
    scores: list[ObjectiveScore] = Field(description="One entry per objective of this actor")
    summary: str = Field(description="How this actor did overall, in one or two sentences")


class Verdict(BaseModel):
    summary: str = Field(description="What happened over the whole simulation, 2 to 5 sentences")
    actors: list[ActorVerdict] = Field(description="One entry per actor that has objectives")
