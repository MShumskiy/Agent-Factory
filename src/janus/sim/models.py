"""What the models are asked to return, as pydantic models.

These become JSON schemas in `response_format`. They are kept flat and small because local models
follow simple schemas far more reliably, and parsing is lenient where small models tend to slip
(probabilities given as percentages, scores given as floats).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, create_model


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


class Action(BaseModel):
    intent: str = Field(description="One sentence: what you are trying to achieve in this step")
    orders: list[str] = Field(description="The concrete things you do in this step, 1 to 5 items")
    rationale: str = Field(description="Why, in one or two sentences")


class OrderOdds(BaseModel):
    actor: str = Field(description="Name of the actor who gave the order")
    order: int = Field(description="Number of the order, as listed under the actor's action")
    p_success: Probability = Field(description="Probability from 0 to 1 that this order achieves what it intends")
    rationale: str = Field(description="Why this probability, in one sentence")


class ProposedEvent(BaseModel):
    name: str = Field(description="Short name of the event")
    description: str = Field(description="What would happen, in one or two sentences")
    p: Probability = Field(description="Probability from 0 to 1 that it happens during this step")


class Discovery(BaseModel):
    by: str = Field(description="The actor who could discover it")
    of: str = Field(description="The actor whose private information it is")
    what: str = Field(description="The private fact that could be uncovered, in a few words")
    p: Probability = Field(description="Probability from 0 to 1 that it is uncovered during this step")


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
