from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from janus.sim.scenario import Scenario, load_scenario

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"


@dataclass
class Call:
    schema: str
    model: str
    system: str
    user: str
    agent: str
    seed: int


class FakeLLM:
    """A scripted model: each schema name maps to a function of the call that returns the reply."""

    def __init__(self, **handlers: Callable[[Call], object]):
        self.handlers = handlers
        self.calls: list[Call] = []

    def structured(self, schema, *, model, system, user, agent, temperature, seed):
        call = Call(schema.__name__, model, system, user, agent, seed)
        self.calls.append(call)
        handler = self.handlers.get(schema.__name__)
        if handler is None:
            raise AssertionError(f"unexpected {schema.__name__} call from {agent}")
        reply = handler(call)
        return schema.model_validate(reply) if isinstance(reply, dict) else reply

    def by_agent(self, agent: str) -> list[Call]:
        return [c for c in self.calls if c.agent == agent]


@pytest.fixture
def tank() -> Scenario:
    return load_scenario(SCENARIOS / "tank_pincer.yaml")


@pytest.fixture
def amp() -> Scenario:
    return load_scenario(SCENARIOS / "guitar_amp_purchase.yaml")
