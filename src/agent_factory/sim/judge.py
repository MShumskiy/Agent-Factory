"""The judge: scores the end of a run against each actor's objectives."""

from __future__ import annotations

from statistics import mean

from agent_factory.llm.client import LLM
from agent_factory.sim import prompts
from agent_factory.sim.models import Verdict
from agent_factory.sim.scenario import Scenario


class Judge:
    def __init__(self, scenario: Scenario, llm: LLM, model: str, policy_choice: dict[str, str]):
        self.scenario = scenario
        self.llm = llm
        self.model = model
        self.briefing = prompts.briefing(scenario, policy_choice)

    @property
    def judged_actors(self) -> list[str]:
        return [name for name, actor in self.scenario.actors.items() if actor.objectives]

    def verdict(self, *, initial_state: str, final_state: str, history: str, ended_by: str, seed: int) -> Verdict:
        user = "\n\n".join([
            self.briefing,
            f"# Initial state\n{initial_state}",
            f"# History\n{history}",
            f"# Final state\n{final_state}",
            f"# How it ended\n{ended_by}",
            f"# Task\n{prompts.JUDGE_TASK.format(actors=', '.join(self.judged_actors))}",
        ])
        return self.llm.structured(Verdict, model=self.model, system=prompts.JUDGE_SYSTEM, user=user,
                                   agent="judge", temperature=0.0, seed=seed)


def actor_scores(verdict: Verdict, actors: list[str]) -> dict[str, float]:
    """Mean objective score per actor, for the actors the scenario expects a verdict on."""
    scores = {}
    for entry in verdict.actors:
        if entry.actor in actors and entry.scores:
            scores[entry.actor] = round(mean(s.score for s in entry.scores), 2)
    return scores
