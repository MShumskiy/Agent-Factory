"""The judge: scores the end of a run against each actor's objectives.

Objectives with a metric are scored in code (see `world.metric_scores`); the judge is for the ones that
need reading the history, and for a summary of what happened.
"""

from __future__ import annotations

from janus.llm.client import LLM
from janus.sim import prompts
from janus.sim.models import Verdict
from janus.sim.scenario import Scenario
from janus.sim.world import weighted_score


class Judge:
    def __init__(self, scenario: Scenario, llm: LLM, model: str, policy_choice: dict[str, str]):
        self.scenario = scenario
        self.llm = llm
        self.model = model
        self.briefing = prompts.briefing(scenario, policy_choice)

    @property
    def judged_actors(self) -> list[str]:
        return [name for name, actor in self.scenario.actors.items() if actor.objectives]

    @property
    def needed(self) -> bool:
        """Whether some objective can only be scored by reading the history."""
        return any(o.metric is None for a in self.scenario.actors.values() for o in a.objectives)

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


def combined_scores(scenario: Scenario, metrics: dict[str, dict[str, float]],
                    verdict: Verdict | None) -> dict[str, float]:
    """Weighted objective score per actor: the metric where there is one, else the judge's score."""
    judged: dict[tuple[str, str], float] = {}
    if verdict:
        for entry in verdict.actors:
            for score in entry.scores:
                judged[(entry.actor, _key(score.objective))] = score.score
    result = {}
    for name, actor in scenario.actors.items():
        scores = {}
        for objective in actor.objectives:
            if (value := metrics.get(name, {}).get(objective.text)) is not None:
                scores[objective.text] = value
            elif (value := judged.get((name, _key(objective.text)))) is not None:
                scores[objective.text] = value
        if (value := weighted_score(actor, scores)) is not None:
            result[name] = value
    return result


def _key(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".")
