"""The game master: judges how likely actions are to work, then turns the dice into consequences.

It never decides outcomes itself. `assess` returns probabilities (and, if allowed, one unlisted
event with its probability); the engine rolls; `resolve` describes what happened and expresses it as
changes to declared variables, which the engine checks before applying.
"""

from __future__ import annotations

from agent_factory.llm.client import LLM
from agent_factory.sim import prompts
from agent_factory.sim.models import Action, Assessment, AssessmentWithEvent, Resolution
from agent_factory.sim.scenario import Scenario
from agent_factory.sim.state import AppliedChange
from agent_factory.sim.world import FiredEvent


class GameMaster:
    def __init__(self, scenario: Scenario, llm: LLM, model: str, policy_choice: dict[str, str]):
        self.scenario = scenario
        self.llm = llm
        self.model = model
        self.briefing = prompts.briefing(scenario, policy_choice)

    def assess(
        self,
        *,
        label: str,
        state_text: str,
        applied: list[AppliedChange],
        history: str,
        events: list[FiredEvent],
        actions: dict[str, Action],
        allow_open_event: bool,
        seed: int,
    ) -> Assessment:
        task = (prompts.ASSESS_TASK.format(actors=", ".join(actions)) if actions
                else prompts.ASSESS_NO_ACTIONS)
        if allow_open_event:
            task += "\n\n" + prompts.OPEN_EVENT_TASK
        schema = AssessmentWithEvent if allow_open_event else Assessment
        return self.llm.structured(
            schema,
            model=self.model,
            system=prompts.GM_SYSTEM,
            user=self._prompt(label, state_text, applied, history, events,
                              prompts.actions_text(actions) or "No actor acts in this step.", task),
            agent="gm.assess",
            temperature=0.2,
            seed=seed,
        )

    def resolve(
        self,
        *,
        label: str,
        state_text: str,
        applied: list[AppliedChange],
        history: str,
        events: list[FiredEvent],
        actions: dict[str, Action],
        outcomes: dict[str, tuple[bool, float]],
        seed: int,
    ) -> Resolution:
        derived = self.scenario.nature.derived
        derived_note = (f"\nDo not change {', '.join(derived)}: they are computed automatically." if derived else "")
        return self.llm.structured(
            Resolution,
            model=self.model,
            system=prompts.GM_SYSTEM,
            user=self._prompt(label, state_text, applied, history, events,
                              prompts.actions_text(actions, outcomes) or "No actor acts in this step.",
                              prompts.RESOLVE_TASK.format(derived=derived_note)),
            agent="gm.resolve",
            temperature=0.4,
            seed=seed,
        )

    def _prompt(self, label: str, state_text: str, applied: list[AppliedChange], history: str,
                events: list[FiredEvent], actions: str, task: str) -> str:
        return "\n\n".join([
            self.briefing,
            f"# Recent history\n{history or 'This is the first step.'}",
            f"# Current state (after this step's rules and events)\n{state_text}",
            f"# Already applied in code this step\n{prompts.changes_text(applied)}",
            f"# This step: {label}\n{prompts.events_text(events, 'Events')}",
            f"# Actions\n{actions}",
            f"# Task\n{task}",
        ])
