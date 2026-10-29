"""The game master: judges how likely orders are to work, then turns the dice into consequences.

It never decides outcomes itself. `assess` returns a probability per order (and, if the scenario allows,
one unlisted event and any private information an order could uncover, each with its probability); the
engine rolls; `resolve` describes what happened and expresses it as changes to declared variables, which
the engine checks before applying.
"""

from __future__ import annotations

from janus.llm.client import LLM
from janus.sim import prompts
from janus.sim.models import Action, Assessment, Resolution, assessment_schema
from janus.sim.scenario import Scenario
from janus.sim.state import AppliedChange
from janus.sim.trace import DiscoveryRoll, Roll
from janus.sim.world import FiredEvent


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
        seed: int,
    ) -> Assessment:
        open_events, secrets = self.scenario.nature.open_events, self.scenario.has_secrets and bool(actions)
        tasks = [prompts.ASSESS_TASK if actions else prompts.ASSESS_NO_ACTIONS]
        if open_events:
            tasks.append(prompts.OPEN_EVENT_TASK)
        if secrets:
            tasks.append(prompts.SECRETS_TASK)
        return self.llm.structured(
            assessment_schema(open_events, secrets),
            model=self.model,
            system=prompts.GM_SYSTEM,
            user=self._prompt(label, state_text, applied, history, events,
                              prompts.actions_text(actions) or "No actor acts in this step.",
                              "\n\n".join(tasks)),
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
        rolls: list[Roll],
        discoveries: list[DiscoveryRoll],
        seed: int,
    ) -> Resolution:
        derived = self.scenario.nature.derived
        derived_note = (f"\nDo not change {', '.join(derived)}: they are computed automatically." if derived else "")
        task = prompts.RESOLVE_TASK.format(derived=derived_note)
        extra = []
        if self.scenario.has_secrets:
            extra.append(f"# Private information\n{prompts.discoveries_text(discoveries)}")
            task += "\n" + prompts.SECRETS_RULE
        return self.llm.structured(
            Resolution,
            model=self.model,
            system=prompts.GM_SYSTEM,
            user=self._prompt(label, state_text, applied, history, events,
                              prompts.actions_text(actions, rolls) or "No actor acts in this step.",
                              task, extra),
            agent="gm.resolve",
            temperature=0.4,
            seed=seed,
        )

    def _prompt(self, label: str, state_text: str, applied: list[AppliedChange], history: str,
                events: list[FiredEvent], actions: str, task: str, extra: list[str] | None = None) -> str:
        return "\n\n".join([
            self.briefing,
            f"# Recent history\n{history or 'This is the first step.'}",
            f"# Current state (after this step's rules and events)\n{state_text}",
            f"# Already applied in code this step\n{prompts.changes_text(applied)}",
            f"# This step: {label}\n{prompts.events_text(events, 'Events')}",
            f"# Actions\n{actions}",
            *(extra or []),
            f"# Task\n{task}",
        ])
