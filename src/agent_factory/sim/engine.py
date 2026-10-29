"""One rollout of a scenario: the step loop that ties nature, actors, the game master and the judge.

Each step: rules, base-rate events, actors decide, the game master assesses odds (and may propose an
unlisted event), seeded dice decide, the game master resolves, derived values are recomputed, and
the step is recorded as a node. Steps where nobody acts and nothing happens make no model calls.
All randomness comes from a generator seeded by the run seed and the path so far, so a run is
reproducible and, later, branches from the same node are independent of each other.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from agent_factory.llm.client import LLM
from agent_factory.llm.privacy import PrivacyMode, check_model
from agent_factory.sim import prompts
from agent_factory.sim.actors import Actor, ActorView, LLMActor, PolicyActor
from agent_factory.sim.gm import GameMaster
from agent_factory.sim.judge import Judge, actor_scores
from agent_factory.sim.models import ActionOdds, StateChange
from agent_factory.sim.scenario import Scenario, ScenarioError
from agent_factory.sim.state import AppliedChange, WorldState
from agent_factory.sim.trace import Node, Roll, RunInfo, RunResult, node_id
from agent_factory.sim.world import FiredEvent, apply_formulas, clock_label, end_reason, roll_events

DEFAULT_MODEL = "ollama/gemma3:4b"
DEFAULT_GM_MODEL = "ollama/gpt-oss:20b"
DEFAULT_JUDGE_MODEL = "ollama/gpt-oss:20b"


@dataclass(frozen=True)
class RoleModels:
    actors: dict[str, str]
    game_master: str
    judge: str

    def as_dict(self) -> dict[str, str]:
        return {**{f"actor.{n}": m for n, m in self.actors.items()},
                "game_master": self.game_master, "judge": self.judge}


def resolve_models(
    scenario: Scenario,
    *,
    model: str | None = None,
    gm_model: str | None = None,
    judge_model: str | None = None,
    env: Mapping[str, str] = os.environ,
) -> RoleModels:
    """Command line, then the scenario (actor, then default), then AF_* variables, then built-ins."""
    default = scenario.models.default or env.get("AF_MODEL") or DEFAULT_MODEL
    actors = {name: model or spec.model or default
              for name, spec in scenario.actors.items() if spec.kind == "llm"}
    return RoleModels(
        actors=actors,
        game_master=gm_model or scenario.models.game_master or env.get("AF_GM_MODEL") or DEFAULT_GM_MODEL,
        judge=judge_model or scenario.models.judge or env.get("AF_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL,
    )


def check_models(models: RoleModels, privacy: PrivacyMode) -> None:
    for model in {*models.actors.values(), models.game_master, models.judge}:
        check_model(model, privacy)


def _is_listed(name: str, scenario: Scenario) -> bool:
    return name.strip().lower() in {e.name.strip().lower() for e in scenario.nature.events}


def derive_seed(*parts: object) -> int:
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class _CountingLLM:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.calls = 0

    def structured(self, schema, **kwargs):
        self.calls += 1
        return self.llm.structured(schema, **kwargs)


class Simulation:
    def __init__(
        self,
        scenario: Scenario,
        llm: LLM,
        models: RoleModels,
        *,
        seed: int,
        privacy: PrivacyMode,
        run_id: str,
        option: int = 0,
        steps: int | None = None,
        history_window: int = 6,
        on_step: Callable[[Node], None] | None = None,
    ):
        check_models(models, privacy)
        self.scenario = scenario
        self.models = models
        self.seed = seed
        self.privacy = privacy
        self.run_id = run_id
        self.steps = steps or scenario.clock.horizon
        self.history_window = history_window
        self.on_step = on_step
        self.llm = _CountingLLM(llm)
        self.base_llm = llm

        self.policy_choice: dict[str, str] = {}
        self.actors: dict[str, Actor] = {}
        for name, spec in scenario.actors.items():
            if spec.kind == "policy":
                if not 0 <= option < len(spec.options):
                    raise ScenarioError(f"option {option} does not exist for {name}: "
                                        f"choose 0 to {len(spec.options) - 1}")
                self.policy_choice[name] = spec.options[option].label
                self.actors[name] = PolicyActor(name, spec.options[option])
            else:
                self.actors[name] = LLMActor(name, scenario, self.llm, models.actors[name])
        self.gm = GameMaster(scenario, self.llm, models.game_master, self.policy_choice)
        self.judge = Judge(scenario, self.llm, models.judge, self.policy_choice)
        self.root_id = node_id("root", 0, [scenario.name, self.policy_choice])

    def run(self) -> RunResult:
        scenario, nature = self.scenario, self.scenario.nature
        state = WorldState(scenario.state)
        initial = WorldState(scenario.state).values
        started_at = datetime.now().isoformat(timespec="seconds")
        nodes: list[Node] = []
        parent = self.root_id
        ended_by = None

        for step in range(1, self.steps + 1):
            rng = random.Random(f"{self.seed}/{parent}/{step}")
            calls_before = self.llm.calls
            label = clock_label(scenario.clock, step)

            rule_changes = apply_formulas(nature.rules, state, rng, "rules")
            events = roll_events(nature.events, state, rng)
            applied = rule_changes + [c for e in events for c in e.changes]

            actions = {}
            for name, actor in self.actors.items():
                visible = state.visible_to(name, scenario.actors, scenario.visibility)
                view = ActorView(step=step, label=label, visible_state=state.render(visible),
                                 history=prompts.actor_history(nodes, name, self.history_window),
                                 events=events, seed=derive_seed(self.seed, parent, step, name))
                if (action := actor.decide(view)) is not None:
                    actions[name] = action

            review = nature.open_events and nature.review_every > 0 and step % nature.review_every == 0
            history = prompts.full_history(nodes, self.history_window)
            rolls: list[Roll] = []
            proposed = None
            if actions or review:
                assessment = self.gm.assess(
                    label=label, state_text=state.render(), applied=applied, history=history,
                    events=events, actions=actions, allow_open_event=nature.open_events,
                    seed=derive_seed(self.seed, parent, step, "gm.assess"))
                rolls = self._roll(actions, assessment.odds, rng)
                proposed = getattr(assessment, "proposed_event", None)
                # A listed event is already rolled from its base rate; rolling it again would double it.
                if proposed is not None and not _is_listed(proposed.name, scenario):
                    roll = rng.random()
                    if roll < proposed.p:
                        events.append(FiredEvent(name=proposed.name, description=proposed.description,
                                                 p=proposed.p, roll=roll, source="open"))

            quiet = not actions and not events
            narrative, observations, gm_changes = "Nothing notable happened.", {}, []
            if not quiet:
                resolution = self.gm.resolve(
                    label=label, state_text=state.render(), applied=applied, history=history,
                    events=events, actions=actions,
                    outcomes={r.actor: (r.success, r.p) for r in rolls},
                    seed=derive_seed(self.seed, parent, step, "gm.resolve"))
                narrative = resolution.narrative
                observations = {o.actor: o.text for o in resolution.observations if o.actor in scenario.actors}
                gm_changes = [self._apply(change, state) for change in resolution.changes]
            derived_changes = apply_formulas(nature.derived, state, rng, "derived")

            content = {"events": [e.name for e in events],
                       "actions": {a: act.model_dump() for a, act in actions.items()},
                       "outcomes": {r.actor: r.success for r in rolls},
                       "state": state.values}
            node = Node(id=node_id(parent, step, content), parent_id=parent, step=step, label=label,
                        quiet=quiet, rule_changes=rule_changes, events=events, actions=actions,
                        rolls=rolls, proposed_event=proposed, narrative=narrative,
                        observations=observations, gm_changes=gm_changes,
                        derived_changes=derived_changes, state=state.copy().values,
                        model_calls=self.llm.calls - calls_before)
            nodes.append(node)
            if self.on_step:
                self.on_step(node)
            parent = node.id
            if (condition := end_reason(scenario.end_when, state, rng)) is not None:
                ended_by = f"end condition met at the end of {label}: {condition}"
                break

        if ended_by is None:
            ended_by = (f"reached the end of {nodes[-1].label}" if self.steps >= scenario.clock.horizon
                        else f"stopped after {nodes[-1].label} (step limit)")

        verdict, scores = None, {}
        if self.judge.judged_actors:
            verdict = self.judge.verdict(
                initial_state=WorldState(scenario.state, initial).render(),
                final_state=state.render(), history=prompts.full_history(nodes), ended_by=ended_by,
                seed=derive_seed(self.seed, parent, "judge"))
            scores = actor_scores(verdict, self.judge.judged_actors)

        info = RunInfo(run_id=self.run_id, scenario=scenario.name, title=scenario.title or scenario.name,
                       seed=self.seed, option="; ".join(self.policy_choice.values()) or None,
                       privacy=self.privacy.value, models=self.models.as_dict(),
                       started_at=started_at, root_id=self.root_id)
        return RunResult(info=info, initial_state=initial, nodes=nodes, ended_by=ended_by,
                         verdict=verdict, scores=scores,
                         model_requests=getattr(self.base_llm, "requests", self.llm.calls),
                         cache_hits=getattr(self.base_llm, "cache_hits", 0))

    def _roll(self, actions: dict, odds: list[ActionOdds], rng: random.Random) -> list[Roll]:
        by_actor = {o.actor.strip().lower(): o for o in odds}
        rolls = []
        for actor in actions:
            estimate = by_actor.get(actor.lower())
            p = estimate.p_success if estimate else 0.5
            rationale = estimate.rationale if estimate else "The game master gave no estimate; 0.5 was used."
            roll = rng.random()
            rolls.append(Roll(actor=actor, p=p, roll=roll, success=roll < p, rationale=rationale))
        return rolls

    def _apply(self, change: StateChange, state: WorldState) -> AppliedChange:
        cause = f"game master: {change.reason}"
        if change.path in self.scenario.nature.derived:
            before = state.get(change.path)
            return AppliedChange(path=change.path, before=before, after=before, cause=cause,
                                 note="computed by the scenario's rules, change ignored")
        if change.op == "add":
            return state.increment(change.path, change.value, cause)
        return state.assign(change.path, change.value, cause)
