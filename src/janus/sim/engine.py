"""Stepping a scenario forward, one transition at a time: along one sampled path, or into a tree.

Each step: rules, base-rate events, actors decide, the game master gives odds per order (and may propose
an unlisted event or a discovery of private information), the dice decide, the game master resolves,
derived values are recomputed, and the step becomes a node. Steps where nobody acts and nothing happens
make no model calls.

Chance goes through one `Dice` per step. `run` samples it from the run seed and the path; the tree
explorer also forces the most likely outcome, or samples other branches from the same node. The models'
seeds come from the node, not the run, so a node always gets the same decisions (replayed from the
cache) and branches differ only where chance differs.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from janus.llm.client import LLM
from janus.llm.privacy import PrivacyMode, check_model
from janus.sim import prompts
from janus.sim.actors import Actor, ActorView, LLMActor, PolicyActor
from janus.sim.gm import GameMaster
from janus.sim.judge import Judge, combined_scores
from janus.sim.models import Action, OrderOdds, StateChange
from janus.sim.scenario import Scenario, ScenarioError
from janus.sim.state import AppliedChange, WorldState
from janus.sim.trace import DiscoveryRoll, Node, Roll, RunInfo, RunResult, node_id
from janus.sim.world import Dice, FiredEvent, apply_formulas, clock_label, end_reason, metric_scores, roll_events

# Part of every tree's identity: bump it when a change to the engine or prompts changes what runs produce.
ENGINE_VERSION = "2"

DEFAULT_MODEL = "ollama/gemma3:4b"
DEFAULT_GM_MODEL = "ollama/gpt-oss:20b"
DEFAULT_JUDGE_MODEL = "ollama/gpt-oss:20b"

ExistingChild = Callable[[str, int, list[tuple[str, bool]]], Node | None]


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
    """Command line, then the scenario (actor, then default), then JANUS_* variables, then built-ins."""
    default = scenario.models.default or env.get("JANUS_MODEL") or DEFAULT_MODEL
    actors = {name: model or spec.model or default
              for name, spec in scenario.actors.items() if spec.kind == "llm"}
    return RoleModels(
        actors=actors,
        game_master=gm_model or scenario.models.game_master or env.get("JANUS_GM_MODEL") or DEFAULT_GM_MODEL,
        judge=judge_model or scenario.models.judge or env.get("JANUS_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL,
    )


def check_models(models: RoleModels, privacy: PrivacyMode) -> None:
    for model in {*models.actors.values(), models.game_master, models.judge}:
        check_model(model, privacy)


def derive_seed(*parts: object) -> int:
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def root_id(scenario: Scenario, policy_choice: dict[str, str]) -> str:
    return node_id("root", 0, [scenario.name, policy_choice])


def _is_listed(name: str, scenario: Scenario) -> bool:
    return name.strip().lower() in {e.name.strip().lower() for e in scenario.nature.events}


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
        privacy: PrivacyMode,
        run_id: str,
        option: int = 0,
        history_window: int = 6,
    ):
        check_models(models, privacy)
        self.scenario = scenario
        self.models = models
        self.privacy = privacy
        self.run_id = run_id
        self.history_window = history_window
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
        self.root_id = root_id(scenario, self.policy_choice)

    @property
    def option_label(self) -> str | None:
        return "; ".join(self.policy_choice.values()) or None

    def step(self, path: list[Node], dice: Dice, *, candidate: int = 0,
             existing: ExistingChild | None = None, reviews: bool = True) -> Node:
        """The node reached from the end of `path` with these dice. With `existing`, a child that already
        has the same chance outcomes is returned before the game master is asked to resolve it again.
        `reviews=False` skips the game master's periodic reviews (used when carrying branches forward)."""
        scenario, nature = self.scenario, self.scenario.nature
        parent = path[-1].id if path else self.root_id
        state = WorldState(scenario.state, path[-1].state if path else None)
        step = len(path) + 1
        label = clock_label(scenario.clock, step)
        calls_before = self.llm.calls

        rule_changes = apply_formulas(nature.rules, state, dice.rng, "rules")
        events = roll_events(nature.events, state, dice)
        applied = rule_changes + [c for e in events for c in e.changes]

        actions: dict[str, Action] = {}
        for name, actor in self.actors.items():
            visible = state.visible_to(name, scenario.actors, scenario.visibility)
            view = ActorView(step=step, label=label, visible_state=state.render(visible),
                             history=prompts.actor_history(path, name, self.history_window),
                             events=events, seed=derive_seed(parent, step, name, candidate), path=path)
            if (action := actor.decide(view)) is not None:
                actions[name] = action

        review = reviews and nature.open_events and nature.review_every > 0 and step % nature.review_every == 0
        history = prompts.full_history(path, self.history_window)
        rolls: list[Roll] = []
        discoveries: list[DiscoveryRoll] = []
        proposed = None
        if actions or review:
            assessment = self.gm.assess(
                label=label, state_text=state.render(), applied=applied, history=history, events=events,
                actions=actions, seed=derive_seed(parent, step, "gm.assess", candidate))
            rolls = self._roll_orders(actions, assessment.odds, dice)
            proposed = getattr(assessment, "proposed_event", None)
            # A listed event is already rolled from its base rate; rolling it again would double it.
            if proposed is not None and not _is_listed(proposed.name, scenario):
                hit, roll = dice.roll(f"open:{proposed.name}", proposed.p)
                if hit:
                    events.append(FiredEvent(name=proposed.name, description=proposed.description,
                                             p=proposed.p, roll=roll, source="open"))
            for found in getattr(assessment, "discoveries", []):
                if found.by in scenario.actors and found.of in scenario.actors and found.by != found.of:
                    hit, roll = dice.roll(f"discovery:{found.by}:{found.of}:{found.what}", found.p)
                    discoveries.append(DiscoveryRoll(by=found.by, of=found.of, what=found.what,
                                                     p=found.p, roll=roll, success=hit))

        if existing is not None and (known := existing(parent, candidate, dice.signature)) is not None:
            return known

        quiet = not actions and not events
        narrative, observations, gm_changes = "Nothing notable happened.", {}, []
        if not quiet:
            resolution = self.gm.resolve(
                label=label, state_text=state.render(), applied=applied, history=history, events=events,
                actions=actions, rolls=rolls, discoveries=discoveries,
                seed=derive_seed(parent, step, "gm.resolve", candidate))
            narrative = resolution.narrative
            observations = {o.actor: o.text for o in resolution.observations if o.actor in scenario.actors}
            gm_changes = [self._apply(change, state) for change in resolution.changes]
        derived_changes = apply_formulas(nature.derived, state, dice.rng, "derived")

        terminal = None
        if (condition := end_reason(scenario.end_when, state, dice.rng)) is not None:
            terminal = f"end: {condition}"
        elif step >= scenario.clock.horizon:
            terminal = "horizon"
        content = {"candidate": candidate, "signature": dice.signature,
                   "actions": {a: act.model_dump() for a, act in actions.items()}, "state": state.values}
        return Node(id=node_id(parent, step, content), parent_id=parent, step=step, label=label, quiet=quiet,
                    candidate=candidate, p_branch=dice.probability, most_likely=dice.most_likely,
                    signature=dice.signature, chances=[(label, p) for label, p, _ in dice.log],
                    rule_changes=rule_changes, events=events, actions=actions,
                    rolls=rolls, proposed_event=proposed, discoveries=discoveries, narrative=narrative,
                    observations=observations, gm_changes=gm_changes, derived_changes=derived_changes,
                    state=state.copy().values, scores=metric_scores(scenario.actors, state),
                    terminal=terminal, model_calls=self.llm.calls - calls_before)

    def run(self, *, seed: int, steps: int | None = None, judge: bool = True,
            on_step: Callable[[Node], None] | None = None) -> RunResult:
        """One path to the horizon (or `steps`, or an end condition), with chance sampled from `seed`."""
        scenario = self.scenario
        started_at = datetime.now().isoformat(timespec="seconds")
        limit = min(steps or scenario.clock.horizon, scenario.clock.horizon)
        path: list[Node] = []
        while len(path) < limit:
            parent = path[-1].id if path else self.root_id
            node = self.step(path, Dice(random.Random(f"{seed}/{parent}/{len(path) + 1}")))
            path.append(node)
            if on_step:
                on_step(node)
            if node.terminal and node.terminal.startswith("end: "):
                break

        last = path[-1]
        if last.terminal and last.terminal.startswith("end: "):
            ended_by = f"end condition met at the end of {last.label}: {last.terminal[5:]}"
        elif last.terminal == "horizon":
            ended_by = f"reached the end of {last.label}"
        else:
            ended_by = f"stopped after {last.label} (step limit)"

        verdict = None
        if judge and self.judge.judged_actors:
            verdict = self.judge.verdict(
                initial_state=self.initial_state().render(), final_state=WorldState(scenario.state, last.state).render(),
                history=prompts.full_history(path), ended_by=ended_by, seed=derive_seed(last.id, "judge"))
        info = RunInfo(run_id=self.run_id, scenario=scenario.name, title=scenario.title or scenario.name,
                       seed=seed, option=self.option_label, privacy=self.privacy.value,
                       models=self.models.as_dict(), started_at=started_at, root_id=self.root_id)
        return RunResult(info=info, initial_state=self.initial_state().values, nodes=path, ended_by=ended_by,
                         verdict=verdict, scores=combined_scores(scenario, last.scores, verdict),
                         model_requests=getattr(self.base_llm, "requests", self.llm.calls),
                         cache_hits=getattr(self.base_llm, "cache_hits", 0))

    def initial_state(self) -> WorldState:
        return WorldState(self.scenario.state)

    def _roll_orders(self, actions: dict[str, Action], odds: list[OrderOdds], dice: Dice) -> list[Roll]:
        by_order = {(o.actor.strip().lower(), o.order): o for o in odds}
        by_actor = {o.actor.strip().lower(): o for o in odds}  # if a model numbered the orders wrongly
        rolls = []
        for actor, action in actions.items():
            for i, text in enumerate(action.orders, 1):
                estimate = by_order.get((actor.lower(), i)) or by_actor.get(actor.lower())
                p = estimate.p_success if estimate else 0.5
                rationale = estimate.rationale if estimate else "The game master gave no estimate; 0.5 was used."
                hit, roll = dice.roll(f"order:{actor}:{i}", p)
                rolls.append(Roll(actor=actor, order=i, text=text, p=p, roll=roll, success=hit, rationale=rationale))
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
