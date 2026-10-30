"""Scenario files: the world, its actors, nature, and an optional personal profile.

A scenario is loaded from YAML, merged with its profile (whose values override the scenario's
defaults), and checked as a whole: every expression must parse and only read or write variables
that exist, so mistakes surface at load time instead of halfway through a run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from janus.llm.privacy import PrivacyMode
from janus.sim import expr

Scalar = int | float | bool | str


class ScenarioError(ValueError):
    """A scenario or profile file is invalid."""


class _Strict(BaseModel):
    """Unknown keys are errors, so a typo in a scenario file is reported instead of ignored."""

    model_config = ConfigDict(extra="forbid")


class VarSpec(_Strict):
    value: Scalar
    min: float | None = None
    max: float | None = None
    choices: list[str] | None = None
    description: str = ""

    @property
    def is_number(self) -> bool:
        return isinstance(self.value, (int, float)) and not isinstance(self.value, bool)

    @model_validator(mode="after")
    def _consistent(self) -> VarSpec:
        if (self.min is not None or self.max is not None) and not self.is_number:
            raise ValueError("min and max only apply to numbers")
        if self.choices is not None and self.value not in self.choices:
            raise ValueError(f"value {self.value!r} is not one of the choices {self.choices}")
        return self


def _var_specs(entities: object) -> object:
    """Accept `strength: 100` as shorthand for `strength: {value: 100}`."""
    if not isinstance(entities, dict):
        return entities
    return {
        entity: {
            name: spec if isinstance(spec, dict) else {"value": spec}
            for name, spec in (variables or {}).items()
        }
        for entity, variables in entities.items()
    }


class PolicyOption(_Strict):
    label: str
    steps: dict[int, str] = Field(default_factory=dict)
    retries: int = Field(default=2, ge=0)  # times a planned order that failed is tried again

    def order_at(self, step: int) -> str | None:
        return self.steps.get(step) if self.steps else (self.label if step == 1 else None)


class Objective(_Strict):
    """What an actor wants. A `metric` scores it in code from the state: true is 10, false 0, a number is
    clamped to 0..10. Objectives without one can only be scored by the judge."""

    text: str
    metric: str | None = None


class ActorSpec(_Strict):
    kind: Literal["llm", "policy"] = "llm"
    brief: str = ""
    private: str = ""
    objectives: list[Objective] = Field(default_factory=list)
    options: list[PolicyOption] = Field(default_factory=list)
    model: str | None = None

    @field_validator("options", mode="before")
    @classmethod
    def _options(cls, options: object) -> object:
        if isinstance(options, list):
            return [{"label": o} if isinstance(o, str) else o for o in options]
        return options

    @field_validator("objectives", mode="before")
    @classmethod
    def _objectives(cls, objectives: object) -> object:
        if isinstance(objectives, list):
            return [{"text": o} if isinstance(o, str) else o for o in objectives]
        return objectives

    @model_validator(mode="after")
    def _policy_has_options(self) -> ActorSpec:
        if self.kind == "policy" and not self.options:
            raise ValueError("a policy actor needs at least one option")
        return self


class EventSpec(_Strict):
    name: str
    p: float = Field(ge=0, le=1)
    when: str | None = None
    description: str = ""
    effects: dict[str, Scalar] = Field(default_factory=dict)


class NatureSpec(_Strict):
    rules: dict[str, str] = Field(default_factory=dict)
    derived: dict[str, str] = Field(default_factory=dict)
    events: list[EventSpec] = Field(default_factory=list)
    open_events: bool = False
    review_every: int = Field(default=0, ge=0)


class ClockSpec(_Strict):
    unit: str = "step"
    horizon: int = Field(ge=1)


class ModelsSpec(_Strict):
    default: str | None = None
    game_master: str | None = None
    judge: str | None = None


class Profile(_Strict):
    sensitive: bool = True
    actor: str | None = None
    about: str = ""
    state: dict[str, dict[str, VarSpec]] = Field(default_factory=dict)

    @field_validator("state", mode="before")
    @classmethod
    def _shorthand(cls, state: object) -> object:
        return _var_specs(state)


class Scenario(_Strict):
    name: str
    title: str = ""
    setting: str
    privacy: PrivacyMode = PrivacyMode.LOCAL
    profile: str | None = None
    clock: ClockSpec
    models: ModelsSpec = Field(default_factory=ModelsSpec)
    visibility: Literal["own", "full"] = "own"
    actors: dict[str, ActorSpec]
    state: dict[str, dict[str, VarSpec]]
    nature: NatureSpec = Field(default_factory=NatureSpec)
    end_when: list[str] = Field(default_factory=list)
    loaded_profile: Profile | None = Field(default=None, exclude=True)

    @field_validator("state", mode="before")
    @classmethod
    def _shorthand(cls, state: object) -> object:
        return _var_specs(state)

    @property
    def sensitive(self) -> bool:
        return self.loaded_profile is not None and self.loaded_profile.sensitive

    def about(self, actor: str) -> str:
        """The profile's free text, if the profile describes this actor."""
        profile = self.loaded_profile
        return profile.about.strip() if profile and profile.actor == actor else ""

    @property
    def has_secrets(self) -> bool:
        """Whether one actor could discover another's private information."""
        return len(self.actors) > 1 and any(a.private.strip() for a in self.actors.values())

    def fingerprint(self) -> str:
        """A hash of everything that shapes a run, including the profile's values and text."""
        content = {"scenario": self.model_dump(mode="json"),
                   "profile": self.loaded_profile.model_dump(mode="json") if self.loaded_profile else None}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    data = _read_yaml(path)
    try:
        scenario = Scenario.model_validate(data)
    except ValidationError as e:
        raise ScenarioError(f"{path}: {e}") from e
    if scenario.profile:
        profile_path = _find(scenario.profile, path)
        try:
            profile = Profile.model_validate(_read_yaml(profile_path))
        except ValidationError as e:
            raise ScenarioError(f"{profile_path}: {e}") from e
        _merge_profile(scenario, profile, profile_path)
    _check(scenario, path)
    return scenario


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ScenarioError(f"file not found: {path}") from None
    except yaml.YAMLError as e:
        raise ScenarioError(f"{path}: invalid YAML: {e}") from e
    if not isinstance(data, dict):
        raise ScenarioError(f"{path}: expected a mapping at the top level")
    return data


def _find(reference: str, scenario_path: Path) -> Path:
    """Profiles are found relative to the working directory, then to the scenario file."""
    candidate = Path(reference)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return scenario_path.parent / candidate


def _merge_profile(scenario: Scenario, profile: Profile, profile_path: Path) -> None:
    if profile.actor is not None and profile.actor not in scenario.actors:
        raise ScenarioError(f"{profile_path}: actor {profile.actor!r} is not in the scenario")
    for entity, variables in profile.state.items():
        target = scenario.state.setdefault(entity, {})
        for name, spec in variables.items():
            if name in target:
                merged = target[name].model_copy(update={"value": spec.value})
                try:
                    target[name] = VarSpec.model_validate(merged.model_dump())
                except ValidationError as e:
                    raise ScenarioError(f"{profile_path}: {entity}.{name}: {e}") from e
            else:
                target[name] = spec
    scenario.loaded_profile = profile


def _check(scenario: Scenario, path: Path) -> None:
    problems: list[str] = []
    known = {f"{e}.{v}" for e, variables in scenario.state.items() for v in variables}

    for name in scenario.actors:
        if not name.isidentifier() or name == "world":
            problems.append(f"actor name {name!r} must be an identifier other than 'world'")

    def check_expr(where: str, source: str) -> None:
        try:
            missing = expr.paths(source) - known
        except expr.ExprError as e:
            problems.append(f"{where}: {e}")
            return
        if missing:
            problems.append(f"{where}: unknown variable(s) {', '.join(sorted(missing))}")

    def check_target(where: str, target: str) -> None:
        if target not in known:
            problems.append(f"{where}: unknown variable {target}")

    nature = scenario.nature
    for target, source in {**nature.rules, **nature.derived}.items():
        check_target(f"rule {target}", target)
        check_expr(f"rule {target}", source)
    for event in nature.events:
        if event.when:
            check_expr(f"event {event.name!r} when", event.when)
        for target, effect in event.effects.items():
            check_target(f"event {event.name!r} effect", target)
            if isinstance(effect, str):
                check_expr(f"event {event.name!r} effect on {target}", effect)
    for condition in scenario.end_when:
        check_expr("end_when", condition)
    for name, actor in scenario.actors.items():
        for objective in actor.objectives:
            if objective.metric:
                check_expr(f"{name} objective {objective.text!r} metric", objective.metric)

    if problems:
        raise ScenarioError(f"{path}:\n  " + "\n  ".join(problems))
