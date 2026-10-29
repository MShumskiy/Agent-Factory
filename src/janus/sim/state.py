"""The world state: typed variables that only change through checked assignments.

Numbers are clamped to their bounds, strings with choices must match one of them, and unknown
variables are refused. Every change comes back as an `AppliedChange`, so the trace shows what
moved, why, and what was adjusted or ignored.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

from pydantic import BaseModel

from janus.sim.scenario import Scalar, VarSpec


class AppliedChange(BaseModel):
    path: str
    before: Scalar | None
    after: Scalar | None
    cause: str
    note: str | None = None

    @property
    def changed(self) -> bool:
        return self.before != self.after


class WorldState:
    def __init__(
        self,
        specs: dict[str, dict[str, VarSpec]],
        values: dict[str, dict[str, Scalar]] | None = None,
    ):
        self.specs = specs
        if values is None:
            values = {e: {n: s.value for n, s in vs.items()} for e, vs in specs.items()}
        self.values = copy.deepcopy(values)

    def copy(self) -> WorldState:
        return WorldState(self.specs, self.values)

    def spec(self, path: str) -> VarSpec | None:
        entity, _, name = path.partition(".")
        return self.specs.get(entity, {}).get(name)

    def get(self, path: str) -> Scalar:
        entity, _, name = path.partition(".")
        return self.values[entity][name]

    def assign(self, path: str, value: object, cause: str) -> AppliedChange:
        spec = self.spec(path)
        if spec is None:
            return AppliedChange(path=path, before=None, after=None, cause=cause,
                                 note="unknown variable, change ignored")
        before = self.get(path)
        after, note = _coerce(spec, value)
        if after is None:
            return AppliedChange(path=path, before=before, after=before, cause=cause, note=note)
        entity, _, name = path.partition(".")
        self.values[entity][name] = after
        return AppliedChange(path=path, before=before, after=after, cause=cause, note=note)

    def increment(self, path: str, amount: object, cause: str) -> AppliedChange:
        spec = self.spec(path)
        if spec is None or not spec.is_number:
            reason = "unknown variable" if spec is None else "only numbers can be added to"
            before = self.get(path) if spec else None
            return AppliedChange(path=path, before=before, after=before, cause=cause,
                                 note=f"{reason}, change ignored")
        number = _to_number(amount)
        if number is None:
            before = self.get(path)
            return AppliedChange(path=path, before=before, after=before, cause=cause,
                                 note=f"{amount!r} is not a number, change ignored")
        return self.assign(path, self.get(path) + number, cause)

    def visible_to(self, actor: str, actors: Iterable[str], visibility: str) -> dict[str, dict[str, Scalar]]:
        """What an actor can see: everything, or everything except other actors' variables."""
        if visibility == "full":
            return self.values
        hidden = set(actors) - {actor}
        return {e: vs for e, vs in self.values.items() if e not in hidden}

    def render(self, values: dict[str, dict[str, Scalar]] | None = None) -> str:
        """One line per variable, with its range or choices, for prompts and reports."""
        lines = []
        for entity, variables in (self.values if values is None else values).items():
            for name, value in variables.items():
                spec = self.specs[entity][name]
                lines.append(f"{entity}.{name} = {_fmt(value)}{_describe(spec)}")
        return "\n".join(lines)


def _coerce(spec: VarSpec, value: object) -> tuple[Scalar | None, str | None]:
    """The value to store and a note if it was adjusted; (None, note) if it can't be stored."""
    if spec.is_number:
        number = _to_number(value)
        if number is None:
            return None, f"{value!r} is not a number, change ignored"
        clamped = number
        if spec.min is not None:
            clamped = max(spec.min, clamped)
        if spec.max is not None:
            clamped = min(spec.max, clamped)
        note = None if clamped == number else f"{_fmt(number)} is out of range, clamped to {_fmt(clamped)}"
        return _tidy(clamped), note
    if isinstance(spec.value, bool):
        if isinstance(value, bool):
            return value, None
        text = str(value).strip().lower()
        if text in ("true", "yes"):
            return True, None
        if text in ("false", "no"):
            return False, None
        return None, f"{value!r} is not true or false, change ignored"
    text = str(value).strip()
    if spec.choices is not None:
        match = next((c for c in spec.choices if c.lower() == text.lower()), None)
        if match is None:
            return None, f"{text!r} is not one of {', '.join(spec.choices)}, change ignored"
        return match, None
    return text, None


def _to_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _tidy(number: float) -> int | float:
    number = round(number, 2)
    return int(number) if number.is_integer() else number


def _fmt(value: object) -> str:
    return str(_tidy(value)) if isinstance(value, float) else str(value)


def _describe(spec: VarSpec) -> str:
    parts = []
    if spec.choices:
        parts.append("one of: " + ", ".join(spec.choices))
    elif spec.min is not None or spec.max is not None:
        lo = "" if spec.min is None else _fmt(spec.min)
        hi = "" if spec.max is None else _fmt(spec.max)
        parts.append(f"range {lo}..{hi}")
    if spec.description:
        parts.append(spec.description)
    return f"  ({'; '.join(parts)})" if parts else ""
