"""Prompt text for actors, the game master and the judge, and the context blocks they share.

Kept in one place so the wording can be tuned for small local models without touching the engine.
"""

from __future__ import annotations

from janus.sim.models import Action
from janus.sim.scenario import Scenario
from janus.sim.state import AppliedChange
from janus.sim.trace import DiscoveryRoll, Node, Roll
from janus.sim.world import FiredEvent

ACTOR_SYSTEM = """\
You are {name}, an actor in a simulation. Stay in your role and pursue your objectives.
You only know what is shown to you; do not invent knowledge of the others' private plans.
Decide what you do in the current step. Be concrete and realistic about what can be done in one {unit}.
Reply with a JSON object: intent (one sentence), orders (1 to 5 concrete actions), rationale."""

GM_SYSTEM = """\
You are the game master of a simulation. You are neutral: you favour no actor and you do not make
things dramatic for the sake of it. You keep the world consistent with the setting, the current state
and what happened before. You see everything, including each actor's private information.
Reply with a JSON object matching the requested schema."""

ASSESS_TASK = """\
For each numbered order above, judge how likely it is to achieve what it intends during this step, given
the state, the setting and what the others are doing at the same time. Routine orders are likely to
work; ambitious, risky or contested ones less so. Answer with one step of this scale:
0.05 almost certainly not, 0.2 unlikely, 0.35 somewhat unlikely, 0.5 even chance,
0.65 somewhat likely, 0.8 likely, 0.95 almost certain.
Give exactly one entry per order, with the actor's name and the order's number."""

ASSESS_NO_ACTIONS = "No actor acts in this step, so `odds` is an empty list."

OPEN_EVENT_TASK = """\
Also consider whether something could plausibly happen during this step on its own (weather, third
parties, accidents, luck, the actors' circumstances) that is NOT one of the events listed under Nature:
those are rolled separately and must not be proposed again. If so, propose one such event with your
honest probability that it happens during this step, as one of 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5.
Most steps nothing unusual happens, so keep it low unless the situation makes it likely. If nothing
fits, set proposed_event to null."""

SECRETS_TASK = """\
If an order in this step could uncover another actor's private information (shown under Actors), list
it under discoveries: who could discover it, whose it is, what it is in a few words, and the probability
that it is uncovered during this step, as one of 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5. Only list what
these orders could really uncover; often nothing. Leave out anything already discovered."""

RESOLVE_TASK = """\
Resolve this step. The dice have already decided which orders worked and which events happened: respect
that order by order. Write what happened as everyone would see it. Express the consequences as changes
to the variables listed under the current state: use "add" for amounts and "set" for new values, and stay
within each variable's range or allowed values. Do not repeat the changes already applied in code.{derived}
If an actor learns something the others don't, add a private observation for that actor."""

SECRETS_RULE = """\
Never reveal an actor's private information in the narrative or to another actor, unless it was
discovered (this step or before). A discovery made this step goes only in the discoverer's private
observation."""

JUDGE_SYSTEM = """\
You are a neutral judge. You score how well each actor achieved each of its objectives by the end of
the simulation, using only the history and the final state shown. You favour no one.
Reply with a JSON object matching the requested schema."""

JUDGE_TASK = """\
For each actor with objectives ({actors}), score every objective from 0 (completely failed) to 10
(fully achieved), citing evidence from the history or the final state, then summarise the simulation."""


def briefing(scenario: Scenario, policy_choice: dict[str, str]) -> str:
    """Everything about the scenario, for the game master and the judge."""
    parts = [f"# Setting\n{scenario.setting.strip()}",
             f"# Time\nEach step is one {scenario.clock.unit}; {scenario.clock.horizon} steps in total."]
    actors = []
    for name, actor in scenario.actors.items():
        lines = [f"## {name}", f"Role: {actor.brief.strip() or '(no description)'}"]
        if actor.kind == "policy":
            lines.append(f"Follows a fixed plan: {policy_choice.get(name, '')}")
        if actor.private.strip():
            lines.append(f"Private information: {actor.private.strip()}")
        if about := scenario.about(name):
            lines.append(f"About them: {about}")
        if actor.objectives:
            lines.append("Objectives:\n" + _bullets(o.text for o in actor.objectives))
        actors.append("\n".join(lines))
    parts.append("# Actors\n" + "\n\n".join(actors))
    nature = scenario.nature
    nature_lines = []
    if nature.rules:
        nature_lines.append("Rules applied in code at the start of every step:\n"
                            + _bullets(f"{t} = {e}" for t, e in nature.rules.items()))
    if nature.derived:
        nature_lines.append("Values computed in code after every step:\n"
                            + _bullets(f"{t} = {e}" for t, e in nature.derived.items()))
    if nature.events:
        nature_lines.append("Events that can happen, with their probability per step:\n"
                            + _bullets(f"{e.name} (p={e.p:g}){': ' + e.description if e.description else ''}"
                                       for e in nature.events))
    if nature_lines:
        parts.append("# Nature\n" + "\n\n".join(nature_lines))
    return "\n\n".join(parts)


def actor_prompt(scenario: Scenario, name: str, visible_state: str, history: str,
                 label: str, events: list[FiredEvent]) -> str:
    actor = scenario.actors[name]
    parts = [f"# Setting\n{scenario.setting.strip()}", f"# You are {name}\n{actor.brief.strip()}"]
    if actor.private.strip():
        parts.append(f"## Only you know\n{actor.private.strip()}")
    if about := scenario.about(name):
        parts.append(f"## About you\n{about}")
    if actor.objectives:
        parts.append("## Your objectives\n" + _bullets(o.text for o in actor.objectives))
    others = [f"{n}: {a.brief.strip()}" for n, a in scenario.actors.items() if n != name]
    if others:
        parts.append("# The others\n" + _bullets(others))
    parts.append(f"# What you can see of the current state\n{visible_state}")
    parts.append(f"# What has happened so far\n{history or 'This is the first step.'}")
    parts.append(f"# Now: {label}\n{events_text(events, 'Events at the start of this step')}\n\n"
                 "What do you do in this step?")
    return "\n\n".join(parts)


def actor_history(nodes: list[Node], name: str, window: int) -> str:
    blocks = []
    for node in nodes[-window:]:
        lines = [f"## {node.label}"]
        if node.events:
            lines.append("Events: " + "; ".join(e.name for e in node.events))
        lines.append(node.narrative)
        if (action := node.actions.get(name)) is not None:
            lines.append(f"You did: {action.intent}")
            lines += [f"- {'worked' if r.success else 'did not work'}: {r.text}" for r in node.rolls_for(name)]
        if note := node.observations.get(name):
            lines.append(f"Only you learned: {note}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def full_history(nodes: list[Node], window: int | None = None) -> str:
    """The history as the game master and judge see it, with quiet steps folded together."""
    blocks: list[str] = []
    quiet_run: list[Node] = []

    def flush() -> None:
        if quiet_run:
            span = quiet_run[0].label if len(quiet_run) == 1 else f"{quiet_run[0].label} to {quiet_run[-1].label}"
            blocks.append(f"## {span}\nNothing notable happened (only the rules applied).")
            quiet_run.clear()

    for node in nodes if window is None else nodes[-window:]:
        if node.quiet:
            quiet_run.append(node)
            continue
        flush()
        lines = [f"## {node.label}"]
        if node.events:
            lines.append("Events: " + "; ".join(e.name for e in node.events))
        for actor, action in node.actions.items():
            worked, total = node.worked(actor)
            lines.append(f"{actor}: {action.intent} ({worked} of {total} orders worked)")
        lines.append(node.narrative)
        for found in node.discoveries:
            if found.success:
                lines.append(f"{found.by} discovered {found.of}'s private information: {found.what}")
        for actor, note in node.observations.items():
            lines.append(f"Privately, {actor} learned: {note}")
        blocks.append("\n".join(lines))
    flush()
    return "\n\n".join(blocks)


def actions_text(actions: dict[str, Action], rolls: list[Roll] | None = None) -> str:
    """Each actor's action with numbered orders, and each order's outcome once the dice are rolled."""
    outcome = {(r.actor, r.order): r for r in rolls or []}
    blocks = []
    for actor, action in actions.items():
        lines = [f"## {actor}", f"Intent: {action.intent}", "Orders:"]
        for i, order in enumerate(action.orders, 1):
            if (roll := outcome.get((actor, i))) is None:
                lines.append(f"{i}. {order}")
            else:
                verdict = "WORKED" if roll.success else "DID NOT WORK"
                lines.append(f"{i}. [{verdict}, p={roll.p:.2f}] {order}")
        blocks.append("\n".join(lines))
    if rolls:
        blocks.append("An order that did not work does not achieve what it intends: it may partly happen, "
                      "stall or backfire.")
    return "\n\n".join(blocks)


def discoveries_text(discoveries: list[DiscoveryRoll]) -> str:
    made = [f"{d.by} discovers {d.of}'s: {d.what}" for d in discoveries if d.success]
    missed = [f"{d.by} does not discover {d.of}'s: {d.what}" for d in discoveries if not d.success]
    parts = []
    if made:
        parts.append("Discovered this step:\n" + _bullets(made))
    if missed:
        parts.append("Not discovered this step (keep it hidden):\n" + _bullets(missed))
    return "\n\n".join(parts) or "Nothing was discovered this step."


def events_text(events: list[FiredEvent], heading: str) -> str:
    if not events:
        return f"{heading}: nothing unusual."
    return f"{heading}:\n" + _bullets(f"{e.name}{': ' + e.description if e.description else ''}" for e in events)


def changes_text(changes: list[AppliedChange]) -> str:
    moved = [c for c in changes if c.changed]
    if not moved:
        return "none"
    return _bullets(f"{c.path}: {c.before} -> {c.after} ({c.cause})" for c in moved)


def _bullets(items) -> str:
    return "\n".join(f"- {item}" for item in items)
