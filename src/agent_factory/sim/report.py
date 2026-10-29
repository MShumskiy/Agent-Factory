"""Writing a run to disk: `trace.jsonl` for machines, `report.md` for people."""

from __future__ import annotations

import json
from pathlib import Path

from agent_factory.sim.scenario import Scenario
from agent_factory.sim.state import AppliedChange, WorldState
from agent_factory.sim.trace import Node, RunResult


def write_run(result: RunResult, scenario: Scenario, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = out_dir / "trace.jsonl"
    with trace.open("w", encoding="utf-8") as f:
        _line(f, {"type": "run", **result.info.model_dump(), "initial_state": result.initial_state})
        for node in result.nodes:
            _line(f, {"type": "node", **node.model_dump()})
        _line(f, {"type": "end", "ended_by": result.ended_by,
                  "verdict": result.verdict.model_dump() if result.verdict else None,
                  "scores": result.scores, "model_requests": result.model_requests,
                  "cache_hits": result.cache_hits})
    report = out_dir / "report.md"
    report.write_text(render_report(result, scenario), encoding="utf-8")
    return trace, report


def render_report(result: RunResult, scenario: Scenario) -> str:
    info = result.info
    meta = [f"scenario `{info.scenario}`", f"run `{info.run_id}`", f"seed {info.seed}", f"privacy: {info.privacy}"]
    if info.option:
        meta.append(f"option: {info.option}")
    out = [f"# {info.title}", "", " · ".join(meta), "",
           "Models: " + ", ".join(f"{role} `{model}`" for role, model in info.models.items()), "",
           f"Ended: {result.ended_by}.", ""]

    if result.verdict:
        out += ["## Outcome", "", result.verdict.summary, "",
                "| Actor | Objective | Score | Evidence |", "|---|---|---|---|"]
        for entry in result.verdict.actors:
            for score in entry.scores:
                out.append(f"| {entry.actor} | {_cell(score.objective)} | {score.score} | {_cell(score.evidence)} |")
        out.append("")
        for entry in result.verdict.actors:
            mean = result.scores.get(entry.actor)
            out.append(f"- **{entry.actor}**{f' ({mean}/10)' if mean is not None else ''}: {entry.summary}")
        out.append("")

    out += ["## Steps", ""]
    listed = {e.name.strip().lower() for e in scenario.nature.events}
    for node in result.nodes:
        out += _render_node(node, listed)
    final = WorldState(scenario.state, result.nodes[-1].state if result.nodes else result.initial_state)
    out += ["## Final state", "", "```", final.render(), "```", "",
            f"Model requests: {result.model_requests}; replies served from cache: {result.cache_hits}.", ""]
    return "\n".join(out)


def _render_node(node: Node, listed: set[str]) -> list[str]:
    if node.quiet:
        moved = _inline(node.rule_changes + node.derived_changes)
        return [f"**{node.label}**: quiet.{' ' + moved if moved else ''}", ""]
    out = [f"### {node.label}", ""]
    for event in node.events:
        origin = "listed event" if event.source == "table" else "proposed by the game master"
        out.append(f"- Event: **{event.name}** ({origin}, p {event.p:.2f}, rolled {event.roll:.2f})"
                   + (f": {event.description}" if event.description else ""))
    proposed = node.proposed_event
    if proposed and all(e.name != proposed.name for e in node.events):
        out.append(f"- Proposed by the game master but did not happen: {proposed.name} (p {proposed.p:.2f})"
                   + (" - already a listed event, so not rolled again" if proposed.name.strip().lower() in listed else ""))
    for actor, action in node.actions.items():
        roll = node.roll_for(actor)
        outcome = (f" (p {roll.p:.2f}, rolled {roll.roll:.2f}: **{'worked' if roll.success else 'did not work'}**)"
                   if roll else "")
        out.append(f"- **{actor}**: {action.intent}{outcome}")
        out += [f"  - {order}" for order in action.orders]
    out += ["", node.narrative, ""]
    for actor, note in node.observations.items():
        out += [f"> Privately, {actor}: {_one_line(note)}", ""]
    changes = node.rule_changes + [c for e in node.events for c in e.changes] + node.gm_changes + node.derived_changes
    shown = [c for c in changes if c.changed or c.note]
    if shown:
        out.append("Changes:")
        out += [f"- `{c.path}` {c.before} → {c.after} ({c.cause})" + (f": {c.note}" if c.note else "")
                for c in shown]
        out.append("")
    return out


def _inline(changes: list[AppliedChange]) -> str:
    return "; ".join(f"`{c.path}` {c.before} → {c.after}" for c in changes if c.changed)


def _cell(text: str) -> str:
    return _one_line(text).replace("|", "\\|")


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _line(f, record: dict) -> None:
    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
