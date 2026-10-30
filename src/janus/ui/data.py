"""What the UI shows, built from scenario files and trees as plain JSON-ready dicts."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from janus.llm.privacy import PrivacyError, effective_mode
from janus.sim.engine import ENGINE_VERSION, resolve_models
from janus.sim.scenario import Scenario, ScenarioError, load_scenario
from janus.sim.trace import Node
from janus.sim.tree import CONTINUATIONS, TreeStore, leaf_score, outcome, tree_file


class Catalog:
    """The scenarios in a folder, by name, reloaded whenever a file changes."""

    def __init__(self, folder: Path, out: Path):
        self.folder = folder
        self.out = out
        self._cache: dict[Path, tuple[float, Scenario | ScenarioError]] = {}

    def entries(self) -> list[tuple[Path, Scenario | ScenarioError]]:
        result = []
        for path in sorted(self.folder.glob("*.yaml")):
            mtime = path.stat().st_mtime
            cached = self._cache.get(path)
            if cached is None or cached[0] != mtime:
                try:
                    loaded: Scenario | ScenarioError = load_scenario(path)
                except ScenarioError as e:
                    loaded = e
                cached = self._cache[path] = (mtime, loaded)
            result.append((path, cached[1]))
        return result

    def get(self, name: str) -> Scenario:
        for _, scenario in self.entries():
            if isinstance(scenario, Scenario) and scenario.name == name:
                return scenario
        raise KeyError(name)

    def tree_path(self, scenario: Scenario) -> Path:
        return tree_file(self.out, scenario, resolve_models(scenario))


def scenario_summary(catalog: Catalog, path: Path, scenario: Scenario | ScenarioError) -> dict:
    if isinstance(scenario, ScenarioError):
        return {"file": path.name, "error": str(scenario)}
    try:
        privacy = effective_mode(None, scenario.privacy, scenario.sensitive).value
    except PrivacyError as e:
        privacy = f"invalid: {e}"
    llm_actors = [n for n, a in scenario.actors.items() if a.kind == "llm"]
    options = next((a.options for a in scenario.actors.values() if a.kind == "policy"), [])
    summary = {
        "file": path.name,
        "name": scenario.name,
        "title": scenario.title or scenario.name,
        "against": "nature" if not llm_actors else "actors",
        "llm_actors": llm_actors,
        "options": [o.label for o in options],
        "clock": {"unit": scenario.clock.unit, "horizon": scenario.clock.horizon},
        "privacy": privacy,
        "sensitive": scenario.sensitive,
        "models": resolve_models(scenario).as_dict(),
        "tree": None,
    }
    file = catalog.tree_path(scenario)
    if file.exists():
        with TreeStore(file) as store:
            roots = store.roots()
            coverages = [outcome(store, root_id).coverage for root_id, _ in roots]
            last = store.db.execute("SELECT MAX(created_at) FROM nodes").fetchone()[0]
            summary["tree"] = {"file": file.name, "nodes": store.count(), "updated": last,
                               "coverage": round(mean(coverages), 3) if coverages else 0.0}
    return summary


def describe(node: Node, scenario: Scenario) -> str:
    """A few words for a step: its events and how the actors' orders went."""
    parts = [e.name for e in node.events]
    for actor in node.actions:
        worked, total = node.worked(actor)
        if scenario.actors[actor].kind == "policy":
            parts.append("plan worked" if worked == total else "plan did not work")
        else:
            parts.append(f"{actor} {worked}/{total} orders worked")
    return ", ".join(parts) or f"quiet {scenario.clock.unit}"


def _node(node: Node, scenario: Scenario, parent: str, p: float, path_p: float, kind: str,
          via: dict | None = None) -> dict:
    return {
        "id": node.id, "parent": parent, "step": node.step, "label": node.label, "kind": kind,
        "p": round(p, 4), "path_p": round(path_p, 4), "what": describe(node, scenario),
        "terminal": node.terminal, "via": via,
        "events": [{"name": e.name, "p": e.p, "source": e.source} for e in node.events],
        "orders": [{"actor": r.actor, "text": r.text, "p": r.p, "worked": r.success} for r in node.rolls],
        "discoveries": [d.model_dump() for d in node.discoveries if d.success],
        "narrative": node.narrative, "observations": node.observations,
        "scores": node.scores, "score": {a: round(s, 2) for a, s in leaf_score(node).items()},
        "state": {f"{entity}.{var}": value for entity, values in node.state.items() for var, value in values.items()},
    }


def tree_payload(catalog: Catalog, scenario: Scenario, opponents: str = "expected",
                 focal: str | None = None) -> dict | None:
    """Every option of the scenario with its outcome and explored nodes. Each projection or rollout is
    collapsed into one end node, standing for its branch, as the roll-up treats it."""
    file = catalog.tree_path(scenario)
    if not file.exists():
        return None
    judged = [n for n, a in scenario.actors.items() if any(o.metric for o in a.objectives)]
    with TreeStore(file) as store:
        options = []
        for root_id, label in store.roots():
            nodes: list[dict] = []

            def walk(parent_id: str, parent_key: str, path_p: float) -> None:
                rows = store.children_with_origin(parent_id)
                explored = [child for child, origin in rows if origin not in CONTINUATIONS]
                if explored:
                    total = sum(c.p_branch for c in explored) or 1.0
                    for child in explored:
                        p = child.p_branch / total
                        nodes.append(_node(child, scenario, parent_key, p, path_p * p, "explore"))
                        walk(child.id, child.id, path_p * p)
                    return
                for child, _ in rows:
                    chain = [child]
                    while below := store.children(chain[-1].id):
                        chain.append(below[0])
                    share = 1.0 / len(rows)
                    nodes.append(_node(chain[-1], scenario, parent_key, share, path_p * share, "end", via={
                        "steps": len(chain), "events": [e.name for n in chain for e in n.events]}))

            walk(root_id, "root", 1.0)
            result = outcome(store, root_id, opponents, focal)
            options.append({
                "root": root_id, "label": label or scenario.title or scenario.name,
                "expected": result.expected, "worst10": result.worst10, "p_end": result.p_end,
                "coverage": result.coverage, "leaves": result.leaves, "unfinished": result.unfinished,
                "nodes": nodes,
            })
        count = store.count()
    return {
        "name": scenario.name,
        "title": scenario.title or scenario.name,
        "clock": {"unit": scenario.clock.unit, "horizon": scenario.clock.horizon},
        "actors": judged,
        "objectives": {n: [o.text for o in a.objectives if o.metric] for n, a in scenario.actors.items()
                       if n in judged},
        "tree": {"file": file.name, "nodes": count, "engine": ENGINE_VERSION},
        "options": options,
    }
