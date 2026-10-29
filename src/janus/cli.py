"""Command line: `janus check | run | explore | tree SCENARIO` and `janus ui`."""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

from janus.llm.client import LLMError, LLMPClient, LLMPSettings
from janus.llm.privacy import PrivacyError, PrivacyMode, check_base_url, effective_mode
from janus.sim.engine import RoleModels, Simulation, check_models, resolve_models
from janus.sim.expr import ExprError
from janus.sim.report import write_run
from janus.sim.scenario import Scenario, ScenarioError, load_scenario
from janus.sim.trace import Node
from janus.sim.tree import Explorer, ExploreSettings, TreeStore, leaf_score, outcome, tree_file


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ScenarioError, PrivacyError, LLMError, ExprError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="janus", description="Decision simulations: how does each option hold up?")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="validate a scenario and show its settings, without calling a model")
    _common(check)
    check.set_defaults(handler=_check)

    run = commands.add_parser("run", help="run a scenario once, along one sampled path")
    _common(run)
    _running(run)
    run.add_argument("--option", type=int, default=0, help="which option policy actors follow (default 0)")
    run.add_argument("--steps", type=int, help="stop after this many steps instead of the scenario's horizon")
    run.add_argument("--no-judge", action="store_true", help="skip the judge's verdict and summary")
    run.set_defaults(handler=_run)

    explore = commands.add_parser("explore", help="grow the scenario's ramification tree within a budget")
    _common(explore)
    _running(explore)
    defaults = ExploreSettings()
    explore.add_argument("--option", type=int, action="append",
                         help="option to explore; repeat for several (default: all)")
    explore.add_argument("--depth", type=int, default=defaults.depth, help=f"steps expanded with branching (default {defaults.depth})")
    explore.add_argument("--branches", type=int, default=defaults.branches,
                         help=f"chance outcomes tried per node: the most likely, then the next most likely (default {defaults.branches})")
    explore.add_argument("--candidates", type=int, default=defaults.candidates,
                         help=f"versions of the LLM actors' decisions per node (default {defaults.candidates})")
    explore.add_argument("--rollouts", type=int, default=defaults.rollouts,
                         help=f"sampled continuations with luck per branch, after its most likely one (default {defaults.rollouts})")
    explore.add_argument("--budget", type=int, default=defaults.budget,
                         help=f"model requests to spend; cached replies are free (default {defaults.budget})")
    _comparison(explore)
    explore.set_defaults(handler=_explore)

    tree = commands.add_parser("tree", help="show the scenario's tree and compare its options, without calling a model")
    _common(tree)
    tree.add_argument("--out", default="runs", help="output folder (default: runs)")
    tree.add_argument("--show-depth", type=int, default=2, help="how many steps of the tree to print (default 2)")
    _comparison(tree)
    tree.set_defaults(handler=_tree)

    ui = commands.add_parser("ui", help="open the web UI on this machine")
    ui.add_argument("--scenarios", default="scenarios", help="folder of scenario files (default: scenarios)")
    ui.add_argument("--out", default="runs", help="output folder (default: runs)")
    ui.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1, this machine only)")
    ui.add_argument("--port", type=int, default=8765, help="port (default 8765)")
    ui.set_defaults(handler=_ui)
    return parser


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("scenario", help="path to a scenario YAML file")
    parser.add_argument("--privacy", choices=[m.value for m in PrivacyMode],
                        help="override the scenario's privacy mode (a sensitive profile always forces local)")
    parser.add_argument("--model", help="model for every LLM actor, as an LLMP id such as ollama/gemma3:4b")
    parser.add_argument("--gm-model", help="model for the game master")
    parser.add_argument("--judge-model", help="model for the judge")


def _running(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, help="random seed (default: a random one, printed)")
    parser.add_argument("--out", default="runs", help="output folder (default: runs)")
    parser.add_argument("--no-cache", action="store_true", help="always call the models, never replay cached replies")


def _comparison(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--opponents", choices=["expected", "worst"], default="expected",
                        help="weigh other actors' decision versions equally, or take the worst for --focal")
    parser.add_argument("--focal", help="the actor whose worst case --opponents worst looks for")


def _prepare(args: argparse.Namespace) -> tuple[Scenario, PrivacyMode, RoleModels]:
    scenario = load_scenario(args.scenario)
    requested = PrivacyMode(args.privacy) if args.privacy else None
    privacy = effective_mode(requested, scenario.privacy, scenario.sensitive)
    models = resolve_models(scenario, model=args.model, gm_model=args.gm_model, judge_model=args.judge_model)
    check_models(models, privacy)
    return scenario, privacy, models


def _client(args: argparse.Namespace, privacy: PrivacyMode, run_id: str) -> LLMPClient:
    out = Path(args.out)
    return LLMPClient(LLMPSettings.from_env(), privacy, run_id,
                      cache_dir=None if args.no_cache else out / ".cache")


def _check(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    requested = PrivacyMode(args.privacy) if args.privacy else None
    privacy = effective_mode(requested, scenario.privacy, scenario.sensitive)
    models = resolve_models(scenario, model=args.model, gm_model=args.gm_model, judge_model=args.judge_model)
    nature = scenario.nature
    kinds = ", ".join(f"{name} ({actor.kind})" for name, actor in scenario.actors.items())
    print(f"Scenario: {scenario.title or scenario.name} ({scenario.name})")
    print(f"Actors:   {kinds}")
    print(f"Nature:   {len(nature.rules)} rules, {len(nature.derived)} derived values, "
          f"{len(nature.events)} events, open events {'on' if nature.open_events else 'off'}"
          + (f", game master review every {nature.review_every} steps" if nature.review_every else ""))
    print(f"Clock:    {scenario.clock.horizon} steps of one {scenario.clock.unit}")
    reason = "forced by a sensitive profile" if scenario.sensitive else "from --privacy" if args.privacy else "from the scenario"
    print(f"Privacy:  {privacy.value} ({reason})")
    print("Models:   " + ", ".join(f"{role} {model}" for role, model in models.as_dict().items()))
    for name, actor in scenario.actors.items():
        for i, option in enumerate(actor.options):
            print(f"Option {i} for {name}: {option.label}")
        for objective in actor.objectives:
            how = f"metric {objective.metric}" if objective.metric else "judge only (not counted in trees)"
            print(f"Objective of {name}: {objective.text} [{how}]")
    check_models(models, privacy)
    try:
        settings = LLMPSettings.from_env()
    except LLMError as e:
        print(f"LLMP:     not configured ({e})")
    else:
        check_base_url(settings.base_url, privacy)
        print(f"LLMP:     {settings.base_url}")
    print("OK: the scenario is valid and its models are allowed by its privacy mode.")
    return 0


def _run(args: argparse.Namespace) -> int:
    scenario, privacy, models = _prepare(args)
    seed = _seed(args)
    has_policy = any(a.kind == "policy" for a in scenario.actors.values())
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-s{seed}" + (f"-o{args.option}" if has_policy else "")
    out = Path(args.out)
    simulation = Simulation(scenario, _client(args, privacy, run_id), models, privacy=privacy,
                            run_id=run_id, option=args.option)
    print(f"{scenario.title or scenario.name}: run {run_id}, privacy {privacy.value}")
    if simulation.option_label:
        print(f"Option: {simulation.option_label}")
    result = simulation.run(seed=seed, steps=args.steps, judge=not args.no_judge, on_step=_print_step)
    trace, report = write_run(result, scenario, out / scenario.name / run_id)
    with TreeStore(tree_file(out, scenario, models)) as store:
        store.add_root(simulation.root_id, simulation.option_label)
        for node in result.nodes:
            store.add(node, simulation.root_id, "run")
        store.record_run(run_id, "run", simulation.root_id, result.nodes[-1].id, seed)
    print(f"\nEnded: {result.ended_by}")
    for actor, score in result.scores.items():
        print(f"  {actor}: {score}/10")
    print(f"Model requests: {result.model_requests}, from cache: {result.cache_hits}")
    print(f"Report: {report}\nTrace:  {trace}\nTree:   {store.file}")
    return 0


def _explore(args: argparse.Namespace) -> int:
    scenario, privacy, models = _prepare(args)
    seed = _seed(args)
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-explore-s{seed}"
    settings = ExploreSettings(depth=args.depth, branches=args.branches, candidates=args.candidates,
                               rollouts=args.rollouts, budget=args.budget)
    llm = _client(args, privacy, run_id)
    with TreeStore(tree_file(Path(args.out), scenario, models)) as store:
        explorer = Explorer(scenario, llm, models, privacy=privacy, store=store, settings=settings,
                            seed=seed, run_id=run_id, on_node=_print_node)
        print(f"{scenario.title or scenario.name}: exploring, privacy {privacy.value}, budget {settings.budget} requests")
        roots = explorer.explore(args.option)
        print(f"\nModel requests: {llm.requests}, from cache: {llm.cache_hits}. Tree: {store.file} ({store.count()} nodes)")
        for failure in explorer.failures:
            print(f"Skipped a branch: {_short(failure, 200)}")
        _print_outcomes(store, roots, args.opponents, args.focal)
    return 0


def _tree(args: argparse.Namespace) -> int:
    scenario, _, models = _prepare(args)
    path = tree_file(Path(args.out), scenario, models)
    if not path.exists():
        print(f"No tree yet for {scenario.name} with these models ({path}). Run `janus explore` first.")
        return 1
    with TreeStore(path) as store:
        roots = {root_id: option or scenario.title or scenario.name for root_id, option in store.roots()}
        for root_id, label in roots.items():
            print(f"\n{label}")
            _print_subtree(store, root_id, 1, args.show_depth)
        _print_outcomes(store, roots, args.opponents, args.focal)
    return 0


def _ui(args: argparse.Namespace) -> int:
    import uvicorn

    from janus.ui.server import create_app

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: listening on {args.host} lets other machines read every scenario, tree and profile value "
              "this UI shows; there is no login", file=sys.stderr)
    app = create_app(Path(args.scenarios), Path(args.out))
    print(f"JANUS on http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _seed(args: argparse.Namespace) -> int:
    return args.seed if args.seed is not None else random.SystemRandom().randrange(1_000_000)


def _print_step(node: Node) -> None:
    if node.quiet:
        print(f"{node.label}: quiet")
        return
    print(f"{node.label}  [{node.model_calls} model calls]")
    for event in node.events:
        print(f"  event: {event.name}" + (" (proposed by the game master)" if event.source == "open" else ""))
    for actor, action in node.actions.items():
        worked, total = node.worked(actor)
        print(f"  {actor}: {_short(action.intent)} -> {worked} of {total} orders worked")


def _print_node(node: Node, origin: str) -> None:
    print(f"  + {_describe(node)}" + ("" if origin == "explore" else f"  ({origin})"))


def _describe(node: Node) -> str:
    parts = [f"{node.label} p={node.p_branch:.2f}"]
    for actor in node.actions:
        worked, total = node.worked(actor)
        parts.append(f"{actor} {worked}/{total}")
    if node.events:
        parts.append("events: " + ", ".join(e.name for e in node.events))
    if scores := leaf_score(node):
        parts.append("score " + ", ".join(f"{a} {s:.1f}" for a, s in scores.items()))
    if node.terminal:
        parts.append("END" if node.terminal == "horizon" else node.terminal)
    return " | ".join(parts)


def _print_subtree(store: TreeStore, parent_id: str, depth: int, max_depth: int) -> None:
    if depth > max_depth:
        return
    for child in store.children(parent_id):
        print("  " * depth + _describe(child))
        _print_subtree(store, child.id, depth + 1, max_depth)


def _print_outcomes(store: TreeStore, roots: dict[str, str], opponents: str, focal: str | None) -> None:
    print("\nOptions compared (objective scores out of 10, weighted by probability):")
    for root_id, label in roots.items():
        result = outcome(store, root_id, opponents, focal)
        print(f"  {label}")
        if not result.leaves:
            print("    not explored yet")
            continue
        for actor, value in result.expected.items():
            print(f"    {actor}: expected {value:.1f}, worst 10% {result.worst10.get(actor, value):.1f}")
        unfinished = f" ({result.unfinished} stop early: budget)" if result.unfinished else ""
        print(f"    ends early {result.p_end:.0%} · {result.leaves} leaves{unfinished} · "
              f"explored paths cover {result.coverage:.0%} of the probability")


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
