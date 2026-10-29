"""Command line: `agent-factory check SCENARIO` and `agent-factory run SCENARIO`."""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

from agent_factory.llm.client import LLMError, LLMPClient, LLMPSettings
from agent_factory.llm.privacy import PrivacyError, PrivacyMode, check_base_url, effective_mode
from agent_factory.sim.engine import RoleModels, Simulation, check_models, resolve_models
from agent_factory.sim.expr import ExprError
from agent_factory.sim.report import write_run
from agent_factory.sim.scenario import Scenario, ScenarioError, load_scenario
from agent_factory.sim.trace import Node


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
    parser = argparse.ArgumentParser(prog="agent-factory", description="Run LLM-driven simulations.")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="validate a scenario and show its settings, without calling a model")
    _common(check)
    check.set_defaults(handler=_check)

    run = commands.add_parser("run", help="run a scenario once")
    _common(run)
    run.add_argument("--seed", type=int, help="random seed (default: a random one, printed)")
    run.add_argument("--option", type=int, default=0, help="which option policy actors follow (default 0)")
    run.add_argument("--steps", type=int, help="stop after this many steps instead of the scenario's horizon")
    run.add_argument("--out", default="runs", help="output folder (default: runs)")
    run.add_argument("--no-cache", action="store_true", help="always call the models, never replay cached replies")
    run.set_defaults(handler=_run)
    return parser


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("scenario", help="path to a scenario YAML file")
    parser.add_argument("--privacy", choices=[m.value for m in PrivacyMode],
                        help="override the scenario's privacy mode (a sensitive profile always forces local)")
    parser.add_argument("--model", help="model for every LLM actor, as an LLMP id such as ollama/gemma3:4b")
    parser.add_argument("--gm-model", help="model for the game master")
    parser.add_argument("--judge-model", help="model for the judge")


def _prepare(args: argparse.Namespace) -> tuple[Scenario, PrivacyMode, RoleModels]:
    scenario = load_scenario(args.scenario)
    requested = PrivacyMode(args.privacy) if args.privacy else None
    privacy = effective_mode(requested, scenario.privacy, scenario.sensitive)
    models = resolve_models(scenario, model=args.model, gm_model=args.gm_model, judge_model=args.judge_model)
    return scenario, privacy, models


def _check(args: argparse.Namespace) -> int:
    scenario, privacy, models = _prepare(args)
    nature = scenario.nature
    kinds = ", ".join(f"{name} ({actor.kind})" for name, actor in scenario.actors.items())
    print(f"Scenario: {scenario.title or scenario.name} ({scenario.name})")
    print(f"Actors:   {kinds}")
    print(f"Nature:   {len(nature.rules)} rules, {len(nature.derived)} derived values, "
          f"{len(nature.events)} events, open events {'on' if nature.open_events else 'off'}"
          + (f", game master review every {nature.review_every} steps" if nature.review_every else ""))
    print(f"Clock:    {scenario.clock.horizon} steps of one {scenario.clock.unit}")
    reason = "forced by a sensitive profile" if scenario.sensitive else "from the scenario" if not args.privacy else "from --privacy"
    print(f"Privacy:  {privacy.value} ({reason})")
    print("Models:   " + ", ".join(f"{role} {model}" for role, model in models.as_dict().items()))
    for name, actor in scenario.actors.items():
        for i, option in enumerate(actor.options):
            print(f"Option {i} for {name}: {option.label}")
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
    check_models(models, privacy)
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1_000_000)
    has_policy = any(a.kind == "policy" for a in scenario.actors.values())
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-s{seed}" + (f"-o{args.option}" if has_policy else "")
    out = Path(args.out)
    llm = LLMPClient(LLMPSettings.from_env(), privacy, run_id,
                     cache_dir=None if args.no_cache else out / ".cache")
    simulation = Simulation(scenario, llm, models, seed=seed, privacy=privacy, run_id=run_id,
                            option=args.option, steps=args.steps, on_step=_print_step)
    print(f"{scenario.title or scenario.name}: run {run_id}, privacy {privacy.value}")
    for name, label in simulation.policy_choice.items():
        print(f"{name} follows: {label}")
    result = simulation.run()
    trace, report = write_run(result, scenario, out / scenario.name / run_id)
    print(f"\nEnded: {result.ended_by}")
    if result.verdict:
        for actor, score in result.scores.items():
            print(f"  {actor}: {score}/10")
    print(f"Model requests: {result.model_requests}, from cache: {result.cache_hits}")
    print(f"Report: {report}\nTrace:  {trace}")
    return 0


def _print_step(node: Node) -> None:
    if node.quiet:
        print(f"{node.label}: quiet")
        return
    print(f"{node.label}  [{node.model_calls} model calls]")
    for event in node.events:
        print(f"  event: {event.name}" + (" (proposed by the game master)" if event.source == "open" else ""))
    for actor, action in node.actions.items():
        roll = node.roll_for(actor)
        outcome = "" if roll is None else f" -> {'worked' if roll.success else 'did not work'} (p {roll.p:.2f})"
        print(f"  {actor}: {_short(action.intent)}{outcome}")


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
