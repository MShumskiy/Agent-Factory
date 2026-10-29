# Simulation engine v2: design

Status: M0 and M1 implemented on branch `sim-v2`. Later milestones are plans.

## Goal

An engine that runs LLM-driven scenarios many times, branches them, and shows what tends to happen
and why. The same engine covers three shapes of scenario:

| Shape | Actors | Examples |
|---|---|---|
| Against an agent | you + one LLM actor (+ nature) | war game, negotiation, a diagnosis debate |
| Against nature | you only | buying an expensive item, changing jobs, moving country, a health plan |
| Global | several actors + nature | a market with competitors, geopolitics, family or workplace dynamics |

It must run fully locally when a scenario touches personal data.

## Why v1 was replaced

v1 (now in `legacy/`) had a working loop (opening move, alternating counter-moves, random events, a
judge) but no simulation underneath it:

- The only state was the transcript. Nothing tracked resources, positions or limits.
- Both players were the same agent with the same prompt and knowledge base; the judge regularly gave
  them identical strengths and weaknesses.
- RAG ran on every move: slow, and it pulled unrelated personas into scenes.
- Random events were whatever the LLM invented; the judge ran once at the end with no rubric.
- "Grand sim" was N independent runs drawn side by side, not a tree.
- Bugs that affected results: Player 2 was told to counter its own move, `str.strip("prefix")` ate
  characters off prompts, RAG overrides leaked into later calls through module globals.

## Concepts

- **World:** a clock, typed state variables (numbers with bounds, or strings with allowed choices), and
  nature.
- **Nature** is not an opponent and has no objective. It has three layers:
  1. *Rules*: expressions applied in code every step (`me.savings: me.savings + me.income - ...`).
     Arithmetic never goes through an LLM.
  2. *Base-rate events*: a table of events with a probability per step, an optional condition, and
     optional effects written as expressions.
  3. *Open events*: when it reviews a step, the game master may propose one event the table doesn't
     list. It has to give a probability, and the engine rolls for it; it can't just declare it.
- **Actors** have a brief, private information and objectives. Kinds: `llm` (decides with a model),
  `policy` (a fixed option you want to test, applied on the first step), `human` (planned, M5).
- **Game master** (GM): sees everything. It *assesses* each action's chance of success, the engine rolls
  seeded dice, then it *resolves* the step into state changes (`set`/`add` on declared variables), a
  public narrative and what each actor privately learns.
- **Judge:** scores the final state and history against each actor's objectives (0–10 with evidence).
- **Quiet steps cost nothing:** if no actor acts, no event fires and it isn't a review step, rules advance
  the state in code and no model is called.

### One step

```
rules → roll base-rate events → actors decide (seeing the events) →
GM assess (odds per action, maybe one open event) → roll dice →
GM resolve (changes + narrative + private observations) → apply with bounds → record node →
stop if an end condition holds
```

Every step is a node with a deterministic id derived from its parent and its content, so identical
prefixes from different runs get identical ids. That is what M2 builds the tree on.

## Scenario files

Scenarios live in `scenarios/*.yaml`. See `scenarios/tank_pincer.yaml` (against an agent) and
`scenarios/guitar_amp_purchase.yaml` (against nature, with a profile) for full examples.

```yaml
name: guitar_amp_purchase
privacy: local                  # local (default) or cloud_ok
profile: profiles/example_me.yaml
clock: {unit: month, horizon: 12}
models: {default: ollama/gemma3:4b, game_master: ollama/gpt-oss:20b, judge: ollama/gpt-oss:20b}
actors:
  me:
    kind: policy
    options: [Buy a new amp now for 1500, Buy a used amp for 800, Wait 6 months and save]
    objectives: [Emergency fund of at least 3 months of fixed costs, Play and record more]
state:
  me:
    savings: {value: 4000, min: -20000}
    stress: {value: 3, min: 0, max: 10}
    amp: {value: none, choices: [none, new, used, broken]}
nature:
  rules:
    me.savings: me.savings + me.income - me.fixed_costs - me.spending
  events:
    - name: Car repair
      p: 0.04
      effects: {me.savings: "me.savings - uniform(300, 1500)", me.stress: "me.stress + 2"}
  open_events: true
  review_every: 3
end_when: ["me.savings < -2000"]
```

Expressions support `+ - * / // % **`, comparisons, `and/or/not`, `x if c else y`, `entity.var` lookups
and the functions `min max abs round clamp uniform normal randint chance`. Effects that are plain numbers
or booleans are literals; strings are expressions, so a string literal needs inner quotes (`"'new'"`).

### Profiles

Personal data goes in a profile YAML, kept in the gitignored `private/` folder:

```yaml
sensitive: true          # forces privacy: local for any scenario that uses this profile
actor: me                # the actor this profile describes
about: |                 # free text the actor and the GM see
  ...
state:                   # values that override the scenario's defaults
  me: {income: 1800, fixed_costs: 1100, savings: 4200}
```

`profiles/example_me.yaml` is a made-up example. Copy it to `private/me.yaml` for real data and point the
scenario's `profile:` at it.

## Privacy

`privacy: local | cloud_ok` per scenario, overridable per run with `--privacy`, except that a sensitive
profile forces `local` and asking for `cloud_ok` with one is an error.

In `local` mode, before any call is made (`src/agent_factory/llm/privacy.py`):

- every model must be `ollama/...` and not one of Ollama's cloud models (`:cloud` / `-cloud` tags run on
  ollama.com even when called through a local Ollama);
- the LLMP address must resolve to loopback or a private network address;
- calls are sent with `llmp.log = "meta"`, so prompts and responses stay out of LLMP's `chat_log`;
- runs, traces and the response cache are written under the gitignored `runs/` folder.

LLMP can't yet limit a key to Ollama, and its OpenRouter allowlist only filters the model listing, so the
engine's guard is the only enforcement until a per-key provider policy lands in LLMP.

Prompts and output schemas are designed for local models first. The strongest local model plays game
master and judge; smaller ones play actors. Responses are cached on disk (keyed by the full request), so a
run with the same seed replays without calling a model.

## Code layout

```
src/agent_factory/
  llm/client.py      LLMP /v1 client: JSON-schema replies, validation + one repair retry, seeds, tags, cache
  llm/privacy.py     privacy modes and the local-only guard
  sim/expr.py        safe expression evaluator for rules, effects and end conditions
  sim/scenario.py    Scenario, ActorSpec, VarSpec, EventSpec, Profile + YAML loading
  sim/state.py       WorldState: typed values, bounds, changes, visibility
  sim/world.py       rules, event sampling, end conditions, clock labels
  sim/models.py      structured LLM inputs/outputs: Action, Assessment, Resolution, Verdict
  sim/actors.py      LLMActor, PolicyActor
  sim/gm.py          game master: assess and resolve
  sim/judge.py       final verdict
  sim/engine.py      one rollout, producing nodes
  sim/report.py      trace.jsonl and report.md
  cli.py             `agent-factory run|check`
scenarios/  profiles/  tests/  legacy/
```

## Milestones

- **M0 – Reset** (done): v1 moved to `legacy/` with its own dependencies, slim `pyproject.toml`, README.
- **M1 – One path, both kinds of scenario** (done): the world model, `llm` and `policy` actors, GM, judge,
  CLI, privacy guard and sensitive profiles, two scenarios, offline tests with a fake LLM.
- **M2 – Ramification trees:** SQLite tree per scenario, repeated runs merged by shared prefix, branching at
  decision nodes (your options, k candidate replies from LLM actors) and chance nodes (event outcomes, dice),
  budgets (nodes, calls, depth), roll-up (probability-weighted mean and worst 10% at chance nodes, expected or
  worst case at opponents' decisions, side-by-side at yours), static HTML viewer with no CDN.
- **M3 – Global scenarios:** N actors, actor memory summaries, a multi-party scenario.
- **M4 – Checking the simulator:** label-swap test, agreement between two judge models, event frequencies vs
  base rates, diversity across branches, spot-checks.
- **M5 – Knowledge and a human seat:** a local RAG tool for actors via LLMP embeddings, playing a seat live.

LLM-assigned probabilities are not calibrated forecasts. Trees show plausible ramifications and their
relative weight: use them to find fragile choices and the events that drive them.

## Decisions taken

1. Local models by default everywhere; cloud models are a per-scenario opt-in and never allowed with
   sensitive data.
2. A per-key provider limit will be added to LLMP (separate change in that repo).
3. In trees, opponents roll up by expected behaviour by default, with a worst-case switch.
4. Actors act simultaneously each step; state is typed; v1 lives in `legacy/`.
