# JANUS: design

JANUS is this repo's name since M2; it was Agent Factory, an agent designer, until the simulation pivot.
Status: M0 and M1 merged (PR #8); M2a (trees) on branch `sim-m2`; M2b (the UI) next.

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
     A proposal that repeats a listed event is recorded but not rolled, so base rates aren't doubled
     (local models do propose listed events despite being told not to).
- **Actors** have a brief, private information and objectives. Kinds: `llm` (decides with a model),
  `policy` (a fixed option you want to test, with orders on planned steps; a planned order that fails is
  retried up to `retries` times), `human` (planned, M5).
- **Game master** (GM): sees everything. It *assesses* each order's chance of success (one roll per
  order, not per action), and any private information an order could uncover; the engine rolls; then it
  *resolves* the step into state changes (`set`/`add` on declared variables), a public narrative and what
  each actor privately learns. Private information may only come out through a discovery that was rolled.
- **Objectives** can carry a `metric`, an expression scored in code (true = 10, false = 0, a number
  clamped to 0..10). Trees score every node this way, with no model call.
- **Judge:** reads the history for the objectives without a metric, and writes the run's summary.
- **Quiet steps cost nothing:** if no actor acts, no event fires and it isn't a review step, rules advance
  the state in code and no model is called.

### One step

```
rules → roll base-rate events → actors decide (seeing the events) →
GM assess (odds per order, maybe one open event, possible discoveries) → roll dice →
GM resolve (changes + narrative + private observations) → apply with bounds → derived values →
score metric objectives → record node → stop if an end condition holds
```

Every step is a node with a deterministic id derived from its parent and its content, so identical
prefixes from different runs get identical ids. That is what the tree is built on.

### Ramification trees (M2)

- **Chance goes through a `Dice` per step**, which logs every roll's label, probability and outcome, so a
  node knows its exact probability given its parent (`p_branch`) and its signature (which outcomes
  happened). Models' seeds come from the node, not the run, so a node always gets the same decisions and
  branches differ only where chance differs.
- **Branching:** the explorer expands the most probable unexpanded node first. At each node it takes the
  most likely outcome of every roll, then the same step with one uncertain outcome flipped, most
  uncertain first (`--branches` in total). That covers far more probability than random samples: a 5%
  failure is found at once instead of rarely. Amounts drawn in code (a repair bill) stay the same across
  a node's branches.
- **Rollouts:** below `--depth`, one sampled continuation per leaf to the end (`--rollouts`).
- **Budget:** model requests, 60% for expansion and the rest for rollouts; cached replies are free.
  Leaves still unfinished when the budget runs out are scored as they stand and counted as such.
- **Decision versions** (`--candidates k`): k versions of the LLM actors' decisions per node, weighed
  equally, or the worst one for a focal actor (`--opponents worst --focal blue`).
- **Roll-up:** leaves are weighted by probability, normalised over the branches explored; per option:
  expected score per actor, the 10th percentile, the chance of ending early, and coverage (how much of
  the probability the explored leaves stand for).
- **Storage:** `runs/<scenario>/tree-<id>.sqlite`, the id hashing the scenario and profile, the models
  and `ENGINE_VERSION`, so trees of different simulators never mix. Single runs land in the same tree.

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
    objectives:
      - text: Keep an emergency fund of at least 3 months of fixed costs
        metric: me.emergency_months >= 3      # scored in code: true 10, false 0
      - Play and record more                  # no metric: only the judge can score it
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

Unknown keys are errors. Inside `{...}` flow mappings, quote any text containing a comma: YAML would
otherwise split `description: rent, bills` into extra keys.

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

In `local` mode, before any call is made (`src/janus/llm/privacy.py`):

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
src/janus/
  llm/client.py      LLMP /v1 client: JSON-schema replies, validation + one repair retry, seeds, tags, cache
  llm/privacy.py     privacy modes and the local-only guard
  sim/expr.py        safe expression evaluator for rules, effects and end conditions
  sim/scenario.py    Scenario, ActorSpec, VarSpec, EventSpec, Profile + YAML loading
  sim/state.py       WorldState: typed values, bounds, changes, visibility
  sim/world.py       Dice, rules, event rolls, end conditions, objective metrics, clock labels
  sim/models.py      structured LLM inputs/outputs: Action, Assessment (per order), Resolution, Verdict
  sim/trace.py       Node (one step), RunResult
  sim/prompts.py     every prompt, and the context blocks they share
  sim/actors.py      LLMActor, PolicyActor (with retries)
  sim/gm.py          game master: assess and resolve
  sim/judge.py       verdict for objectives without a metric; combined scores
  sim/engine.py      Simulation.step (one transition from any node) and run (one sampled path)
  sim/tree.py        TreeStore (SQLite), Explorer, roll-up
  sim/report.py      trace.jsonl and report.md
  cli.py             `janus check|run|explore|tree`
scenarios/  profiles/  tests/  legacy/
```

## Milestones

- **M0 – Reset** (done): v1 moved to `legacy/` with its own dependencies, slim `pyproject.toml`, README.
- **M1 – One path, both kinds of scenario** (done): the world model, `llm` and `policy` actors, GM, judge,
  CLI, privacy guard and sensitive profiles, two scenarios, offline tests with a fake LLM.
- **M2a – Ramification trees** (done on `sim-m2`): see "Ramification trees" above, plus odds per order,
  discovery rolls for private information, plan retries and objective metrics. The repo became JANUS.
- **M2b – The UI** (next): a local web app in the MS Intelligence identity (`Business-Study/brand`):
  scenarios, the tree explorer and run reports; FastAPI plus plain HTML, CSS and JS, the font and logo
  served locally and a content security policy that blocks anything external. A clickable mockup comes
  first, for approval.
- **M3 – Global scenarios:** N actors, actor memory summaries, a multi-party scenario.
- **M4 – Checking the simulator:** label-swap test, agreement between two judge models, event frequencies vs
  base rates, diversity across branches, spot-checks.
- **M5 – Knowledge and a human seat:** a local RAG tool for actors via LLMP embeddings, playing a seat live.

## Findings from the first local runs

Both scenarios ran on `gemma3:4b` actors and a `gpt-oss:20b` game master and judge. On this machine a
`gpt-oss:20b` call takes about 30–40 s (it spends ~1,800 tokens reasoning); `gemma3:12b` as game master
takes about 20 s but follows the schema less reliably (it tried to write variables that don't exist), so
`gpt-oss:20b` stays the default. What they showed, and what M2 did about it:

- **One roll per action was too coarse:** one failed roll made the game master fail all five orders.
  M2: odds and a roll per order.
- **Private information leaked too easily:** a successful recon by Red "confirmed" Blue's hidden
  anti-tank companies in the public narrative. M2: the game master lists what an order could uncover,
  each discovery is rolled, and the resolution prompt forbids revealing anything not discovered. This is
  enforced through the prompt only; the text itself is not checked.
- **Failed plans were never retried:** the used-amp purchase rolled its 5% failure and stopped there.
  M2: planned orders are retried (`retries`, default 2).
- Small actor models drift on details (Red's northern column was "Column 1", then "Column 2"). Actor
  memory summaries (M3) should carry such facts forward.

LLM-assigned probabilities are not calibrated forecasts. Trees show plausible ramifications and their
relative weight: use them to find fragile choices and the events that drive them.

## Decisions taken

1. Local models by default everywhere; cloud models are a per-scenario opt-in and never allowed with
   sensitive data.
2. A per-key provider limit will be added to LLMP (separate change in that repo).
3. In trees, opponents roll up by expected behaviour by default, with a worst-case switch.
4. Actors act simultaneously each step; state is typed; v1 lives in `legacy/`.
