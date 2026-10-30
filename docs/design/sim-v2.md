# JANUS: design

JANUS is this repo's name since M2; it was Agent Factory, an agent designer, until the simulation pivot.
Status: M0 and M1 merged (PR #8); M2a (trees) and M2b (the UI) in PR #9 (`sim-m2`); phase A on
`sim-phase-a`.

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
  clamped to 0..10; graded expressions are preferred, since pass/fail hides differences) and a `weight`
  (default 1) for the actor's weighted overall score. Trees score every node this way, from its stored
  state with the scenario's current metrics and weights, so changing either re-scores existing trees
  without any model call.
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
- **Continuations:** below `--depth`, every leaf is carried to the end along its most likely course (a
  projection: no surprises and no game-master reviews, so free when nobody acts), plus `--rollouts`
  sampled continuations.
- **Budget:** model requests, 60% for expansion and the rest for continuations; cached replies are free.
  Leaves still unfinished when the budget runs out are scored as they stand and counted as such.
- **Decision versions** (`--candidates k`): k versions of the LLM actors' decisions per node, weighed
  equally, or the worst one for a focal actor (`--opponents worst --focal blue`).
- **Roll-up:** explored branches are weighted by probability, normalised over the branches explored.
  Continuations are weighted by their full chain probability (a rare job loss inside a sampled rollout
  keeps its rarity) and together stand for the node they continue. Per option: expected (weighted)
  score per actor, the 10th percentile, the chance of ending early, and coverage (how much of the
  probability the explored leaves stand for).
- **Odds scales (engine v3):** the game master gives an order's odds as one of 0.05, 0.2, 0.35, 0.5,
  0.65, 0.8, 0.95 (from "almost certainly not" to "almost certain"), and surprises and discoveries as one
  of 0.01 to 0.5. The schema lists the steps; other numbers are snapped to the nearest one on the odds
  scale. This removes differences that are only noise (0.85 against 0.80).
- **Storage:** `runs/<scenario>/tree-<id>.sqlite`, the id hashing the scenario and profile (without how
  objectives are scored), the models and `ENGINE_VERSION`, so trees of different simulators never mix.
  Single runs land in the same tree. Trees built by an earlier engine are still found and read (and the
  UI says so); new explorations start the current engine's tree.

### Sensitivity: what would change the answer (phase A)

`janus sensitivity` (and section 03 of the explorer) compares the options again under standard changes
of assumption: each listed event at half and double its rate, each actor's orders working less and more
often (odds halved and doubled), unlisted surprises twice as likely, secrets discovered more easily. It
re-weights the stored tree: every roll is stored with its label, probability and outcome, so each
branch's probability can be recomputed for any change, without a model call (milliseconds for a
300-node tree). It reports the best option and its lead under each change, which changes flip the
best option, and the closest call. Limits: unexplored branches stay unexplored (a change that makes
them likely lowers the coverage), and projections followed the most likely course under the original
probabilities.

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
  sim/tree.py        TreeStore (SQLite), TreeView, Explorer, roll-up, scoring from state
  sim/sensitivity.py what would change the answer: re-weighting stored trees
  sim/report.py      trace.jsonl and report.md
  cli.py             `janus check|run|explore|tree|sensitivity|ui`
scenarios/  profiles/  tests/  legacy/
```

## Milestones

- **M0 – Reset** (done): v1 moved to `legacy/` with its own dependencies, slim `pyproject.toml`, README.
- **M1 – One path, both kinds of scenario** (done): the world model, `llm` and `policy` actors, GM, judge,
  CLI, privacy guard and sensitive profiles, two scenarios, offline tests with a fake LLM.
- **M2a – Ramification trees** (done on `sim-m2`): see "Ramification trees" above, plus odds per order,
  discovery rolls for private information, plan retries and objective metrics. The repo became JANUS.
- **M2b – The UI** (done on `sim-m2`): `janus ui`, a local web app in the "dark instrument" style chosen
  from the mockup: the MS Intelligence dark palette (near-black, ivory, muted gold), Playfair Display for
  titles and figures, a monospaced data layer (Cascadia Mono / Consolas, already on Windows), hairline
  grids, square corners. `src/janus/ui/`: `data.py` (scenario summaries, tree payloads), `jobs.py`
  (explorations and runs in a background thread, one at a time, same privacy checks as the CLI),
  `server.py` (FastAPI; content security policy `default-src 'self'`, no inline styles or scripts),
  `static/` (vanilla JS building the DOM directly, the brand font and logos). Not yet: run reports in
  the UI, editing scenarios and profiles, a human seat.
  Finding: with two model-played sides issuing five orders each, every step has ten or more rolls, so a
  few branches per node cover little of the probability (the tank tree: 5% at a 30-request budget). Next
  step for such scenarios: branch on a summary of each actor's orders (how many worked) instead of on
  every order.
### Next: the Westworld direction

The aim is the capability of the simulation systems in *Westworld*, for decision-making: Rehoboam's
living forecast and steering, the Forge's fidelity-tested copies of people, the Cradle as a rehearsal
sandbox, and hosts that can be questioned in "analysis mode". For your own decisions, or clients who
agree to be modelled; nobody predicts people the way Rehoboam does, so the value is structured
exploration, checks against reality, and knowing which assumptions the answer rests on. Built around a
free GPU: code-side dynamics, re-weighting stored trees and cached replays first; model calls only where
they change the answer.

- **Phase A – numbers you can trust** (done on `sim-phase-a`): the sensitivity check, objective weights
  and graded metrics re-scoring existing trees, odds from fixed scales (engine v3), and the corrected
  weighting of continuations.
- **Phase B – the hosts:** more than two actors, each with a running memory; crowds defined in code
  (customers, a market) with the model only at pivotal moments; analysis mode (ask an actor why, at any
  node, one call on demand); branching on a summary of each side's orders.
- **Phase C – the fidelity test:** replay a past period, score how closely the simulation matched what
  happened, tune base rates to it; for personal scenarios, compare each month's forecast with reality.
- **Phase D – the living forecast:** record what actually happens, project again from the present, flag
  when reality leaves the forecast band; futures shown as streams of probability over time.
- **Phase E – steering:** conditional plans ("if the car repair hits, sell the amp"), a search for the
  plan that does best across the futures, and the decision that matters most now.
- Later: checking the simulator (label-swap test, judge agreement, event frequencies against base
  rates), a local knowledge tool for actors, a seat for you to play, branded decision reports in the
  MS Intelligence document style.

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

## Findings from phase A

- **Continuations were weighted wrongly until phase A.** The roll-up normalised a continuation chain step
  by step, so every chain counted as probability 1: a sampled rollout that hit a 1-in-700 job loss
  weighed as much as the most likely course. With chains weighted by their full probability, the amp
  comparison changed from "save first 9.0, buy used 8.9, buy new 7.6" to a near tie: buy new 8.41,
  save first 8.30, buy used 8.28, with no standard change flipping the order (closest call: paid gigs
  twice as likely, lead 0.01).
- **The amp scenario barely separates its options.** In the most likely course all three end the year
  the same way (fund above 3 months, an amplifier, no extra playing, no stress); only rare events differ.
  Owning an amplifier never raises playing time in its rules, only paid gigs do. The scenario, not the
  engine, needs that dynamic for the comparison to mean something.

## Decisions taken

1. Local models by default everywhere; cloud models are a per-scenario opt-in and never allowed with
   sensitive data.
2. No per-key provider limit in LLMP for now: the engine's privacy guard is enough (decided 2026-10-30).
3. In trees, opponents roll up by expected behaviour by default, with a worst-case switch.
4. Actors act simultaneously each step; state is typed; v1 lives in `legacy/`.
