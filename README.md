# Agent Factory

LLM-driven simulations. A scenario describes a world (state, rules, chance events), the actors in it and
what each of them wants. The engine plays it forward step by step: actors decide, a game master judges
how likely each action is to work, seeded dice decide, and the game master turns the result into state
changes and a narrative. A judge scores the outcome against everyone's objectives.

It covers scenarios against another agent (a war game, a negotiation), against nature (a personal
decision under uncertainty) and, later, many actors at once. Runs can be kept fully local: when a
scenario uses personal data, every call goes to local models through LLMP and nothing else.

The design and the roadmap (ramification trees, multi-actor scenarios, evaluation) are in
[docs/design/sim-v2.md](docs/design/sim-v2.md). The previous RAG-centred version is frozen in
[legacy/](legacy/README.md).

## Setup

Requires Python 3.12 and a running LLMP server (the companion `LLMP` repo: an OpenAI-compatible `/v1`
API over Ollama and OpenRouter).

```bash
poetry install
```

Copy `.env.example` to `.env` and set the LLMP address and key. `LLMP_BASE_URL` and `LLMP_API_KEY` are
read first; the v1 names `LLMP_URL` and `LLMP_PASSWORD` still work.

## Running a scenario

```bash
poetry run agent-factory check scenarios/tank_pincer.yaml
```

`check` validates the scenario and shows the privacy mode and the models each role will use, without
calling any model.

```bash
poetry run agent-factory run scenarios/tank_pincer.yaml --seed 7
```

```bash
poetry run agent-factory run scenarios/guitar_amp_purchase.yaml --option 1 --seed 3
```

Each run writes `runs/<scenario>/<run id>/trace.jsonl` (every step, roll and model reply) and
`report.md` (the readable version). Model replies are cached in `runs/.cache/`, so running again with
the same seed replays without calling a model; `--no-cache` turns that off. Other options: `--steps`,
`--privacy`, `--model`, `--gm-model`, `--judge-model`.

## Privacy

Every scenario runs in `local` mode unless it says `privacy: cloud_ok` (or the run passes `--privacy
cloud_ok`). In `local` mode the engine refuses to start unless every model is a non-cloud `ollama/`
model and LLMP is on this machine or the local network, and it asks LLMP not to store prompts or
replies. A profile marked `sensitive: true` forces `local`, whatever the scenario or command line says.

Put personal profiles in `private/`, which is gitignored, like `runs/`.

## Tests

```bash
poetry run pytest
```

The tests use a scripted fake model and need neither LLMP nor Ollama.
