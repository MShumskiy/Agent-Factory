# Agent Factory v1 (frozen)

The first version of Agent Factory, kept for reference. Nothing here is imported by the new
simulation engine in `src/agent_factory/`.

## What is here

| Path | Contents |
|---|---|
| `src/agents/` | Agent 0 (tool router), KB agent, adversary agent, `SIMAgent` (the first simulation loop) |
| `src/pipelines/` | PDF ingestion into Postgres embeddings, RAG with cross-encoder reranking |
| `src/app/` | Gradio apps for the KB agent and the simulation |
| `config/` | RAG, ingestion and image-generation settings |
| `research/`, `grand_sim_test.ipynb` | Experiment notebooks, including the "grand simulation" post-processing and graph drawing |
| `data/datasets/` | LLM call history exports and RAG evaluation results |
| `data/KBs/` | Knowledge-base documents (gitignored) |
| `baselines/` | Output of a v1 grand simulation run (dialogues and judge verdicts), kept to compare v2 against |
| `docs/` | v1 architecture diagrams and screenshots |

## Running it

v1 has its own dependencies (`pyproject.toml` and `poetry.lock` in this folder) and calls the legacy
LLMP `/generate` endpoint. The code imports `src.*`, so run notebooks and scripts with `legacy/` as the
working directory. Known problems are listed in `docs/design/sim-v2.md` at the repo root; they are not
going to be fixed here.
