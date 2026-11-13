# Research Agent

An agentic LLM pipeline for literature-driven research: given a research problem or idea, it surveys relevant literature, detects research gaps, proposes a novelty-assessed idea backed by evidence, and produces a provenance-aware experiment plan where every element traces back to supporting papers.

## Features

- **Literature search**: turns the question into a research brief, searches arXiv, OpenAlex and Semantic Scholar, follows citations, ranks the candidates and screens them for relevance.
- **Grounded extraction**: parses each kept paper into addressed sentences and extracts its tasks, methods, datasets, metrics, results, claims and limitations, each backed by quotes that code checks against the paper.
- **Knowledge graph**: merges names across papers and links findings that support or contradict each other.
- **Gap detection**: picks the closest related work, builds a grid of the most-studied methods against datasets, and ranks research gaps (stated limitations, untested method-dataset pairs, conflicting findings), each with a warrant and quoted evidence.
- **Novelty assessment**: writes ideas from the top gaps and checks each one with a fresh search, a facet-by-facet comparison with its 10 closest papers (the 3 closest read in full) and a novelty level from 1 to 5 set by rules that fail closed. An idea rated 1-2 is revised and checked again.
- **Provenance-aware experiment plan**: for the best idea, writes a plan where every element cites the sentences it rests on and records how it was derived and why it was chosen. A verifier model from another model family critiques the plan, a targeted search fills what is missing, the plan is revised, and the verifier then checks every cited claim. The full provenance graph is exported as PROV-JSON.
- **Dependable runs**: a run that stops resumes from its last finished stage; LLM replies and search responses are kept, so a rerun of the same question costs little; each run's time, model calls, tokens and API requests are measured.
- **Web UI**: a local React UI to start, watch, resume and browse runs.

## How it works

A run is a LangGraph workflow with checkpoints after every stage:

```
brief -> search -> expand -> rank -> fulltext -> extract -> build_graph
  -> related_work -> landscape -> gaps -> ideate -> novelty_search -> novelty_assess
  novelty_assess -> ideate                (idea rated too low: revise it)
  novelty_assess -> plan -> critique
  critique -> targeted_search -> plan     (the critique found problems: search and revise)
  critique -> verify -> report
```

LLMs are assigned by role: an extraction model for high-volume work (relevance checks, paper extraction), a reasoning model for the brief, gaps, ideas and plans, and a verifier from a different model family that critiques the plan and checks its claims. Papers, sentences, the graph, ideas and plans are stored in SQLite.

Each run writes to `var/runs/<run_id>/`:

- `report.md` and `report.json`: the Novel Evidence Report (the best idea, its novelty level, its closest papers and the quoted evidence), the experiment plan with each element's verdict and Provenance Coverage, and every idea checked.
- `provenance.json`: the plan's provenance graph in PROV-JSON.

A full run takes about 75-140 minutes on free-tier API keys, mostly spent waiting on per-minute rate limits.

## Project structure

```
Research-Agent/
├── src/research_agent/
│   ├── cli.py               # the `agent` command
│   ├── config.py            # settings read from .env
│   ├── llm.py               # LLM roles, retries and fallbacks
│   ├── llm_providers.py     # Google, OpenRouter and Groq clients
│   ├── embeddings.py        # text embeddings for linking findings
│   ├── trace.py             # per-stage trace of what each step used and produced
│   ├── costs.py             # time, model calls, tokens and API requests per run
│   ├── report.py            # report.md and report.json
│   ├── pipeline/            # the LangGraph workflow, runner, checkpoints and resume
│   ├── clients/             # arXiv, OpenAlex, Semantic Scholar and Unpaywall clients, response cache
│   ├── literature/          # brief, search, citation expansion, ranking, relevance, PDF parsing
│   ├── graph/               # grounded extraction, quote checks, name merging, knowledge graph
│   ├── gaps/                # related work, methods x datasets grid, ranked gaps
│   ├── ideas/               # idea generation, novelty search, facet comparison, novelty levels
│   ├── plans/               # experiment plan, critique, targeted search, verification, PROV-JSON
│   ├── storage/             # SQLite database and its migrations
│   └── web/                 # FastAPI backend of the web UI
├── frontend/                # React + TypeScript web UI (Vite)
├── tests/                   # offline tests with fake services, plus live API tests
├── scripts/                 # quality checks: key-paper recall, graph, gaps, novelty, plans, end-to-end regression
├── data/                    # sample research topics with their key papers, planted ideas for the duplicate test
├── docs/
│   ├── project/             # design docs
│   └── literature/          # notes on the papers the design builds on
├── COMMANDS.md              # every command, option and setting
├── .env.example             # settings and API keys
└── var/                     # runtime data: database, checkpoints, run reports (gitignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```sh
uv sync
cp .env.example .env               # then fill in the API keys
```

`.env.example` lists every setting. The keys it needs:

- LLMs: [Google AI Studio](https://aistudio.google.com/apikey), [OpenRouter](https://openrouter.ai/settings/keys) and [Groq](https://console.groq.com/keys). Each role (`RA_LLM_EXTRACTION`, `RA_LLM_REASONING`, `RA_LLM_VERIFIER`) is set as `provider:model`, with optional fallbacks.
- Literature: a Semantic Scholar API key, an OpenAlex API key and a contact email (for Unpaywall and OpenAlex).

## Usage

```sh
uv run agent run "How can RAG systems reduce hallucinations when retrieved documents conflict?"
uv run agent inspect               # recent runs
uv run agent inspect <run_id>      # a run's trace
uv run agent resume <run_id>       # continue a failed run
uv run agent graph <run_id>        # the knowledge graph; --gaps for the ranked research gaps
uv run agent ideas <run_id>        # the ideas a run checked, with their novelty levels
uv run agent novelty "<idea>"      # check one idea of your own
uv run agent plan <run_id>         # the experiment plan, each element's verdict; --element traces one
uv run agent cost <run_id>         # time, model calls, tokens and API requests of a run
uv run agent report <run_id>       # write a run's report again from what is stored
uv run agent ui                    # the web UI (build it first, below)
```

The web UI (React, in `frontend/`) is built once with Node 20+: `cd frontend && npm install && npm run build`. It runs on 127.0.0.1 only.

Runtime data (database, checkpoints, reports) goes to `var/`.

## Tests

```sh
uv run pytest                      # offline tests
uv run pytest -m live              # one real call per literature API and LLM model
```

## Docs

- [COMMANDS.md](COMMANDS.md): every command, option and setting.
- [docs/project/basic-explanation.md](docs/project/basic-explanation.md): what the agent does, in plain words.
- [docs/project/phase-plan.md](docs/project/phase-plan.md): the design of each stage and its measured results.
- [docs/project/short-phase-plan.md](docs/project/short-phase-plan.md): a short version of the design.
- `docs/literature/`: summaries and notes of the papers the design borrows from.
