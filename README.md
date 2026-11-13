# Research Agent
 
A multi-agent system for literature-driven research: given a research problem or idea, it surveys relevant literature, detects research gaps, proposes a novelty-assessed idea backed by evidence, and produces a provenance-aware experiment plan where every element traces back to supporting papers.
 
## Status
 
Phases 0 (Foundations), 1 (Research input + literature acquisition), 2 (Research knowledge graph), 3 (Related work + gap detection), 4 (Idea generation + novelty assessment), 5 (Provenance-aware experiment plan) and 6 (Hardening + integration) are done. A run turns the question into a research brief, searches arXiv, OpenAlex and Semantic Scholar, follows citations, ranks the candidates and screens them for relevance, and parses the kept papers into addressed sentences. It then extracts each paper's tasks, methods, datasets, metrics, results, claims and limitations, each backed by quotes that code checks against the paper, and builds a knowledge graph with names merged across papers and supporting and contradicting findings linked (`agent graph`). From the graph it picks the closest related work, builds a grid of the most-studied methods against datasets, and ranks research gaps (stated limitations, untested method-dataset pairs, conflicting findings), each with a warrant and quoted evidence. From the top gaps it writes ideas and checks three of them: a fresh search for each idea, a facet-by-facet comparison with its 10 closest papers (the 3 closest read in full), and a novelty level from 1 to 5 set by rules that fail closed, reported as potential novelty in the Novel Evidence Report; an idea rated 1-2 is revised and checked again (`agent ideas`, and `agent novelty` for one idea of your own). For the best idea it writes an experiment plan (Phase 5) from facts of its closest papers and the graph: every element cites the sentences it rests on and records how it was derived and why it was chosen; code and a verifier model from another model family critique it, a targeted search fills what is missing and the plan is revised; the verifier then checks every cited claim, and the report gives the plan's Provenance Coverage, with the full provenance graph exported as PROV-JSON (`agent plan` traces any element back to its paper and section). Phase 6 made runs dependable: a run that stops resumes from its last finished stage, every parse of a paper is kept, LLM replies and search responses are kept so reruns cost little, each run's cost is measured (`agent cost`), and a local web UI (`agent ui`) starts, watches, resumes and browses runs. What is deliberately left in a development state is listed in `END_TODO.md`. See `docs/project/phase-plan.md`.

## Usage

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```sh
uv sync
cp .env.example .env               # then fill in the API keys
uv run agent run "How can RAG systems reduce hallucinations when retrieved documents conflict?"
uv run agent inspect               # recent runs
uv run agent inspect <run_id>      # a run's trace
uv run agent resume <run_id>       # continue a failed run
uv run agent ideas <run_id>        # the ideas a run checked, with their novelty levels
uv run agent novelty "<idea>"      # check one idea of your own
uv run agent plan <run_id>         # the experiment plan, each element's verdict; --element traces one
uv run agent ui                    # the web UI: start, watch, resume and browse runs (build it first, below)
uv run pytest                      # offline tests
uv run pytest -m live              # one real call per literature API and LLM model
```

The web UI (React, in `frontend/`) is built once with Node 20+: `cd frontend && npm install && npm run build`. It runs on 127.0.0.1 only.

Runtime data (database, checkpoints, reports) goes to `var/`. Dev topics for checking each phase are in `data/dev_topics.toml`.

See [COMMANDS.md](COMMANDS.md) for every command, option and setting.
