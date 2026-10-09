# Research Agent
 
A multi-agent system for literature-driven research: given a research problem or idea, it surveys relevant literature, detects research gaps, proposes a novelty-assessed idea backed by evidence, and produces a provenance-aware experiment plan where every element traces back to supporting papers.
 
## Status
 
Phases 0 (Foundations) and 1 (Research input + literature acquisition) are done. A run turns the question into a research brief, searches arXiv, OpenAlex and Semantic Scholar, follows citations, ranks the candidates and screens them for relevance, and parses the kept papers into addressed sentences. The stages from Phase 2 on are still stubs. What is deliberately left in a development state is listed in `END_TODO.md`. See `docs/project/phase-plan.md`.

## Usage

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```sh
uv sync
cp .env.example .env               # then fill in the API keys
uv run agent run "How can RAG systems reduce hallucinations when retrieved documents conflict?"
uv run agent inspect               # recent runs
uv run agent inspect <run_id>      # a run's trace
uv run agent resume <run_id>       # continue a failed run
uv run pytest                      # offline tests
uv run pytest -m live              # one real call per literature API and LLM model
```

Runtime data (database, checkpoints, reports) goes to `var/`. Dev topics for checking each phase are in `data/dev_topics.toml`.

See [COMMANDS.md](COMMANDS.md) for every command, option and setting.
