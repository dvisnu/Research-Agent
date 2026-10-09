# Commands

Everything you can run from the command line, as of Phase 1. Run all commands from the project root.

## One-time setup

```sh
uv sync                          # creates .venv with Python 3.13 and installs everything
cp .env.example .env             # PowerShell: Copy-Item .env.example .env
```

Then fill in the keys in `.env`. A run needs the extraction and reasoning models and their providers' keys; the rest is optional:

| Variable | What it is |
|---|---|
| `RA_DATA_DIR` | Where runs are stored. Default `var`. |
| `RA_LLM_EXTRACTION`, `RA_LLM_REASONING`, `RA_LLM_VERIFIER` | The model for each LLM role, as `provider:model`. A comma-separated list adds fallbacks: when a model is unavailable after retries, the call moves to the next one, and a used-up daily quota rules that model out for the rest of the session. The trace records which model answered. The verifier cannot share a model with reasoning. `.env.example` has the free-tier development models and their limits. |
| `RA_GOOGLE_API_KEY`, `RA_OPENROUTER_API_KEY`, `RA_GROQ_API_KEY` | API keys for the providers `google` (Gemini and Gemma, via Google AI Studio), `openrouter` and `groq`. Only the providers your roles use are needed; Groq is first used by the verifier in Phase 5. |
| `RA_S2_API_KEY` | Your Semantic Scholar API key. Its limit is **1 request per second across all endpoints**. The client enforces it machine-wide: one request at a time across all processes, each sent at least 1.1 s after the previous response. Never call the API outside the client. Without a key, most requests are rejected with HTTP 429. |
| `RA_OPENALEX_API_KEY` | Your OpenAlex API key (free). Without it, OpenAlex requests share a small daily budget with everyone on your IP address; once that is used up, OpenAlex search is skipped until midnight UTC. |
| `RA_CONTACT_EMAIL` | Your email. Unpaywall requires it; OpenAlex gives faster service with it. |
| `RA_FULL_READ` | How many of the kept papers are read in full. Default 10 during development (see `END_TODO.md`). |
| `RA_ABSTRACT_ONLY` | How many papers after those are kept and read by abstract. Default 30 during development. The rest are excluded, and the trace records why. |
| `RA_REUSE_TEXTS` | Development only. `true` reuses full texts parsed by an earlier run instead of downloading them again, which makes repeated runs much faster. Default `false`: the product always fetches fresh. |

`.env` is gitignored, so keys stay on your machine. A setting can also be given as an ordinary environment variable, which overrides `.env`.

## Running the `agent` command

Prefix each command with `uv run`:

```sh
uv run agent --help
```

Or activate the virtual environment once per terminal and drop the prefix:

```sh
.venv\Scripts\Activate.ps1       # PowerShell
source .venv/Scripts/activate    # Git Bash
agent --help
```

Every command takes `--help`.

---

## `agent run`

Runs the pipeline on a research question or idea.

```sh
uv run agent run "How can RAG systems reduce hallucinations when retrieved documents conflict?"
uv run agent run "I want to improve retrieval for scientific RAG."
uv run agent run "I want to improve retrieval for scientific RAG." --upload my-draft.pdf --upload related.pdf
```

`--upload` / `-u` adds a PDF you already have, such as your own draft or a paper you want considered. Uploaded papers are tagged as user-provided and always read in full, without using up the `RA_FULL_READ` quota. An arXiv PDF is recognised by its margin stamp and merged with the online copy. A file that can't be read as a PDF fails the run.

What happens:

1. A new run ID is created, e.g. `run_75e57bfa021a422d`.
2. The question goes through every pipeline stage. Each stage writes an event to the run trace, and a checkpoint is saved after it.
3. The report is written to `var/runs/<run_id>/report.md` and `report.json`, then printed.
4. The last lines show the run ID and the commands to inspect it.

The literature stages (`brief` to `fulltext`) do real work. The reasoning model writes a research brief: the problem, two or three search angles in the field's vocabulary, and the facets of an idea. The question's own words are always searched as well, as the first angle. Each angle is searched on arXiv, OpenAlex and Semantic Scholar. The run follows references, citations and recommendations from the best hits and ranks everything with SPECTER2. The extraction model then screens the best-ranked papers and drops off-topic ones. Finally the kept papers are parsed into addressed sentences. The later stages are still stubs, so the report sections all say *Not produced yet*.

Timing: a run takes several minutes, mostly because arxiv.org asks for 15 s between full-text downloads (its robots.txt), Semantic Scholar allows 1 request per second, and the free LLM tiers are paced to their per-minute limits. A busy free model is retried for about a minute before the call fails. The first run also downloads the SPECTER2 model (about 440 MB). Every run downloads its full texts fresh, unless `RA_REUSE_TEXTS=true` is set for development.

If a stage fails, the run is marked `failed`, and the output gives the commands to inspect and resume it.

## `agent resume <run_id>`

Continues a failed or interrupted run from the stage where it stopped. Stages that already finished are not run again.

```sh
uv run agent resume run_165c14f256c8437a
```

- A run that already completed can't be resumed; you'll get `Run ... has already completed.`
- An unknown ID gives `No run with ID ...`.

## `agent inspect [run_id]`

Without a run ID, lists the 20 most recent runs:

```sh
uv run agent inspect
```

With a run ID, shows that run's trace as a tree:

```sh
uv run agent inspect run_165c14f256c8437a
```

```
run_165c14f256c8437a  completed  How can RAG systems ...
+--   1  brief  ok  0 ms  used 1, generated 0, dropped 0
+--   2  search  ok  0 ms  used 0, generated 0, dropped 0
...
+--  10  gaps  ForcedFailure: forced failure in stage 'gaps'  0 ms  ...
+--  11  gaps (attempt 2)  ok  0 ms  used 0, generated 0, dropped 0
...
`--  18  report  ok  2 ms  used 0, generated 2, dropped 0
```

How to read a line:

| Part | Meaning |
|---|---|
| `11` | Order the event was written in |
| `gaps (attempt 2)` | The stage. The attempt number appears when a stage ran more than once, after a failure or in a loop |
| `ok` or an error | The outcome. Errors show the exception message |
| `0 ms` | How long it took |
| `used` / `generated` / `dropped` | How many entities (question, papers, sentences, files) the stage read, produced, or considered and rejected |

Calls made inside a stage appear indented under it: `api_call semantic_scholar` for a literature API and `llm_call reasoning/<model>` for an LLM.

Add `--json` to print the raw events, with every field (`used`, `generated`, `dropped` with reasons, `prompt_version`, HTTP status, token usage, and so on):

```sh
uv run agent inspect run_165c14f256c8437a --json > trace.json
```

---

## Debug options

`agent run` has two hidden options for testing the machinery. They don't appear in `--help`.

| Option | Effect |
|---|---|
| `--fail-at <stage>` | Makes that stage fail, so you can try `agent resume`. Applies to this invocation only, so the resume runs normally. |
| `--force-loops` | Takes the novelty retry loop and the critique/targeted-search loop once each. |

```sh
uv run agent run "test question" --fail-at gaps
uv run agent resume <run_id printed above>
uv run agent run "test question" --force-loops
```

The stages, in order, and the phase that builds each:

| Stage | Phase | Does |
|---|---|---|
| `brief` | 1 | Turn the question or idea into a research brief: the problem, the question's own words plus 2-3 search angles, and an idea's facets |
| `search` | 1 | Keyword, snippet and arXiv search, once per angle |
| `expand` | 1 | Follow references, citations and recommendations |
| `rank` | 1 | Rank papers, drop the ones the relevance check calls off-topic, split the rest into full-read and abstract-only sets |
| `fulltext` | 1 | Fetch and parse full texts into sections and sentences |
| `extract` | 2 | Extract each paper's facts, with quoted evidence |
| `build_graph` | 2 | Merge names, build the graph, link supporting and contradicting claims |
| `related_work` | 3 | Find the work closest to the question |
| `landscape` | 3 | Build the methods x datasets x metrics grid |
| `gaps` | 3 | Find and rank evidence-backed gaps |
| `ideate` | 4 | Generate ideas from the top gaps |
| `novelty_search` | 4 | Search independently for work close to the idea |
| `novelty_assess` | 4 | Compare the idea with the closest work; assign a level 1-5. Can loop back to `ideate` |
| `plan` | 5 | Draft the experiment plan from the graph |
| `critique` | 5 | Review feasibility and missing baselines. Can loop through `targeted_search` back to `plan` |
| `targeted_search` | 5 | Search for what the critique found missing |
| `verify` | 5 | Check every cited claim against its source |
| `report` | 0 | Write the Novel Evidence Report and the experiment plan |

---

## Tests

```sh
uv run pytest                       # offline tests (no network)
uv run pytest -m live               # one real call to each literature API and each configured LLM model
uv run pytest tests/test_pipeline.py -v
```

The live tests read `.env`. The LLM tests ask every model of every role, fallbacks included, for a research brief; a model whose provider has no key is skipped. Unpaywall's test is skipped unless `RA_CONTACT_EMAIL` is set. Semantic Scholar's test may fail with HTTP 429 until `RA_S2_API_KEY` is set.

## Phase 1 check: key papers

```sh
uv run python scripts/key_paper_recall.py                    # every dev topic
uv run python scripts/key_paper_recall.py gnn-oversmoothing  # chosen topics
uv run python scripts/key_paper_recall.py --full-text        # also download full texts
```

For each dev topic in `data/dev_topics.toml`, this runs the pipeline and reports how many of the topic's key papers were found, how many were among the first `RA_FULL_READ` papers kept (read in full), how many were kept at all, and how many the relevance check wrongly called off-topic. Each key paper is shown as `rank>place`: its rank, then its place among the kept papers, `off-topic`, or `cut`. Without `--full-text` it skips the slow downloads; the ranking is the same either way. A topic whose run fails is reported and skipped, and can be resumed with `agent resume`. Results are saved to `var/checks/`.

Each topic uses one reasoning call (the brief) and three to six extraction calls (the relevance check). The free reasoning model allows 50 calls a day, so the full check fits in a day's quota.

## Where runs are stored

| Path | Contents |
|---|---|
| `var/agent.db` | Runs, papers, sentences, graph, and the trace |
| `var/checkpoints.db` | LangGraph checkpoints, used by `agent resume` |
| `var/runs/<run_id>/report.md`, `report.json` | Each run's report |

`var/` is gitignored. Deleting it removes every run and starts fresh.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | The run failed (resumable), the run ID wasn't found, or the run can't be resumed |
| `2` | Bad usage: invalid `.env` settings, an unknown `--fail-at` stage, or a missing argument |
