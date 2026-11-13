# Commands

Everything you can run from the command line, as of Phase 6. Run all commands from the project root.

## One-time setup

```sh
uv sync                          # creates .venv with Python 3.13 and installs everything
cp .env.example .env             # PowerShell: Copy-Item .env.example .env
cd frontend; npm install; npm run build; cd ..   # only for the web UI (agent ui); needs Node 20+
```

Then fill in the keys in `.env`. A run needs the extraction and reasoning models and their providers' keys; the rest is optional:

| Variable | What it is |
|---|---|
| `RA_DATA_DIR` | Where runs are stored. Default `var`. |
| `RA_LLM_EXTRACTION`, `RA_LLM_REASONING`, `RA_LLM_VERIFIER` | The model for each LLM role, as `provider:model`. A comma-separated list adds fallbacks: when a model is unavailable after retries, the call moves to the next one, and a used-up daily quota rules that model out for the rest of the session. The trace records which model answered. The verifier cannot share a model with reasoning. `.env.example` has the free-tier development models and their limits. |
| `RA_EMBEDDING`, `RA_EMBEDDING_DIMS` | The text-embedding model, as `provider:model` (`google:gemini-embedding-2`, using `RA_GOOGLE_API_KEY`), and its vector size (default 768). Used from Phase 2 to find findings in different papers that say similar things. Its free tier counts every text as a request (100 a minute, 1,000 a day); each text is embedded once and kept. Unset: findings are paired only when they report results on the same dataset and metric. |
| `RA_GOOGLE_API_KEY`, `RA_OPENROUTER_API_KEY`, `RA_GROQ_API_KEY` | API keys for the providers `google` (Gemini and Gemma, via Google AI Studio), `openrouter` and `groq`. Only the providers your roles use are needed; Groq is first used by the verifier in Phase 5. |
| `RA_S2_API_KEY` | Your Semantic Scholar API key. Its limit is **1 request per second across all endpoints**. The client enforces it machine-wide: one request at a time across all processes, each sent at least 1.1 s after the previous response. Never call the API outside the client. Without a key, most requests are rejected with HTTP 429. |
| `RA_OPENALEX_API_KEY` | Your OpenAlex API key (free). Without it, OpenAlex requests share a small daily budget with everyone on your IP address; once that is used up, OpenAlex search is skipped until midnight UTC. |
| `RA_CONTACT_EMAIL` | Your email. Unpaywall requires it; OpenAlex gives faster service with it. |
| `RA_FULL_READ` | How many of the kept papers are read in full. Default 10 during development (see `END_TODO.md`). |
| `RA_ABSTRACT_ONLY` | How many papers after those are kept and read by abstract. Default 30 during development. The rest are excluded, and the trace records why. |
| `RA_REUSE_TEXTS` | `true` reuses full texts parsed by an earlier run instead of downloading them again, which makes repeated runs much faster. Default `false`: every run fetches its full texts fresh. Either way every parse of a paper is kept: a changed text (an abstract later read in full, a new arXiv version) is stored as a new version, and evidence earlier runs quote from the old one stays valid. |
| `RA_RESPONSE_CACHE_DAYS` | How many days the search APIs' responses (arXiv search, OpenAlex, Semantic Scholar, Unpaywall) are kept and reused, so a rerun of a question finds the same papers. Default 30. `0`, or `agent run --refresh`, searches afresh (and keeps what it found). Full texts are not kept as responses. |

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

`--refresh` searches afresh instead of reusing kept search responses (see `RA_RESPONSE_CACHE_DAYS`). Without it, asking a question again reuses the earlier run's brief, search results, relevance labels and every other kept LLM reply, and ranks the papers by what its own (kept) searches returned. It reads the same papers, and costs little, unless a search failed in the earlier run: failures aren't kept, so the rerun tries that search again, and if it now answers the corpus can differ, and everything after it.

`--upload` / `-u` adds a PDF you already have, such as your own draft or a paper you want considered. Uploaded papers are tagged as user-provided and always read in full, without using up the `RA_FULL_READ` quota. An arXiv PDF is recognised by its margin stamp and merged with the online copy. A file that can't be read as a PDF fails the run.

What happens:

1. A new run ID is created, e.g. `run_75e57bfa021a422d`.
2. The question goes through every pipeline stage. Each stage writes an event to the run trace, and a checkpoint is saved after it.
3. The report is written to `var/runs/<run_id>/report.md` and `report.json`, with the plan's provenance in `provenance.json`, then printed.
4. The last lines show the run ID and the commands to inspect it.

The literature stages (`brief` to `fulltext`) do real work. The reasoning model writes a research brief: the problem, two or three search angles in the field's vocabulary, and the facets of an idea. The question's own words are always searched as well, as the first angle. Each angle is searched on arXiv, OpenAlex and Semantic Scholar. The run follows references, citations and recommendations from the best hits and ranks everything with SPECTER2. The extraction model then screens the best-ranked papers and drops off-topic ones. Finally the kept papers are parsed into addressed sentences.

The knowledge-graph stages (`extract`, `build_graph`) come next. The extraction model fills a fixed form for each paper: tasks, methods, datasets, metrics, results with their conditions, claims, limitations, assumptions, future work and the paper's facets. Full-read papers are read in chunks of whole sections (about 7k tokens each, appendices left out), abstract-read papers six abstracts to a call. Every item cites the sentences it rests on, with exact quotes, and code checks every quote against the paper: an item whose quotes can't be found is dropped. Names written differently by different papers are merged. Findings in different papers that say similar things, or report results on the same dataset and metric, are labelled supports or contradicts by the extraction model. Each extraction, name grouping, pair label and embedding is kept, so a later run on the same papers makes none of those requests again. The report's *Knowledge graph* section sums up what was built; `agent graph` shows it.

The Phase 3 stages (`related_work`, `landscape`, `gaps`) work from that graph. *Related work* is the 10 corpus papers closest to the question: SPECTER2 similarity, lifted for papers that share tasks, methods and datasets with the closest ones. The *landscape* is a grid of the most-studied methods (12) against the most-studied datasets (10), marking for each pair whether a paper reports a result, a paper uses both without a result for the pair, or nothing does. *Gaps* come in three kinds: stated (a limitation or future-work item a paper reports), structural (an empty grid cell the reasoning model judges worth studying, dropped when any corpus paper names both the method and the dataset, since the graph may just have missed that paper's result) and conflict (findings that contradict each other in the same setting). Candidates of one kind that say the same thing are merged (Gemini Embedding 2 similarity of at least 0.94), then code ranks them by relevance to the question (the same embeddings, in tiers; SPECTER2 if any vector is missing), then by how many independent papers their evidence comes from; the reasoning model writes the best 15 up as gaps with a one-line warrant, drops the ones that aren't research gaps and merges duplicates; the top 5 are kept as graph nodes with their evidence. The report shows the gaps, the related work and the grid.

The Phase 4 stages (`ideate`, `novelty_search`, `novelty_assess`) turn the gaps into ideas and check how novel each is. The reasoning model writes 6 ideas from the top gaps, each fixing a reported limitation, combining methods or borrowing a technique from another area, with a purpose, a mechanism and an evaluation; code drops near-duplicates and checks 3, one per gap in gap order. When your input is an idea, it is checked too. Each idea gets its own fresh search: keyword queries and likely paper titles the extraction model writes for it (Semantic Scholar, OpenAlex, arXiv), Semantic Scholar snippet search on the whole idea, and the run's corpus; SPECTER2 keeps the 30 most similar papers and the extraction model reranks them by how many facets they share with the idea; recommendations for the 3 closest are added and the ranking is done again, down to 10. The extraction model compares the idea with each of the 10 separately, facet by facet, quoting the paper (every quote goes through the quote check), and says whether the paper already solves the idea's gap. Before an idea can be rated 3 or higher, its 3 closest papers are read in full. Code then assigns the level from rules: 1 a paper has the same purpose and mechanism; 2 a close purpose and mechanism; 3 no paper matches both; 4 also answers a stated or conflict gap that no paper solves; 5 also the gap rests on two or more papers and every paper's mechanism differs. The rules fail closed: weak search coverage, a failed comparison with one of the 3 closest, or a paper that already solves the gap caps the level at 3 (marked inconclusive, or the gap "possibly addressed"). An idea rated 1 or 2 has one facet changed by the reasoning model and is checked again, at most twice; rejected versions are remembered so they aren't proposed again. The report's Novel Evidence Report holds the best idea; *Ideas checked* lists all of them, best first.

The Phase 5 stages (`plan`, `critique`, `targeted_search`, `verify`) write a provenance-aware experiment plan for the best idea. Its facts come from the idea's 10 closest papers, its gap's papers and the run's related work; papers the run has not extracted are extracted first (in full when Phase 4 read them in full). Code builds a numbered list of facts from them and from the knowledge graph: the gap, the closest papers' own approach, datasets, methods and metrics (merged by name), reported results with their conditions, and limitations, each with its evidence sentences. The reasoning model writes the hypothesis, research question, method steps, datasets, baselines, metrics, conditions, ablations, expected results and risks, citing facts by number; datasets, baselines and metrics must be listed facts, and each expected result is stated against a reported one. Every element keeps three provenance records: its *source* (the sentences and quotes it rests on), its *derivation* (the step, round, model and the origin of each fact it cites) and its *decision* (why it was chosen, over which listed alternatives, and which critique points it answers). The critique comes next: code checks that the closest work is a baseline, that every dataset has a reported result of a chosen baseline and that nothing is uncited, then the verifier model reviews feasibility, missing baselines, ablations, confounds and whether the plan answers the gap. Points that need facts the sources lack are searched for (Semantic Scholar keyword and snippet search, up to 3 queries a round), the new papers are extracted, and the reasoning model revises the plan, answering each point; at most 2 critiques. Finally the verifier model checks each element's basis against the full sentences it cites: supported elements are kept, unsupported ones flagged, contradicted ones removed. The report's Experiment Plan shows each element with numbered sources and its reason (a flagged or removed element also with its provenance in full: what it rests on, the quotes it cites and how it was derived), the Provenance Coverage (supported elements out of all elements), what verification removed, the critique points left open and the revisions; `provenance.json` holds the whole provenance graph in PROV-JSON, and `agent plan` shows it element by element.

Timing: a run takes several minutes, mostly because arxiv.org asks for 15 s between full-text downloads (its robots.txt), Semantic Scholar allows 1 request per second, and the free LLM tiers are paced to their per-minute limits. A busy free model is retried for about a minute before the call fails. The first run also downloads the SPECTER2 model (about 440 MB). Every run downloads its full texts fresh, unless `RA_REUSE_TEXTS=true` is set. On new papers the graph stages add several minutes: about 70-100 extraction-model requests at 15 a minute, and up to about 400 embedding requests at 90 a minute. Phase 4 adds about 10-15 minutes: 1-3 reasoning calls, about 15 extraction calls and 12 Semantic Scholar requests per idea checked, and up to 3 full texts per idea from arxiv.org. Phase 5 adds about 10-20 minutes: 1-2 reasoning calls, 10-20 extraction calls for the closest papers the run had not extracted, about 8-16 verifier calls (Groq's free tier allows 8k tokens a minute, so about one call a minute, and 200k a day, about 4-6 plans) and a few Semantic Scholar requests and full texts for the targeted searches.

If a stage fails, the run is marked `failed`, and the output gives the commands to inspect and resume it. A model whose calls fail three times in a row (its provider down, say) is skipped for the rest of the run, as one whose daily quota ran out is; when a role has no model left, the stage stops and the run can be resumed once the quota is back or its key is switched.

## `agent resume <run_id>`

Continues a failed or interrupted run from the stage where it stopped. Stages that already finished are not run again.

```sh
uv run agent resume run_165c14f256c8437a
```

A stage that stopped partway (a quota running out mid-stage, say) runs again from its start, reusing every reply and response it had already got, so the resumed run ends as an uninterrupted one would (`tests/test_resume.py` checks this for every stage).

- A run that already completed can't be resumed; you'll get `Run ... has already completed.`
- An unknown ID gives `No run with ID ...`.

## `agent novelty "<idea>"`

Checks one idea's novelty on its own, without a literature review, gaps or plan: the extraction model splits the idea into a title, purpose, mechanism and evaluation, then it gets the same novelty search, comparisons and level as an idea in a full run (it is never revised). The report has the Novel Evidence Report for it.

```sh
uv run agent novelty "Distil an LLM's retrieval-quality judgements into a small evaluator for corrective RAG, tested on PopQA."
```

Such a run is not checkpointed, so `agent resume` can't continue it; check the idea again instead. It costs about 15 extraction calls and 12 Semantic Scholar requests, plus up to 3 full-text downloads.

## `agent plan <run_id>`

Shows a run's experiment plan: every element with its verdict (`supported`, `not_supported`, `contradicted`, `no_source`, `unverified`) and the facts it cites, then the Provenance Coverage.

```sh
uv run agent plan run_165c14f256c8437a
uv run agent plan run_165c14f256c8437a --element "baselines 2"
```

`--element` / `-e` traces one element, by its ID or its field and position, through the whole chain: the element and its verdict, the decision (its reason, the alternatives and the critique points it answers), how it was derived (round, model, prompt, checks), its basis, and for each cited fact where it came from (the extraction or graph query, and why its paper was read), each evidence quote with its full sentence, the section and the paper.

## `agent ideas <run_id>`

Lists every idea a run wrote or checked: its rank in the report, its round (each retry adds one), what became of it (`unchecked`, `assessed`, `rejected`), its level and why.

```sh
uv run agent ideas run_165c14f256c8437a
```

## `agent graph <run_id>`

Shows a run's knowledge graph: how many nodes and edges of each kind, and the most connected tasks, methods, datasets and metrics (with how many papers link to each).

```sh
uv run agent graph run_165c14f256c8437a
```

Or asks it one question. Names match however a paper wrote them (`lora`, `LoRA`, `Low-Rank Adaptation` once merged), and each answer shows the sentences it rests on:

| Option | Answers |
|---|---|
| `--methods-on DATASET` | Methods evaluated on a dataset: from results reported on it, then from papers that use it |
| `--limitations-for TASK` | Limitations reported by the papers that address a task |
| `--about METHOD` | Results about a method, each with the findings that support or contradict it |
| `--conflicts` | Findings that contradict each other in the same setting |
| `--sentence SENTENCE_ID` | Every node and edge that rests on a sentence, e.g. `p_1a2b3c4d5e6f7a8b:s3:12` |
| `--gaps` | The run's ranked research gaps (Phase 3), each with its warrant and evidence |

```sh
uv run agent graph run_165c14f256c8437a --methods-on GLUE
uv run agent graph run_165c14f256c8437a --limitations-for "knowledge conflicts"
```

## `agent report <run_id>`

Writes a run's `report.md` and `report.json` again from what is stored: the graph summary and related work from the run's latest stages in the trace, the grid from its graph, the gaps from their nodes, the ideas with their assessments, and the plan. It also writes the plan's `provenance.json` again. Use it after `scripts/graph_check.py --rebuild` or `scripts/gap_check.py --rerun`, which update the database and trace but not the report files (`scripts/novelty_check.py --rerun` and `scripts/plan_check.py --rerun` rewrite them themselves).

```sh
uv run agent report run_165c14f256c8437a
```

## `agent cost <run_id>`

Shows what a run cost, from its trace: time per stage (failed attempts included), each model's calls and tokens, each API's requests and how many came from kept responses, and how much earlier work it reused (LLM replies, embeddings). The same summary ends the report (`## Run cost`).

```sh
uv run agent cost run_165c14f256c8437a
```

## `agent ui`

Opens the web UI in the browser, at http://127.0.0.1:8000. Build it once first (see One-time setup).

```sh
uv run agent ui                    # --port 8001 for another port, --no-open to not open the browser
```

- **Runs**: every run with its state, best idea, plan coverage, and a badge when a quota stopped it.
- **New run**: a question or idea, optional PDFs to read in full, and "search afresh". It runs `agent run` in a child process, so closing the UI (Ctrl+C) leaves the run going. One run at a time.
- **A run's page**: its stages live (each stage's status, time and calls, and the latest model and API calls), then its overview (the Novel Evidence Report), gaps, ideas with their closest papers, the experiment plan (click an element to trace it through its facts to the quoted sentences, paper and section), the knowledge graph (click a node or edge for the sentences it rests on), the papers it found and why each was read or not, and its cost. Every quote opens its sentence in context.
- **Stops**: when a quota runs out, the page says which key to switch in `.env`; then **Resume** continues the run, reading the new key. A run started from the UI can be stopped there and resumed later.

It listens on 127.0.0.1 only, and refuses requests from other sites. Developing the UI: `cd frontend; npm run dev` serves it with hot reload at http://127.0.0.1:5173, passing `/api` to a running `agent ui`.

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
| `plan` | 5 | Write the experiment plan for the best idea from facts of its closest papers and the graph; later rounds revise it |
| `critique` | 5 | Check the plan (code, then the verifier model) for missing baselines, unsupported elements, feasibility and the gap. Can loop through `targeted_search` back to `plan` |
| `targeted_search` | 5 | Search for what the critique found missing and extract the papers found |
| `verify` | 5 | Check every cited claim against its sentences; compute Provenance Coverage |
| `report` | 0 | Write the Novel Evidence Report and the experiment plan |

---

## Tests

```sh
uv run pytest                       # offline tests (no network)
uv run pytest -m live               # one real call to each literature API and each configured LLM model
uv run pytest tests/test_pipeline.py -v
```

The live tests read `.env`. The LLM tests ask every model of every role, fallbacks included, for a research brief; a model whose provider has no key is skipped. Unpaywall's test is skipped unless `RA_CONTACT_EMAIL` is set. Semantic Scholar's test may fail with HTTP 429 until `RA_S2_API_KEY` is set.

## Phase 6 check: regression on dev topics

```sh
uv run python scripts/regression.py                          # new runs on the 7 dev topics checked since Phase 2
uv run python scripts/regression.py moe-load-balancing       # new runs on chosen topics (data/dev_topics.toml)
uv run python scripts/regression.py --resume run_ab12        # finish a run that stopped, then check it
uv run python scripts/regression.py --runs run_ab12          # check finished runs, no requests
```

Runs the whole pipeline as `agent run` does and checks each run: it completed (resuming after a stop counts), its report has every part and `provenance.json` exists, every evidence item it holds resolves to a stored sentence containing its quote, and every cited plan element traces to a section and a paper; and it records the run's cost. It stops at the first quota that runs out and prints the `--resume` command to continue. A topic run before reuses its kept work; one never run costs a full run. Results are saved to `var/checks/`.

## Phase 5 check: experiment plans

```sh
uv run python scripts/plan_check.py rag-knowledge-conflicts            # new runs on chosen dev topics
uv run python scripts/plan_check.py --runs run_ab12 run_cd34           # measure finished runs, no requests
uv run python scripts/plan_check.py --runs run_ab12 run_cd34 --rerun   # run Phase 5 again on them first
```

For each run this prints the plan element by element with its verdict, the Provenance Coverage, whether every element has its three provenance records, whether every evidence quote resolves to a real sentence containing it, whether the PROV export traces every cited element to a section and a paper, the critique points (code's and the verifier's) and the targeted searches, and Phase 5's LLM calls, tokens, API requests and time. `--rerun` runs Phase 5 again on runs whose ideas are assessed and rewrites their reports and `provenance.json`; it stops early, saying so, when a Google or Groq quota runs out. Results are saved to `var/checks/`.

## Phase 4 check: ideas and novelty

```sh
uv run python scripts/novelty_check.py rag-knowledge-conflicts            # new runs on chosen dev topics
uv run python scripts/novelty_check.py --runs run_ab12 run_cd34           # measure finished runs, no requests
uv run python scripts/novelty_check.py --runs run_ab12 run_cd34 --rerun   # run Phase 4 again on them first
```

For each run this prints every idea checked (and each rejected version), with its level, the reasons, the search coverage and the closest papers with their facet matches, checks that every comparison quote resolves to a real sentence containing it, and shows Phase 4's LLM calls, API requests and time. `--rerun` runs Phase 4 again on runs whose gaps exist and rewrites their reports; it stops early, saying so, when a Google quota runs out. Results are saved to `var/checks/`.

## Phase 4 check: planted duplicates

```sh
uv run python scripts/duplicate_test.py plant --runs run_ab12 run_cd34    # once: pick papers, write their ideas
uv run python scripts/duplicate_test.py check --filter specter2           # check every planted idea
uv run python scripts/duplicate_test.py brief                             # once: the same papers as brief ideas
uv run python scripts/duplicate_test.py check --filter specter2 --set brief
uv run python scripts/duplicate_test.py filters var/checks/duplicates-specter2-<time>.json
```

Tests whether the novelty check recognises ideas that published papers already present. `plant` picks 3 papers per run from 2025 on, outside the run's corpus, and has the extraction model restate each as an idea in its own words, without the paper's title words or method names, into `data/planted_ideas.json`. `brief` restates the same papers as short, high-level ideas, a harder test, into `data/planted_ideas_brief.json`. `check` checks each planted idea of one set (`--set exact`, the default, or `brief`; `--only` picks IDs such as `run_ab12:1`) in a run of its own (as `agent novelty` does) with the chosen embedding filter, and reports how often the source paper was found, kept by the filter, among the 10 closest, and how often the idea was rated 1 or 2. It stops when a Google daily quota runs out. `filters` scores each idea's candidate pool with both SPECTER2 and Gemini Embedding 2 and compares the source paper's rank under each (Gemini needs an embedding request for each paper text not embedded before).

## Phase 3 check: research gaps

```sh
uv run python scripts/gap_check.py rag-knowledge-conflicts              # new runs on chosen dev topics
uv run python scripts/gap_check.py --runs run_ab12 run_cd34             # measure finished runs, no requests
uv run python scripts/gap_check.py --runs run_ab12 run_cd34 --rerun     # run Phase 3 again on them first
```

For each run this prints the ranked gaps with their kind, warrant and evidence, checks that every piece of evidence resolves to a real sentence containing its quote, and shows how many candidates of each kind were found, the closest papers, the grid's cell counts, and the Phase 3 stages' LLM calls and time. `--rerun` runs the three Phase 3 stages again on runs whose graph is already built, for instance runs made before Phase 3 existed; it takes about 2 reasoning-model calls and 40-60 embedding requests a run (fewer when the same input was seen before). Results are saved to `var/checks/`.

## Phase 2 check: knowledge graphs

```sh
uv run python scripts/graph_check.py rag-knowledge-conflicts            # new runs on chosen dev topics
uv run python scripts/graph_check.py --runs run_ab12 run_cd34           # measure finished runs, no requests
uv run python scripts/graph_check.py --runs run_ab12 --sample 20        # also write a hand-check sample
```

For each run this reports the quote check's pass rate, how often each field of the form was filled (full-read and abstract papers apart), the nodes and edges, names merged, the cross-paper links, LLM calls and time, and three example queries. `--sample N` writes N extracted items, picked at random, with the sentences they cite, as a Markdown checklist in `var/checks/`, for checking by hand. A new run on a topic costs about 70-100 extraction-model requests (the two Flash-Lite models allow 500 a day each) and up to about 400 embedding requests (1,000 a day); work kept from earlier runs is not repeated. Results are saved to `var/checks/`.

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
| `var/agent.db` | Runs, papers, every parse of their texts with addressed sentences, graph, kept LLM replies, and the trace |
| `var/responses.db` | Kept search API responses (see `RA_RESPONSE_CACHE_DAYS`); delete it to search everything afresh |
| `var/checkpoints.db` | LangGraph checkpoints, used by `agent resume` |
| `var/runs/<run_id>/report.md`, `report.json` | Each run's report |
| `var/runs/<run_id>/provenance.json` | The plan's provenance graph, as PROV-JSON |
| `var/runs/<run_id>/console.log` | What a run started from the web UI printed |
| `var/uploads/<run_id>/` | PDFs uploaded in the web UI |

`var/` is gitignored. Deleting it removes every run and starts fresh.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | The run failed (resumable), the run ID wasn't found, or the run can't be resumed |
| `2` | Bad usage: invalid `.env` settings, an unknown `--fail-at` stage, or a missing argument |
