# Research Agent — Phase Plan (rough)

v3 · 2026-11-13 (v2: 2026-10-07, v1: 2026-10-02)

This replaces the phase plan (Section 5) in `architecture-overview-and-project-plan.docx`. The rest of that document still applies: the components, the two outputs, the 5-level novelty scale, the three provenance types, and the evaluation metrics.

This is a high-level plan. Details for each phase get decided when we reach it.

**Where things stand (2026-11-13):** Phases 0-6 are done, so the agent works end to end, from the CLI and from the local web UI. What is left is fine-tuning (*Remaining work*, after Phase 6), then the Phase 7 evaluation.

## Scope decisions

| Decision | Choice |
| --- | --- |
| Domain | CS/ML only, permanently. A domain-specific agent by design. |
| Interface | CLI plus reports (Markdown + JSON) with a browsable provenance trace, and (added 2026-10-14, at the user's request) a local web UI, `agent ui`: start, watch, resume and browse runs, down to each quoted sentence. Local only, one user. |
| Evaluation | Fully automated. No human raters. |
| Timeline | No fixed deadline. Phases end at milestones, not dates. |
| Agent shape | A fixed pipeline with a few loops, run by LangGraph. Not a free-roaming tool-picking agent. |
| Relation to prior work | We don't replicate any one system. We borrow mechanisms from the sources in `docs/literature/` only where they help. |
| Python | 3.13. torch, transformers, sentence-transformers and PyMuPDF install and import on it (checked 2026-10-07). |
| Caching | Allowed: this is a research project (decided 2026-10-09, replacing the 2026-10-07 "no caching" rule). Work is kept and reused only when a rerun would ask for exactly the same thing: paper extractions, name groupings and claim-pair labels when the input, prompt version and model match; embeddings per text and model. `RA_REUSE_TEXTS=true` reuses parsed full texts (off by default). The trace records what was reused. |
| Embeddings | `gemini-embedding-2` on Google's free tier (decided 2026-10-09): 100 requests/min, 30k tokens/min, 1,000 requests/day, and every text in a batch counts as one request, so only findings are embedded, each once. No local embedding or NLI model in Phase 2: the extraction model labels claim pairs, which suits scientific claims and their conditions better. SPECTER2 stays for ranking papers. |
| Literature sources | arXiv and OpenAlex are the base sources. Semantic Scholar is added when `RA_S2_API_KEY` is set (key requested 2026-10-07); without a key most of its requests are rejected with HTTP 429. |
| PDF parsing | PyMuPDF only, no GROBID. Most CS/ML papers have arXiv HTML, so PDFs are the fallback. |
| LLMs during development | Free tiers (decided 2026-10-09): extraction = Gemini 3.5 Flash-Lite, falling back to Gemini 3.1 Flash-Lite and Gemma 4 31B (Google AI Studio); reasoning = Nemotron 3 Ultra (OpenRouter, free); verifier = gpt-oss-120b (Groq). Stronger models may replace them before Phase 7 (*Remaining work* 1). |
| Corpus size during development | 10 papers read in full + 30 by abstract (decided 2026-10-09), to fit the free LLM limits. On the dev topics, the top 40 hold 86% of the key papers against 88% for the top 100. |

## Principles

1. **Every phase ends runnable.** Each milestone is something you can run from the CLI on real research questions.
2. **Provenance starts in Phase 0.** The run trace and sentence-level addresses (paper → section → sentence) exist before any component needs them. They are hard to add afterwards.
3. **The two contributions sit in the middle.** Novelty assessment (Phase 4) and provenance-aware planning (Phase 5) are built on top of a working corpus and knowledge graph, with hardening and evaluation after them.
4. **Free first.** Free literature APIs, free model tiers, a small local model where it fits (SPECTER2 for ranking), embedded storage. LLMs are the only paid part. They are called through one provider-neutral interface with three roles:
   - *extraction*: cheap, high volume
   - *reasoning*: ideas, gaps, plans
   - *verifier / judge*: a different provider or model family from the reasoning role
5. **Reuse what is unchanged.** LLM work and embeddings are kept and reused when the input, prompt version and model are the same; searches are made again on every run, so a rerun sees the literature as it is then. Request pacing and backoff keep us inside rate limits, and the trace records what each run fetched and what it reused.

## Overview

| Phase | Builds | Milestone | Status |
| --- | --- | --- | --- |
| 0. Foundations | Project skeleton, clients, storage, run trace | A stub pipeline runs end to end and writes a trace | done 2026-10-07 |
| 1. Input + literature acquisition | C1, C2 | A ranked, parsed corpus for any CS/ML question | done 2026-10-09 |
| 2. Research knowledge graph | C3 | An evidence-backed graph for any corpus | done 2026-10-10 |
| 3. Related work + gap detection | C4 stages 1–2 | Ranked gaps, each backed by quoted evidence | done 2026-10-10 |
| 4. Idea generation + novelty assessment | C4 stage 3 | A Novel Evidence Report with a novelty level 1–5 | done 2026-10-11 |
| 5. Provenance-aware experiment plan | C5 | A verified plan where every element traces to a paper section | done 2026-10-13 |
| 6. Hardening + integration | Whole system | Stable, resumable, cost-measured runs | done 2026-10-14 |
| 7. Evaluation | Evaluation | Metrics vs baselines, with an evaluation report | next, after fine-tuning |
| 8. Extensions (optional) | Improvements | Chosen "later" items, each measured | not started |

```
0 ──► 1 ──► 2 ──► 3 ──► 4 ──► 5 ──► 6 ──► 7 ──► 8
            └─ benchmark topics + simple baselines can start here, in parallel ─┘
```

---

## Phase 0 — Foundations

**Status:** done (2026-10-07).

**Goal:** The skeleton every later phase plugs into.

- Project layout with `uv` on Python 3.13. Confirm the local ML libraries install on it.
- Config and secrets through `.env`, and a CLI skeleton with Typer (`run`, `resume`, `inspect`).
- **LLM interface:** one internal module with the three roles above and structured outputs validated with Pydantic. Providers can be swapped by config; each role gets its model when a phase first needs it.
- **API clients** for Semantic Scholar, arXiv, OpenAlex and Unpaywall, with retry/backoff (`httpx` + `tenacity`) and per-host request pacing.
- **Storage:** a first SQLite schema covering papers, sections, sentences, graph nodes and edges, and trace events.
- **Run trace:** a PROV-DM-shaped event log (inputs, outputs, step, prompt version, sources used and dropped). Every later component writes to it.
- **LangGraph skeleton** with a stub node for every stage, plus SQLite checkpointing so runs can resume.
- **Dev topic set:** 10–20 real CS/ML research questions, used to check each phase by hand.

**Done when:** `agent run "<question>"` passes through every stub stage, writes a trace, and prints an empty report.

## Phase 1 — Research input + literature acquisition (C1, C2)

**Status:** done (2026-10-09). The LLM research brief (the question's own words plus 2-3 angles in the field's vocabulary) and the LLM relevance check are in, on the free development models. Checked end to end on one dev topic: 545 candidates, 3 of 4 key papers found, all 10 full-read papers parsed, about 7 minutes. The full 18-topic key-paper check, and tuning retrieval with it, are deferred to the end (*Remaining work* 3); they affect retrieval quality, not the pipeline. Before the LLM brief, the check gave 57/64 key papers found (89%), 48/64 in the top 20 (75%) and 56/64 in the top 100 (88%).

**Goal:** For any CS/ML question, find the right papers and turn them into addressable text.

- **Research brief:** the LLM turns the question or idea into a problem statement, a few search angles, keywords, and facets (purpose / mechanism / evaluation) when the user gives an idea.
- **Search:** arXiv API and OpenAlex search. With a Semantic Scholar key, also Semantic Scholar keyword search and snippet search (passages inside papers). Deduplicate by arXiv, DOI, OpenAlex and Semantic Scholar IDs, and by normalised title.
- **Expand** from the best hits through OpenAlex references, citations and related works. With a key, also Semantic Scholar references, citations and recommendations.
- **Rank:** SPECTER2 similarity (computed locally; the vectors match Semantic Scholar's), combined with a citation-rate prior and the number of channels and seed papers that led to each paper, then an LLM relevance check. Similarity alone buried the foundational papers (12% of key papers in the top 20). Split the results into an *abstract-only* set and a *full-read* set.
- **Full text:** use arXiv HTML first. Otherwise use open-access PDFs (from arXiv, or found via OpenAlex, Unpaywall or Semantic Scholar), parsed with PyMuPDF. Each paper becomes sections and sentences with stable IDs.
- **User uploads** go through the same parsing, tagged "user-provided".
- **Retrieval log:** every query, every result, and what was kept or dropped, with the reason.

**Done when:** each dev topic yields a ranked corpus with its full-read papers parsed into sentences, and the obvious key papers for each topic are found.

## Phase 2 — Research knowledge graph (C3)

**Status:** done (2026-10-10). After the check below, two fixes: an identical re-parse of a paper no longer deletes evidence, and extraction (prompt v2, plus a filter in code) skips vague names such as "several benchmarks" or "large language models". Two smaller issues are left for the end (*Remaining work* 4). Checked on 6 dev topics (2026-10-10): RAG with conflicting documents, PEFT, scientific RAG, long context, offline RL, vision under distribution shift. Quote check 96.7-98.7% per topic, 97.97% over 5,310 quotes; failures are mostly LaTeX-heavy sentences the model can't copy exactly, small rewordings, and abstract sentences replaced when a later run parsed the paper's full text. 40 hand-checked items all quote their sentences faithfully; 5 are weak (tasks or entities that are too generic or taken from related work). Per topic: 23-29 extraction calls in 3-4.5 min, then 19-21 calls and 2-4 min for the graph, about 130-270 embedding requests, so the 1,000 a day cover about 5 new topics. 0-3 contradictions per topic.

**Goal:** Turn the corpus into a graph in which every fact points to the sentence it came from.

- **Extraction form** for each paper and section:
  - task, methods, datasets, metrics
  - results, with their conditions
  - limitations, assumptions, future work
  - purpose / mechanism / evaluation facets
  - atomic claims, each quoting its evidence sentences
- **How papers are read:** full-read papers in chunks of whole sections, about 7k tokens each, without appendices; the first chunk also gives the facets. Abstract-read papers get a shorter form (no limitations, assumptions or future work), six abstracts to a call. Evidence is cited as sentence addresses shown to the model (`[s3:12]`) with exact quotes.
- **Quote check:** code confirms that every quoted span exists in the paper, after normalising Unicode, dashes, quotation marks, spacing and case. A quote found in another sentence the model was shown is moved there; anything else is dropped, and so is an item left without evidence. Missing fields stay empty rather than being filled with placeholders.
- **Name merging** for methods, datasets, metrics and tasks: a normalised key first (case, punctuation, generic words such as "dataset"), then the extraction model groups the distinct names of each kind, keeping versions and sizes apart. Embeddings are not used here: each text counts as a request on the free tier.
- **Graph:** nodes and edges stored in SQLite under the run's ID, with evidence rows, and loaded into NetworkX for queries.
- **Cross-paper links:** candidate pairs are findings with similar text (`gemini-embedding-2`) or results on the same dataset and metric; the extraction model labels each pair supports, contradicts or unrelated. A contradiction is kept only in the same setting.
- **Reuse:** each paper chunk is extracted once and reused across questions; name groupings, pair labels and embeddings are kept too.
- **Graph queries:** `agent graph <run_id>` with `--methods-on`, `--limitations-for`, `--about`, `--conflicts`, `--sentence`.

**Done when:** every dev topic has a graph, the quote-check pass rate is measured, a sample of extractions has been checked by hand, and the example queries return sensible answers.

## Phase 3 — Related work + gap detection (C4, stages 1–2)

**Status:** done (2026-10-10). Checked on 7 dev topics: the 6 Phase 2 runs (Phase 3 run on their graphs) and a new end-to-end `agent run` on LLM-as-judge bias. Final check, with Gemini Embedding 2 for gap relevance and merging: every topic has 5 ranked gaps, all 47 pieces of gap evidence resolve to real sentences containing their quotes, and no embedding vector is missing. Most gaps are stated limitations or future work; vision has two conflict gaps (scale vs data diversity; whether synthetic-shift robustness transfers to natural shifts) and RAG one structural gap (CRAG never reported on 2WikiMultihopQA). The co-mention check dropped 20 of the 35 empty cells the reasoning model picked, including Longformer on Long Range Arena, which that benchmark's paper does test. Switching gap relevance from SPECTER2 to Gemini replaced offline RL's generic 2019-2020 problem statements with specific open issues. Cost per run: 2 reasoning calls (1.5-10 min with free Nemotron, which fell back to Gemini 3.1 Flash-Lite twice when overloaded) and 30-60 embedding requests, kept. Known weak spots, left for the end (*Remaining work* 5): stated gaps can still be problems newer corpus papers have worked on, and near-duplicate gaps can both appear.

**Goal:** Map what has and hasn't been studied, and produce gaps backed by evidence.

- **Related work:** the 10 papers closest to the question: z(SPECTER2 similarity) + 0.5 · z(log personalized PageRank over the paper-entity graph, seeded by that similarity). On the dev topics the graph term changes about 1 paper in 10.
- **Landscape grid:** the most-studied 12 methods × 10 datasets (decided 2026-10-10), from the graph. A cell holds a result for the pair, a paper using both without one, or nothing; the metrics of its results are kept.
- **Gap candidates:**
  - *stated*: limitation and future-work items
  - *structural*: empty grid cells, of which the reasoning model picks at most 5 worth studying; a pick is dropped when any corpus paper names both the method and the dataset (an empty cell can just mean extraction missed the result: Long Range Arena benchmarks Longformer, yet that cell was empty)
  - *conflicts*: findings linked by "contradicts" (same setting only)
- **Ranking (code, decided 2026-10-10):** relevance to the question first (Gemini Embedding 2 similarity, in tiers of half a standard deviation; switched from SPECTER2 at the user's suggestion on 2026-10-10, as SPECTER2 squeezed short sentences into a narrow band and couldn't tell paraphrases apart; SPECTER2 remains the fallback when any vector is missing), then the number of independent papers in the evidence, then raw relevance. Before ranking, candidates of one kind with similarity of at least 0.94 are merged (distinct gaps within a topic reach 0.90-0.93; clear paraphrases about 0.95), so a gap several papers raise counts them all. About 40-60 embedding requests a run, kept.
- **Gap record:** the reasoning model writes up the best 15 as a gap and a one-line warrant (claim, grounds, warrant as in GAPMAP), drops non-gaps (implementation details, problems newer work has addressed, restatements of the question's broad problem) and merges duplicates. The top 5 (decided 2026-10-10) are stored as `gap` nodes with their evidence and `rests_on` edges, shown in the report and by `agent graph --gaps`; `agent report` rewrites a run's report after a rebuild. About 2 reasoning calls a run.

**Done when:** each dev topic has a ranked gap list, and every gap's evidence resolves to real sentences.

## Phase 4 — Idea generation + novelty assessment (C4, stage 3) — contribution 1

**Status:** done (2026-10-11). Checked on the 7 dev topics (the Phase 3 runs) with the free development models. Every topic gets a Novel Evidence Report, and all 1,369 evidence items (470 comparisons, 47 assessed ideas) resolve to real sentences containing their quotes. Duplicate test, 21 papers from 2025 on outside the corpora: restated in full, all 21 caught at level 1, each by its own source; restated briefly at a high level, all 21 caught by the source (20 at level 1, 1 at level 2), which the filter ranked in its top 9 and the rerank put first in 20. Filter: SPECTER2, not Gemini Embedding 2 (decided 2026-10-11). Both kept every source among the 30, but on the 4 ideas where SPECTER2 didn't rank the source first, its median rank was 3, against 15 for Gemini and 14 for a mix of the two; Gemini would also cost about 1,200 embedding requests per run. Final levels of the 3 ideas per topic: RAG 4/3/2, PEFT 4/4/2, scientific RAG 4/4/2 (the user's own idea 2), long context 3 (inconclusive)/2/2, offline RL 4/3 (inconclusive)/2, vision 4/4/2, LLM-as-judge 4/4/4; no level 5. 15 of the 21 first ideas came out at level 1-2, mostly against 2025-2026 papers, and went round the retry loop; 7 lineages stay at level 2 after both revisions (on RAG, three versions of hallucination-aware decoding each matched an existing paper). The "possibly addressed" cap fired once (long context: a 2026 paper solves a gap whose evidence is from 2023), and 3 of 470 comparisons judged a paper to solve the gap. A first version rejected revisions whose changed facet scored 0.94-0.97 against the one it replaced, though they named a genuinely different technique; the threshold for repeats within an idea is now 0.98. Cost per run: 2-3 reasoning calls, 50-112 Gemini 3.5 Flash-Lite calls (500 a day per key, so about 5-7 runs), 2-11 embedding requests, 16-30 min with the retry loop. Known weak spots, left for the end (*Remaining work* 6): the comparison model can mislabel a facet either way and the quote check doesn't catch it, an idea can answer its gap only in part, and the thresholds come from 7 topics.

**Goal:** Propose ideas and rate how novel they are, honestly.

- **Idea generation** from the top gaps: one reasoning call writes 6 ideas, each fixing a reported limitation, combining methods or borrowing a technique from another CS/ML area, and each specific, feasible for an academic lab within months and testable on named datasets, metrics and baselines. An idea has a title, the three facets (purpose, mechanism, evaluation), the gap it answers, what it builds on and a feasibility note. Code drops ideas naming no real gap and near-duplicates (Gemini Embedding 2 similarity of at least 0.94, as for gaps), then checks **3 per run** (decided 2026-10-10), one per gap in gap order. When the user's input is an idea, it is checked too, with the facets the brief gave it, and never revised.
- **Independent novelty search,** after the Idea Novelty Checker (Shahid et al. 2025): keyword queries and likely paper titles that the extraction model writes for the idea (Semantic Scholar, OpenAlex, arXiv), Semantic Scholar snippet search on the whole idea, and the run's corpus. An embedding filter keeps the 30 most similar, the extraction model reranks them by facet match (papers that may already be the idea, then same purpose and domain, then one shared facet), and Semantic Scholar recommendations for the 3 closest (OpenAlex related works without a key) are added before a second filter and rerank down to the 10 closest. Coverage is recorded: the candidates, the channels that answered, how many of the closest more than one channel found, and how many came only from the recommendations round.
- **Closest-work comparison:** the extraction model compares the idea with each of the 10 separately, by abstract: for each facet, the paper's own version with a quote, and same / similar / different / not stated; whether the paper already solves the idea's gap; and how the idea differs. Every quote goes through the Phase 2 quote check, and comparisons are stored with their evidence. Before an idea can be rated 3 or higher, its 3 closest papers are read in full (decided 2026-10-10): arXiv HTML or a PDF, the first 60k characters. A paper stored as an abstract only gets its body added after the abstract, so evidence other runs hold on the abstract stays valid.
- **Rule-based levels 1–5,** computed by code: 1 some paper has the same purpose and the same mechanism; 2 some paper has a same or similar purpose and a same or similar mechanism; 3 no paper matches both; 4 as 3, and the idea answers a stated or conflict gap that no compared paper solves; 5 as 4, the gap rests on at least 2 independent papers, and every compared paper's mechanism is different. The rules fail closed: a "different" no quote supports counts as "similar", an unstated mechanism blocks level 5, and weak coverage (no Semantic Scholar key, fewer than 30 candidates, or 3 or more of the closest found only by the recommendations round), a failed comparison with one of the 3 closest, or fewer than 5 comparisons cap the level at 3 and mark it inconclusive. A paper judged to solve the gap caps it at 3 too, and the report names it ("possibly addressed by"); this is how Phase 4 catches the stale gaps Phase 3 can leave. The result is always worded as potential novelty.
- **Retry loop:** an idea at level 1–2 goes back to the reasoning model with the papers that matched it; it changes exactly one facet (code keeps the other two), and the new version is searched and checked again, at most twice. Rejected versions are remembered: they are shown to the model as ideas not to propose, and a revision repeating one (the changed facet at 0.98 against earlier versions of the same idea, since a genuinely different technique still scored 0.87-0.97 against the one it replaced; or the whole idea at 0.94 against other ideas) is rejected unchecked. Each idea's best version is ranked: level, conclusive before inconclusive, the user's own idea first on a tie, then gap rank.
- **Novel Evidence Report:** Proposed Idea, Novelty Assessment, Evidence, Research Gap, Closest Existing Work, Difference from Existing Work, filled from the best idea; *Ideas checked* lists every idea's best version and its earlier versions. `agent novelty "<idea>"` checks one idea on its own; `agent ideas <run_id>` lists a run's ideas.
- **Duplicate test** (`scripts/duplicate_test.py`): 3 papers per dev topic from 2025 on, outside its corpus, restated by the extraction model as ideas in their own words without title words or method names, each checked on its own. Caught means level 1 or 2. Only "not novel" ideas are tested, since a "novel" label depends on what a search happened to find. The same test compared the two embedding filters (decided 2026-10-10: test both, keep the better).

**Done when:** each dev topic gets a report, the duplicate test catches most planted ideas, and every evidence item resolves to a sentence.

## Phase 5 — Provenance-aware experiment plan (C5) — contribution 2

**Status:** done (2026-10-13). Checked on the 7 dev topics (the Phase 4 runs) with the free development models: each run plans its best idea and gets both linked outputs, the report's Experiment Plan and `provenance.json`. Provenance Coverage: RAG 30/45 (67%), PEFT 24/41 (59%), scientific RAG 28/47 (60%), long context 22/34 (65%, plus 1 element removed as contradicted), offline RL 22/31 (71%), vision 18/37 (49%), LLM-as-judge 19/31 (61%); 163 of 266 elements overall (61%). Every element has its derivation and decision records and every cited one its sources: all 813 evidence items resolve to real sentences containing their quotes, and the PROV walk reaches a section and a paper from 259 of the 266 elements (the other 7 cite nothing and are flagged "no source"). Two verifier fixes came from the dev runs. The first prompt judged the plan's own design choices (51% on the first RAG run), so it now judges only what an element says the literature states, and the writer's basis states only what its facts say. The writer names facts by number in its bases ("F34 ..."), and the verifier, not told what a number named, marked such claims unsupported for that alone; it now sees each cited fact with its papers and why the plan uses them (vision went from 35% to 49%). Most remaining "not supported" verdicts are the free writer claiming more than its facts say ("F11 provides the WordNet hierarchy"). Each plan went through 2 critiques of 6-17 points; the targeted searches added 6-9 papers a plan; and all 14 critiques judged that the plan answers its gap only "partly". Cost per plan: 2 reasoning calls, 11-16 Gemini 3.5 Flash-Lite calls, 8-16 verifier calls (35-60k Groq tokens, so a free key covers 4-6 plans a day), a few Semantic Scholar requests and full texts, 10-20 min. Known weak spots, left for the end (*Remaining work* 7): for a dataset, baseline or metric "supported" often only means the sources name it; the free verifier sees shortened facts and sentences; the gap is only partly answered and Phase 5 can only report it. A rerun can also rewrite a plan when a live search returns different papers (Phase 6).

**Goal:** An experiment plan in which every element is traceable to the literature.

- **Plan structure:** hypothesis, research question, method, datasets, baselines, metrics, conditions, ablations, expected results, risks and limitations, for the run's best idea (Phase 4's rank 1).
- **Facts the plan can cite:** the plan papers are the idea's 10 closest papers (from its novelty search), its gap's evidence papers and the run's related work. Most closest papers are outside the run's corpus (9 of 10 on the RAG topic), so those the run has not extracted go through the Phase 2 extraction (kept, so a rerun asks nothing again), in full when Phase 4 read them in full. From them and the run's graph, code builds a numbered menu: the gap, the closest papers' own approach (as the novelty comparison quoted it), datasets, methods and metrics merged by name with the graph's names, results with their conditions (the plan papers' own, and any corpus paper's on a menu dataset), limitations and findings. Each fact keeps its evidence sentences and its origin. Code ranks the menu: entities the idea names first, then by how many plan papers use them, the 3 closest counting most.
- **Fields filled from the graph:** the reasoning model writes the plan from the menu, citing facts by number: datasets, baselines and metrics are listed facts, an expected result is stated against a reported result, a risk preferably repeats a reported limitation, and the other fields cite what they rest on. Code checks the citations: an entity citing no listed one of its kind is marked "not from the literature".
- **Three provenance records per element** (`plan_elements`, with evidence rows):
  - *source*: the cited facts' evidence sentences, with the exact quotes
  - *derivation*: the step, round, model and prompt that wrote it, and each cited fact's origin (the extraction or graph query, and why its paper was read: closest paper n, gap evidence, related work, or the targeted search with its query)
  - *decision*: the model's reason, the listed alternatives of the same kind it was chosen over, and the critique points it answers
- **Critique loop:** code checks first (each of the 3 closest papers whose purpose the novelty check found the same or similar is a baseline; every dataset has a reported result of a chosen baseline; nothing is uncited), then the verifier model reviews feasibility, missing baselines, whether the datasets and metrics test the hypothesis, whether each ablation isolates one part of the mechanism, confounds, and whether the plan answers the gap in full. A point needing facts the sources lack carries a query: up to 3 a round go to a targeted search (Semantic Scholar keyword and snippet search; OpenAlex and arXiv without a key), whose best 3 new papers per query are extracted by abstract, and a snippet hit from a paper's body also by the chunk holding it. The reasoning model then revises the plan, answering each point (fixed or declined). At most 2 critiques; the last one's points are reported as open.
- **Verification pass:** the verifier-role LLM (decided 2026-10-11: no local NLI model, as in Phase 2) labels every element's basis, the statement about the literature it rests on, against the facts it cites (what each fact number names, its papers and why the plan uses them) and their full sentences: *supported*, *not supported* or *contradicted*. Contradicted elements are removed, unsupported ones flagged, uncited ones flagged as having no source. Provenance Coverage = supported elements ÷ elements of the final plan (the architecture document's supported decisions ÷ total decisions).
- **Output:** the plan is linked to the Novel Evidence Report (the same idea, named in the plan's header). The provenance graph is exported as PROV-JSON (with the `prov` package) to `var/runs/<run_id>/provenance.json`: element → decision and generating step (with its models) → fact → sentence → section → paper, plus the plan versions, critiques and verdicts. `agent plan <run_id>` lists the plan, and `--element` traces one element: recommendation → reasoning → evidence → paper → section. `scripts/plan_check.py` measures coverage, the three records, quote resolution and traceability.

**Done when:** every dev topic produces both linked outputs end to end, every plan element has provenance, and a coverage figure is reported.

## Phase 6 — Hardening + integration

**Status:** done (2026-10-14). Checked live on two new topics, as the user chose; the 7 dev topics run again with the final models (*Remaining work* 1), since each would cost about a full run on the free tiers. Both ran end to end with no manual step but a resume, every report part present:
- *moe-load-balancing* (a question): 137 min in stages. The process was killed once in the novelty search (the machine ran low on memory) and `resume` finished the run. 136 Gemini Flash-Lite calls (585k tokens in, 158k out), 8 Nemotron, 10 verifier calls (41k Groq tokens), 168 Semantic Scholar requests. Provenance Coverage 17/36 (47%); all 1,602 evidence items resolve to sentences containing their quotes; all 36 cited plan elements trace to a section and a paper. Best idea at level 4.
- *draft-free-speculative-decoding* (an idea): 77 min. 166 Gemini Flash-Lite calls (675k in, 208k out), 8 Nemotron, 12 verifier calls (47k Groq tokens), 168 Semantic Scholar requests. Coverage 16/36 (44%); all 1,752 evidence items resolve; 34 of 34 cited elements trace (2 cite nothing, flagged "no source"). Ideas at levels 4, 3 and 2.

Time goes mostly to extraction and the graph (68 min on the bigger topic, at Gemini's 15 requests a minute), the novelty loop (about 30 min) and the gaps (8-13 min). A free Google key (500 calls a day per model) covers about 3 such runs a day, a Groq key 4-6 plans, OpenRouter's 50 free calls about 6 runs. A rerun reused the brief, the kept searches, 35-47 of 49 extractions and most embeddings and claim-pair labels, so its literature and graph stages took minutes; its ideas and plan can differ from the first run's (accepted, see *Reruns* below). Coverage on these two (44-47%) is below the dev topics' 61%: mostly the free writer claiming more than its facts say (*Remaining work* 7).

Two more runs of the finished agent followed the same day, on model merging (killed once for low memory and resumed) and time-series forecasting: both pass `scripts/regression.py`, with coverage 67% and 55%. Reading the four runs' reports found the weak spots listed in *Remaining work* 5-6.

**Goal:** From a working prototype to a dependable tool.

- **Robustness:**
  - *Resuming:* a run that stops (a quota used up, a crash) resumes from its last finished stage with `agent resume`; a stage that stopped partway runs again from its start, reusing every reply and response it had already got. `tests/test_resume.py` stops a run (on fake models and APIs) at the first and the last model call of every stage, resumes it, and checks that it ends exactly as an uninterrupted run: the same report and no duplicate rows.
  - *Failing models:* a model whose calls fail 3 times in a row (its provider down; the dead Gemma endpoint cost a minute a call) is skipped for the rest of the run, as one whose daily quota ran out is. When a role has no model left, the stage stops; the user switches the key and resumes.
  - *Reruns reuse earlier work:* the brief, the relevance labels and the idea facets are kept like every other LLM reply, and the search APIs' responses for 30 days (`RA_RESPONSE_CACHE_DAYS`; `agent run --refresh` searches afresh and keeps what it finds), so asking a question again costs a fraction of the first run. A run ranks its candidates by what its own searches returned (`run_papers.seen`, migration 0008), not by the paper rows, which merge every record fetched since (in the first live rerun, a citation count a later request had fetched, 0 → 9, lifted a paper from rank 124 into the relevance check). A rerun can still differ from the first run: a search that failed then is tried again (failures aren't kept), and if it answers, the corpus and everything after it change. That is accepted (user, 2026-10-14): results may vary between runs.
- **Versioned paper texts:** every parse of a paper is kept (`paper_texts`, migration 0007). A changed text (an abstract later read in full, a new arXiv version) is stored as a new version whose sections are numbered on from the old one's, and the old sections are retired, not deleted, so an earlier run's evidence keeps pointing at the text it quoted; readers see the current text.
- **Measurement:** `agent cost <run_id>` and the report's *Run cost* section, from the trace: time per stage (failed attempts included), each model's calls and tokens, each API's requests and how many came from kept responses, and how much earlier work was reused.
- **CLI polish:** run, resume, inspect a trace, export a report, and now the cost. Reports show plain numbered citations; a flagged or removed plan element also shows its provenance in full (what it rests on, the quotes it cites, how it was derived), and `agent plan --element` traces any element.
- **Web UI** (added at the user's request): `agent ui`, a local FastAPI server and React app. It starts runs (one at a time, as child processes), shows each stage live, says which key to switch when a quota stops a run and resumes it, and browses every result: the report, gaps, ideas with their closest papers, the plan with each element traced to its quoted sentences, the knowledge graph, the corpus and the cost.
- **Tests:** unit tests for the parsers, the quote check and the novelty-level rules (`test_parse`, `test_graph`, `test_ideas`), the resume test, and a regression run on the dev topics (`scripts/regression.py`), which checks every run's report parts, evidence and traceability and records its cost.

**Done when:** all dev topics run without manual intervention, and cost and time per run are documented.

## Remaining work: fine-tuning

The agent works end to end; what is left changes the quality of its output, not the pipeline. It comes before Phase 7, so the evaluation measures the tuned agent. Other sections refer to its items by number.

1. **Final models.** Every check so far ran on free tiers (see `.env.example`). Choose the final model for each role, keeping the verifier in a different model family from reasoning, and the cost budget per run, then run the 7 dev topics again with them (`scripts/regression.py`). Still open: run Phase 7 on the free models, which makes them the final setup, or swap first.
2. **Corpus size.** `RA_FULL_READ` / `RA_ABSTRACT_ONLY` are 10 + 30 to fit the free LLM limits. Go back to 20 + 80, or whatever the re-run checks support.
3. **Retrieval** (Phase 1). Run the full key-paper check (`scripts/key_paper_recall.py --full-text` on all 18 dev topics, about 2 hours) and tune with it:
   - the ranking weights (`CITATION_WEIGHT`, `LINK_WEIGHT` in `literature/stages.py`), chosen before the relevance check existed and not yet confirmed;
   - giving full-read places to papers the relevance check labels core first (on one topic, 8 of the 10 went to broad surveys labelled related); the stored labels let this be measured without new LLM calls;
   - the brief prompt, which tends to paraphrase the problem instead of using the literature's name for it ("conflicting evidence" for "knowledge conflicts");
   - how strict the arXiv queries are (two ANDed phrases often return nothing).
4. **Extraction** (Phase 2). Quotes of LaTeX-heavy sentences fail the quote check, since the model can't copy them exactly, and some extracted tasks and entities are too generic or taken from a paper's related work.
5. **Gaps** (Phase 3). A stated gap can be a problem newer corpus papers have already worked on (2021's "merging more than two models"), a paper's own motivation can be read as a gap (WEMoE, Autoformer), and near-duplicate gaps can both appear.
6. **Novelty check** (Phase 4).
   - A paper stored with a title only (no abstract or text) can't be compared. The failed comparison caps the idea at level 3, inconclusive, which then outranks checked level-2 ideas, so Phase 5 can plan a likely duplicate (model merging: "Model Merging using Geometric Median of Task Vectors").
   - Reading the 3 closest papers in full before a level of 3 or higher fails open: when no full text can be fetched (ACM 403, IEEE, no IDs) the abstract verdicts stand (time series: level 4 with none of the 3 read in full).
   - The search can miss the closest work: the rerank dropped Kangaroo (draft-free speculative decoding), and forecast combination (FFORMA) wasn't found (time series).
   - The comparison model can mislabel a facet either way and the quote check doesn't catch it, an idea can answer its gap only in part, and the thresholds come from 7 topics.
7. **Experiment plan** (Phase 5). The free writer claims more than its facts say ("F11 provides the WordNet hierarchy"), the main reason Provenance Coverage stays at 44-71%. For a dataset, baseline or metric, "supported" often only means the sources name it; the free verifier sees shortened facts and sentences; and every critique so far judged that the plan answers its gap only partly, which Phase 5 can only report.
8. **Phase 7 benchmark topics** are chosen after the final models are fixed: they must come from papers published after every model's training cutoff.

## Phase 7 — Evaluation (automated)

**Goal:** Measure novelty and provenance against baselines.

- **Benchmark set:** about 50 CS/ML topics drawn from papers published after the LLMs' training cutoff, so the models can't have memorised the answers. Each real paper serves as a reference idea and upper bound.
- **Baselines:**
  - LLM only
  - plain RAG (same retrieval, no graph)
  - ablations of our system: without the graph, without the novelty search, without the verifier, and a random-context control
- **Novelty:**
  - a pairwise LLM-judge tournament scored as ELO; each pair is judged in both orders, by a judge from a different provider than the generator
  - a temporal hold-out: do the ideas anticipate papers published after the cutoff?
  - calibration of the novelty levels on planted duplicates
- **Provenance:** Provenance Coverage, citation correctness, citation completeness, provenance consistency, and traceability (checked automatically by walking every chain). The verifier is checked separately on the SciFact benchmark.
- **Reporting:** bootstrap error bars, plus a self-audit: traces, prompts, selection policy, and stated limitations.

**Known limitation:** without human raters we can't check whether the LLM judge agrees with experts. The cross-provider judge, the order swapping and the real-paper reference reduce this risk, and the write-up must state it.

**Done when:** an evaluation report shows every metric against every baseline.

## Phase 8 — Extensions (optional)

Each extension is added only if the Phase 7 numbers show room for it, and each is measured with the same harness:

- "what changed" edges along citation chains
- analogies from more distant CS/ML areas (distance-controlled retrieval)
- running several models together for gap extraction
- mining GitHub code and READMEs for datasets and configs
- adaptive provenance display

## Cross-cutting

- **Dev topics** (Phase 0) are used to check every phase. **Benchmark topics** (Phase 7) are kept separate so we don't tune on the test set.
- **Parallel track:** the benchmark set and the LLM-only and plain-RAG baselines can be built any time after Phase 1.
- **Trace discipline:** every new component writes trace events from its first commit.

## Open questions (settle during the build)

- ~~One idea per run, or the top 3 ranked by novelty level.~~ Settled 2026-10-10: 3 ideas checked and ranked; Phase 5 plans the best.
- Optional CLI pause points (confirm the brief, pick the idea before planning).
- Which models fill the three LLM roles in the final system, and the cost budget per run (development uses free tiers; *Remaining work* 1).
