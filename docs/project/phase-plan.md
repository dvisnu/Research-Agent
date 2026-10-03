# Research Agent — Phase Plan (rough)

Draft v1 · 2026-10-02

This replaces the phase plan (Section 5) in `architecture-overview-and-project-plan.docx`. The rest of that document still applies: the components, the two outputs, the 5-level novelty scale, the three provenance types, and the evaluation metrics.

This is a high-level plan. Details for each phase get decided when we reach it.

## Scope decisions

| Decision | Choice |
| --- | --- |
| Domain | CS/ML only, permanently. A domain-specific agent by design. |
| Interface | CLI plus reports (Markdown + JSON) with a browsable provenance trace. No web UI. |
| Evaluation | Fully automated. No human raters. |
| Timeline | No fixed deadline. Phases end at milestones, not dates. |
| Agent shape | A fixed pipeline with a few loops, run by LangGraph. Not a free-roaming tool-picking agent. |
| Relation to prior work | We don't replicate any one system. We borrow mechanisms from the sources in `docs/literature/` only where they help. |

## Principles

1. **Every phase ends runnable.** Each milestone is something you can run from the CLI on real research questions.
2. **Provenance starts in Phase 0.** The run trace and sentence-level addresses (paper → section → sentence) exist before any component needs them. They are hard to add afterwards.
3. **The two contributions sit in the middle.** Novelty assessment (Phase 4) and provenance-aware planning (Phase 5) are built on top of a working corpus and knowledge graph, with hardening and evaluation after them.
4. **Free first.** Free literature APIs, small local models (embeddings, NLI), embedded storage. LLMs are the only paid part. They are called through one provider-neutral interface with three roles:
   - *extraction*: cheap, high volume
   - *reasoning*: ideas, gaps, plans
   - *verifier / judge*: a different provider or model family from the reasoning role
5. **Cache everything.** API responses and per-paper extractions are cached. This keeps us inside free-tier limits, controls cost, and makes reruns reproducible.

## Overview

| Phase | Builds | Milestone |
| --- | --- | --- |
| 0. Foundations | Project skeleton, clients, storage, run trace | A stub pipeline runs end to end and writes a trace |
| 1. Input + literature acquisition | C1, C2 | A ranked, parsed corpus for any CS/ML question |
| 2. Research knowledge graph | C3 | An evidence-backed graph for any corpus |
| 3. Related work + gap detection | C4 stages 1–2 | Ranked gaps, each backed by quoted evidence |
| 4. Idea generation + novelty assessment | C4 stage 3 | A Novel Evidence Report with a novelty level 1–5 |
| 5. Provenance-aware experiment plan | C5 | A verified plan where every element traces to a paper section |
| 6. Hardening + integration | Whole system | Stable, resumable, cost-measured runs |
| 7. Evaluation | Evaluation | Metrics vs baselines, with an evaluation report |
| 8. Extensions (optional) | Improvements | Chosen "later" items, each measured |

```
0 ──► 1 ──► 2 ──► 3 ──► 4 ──► 5 ──► 6 ──► 7 ──► 8
            └─ benchmark topics + simple baselines can start here, in parallel ─┘
```

---

## Phase 0 — Foundations

**Goal:** The skeleton every later phase plugs into.

- Project layout with `uv`. Confirm the local ML libraries install on the chosen Python version.
- Config and secrets through `.env`, and a CLI skeleton with Typer (`run`, `resume`, `inspect`).
- **LLM interface:** one internal module with the three roles above, structured outputs validated with Pydantic, and response caching. Providers can be swapped by config.
- **API clients** for Semantic Scholar, arXiv, OpenAlex and Unpaywall, with caching (`diskcache`) and retry/backoff (`httpx` + `tenacity`). Semantic Scholar works without a key; add a key when one arrives.
- **Storage:** a first SQLite schema covering papers, sections, sentences, graph nodes and edges, and trace events.
- **Run trace:** a PROV-DM-shaped event log (inputs, outputs, step, prompt version, sources used and dropped). Every later component writes to it.
- **LangGraph skeleton** with a stub node for every stage, plus SQLite checkpointing so runs can resume.
- **Dev topic set:** 10–20 real CS/ML research questions, used to check each phase by hand.

**Done when:** `agent run "<question>"` passes through every stub stage, writes a trace, and prints an empty report.

## Phase 1 — Research input + literature acquisition (C1, C2)

**Goal:** For any CS/ML question, find the right papers and turn them into addressable text.

- **Research brief:** the LLM turns the question or idea into a problem statement, a few search angles, keywords, and facets (purpose / mechanism / evaluation) when the user gives an idea.
- **Search:** Semantic Scholar keyword search, Semantic Scholar snippet search (passages inside papers), and the arXiv API for recent preprints. Deduplicate by arXiv, DOI and Semantic Scholar IDs.
- **Expand** from the best hits through Semantic Scholar references, citations and recommendations.
- **Rank:** SPECTER2 similarity (Semantic Scholar supplies the embeddings), then an LLM relevance check. Split the results into an *abstract-only* set and a *full-read* set.
- **Full text:** use arXiv HTML first. Otherwise use open-access PDFs found via Semantic Scholar or Unpaywall, parsed with GROBID or PyMuPDF. Each paper becomes sections and sentences with stable IDs.
- **User uploads** go through the same parsing, tagged "user-provided".
- **Retrieval log:** every query, every result, and what was kept or dropped, with the reason.

**Done when:** each dev topic yields a ranked corpus with its full-read papers parsed into sentences, and the obvious key papers for each topic are found.

## Phase 2 — Research knowledge graph (C3)

**Goal:** Turn the corpus into a graph in which every fact points to the sentence it came from.

- **Extraction form** for each paper and section:
  - task, methods, datasets, metrics
  - results, with their conditions
  - limitations, assumptions, future work
  - purpose / mechanism / evaluation facets
  - atomic claims, each quoting its evidence sentences
- **Quote check:** code confirms that every quoted span exists in the paper. Claims that fail are dropped. Missing fields stay empty rather than being filled with placeholders.
- **Name merging** for methods, datasets, metrics and tasks: string similarity plus embeddings, with the LLM deciding only the unclear cases.
- **Graph:** nodes and edges stored in SQLite and loaded into NetworkX for queries. Claim embeddings come from a small local model.
- **Cross-paper links:** "supports" and "contradicts" between similar claims, found by a small local NLI model.
- **Per-paper cache:** each paper is extracted once and reused across questions.
- **Graph queries:** e.g. "methods evaluated on dataset X", "limitations reported for task T".

**Done when:** every dev topic has a graph, the quote-check pass rate is measured, a sample of extractions has been checked by hand, and the example queries return sensible answers.

## Phase 3 — Related work + gap detection (C4, stages 1–2)

**Goal:** Map what has and hasn't been studied, and produce gaps backed by evidence.

- **Related work:** the papers closest to the question, by embedding similarity and graph neighbourhood.
- **Landscape grid:** methods × datasets × metrics (× conditions), built from the graph. Filled cells were studied; empty cells weren't.
- **Gap candidates:**
  - *stated*: limitation and future-work sentences
  - *structural*: empty grid cells, filtered for meaningfulness by the LLM
  - *conflicts*: claims linked by "contradicts"
- **Gap record:** the gap, its evidence sentences, and a one-line warrant linking them. Gaps are ranked.

**Done when:** each dev topic has a ranked gap list, and every gap's evidence resolves to real sentences.

## Phase 4 — Idea generation + novelty assessment (C4, stage 3) — contribution 1

**Goal:** Propose ideas and rate how novel they are, honestly.

- **Idea generation** from the top gaps. The prompt asks the LLM to fix a reported limitation, combine methods, or borrow from another CS/ML area, and makes feasibility a requirement. Ideas come out in a fixed structure with facets.
- **Independent novelty search:** a fresh search driven by the idea itself (keyword queries, whole-idea snippet search, recommendations from the closest papers). It never reuses only the corpus that inspired the idea.
- **Closest-work comparison:** rerank by facet match down to about 10 papers, then compare the idea with each one separately, quoting the paper.
- **Rule-based levels 1–5.** If the result is unclear or the search found little, assign the lower level and mark it "inconclusive". Report search coverage, and always word the result as "potential novelty".
- **Retry loop:** an idea at level 1–2 has one facet changed and is rechecked. Rejected ideas are remembered so they aren't regenerated. The number of retries is capped.
- **Novel Evidence Report:** Proposed Idea, Novelty Assessment, Evidence, Research Gap, Closest Existing Work, Difference from Existing Work.
- **Duplicate test:** feed in ideas taken from real published papers. The checker should flag them as duplicates when those papers are findable.

**Done when:** each dev topic gets a report, the duplicate test catches most planted ideas, and every evidence item resolves to a sentence.

## Phase 5 — Provenance-aware experiment plan (C5) — contribution 2

**Goal:** An experiment plan in which every element is traceable to the literature.

- **Plan structure:** hypothesis, research question, method, datasets, baselines, metrics, conditions, ablations, expected results, risks and limitations.
- **Fields filled from the graph:** datasets, baselines, metrics, expected results (with their conditions) and risks are taken from the closest papers. The LLM writes the remaining fields and must cite what each one rests on.
- **Three provenance records per element:**
  - *source*: the evidence sentences
  - *derivation*: the step or query that produced it
  - *decision*: why it was chosen over the alternatives
- **Critique loop:** review for feasibility and missing baselines. When information is missing, run a targeted search and then refine.
- **Verification pass:** the verifier-role LLM and the local NLI model label every cited claim *supported*, *not supported* or *contradicted*. Unsupported elements are flagged or removed, and Provenance Coverage is computed.
- **Output:** the plan is linked to the Novel Evidence Report. The provenance graph is exported as PROV-JSON (with the `prov` package). Every element can be traced: recommendation → reasoning → evidence → paper → section.

**Done when:** every dev topic produces both linked outputs end to end, every plan element has provenance, and a coverage figure is reported.

## Phase 6 — Hardening + integration

**Goal:** From a working prototype to a dependable tool.

- **Robustness:** rate limits, partial failures, and resuming a run from its checkpoint.
- **Measurement:** cost and time per run, and cache hit rates.
- **CLI polish:** run, resume, inspect a trace, export a report. Reports show simple citations by default and full provenance detail for contested claims.
- **Tests:** unit tests for the parsers, the quote check and the novelty-level rules, plus a regression run on the dev topics.

**Done when:** all dev topics run without manual intervention, and cost and time per run are documented.

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

- How many papers per question to read in full, and how many by abstract only.
- One idea per run, or the top 3 ranked by novelty level.
- Optional CLI pause points (confirm the brief, pick the idea before planning).
- GROBID (needs Docker) or PyMuPDF alone for non-arXiv PDFs.
- Which providers fill the three LLM roles, and the cost budget per run.
- The Python version, depending on wheel support for the local ML libraries.
