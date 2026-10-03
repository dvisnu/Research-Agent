# From Agent Traces to Trust — Summary

> *From Agent Traces to Trust: A Survey of Evidence Tracing and Execution Provenance in LLM Agents* — Wang, Zhang, Wu, Cai, Liu, Sun, Sun, Dong, Zheng, Yin, Zhu · arXiv 2606.04990v4 [cs.CR] (Jun 2026) · 28 pages
> A **survey**, not a system paper — its value is the vocabulary, taxonomy and map of what is unsolved. It is the closest thing to a specification for the provenance layer our C5 describes, and it names our project's territory as an open problem. It cites **PaperTrail** as a representative claim-support-graph system.

## Summary

**The core argument, which is also ours:** final-answer accuracy evaluates only the endpoint of an execution. It does not explain how an output was produced, which evidence supported each claim, whether tool calls were justified, or which step caused a failure. This is a **process-level accountability gap**.

Two definitions worth adopting verbatim:

- **Execution provenance** — the complete typed representation of an agent run: evidence units, execution units (retrieved documents, tool calls, parameters, observations, memory accesses, intermediate claims, actions, inter-agent messages, outputs) and their causal, procedural, dependency, update, contradiction and invalidation relations.
- **Evidence tracing** — the projection of that structure onto evidence-support relations between evidence units and agent claims.

**Evidence tracing is a sub-problem of execution provenance.** This reorganises our plan: our three provenance types map cleanly — *source* provenance is evidence tracing; *derivation* and *decision* provenance are execution provenance.

**The typed relation vocabulary** (deliberately minimal): PROV-compatible `USE` / `GENERATE` / `DERIVE`, plus agent-specific **`SUPPORT`** (a passage supports a claim), **`DEPEND-ON`**, **`CONTRADICT`**, **`INVALIDATE`** (a new observation invalidates a plan), **`TRIGGER`**, **`UPDATE`**.

**Granularity and timing are a risk-proportionate policy**, not a maximisation: run-level for coarse audit, step-level for debugging, tool-call/parameter-level for safety enforcement, **claim-level for factual attribution**, token/span-level finest but most expensive; timing pre-execution / runtime / post-hoc / continuous. *"Low-risk RAG may require claim-level post-hoc attribution, whereas high-impact tool use may require runtime parameter-level provenance."* **This settles a design question for us: our agent executes nothing irreversible, so claim-level post-hoc is the right — and cheapest — operating point.**

**Four design tensions** read as a checklist: chronological logging ≠ semantic provenance; component-level tracing ≠ end-to-end accountability; static schemas standardise but runtime safety needs online tracking; and **trace completeness vs deployability** — fine-grained provenance improves verification but increases storage, privacy exposure, annotation burden and complexity. PaperTrail is the empirical proof of that last one.

**The most actionable paragraph for our C5:** claim-level provenance should distinguish **five relations that current systems collapse into a single "citation" link** — (1) citation presence, (2) source relevance, (3) claim support, (4) contradiction, (5) **evidence omission** (a relevant source was available but not used). *"Evidence graphs therefore distinguish evidence availability from evidence use."* Our plan lists "citation correctness" and "citation completeness" as two metrics; this is a four-or-five-way distinction, and collapsing it loses the most common real failure — **relevant but non-supporting**.

**Mapping to W3C PROV-DM** matters because it means our graph can be built on a tooled standard. Entity/Activity/Agent map onto evidence artifacts, execution units and agents; `used`, `wasGeneratedBy`, `wasDerivedFrom`, `wasInformedBy`, `wasRevisionOf` map onto the agent relations. Two things have **no PROV counterpart: `SUPPORT` and `CONTRADICT`**, because they *"require semantic comparison, not provenance bookkeeping."* The two relations our whole project turns on are precisely the ones no standard expresses. Also note the subtle `INVALIDATE` distinction: PROV invalidation ends an entity's existence, while the agent version marks a claim **epistemically invalid while keeping the record** — exactly what we need when a gap that was real in 2019 is closed by 2023.

**Trace schema requirements** (a spec to build against): trace sources, evidence units, execution units, typed relations, **trust metadata** (source trust, timestamps, actor identity, policy constraints), and **replay/recovery fields** — checkpoints, rollback targets, **invalidation events**, human approvals — supporting intervention, not only post-hoc explanation.

**Metric maturity — the single most useful table for calibrating our ambition:**

| Aspect | Status |
|---|---|
| **Evidence attribution** (evidence recall, citation precision, source supportiveness, claim support accuracy) | **Established** |
| **Execution provenance** (trace completeness, provenance accuracy, dependency coverage, temporal consistency) | **Proposed** — *"natural desiderata, but the field does not yet have agreed definitions or broadly adopted datasets"* |
| Safety and robustness | Established |
| Debugging and recovery | Partly established |

Our citation correctness/completeness sit in the **Established** row — reuse existing definitions and datasets (ALCE, FActScore, RAGChecker, SourceCheckup, SciFact), and expect comparable numbers. Our provenance coverage/consistency/traceability sit in the **Proposed** row — no agreed definitions exist, which is both our contribution space and an obligation to define them rigorously. Their benchmark heatmap makes the same point: **no benchmark family rates strong on provenance relations**, and none gives end-to-end coverage across evidence, tools, memory, multi-agent, attacks, relations and recovery.

**Where the other papers sit on this map:** all of PaperTrail, SciGraph-LLM, SciFact, GAPMAP, the Idea Novelty Checker, Scideator and CoI/ResearchAgent are **post-hoc, claim-or-coarser, and evidence-only**. Not one records *execution* provenance. Our pipeline is C2 → C3 → C4 → C5, so **an incorrect experiment plan currently has no way to be localised to the stage that produced the error.**

## Gaps

**As a survey — what it doesn't give us**

- **No system, no implementation, no numbers.** It is a map; every recommendation is unvalidated by this paper.
- **Its own core metrics are "Proposed"** — adopting them means defining them ourselves.
- **Security/CS-agent framing.** Written from a cs.CR perspective centred on tool-use safety, prompt injection and information-flow control; roughly half of it (runtime guardrails, taint tracking, access control) **does not apply to us** since our agent retrieves and writes rather than acting on the world.
- **No scientific-domain specifics** — nothing on rhetorical roles, evidence strength, or the reasoning types where scientific claim verification actually fails.

**The open problems that are ours to attack**

- **`SUPPORT` and `CONTRADICT` have no standard representation.**
- **Semantic provenance across transformations is open** — evidence gets paraphrased, summarised, aggregated and reused before influencing a claim, so *"string-level matching and citation presence are insufficient."* Our C4 synthesises a gap from multiple papers and C5 builds a plan on it: **two transformation hops, neither traceable by string matching.**
- **Multi-claim provenance for a single output** — a response can be partly supported, partly inferred beyond the source, partly contradicted. A single global "provenance coverage" number hides exactly this.
- **Evidence omission is barely operationalised** — only PaperTrail implements it, over a fixed four-paper corpus. For a retrieval-based agent it is both more important and much harder: you must record what you retrieved **and discarded** to measure what you ignored.
- **Recovery-oriented provenance is the least developed area of all** — does provenance *drive* the repair when a plan element loses its support? Scideator's facet-swap is the only mechanism in the whole reading set that does this, and it isn't framed as recovery.
- **Cross-stage provenance in a pipeline** — building it, and being able to localise an error to a stage, is a genuine contribution that no paper in the set does.
- **Privacy and governance are entirely unaddressed in our plan** — if the agent ingests user-uploaded papers, the trace contains them; trace minimisation, retention and access control are live concerns.

## What We Can Use

| Component | What this survey gives us |
|---|---|
| **C2 — Literature Acquisition** | Retrieval is a **trace source**: record the queries issued, results returned, ranking, **and what was discarded**. Our design records only what was kept — and without the discard record, **evidence omission cannot be computed.** |
| **C3 — Research Knowledge Layer** | The **static schema / runtime provenance** split maps onto our design: C3 *is* the static schema; the per-run graph instantiates it. Add **trust metadata** (source, timestamp, actor) to every evidence node, and **`INVALIDATE` as a first-class event** so superseded claims are marked, not deleted. |
| **C4 — Gap / Novelty Detection** | **`CONTRADICT` as a first-class relation** — cross-paper contradiction is GAPMAP's third implicit-gap type and SciFact's dual-label case. And our 5-level novelty rating is itself **a claim about the corpus**, so it needs its own provenance (which searches, which coverage, which stage) or it cannot be audited. |
| **C5 — Provenance-Aware Experiment Planning** | **This is the specification.** Adopt (a) the typed relation vocabulary; (b) **W3C PROV-DM as the base standard** with `SUPPORT`/`CONTRADICT` as documented extensions; (c) their six schema requirements as our field list; (d) the resolution of our three provenance types (source = evidence tracing, derivation = `DERIVE` over execution units, decision = `DEPEND-ON` plus the reasoning step). |
| **Provenance Evaluation** | The **metric-maturity table**. Our citation metrics are Established — reuse existing definitions and datasets rather than inventing our own. Our provenance-coverage/consistency/traceability metrics are Proposed — define them rigorously, and that is where a contribution lives. Also: **replace our two-way correctness/completeness split with the five-relation distinction.** |
| **Architecture** | **Risk-proportionate granularity.** Nothing our agent does is irreversible, so **claim-level post-hoc** provenance is the right operating point — a deliberate, defensible scoping decision this survey justifies. |

**Steal first:** (1) the typed relation vocabulary; (2) PROV-DM as the base model; (3) the schema field list; (4) the five-relation refinement of citation quality; (5) recording discarded retrieval results so omission becomes computable; (6) `INVALIDATE` events for superseded claims; (7) claim-level post-hoc as our stated granularity/timing choice.
