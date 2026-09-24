# From Agent Traces to Trust — Extraction Notes

> **Paper:** *From Agent Traces to Trust: A Survey of Evidence Tracing and Execution Provenance in LLM Agents*
> **Authors:** Yiqi Wang (Griffith), Jiaqi Zhang (Jiangsu), Zhangkai Wu (Sydney), Taotao Cai (USQ), Zirui Liu (Peking), Qingqiang Sun (Great Bay), Zequn Sun (Nanjing), Manqing Dong (Sydney), Mingkai Zheng (SUSTech), Xuefei Yin, Yanming Zhu (Griffith)
> **arXiv:** 2606.04990v4 [cs.CR], 28 Jun 2026 · 28 pages · **Survey**
> **Resources:** accompanying GitHub repo, continuously updated
> **Type:** this is a **survey**, not a system paper — its value is the **vocabulary, taxonomy, and the map of what is unsolved.** It is the closest thing to a specification for the provenance layer our C5 describes, and it explicitly names our project's territory as an open problem.
> **Notable:** it cites **PaperTrail** (Martin-Boyle et al. 2026) as a representative claim-support-graph representation — so the paper we have notes on is a data point inside this map.

---

## 1. TL;DR

The survey's core argument, which is also the argument our project rests on:

> **Final-answer accuracy evaluates only the endpoint of an execution.** It does not explain how an output was produced, which evidence supported each claim, whether tool calls were justified, whether memory items were trustworthy, or which step caused a failure. This is a **process-level accountability gap**.

Two definitions organise everything, and they are worth adopting verbatim:

- **Execution provenance** — *"the complete typed representation of an agent run, including evidence units, execution units such as retrieved documents, tool calls, parameters, observations, memory accesses, intermediate claims, actions, inter-agent messages, and final outputs, and their causal, procedural, dependency, update, contradiction, and invalidation relations."*
- **Evidence tracing** — *"the projection of this provenance structure onto evidence-support and influence relations between evidence units and agent claims, decisions, or actions."*

**Evidence tracing is a sub-problem of execution provenance**: every evidence-support link belongs to the provenance structure, but execution provenance *additionally* records how tools, memory, observations, actions, and inter-agent messages shape the end-to-end execution.

> **This distinction reorganises our docx.** Our §3.5 defines three provenance types — *source* ("where did this come from?"), *derivation* ("how did the agent derive this?"), and *decision* ("why this configuration?"). In this survey's terms, **source provenance is evidence tracing; derivation and decision provenance are execution provenance.** Our instinct to separate them was right, and this survey supplies the formal vocabulary and the typed relation set to implement it.

---

## 2. The Six-Dimensional Taxonomy (Table 1)

| Dimension | Categories |
|---|---|
| **Trace sources** | Reasoning, retrieval, tool use, MCP server/host boundaries, memory, environment, multi-agent communication |
| **Evidence units** | Documents, passages, observations, tool outputs, memory items, claims, policies, final answers |
| **Execution units** | Reasoning steps, retrieval calls, tool invocations, parameters, tool manifests, permission scopes, memory operations, actions, inter-agent messages |
| **Provenance relations** | `SUPPORT`, `DERIVE`, `DEPEND-ON`, `CONTRADICT`, `INVALIDATE`, `TRIGGER`, `UPDATE`, `USE`, `GENERATE` |
| **Tracing granularity** | Run-level, step-level, tool-call-level, parameter-level, **claim-level**, token/span-level |
| **Tracing timing** | Pre-execution, runtime, post-hoc, continuous |
| **Representation forms** | Structured logs, execution graphs, evidence graphs, claim-support graphs, provenance graphs, runtime state |
| **Trust functions** | Verification, attribution, debugging, safety enforcement, audit, failure attribution, **recovery** |

### Evidence units vs execution units

- **Evidence units** = *semantic* objects that can **support, contradict, invalidate, or contextualise** claims and actions.
- **Execution units** = *procedural* objects describing **what the agent did**, in what order, under which context, across which execution boundary.

The distinction is *"useful but not absolute: a tool output is both the result of an execution step and possible evidence for a later claim."* And the design conclusion: *"Provenance-aware agents therefore need to connect semantic support with procedural execution rather than recording only one side of the trace."*

> Our current docx records only the semantic side (paper → evidence → claim). It has no representation of the *agent's own execution* — which search was run, which retrieval returned what, which step consumed which result. **For a multi-stage agent, that omission means an incorrect output cannot be localised to a stage.**

### Provenance relations (the typed edge vocabulary)

The set is deliberately minimal, split into PROV-compatible base relations plus agent-specific additions:

| Relation | Example from the paper |
|---|---|
| `USE` / `GENERATE` / `DERIVE` | PROV-compatible base relations |
| **`SUPPORT`** | a passage supports a claim |
| **`DEPEND-ON`** | a tool call depends on generated parameters |
| **`CONTRADICT`** | a tool output contradicts a memory item |
| **`INVALIDATE`** | a new observation invalidates a plan |
| **`TRIGGER`** | a failed action triggers recovery |
| **`UPDATE`** | memory or external state revised across turns |

*"These relations separate semantic grounding from procedural dependency."*

### Granularity and timing — a risk-proportionate policy

- **Granularity:** run-level → coarse audit; step-level → debugging; tool-call/parameter-level → safety enforcement; **claim-level → factual attribution**; token/span-level → finest but most expensive.
- **Timing:** pre-execution → block unauthorised actions; runtime → policy enforcement and information-flow control; post-hoc → audit and diagnosis; continuous → persistent memory and long-horizon workflows.

> *"The appropriate choice depends on the risk and persistence of the workflow: low-risk RAG may require claim-level post-hoc attribution, whereas high-impact tool use may require runtime parameter-level provenance."*
>
> **For our agent this settles a design question**: our output is a report and a plan — nothing is executed, nothing is irreversible. So we want **claim-level, post-hoc** provenance, not runtime parameter-level enforcement. That's the cheap end of the spectrum, and it's the right end for us.

---

## 3. Four Design Tensions (§2.2)

These are the survey's sharpest contribution and they read as a design checklist:

1. **Chronological logging ≠ semantic provenance.** *"A log can record what happened, but it may not show whether evidence supported a claim or whether memory influenced a tool call."*
2. **Component-level tracing ≠ end-to-end accountability**, because failures propagate across retrieval, tools, memory, environment states, and other agents.
3. **Static schemas standardise; runtime safety needs online tracking.** Schemas help comparison, but enforcement requires live source tracking, policy checking, and dependency-aware enforcement.
4. **Trace completeness vs deployability.** *"Fine-grained provenance improves verification, debugging, audit, and recovery but also increases storage cost, privacy exposure, annotation burden, and system complexity."*

> Tension 4 is the one that will bite us. PaperTrail is the empirical proof: full claim-evidence provenance cost 90 s/query and *reduced usability*. **Provenance granularity is a budget to be allocated, not a quantity to be maximised.**

---

## 4. Provenance Representation (§3) — a layered view

Three layers, each answering a different question:

| Layer | Answers | Role |
|---|---|---|
| **Structured logs** | *what happened* | Execution observability and reconstruction |
| **Execution graphs** | *what depended on what* | Influence analysis for debugging, audit, recovery |
| **Evidence / claim-support graphs** | *what supports what* | Separation of citation presence, relevance, support, contradiction, and **omission** |

Plus two cross-cutting forms (Table 2):
- **Static schemas** — object types, relation types, trust labels, required provenance fields. *"A reusable vocabulary for trace representation and comparison."*
- **Runtime provenance** — concrete source-to-sink, tool, memory, and action dependencies during execution.

*"Logs make executions reconstructable, graphs make dependencies inspectable, and claim-support links make grounding and accountability measurable."*

### §3.2 — the five distinct relations current systems conflate

**This is the most directly actionable paragraph in the survey for our C5.** Claim-level provenance should distinguish **at least five relations that are often collapsed into a single "citation" link**:

1. **Citation presence** — a source is cited.
2. **Source relevance** — the source is topically relevant. *(SourceCheckup shows a cited source may be relevant **without supporting the specific claim**.)*
3. **Claim support** — the source actually supports the claim.
4. **Contradiction** — a source contradicts the claim.
5. **Evidence omission** — a relevant source was available but **not used**. *(This is the relation PaperTrail operationalises.)*

The paper's framing: *"A source may be `CITED` but not `SUPPORT` a claim; a relevant passage may be available but `OMITTED` from the answer; a tool output may `CONTRADICT` a memory item; and new evidence may `INVALIDATE` an earlier belief. **Evidence graphs therefore distinguish evidence availability from evidence use.**"*

> **Our docx §4.2 lists "citation correctness" and "citation completeness" as two metrics. This survey says that's a four-or-five-way distinction.** A recommendation can cite a paper that is (a) not cited at all, (b) cited and irrelevant, (c) cited and relevant but non-supporting, (d) cited and supporting, or (e) contradicted by a paper the agent found and ignored. Collapsing (b)/(c)/(d) into "correct" loses the most common real failure — *relevant but non-supporting* — which is exactly the failure SciGraph-LLM's inverted-claim example and Fact-or-Fiction's directionality errors produce.

### Representation trade-offs

*"Fine-grained provenance enables claim verification, failure localization, rollback, audit, and policy enforcement, but increases storage cost, logging complexity, privacy exposure, and annotation burden. Coarse-grained provenance is easier to collect, but may miss the evidence that supported a claim, the memory item that caused an error, or the tool argument that crossed a trust boundary."*

---

## 5. Mapping to W3C PROV-DM (Table 6) — the reusable standard

The survey maps its model onto **W3C PROV-DM**, which matters because it means our provenance graph can be built on an existing, tooled standard rather than invented from scratch.

| PROV construct | Agent counterpart | Agent-specific extension |
|---|---|---|
| **Entity** | Evidence / execution artifact | Documents, passages, tool outputs, memory items, generated claims, messages |
| **Activity** | Execution unit | Reasoning steps, retrieval and tool calls, memory operations, environment actions |
| **Agent** | Agent | LLM, tool/API, user, or sub-agent; **multi-agent roles and delegation become first-class** |
| `used` | `DEPEND-ON`, `USE` | agent provenance adds **trust/taint labels** on the used value |
| `wasGeneratedBy` | `GENERATE` | a tool output generated by a tool call; a claim by a reasoning step |
| `wasDerivedFrom` | `DERIVE` | **captures summarization and transformation, not only copying** |
| `wasInformedBy` | `TRIGGER` | an observation triggers a tool call; an error triggers replanning |
| `wasInvalidatedBy` | `INVALIDATE` | *PROV invalidation ends an entity's existence; agent `INVALIDATE` marks a claim, plan, or memory item as **epistemically invalid while the record persists*** |
| `wasRevisionOf` | `UPDATE` | memory or external-state updates across turns/sessions |
| **— (no counterpart)** | **`SUPPORT`** | *"central to attribution, but PROV has no support/justification semantics"* |
| **— (no counterpart)** | **`CONTRADICT`** | *"requires semantic comparison, not provenance bookkeeping"* |
| **— (no counterpart)** | **Semantic units** | generated claims, tool-call rationales, natural-language observations, inter-agent messages, trust labels — *"which PROV treats as opaque entities"* |

> **The `INVALIDATE` distinction is subtle and important for us.** PROV's invalidation destroys the entity; the agent version marks a claim **epistemically invalid while keeping the record.** That is exactly what we need when literature supersedes a claim: a gap that was real in 2019 and closed in 2023 should be *marked closed*, not deleted — our C4's evidence trail depends on the history remaining visible.
>
> And the headline: **`SUPPORT` and `CONTRADICT` have no standard counterpart.** The two relations our whole project turns on are precisely the ones no provenance standard expresses, because they require *semantic judgement* rather than *bookkeeping*. That is the gap our C5 lives in.

---

## 6. Trace Schema Requirements (Table 9) — a specification we can build against

| Requirement | What should be represented | Why it matters |
|---|---|---|
| **Trace sources** | User instructions, model outputs, retrieval calls, tool calls, memory operations, environment observations, inter-agent messages, final responses | End-to-end reconstruction of agent execution |
| **Evidence units** | Documents, passages, tool outputs, observations, memory items, policies, intermediate claims, final claims | Claim-level verification and attribution |
| **Execution units** | Reasoning steps, retrieval operations, tool invocations, parameter-generation steps, memory reads/writes, environment actions, message exchanges | Debugging, safety enforcement, audit |
| **Typed relations** | `SUPPORT`, `CONTRADICT`, `DEPEND-ON`, `DERIVE`, `TRIGGER`, `UPDATE`, `INVALIDATE`, `USE`, `GENERATE` | Turns raw logs into provenance graphs that expose influence and responsibility |
| **Trust metadata** | Source trust, permission scope, confidentiality/integrity labels, **timestamps**, actor identity, policy constraints | Runtime enforcement, privacy-aware audit, governance |
| **Replay and recovery fields** | Checkpoints, rollback targets, **invalidation events**, repair steps, human approvals, state-change records | Supports intervention and recovery, **not only post-hoc explanation** |

Existing standards give partial foundations: **W3C PROV-DM** (entities/activities/agents/derivation), **OpenTelemetry** (distributed traces and spans), **OpenLineage** (data workflow lineage), **PROV-AGENT** (provenance modeling adapted to agentic workflows), **AgentOps** and **AgentTrace** (structured agent artifacts and execution context).

> **Concretely for us:** our provenance graph should carry `SUPPORT` edges (evidence → plan element), `DERIVE` edges (plan element → the reasoning step that produced it), `CONTRADICT` edges (paper ↔ paper), timestamps and source-trust metadata on every evidence node, and — the field we'd never have thought to include — **invalidation events**, so a superseded gap or closed research question is recorded as closed rather than silently dropped.

---

## 7. Benchmarks and Metrics (§6) — the evaluation map

### Coverage heatmap (Figure 5) — S = strong, P = partial, L = limited/absent

| Benchmark family | Evidence labels | Tool calls | Memory | Multi-agent | Safety attacks | Provenance relations | Recovery |
|---|---|---|---|---|---|---|---|
| RAG benchmarks | **S** | L | L | L | P | P | L |
| Tool-use benchmarks | L | **S** | L | P | P | P | L |
| Memory benchmarks | P | L | **S** | L | P | P | P |
| Multi-agent benchmarks | L | P | L | **S** | P | P | L |
| Trace-debugging benchmarks | P | P | P | P | L | P | P |
| Provenance/security benchmarks | P | **S** | P | P | **S** | **S** | P |

> **Key gap, in their words:** *"No benchmark family provides strong end-to-end coverage across evidence labels, tool calls, memory, multi-agent communication, safety attacks, provenance relations, and recovery."*
>
> Note the column that is weak everywhere: **provenance relations** — no family rates strong except security benchmarks. This is the empty cell our §4.2 metrics would fill.

### Three core gaps in provenance evaluation (§6.2)

1. **Cross-component provenance.** *"A retrieved passage may support an intermediate claim, the claim may determine a tool argument, the tool output may update memory, and the memory item may later influence a final action. Few benchmarks evaluate this complete chain."*
2. **Relation annotation.** Many benchmarks expose traces; few label typed relations. *"Without such labels, evaluation often measures correctness, faithfulness, or attack success only indirectly, rather than directly measuring trace completeness, provenance accuracy, dependency coverage, memory validity, or cross-agent responsibility."*
3. **Recovery-oriented evaluation.** Benchmarks check whether an agent succeeds or fails; *"they rarely evaluate whether provenance helps the system repair execution."*

### Metric maturity (Table 5) — read this before writing our §4.2

| Aspect | Core question | Representative metrics | **Status** |
|---|---|---|---|
| **Evidence attribution** | Are claims supported by reliable evidence? | Evidence recall, citation precision, source supportiveness, faithfulness, claim support accuracy | **Established** |
| **Execution provenance** | Is the process completely and correctly traceable? | **Trace completeness, provenance accuracy, dependency coverage, temporal consistency** | **Proposed** — *"natural desiderata, but the field does not yet have agreed definitions or broadly adopted datasets"* |
| **Safety and robustness** | Can unsafe/untrusted influence be detected? | Unsafe influence detection, attack success rate, policy violation rate, intervention precision | **Established** |
| **Debugging and recovery** | Can traces identify and repair failures? | Failure localization accuracy, diagnosis correctness, auditability, recovery success | **Partly established** |

> **This directly calibrates our project's ambition.** Our five §4.2 metrics split cleanly across this table:
> - *Citation correctness*, *citation completeness* → **evidence attribution → Established.** We should reuse existing metrics and datasets (ALCE, FActScore, RAGChecker, SourceCheckup, SciFact) rather than invent our own; there are numbers to compare against.
> - *Provenance coverage*, *provenance consistency*, *traceability* → **execution provenance → Proposed.** No agreed definitions exist. **That is our contribution space** — but it also means we must define our metrics precisely and defend them, because nobody will recognise them by name.
>
> The survey is effectively telling us: the evidence half of our evaluation is standing on solid ground; the process half is genuinely open, and therefore genuinely publishable.

### What a full-stack provenance benchmark needs (§6.4)

*"At minimum, it should record user requests, retrieved evidence, tool calls and outputs, memory reads and writes, inter-agent messages, intermediate claims, final responses, external state changes, and policy or permission context. It should also annotate typed relations such as `SUPPORT`, `CONTRADICT`, `DEPEND-ON`, `UPDATE`, `INVALIDATE`, and `TRIGGER`, so that provenance quality can be measured directly rather than inferred from task success."*

Plus three condition types: **adversarial** (can untrusted content influence outputs?), **temporal** (can the agent distinguish current evidence from stale memory, handle updates, preserve lineage?), and **recovery** (can provenance drive retry, rollback, quarantine, human approval, compensation?).

---

## 8. Open Problems (§7) — where the field is unsolved

**7.1 Unified and interoperable trace schemas.** *"Few schemas jointly represent reasoning steps, retrieval, tool calls and outputs, memory reads and writes, inter-agent messages, intermediate claims, final responses, and external state changes."* Future schemas should connect operational trace structures with **semantic evidence relations**.

**7.2 Claim-level and semantic provenance.** Current attribution is coarse (answer-level citation, context-level faithfulness, step-level logging), but *"agent outputs frequently contain multiple claims… one claim may be supported by retrieved evidence, another may be inferred beyond the source, and another may be contradicted by a tool output or memory item."*

The harder problem named: **semantic provenance across transformations.**
> *"Evidence may be paraphrased, summarized, aggregated, compressed into memory, passed through tools, or reused by another agent before influencing a claim or action. **String-level matching and citation presence are therefore insufficient.**"*

Cases they enumerate as important and unsolved: a claim derived from **multiple** retrieved passages; a tool output that **contradicts** a memory item; a memory item **summarized** from a previous conversation; a policy constraint that **justifies or blocks** an action; an inter-agent message that **propagates an unsupported assumption**.

> **The first and last of those are ours.** A gap claim synthesised from five papers, and a claim propagated between our pipeline stages (C4's gap → C5's hypothesis) without re-verification, are exactly the transformation cases the survey says nobody has solved.

**7.3 Memory and multi-agent provenance.** Memory items should carry *"source attribution, write time, update history, retrieval context, downstream usage, conflict status, and invalidation events."* *"Memory should not be treated as an unqualified context reservoir."* Multi-agent provenance adds a **responsibility problem**: *"which agent introduced the unsupported claim, which agent failed to verify it, and which message propagated it."*

**7.4 Runtime safety, enforcement, and recovery.** *"Most systems focus on detecting or blocking unsafe behavior, but trustworthy agents also need to repair unsupported or unsafe executions."* Provenance should support invalidating stale memory, quarantining contaminated evidence, retrying with corrected parameters, requesting human approval, rolling back, replaying from safe checkpoints. *"This requires provenance representations that support intervention and recovery, not only explanation."*

**7.5 Realistic benchmarks, privacy, and governance.** Beyond benchmark realism: *"provenance traces may contain sensitive user data, confidential documents, credentials, tool outputs, API arguments, internal reasoning artifacts, and long-term memory items. Future systems must therefore support **trace minimization, access control, anonymization, retention policies, encryption, and selective disclosure.** Without privacy-aware governance, provenance infrastructure itself may become a source of compliance and security risk."*

---

## 9. Where the other six papers sit on this map

The survey provides the frame; our other notes provide the instances. Useful cross-reference:

| Paper | Trace source | Granularity | Timing | Representation | Trust function |
|---|---|---|---|---|---|
| **SciGraph-LLM** | Retrieval (documents) | Claim | Post-hoc | Provenance/evidence graph | Verification, audit |
| **PaperTrail** *(cited in Table 1)* | Retrieval | Claim | Post-hoc | **Claim-support graph** | Verification, attribution |
| **Fact or Fiction (SciFact)** | Retrieval | Claim + sentence | Post-hoc | Claim–evidence | Verification |
| **GAPMAP** | Retrieval | Claim (+ Warrant) | Post-hoc | Claim/Grounds/Warrant | Verification |
| **Idea Novelty Checker (LGNA)** | Retrieval | Answer/idea | Post-hoc | Citation links | Attribution |
| **Scideator** | Retrieval | Facet | Runtime (interactive) | Facet links | Attribution |
| **CoI / ResearchAgent** | Reasoning, retrieval | **Run-level** | — | Structured log at best | *(none)* |

> **Everything we have read is post-hoc, claim-or-coarser, and evidence-only.** Not one of the seven records *execution* provenance — which retrieval was run, which stage consumed which result, what depended on what. Our agent is a multi-stage pipeline (C2 → C3 → C4 → C5), so **an incorrect experiment plan currently has no way to be localised to the stage that produced the error.** That is a real, concrete gap, and the survey names it as an open problem rather than a solved engineering detail.

---

## 10. What to take into our Research Agent

| Our component | What this survey gives us |
|---|---|
| **C2 — Literature Acquisition** | Retrieval is a **trace source**: record the queries issued, the results returned, the ranking, and what was discarded. Currently our design records only what was *kept*. Without the discard record we cannot compute **evidence omission** — the fifth relation of §3.2 and the one PaperTrail found most valuable. |
| **C3 — Research Knowledge Layer** | The **static schema / runtime provenance** split maps onto our design: C3 *is* the static schema (object types, relation types, required fields); the per-run provenance graph instantiates it. Add **trust metadata** to every evidence node — source, timestamp, actor — and **`INVALIDATE` as a first-class event** so superseded claims are marked, not deleted. |
| **C4 — Gap / Novelty Detection** | **`CONTRADICT` is a first-class relation** with no PROV counterpart — and cross-paper contradiction is GAPMAP's third implicit-gap type, SciFact's dual-label case, and our "conflicting findings" scenario. Also: our 5-level novelty ladder is a **claim about the corpus**, so it needs its own provenance — which searches, which coverage, which stage produced it — or it cannot be audited. |
| **C5 — Provenance-Aware Experiment Planning** | **This is the specification.** Adopt: (a) the **typed relation vocabulary** (`SUPPORT`/`CONTRADICT`/`DEPEND-ON`/`DERIVE`/`TRIGGER`/`UPDATE`/`INVALIDATE`); (b) **W3C PROV-DM as the base standard** with `SUPPORT`/`CONTRADICT` as our documented extensions; (c) **Table 9's six schema requirements** as our field list; (d) our docx's three provenance types resolve cleanly — *source* = evidence tracing, *derivation* = `DERIVE` edges over execution units, *decision* = `DEPEND-ON` + the reasoning step that generated the choice. |
| **C-Eval — Provenance Evaluation (§4.2)** | The **metric-maturity table is the single most useful thing here.** Our citation-correctness and citation-completeness metrics are **Established** — reuse existing definitions and datasets, and expect comparable numbers. Our provenance-coverage, provenance-consistency and traceability metrics are **Proposed** — no agreed definitions exist, so we must define them rigorously, and that is where a contribution lives. Also: **the five-relation distinction** (citation presence / source relevance / claim support / contradiction / omission) should replace our two-way correctness/completeness split. |
| **Architecture (system-level)** | **Risk-proportionate granularity.** Our agent executes nothing irreversible, so **claim-level post-hoc** provenance is the right operating point — not runtime parameter-level enforcement. That is a deliberate, defensible scoping decision, and this survey provides the justification for it. |

**Cheap wins to steal first:** (1) the typed relation vocabulary; (2) PROV-DM as the base model; (3) Table 9's schema field list; (4) the five-relation refinement of citation quality; (5) recording discarded retrieval results so omission becomes computable; (6) `INVALIDATE` events for superseded claims; (7) claim-level post-hoc as our stated granularity/timing choice.

---

## 11. Gaps, Limitations & Future Scope

### As a survey — what it does and doesn't give us
1. **No system, no implementation, no numbers.** It is a map, not a machine. Every design choice it recommends is unvalidated by this paper.
2. **Its own core metrics are "Proposed"** — trace completeness, provenance accuracy, dependency coverage, temporal consistency have **no agreed definitions**. Adopting them means defining them ourselves.
3. **Security/CS-agent framing.** Written from a cs.CR perspective centred on tool-use safety, prompt injection, and information-flow control. **Roughly half the survey (tool provenance, runtime guardrails, taint tracking, access control) does not apply to us** — our agent retrieves and writes, it does not act on the world. Its *scholarly* content is essentially PaperTrail plus the RAG-attribution line.
4. **No coverage of scientific-domain provenance specifics** — nothing on rhetorical roles, evidence strength, or the reasoning types (directionality, numerical, cause/effect) that Fact-or-Fiction shows are where scientific claim verification actually fails.

### The open problems that are ours to attack
1. **`SUPPORT` and `CONTRADICT` have no standard representation** and *"require semantic comparison, not provenance bookkeeping."* The two relations our project is built on are the two the field has no standard for.
2. **Semantic provenance across transformations is open.** *"String-level matching and citation presence are therefore insufficient."* Our C4 synthesises a gap from multiple papers, then C5 builds a plan on that synthesised gap — **two transformation hops, neither traceable by string matching.** This is named as unsolved.
3. **Multi-claim provenance for a single output.** *"A final response may be partially correct: one claim may be supported, another inferred beyond the source, another contradicted."* Our Novel Evidence Report and experiment plan are exactly multi-claim outputs, and a single global "provenance coverage" number hides precisely this.
4. **Evidence omission is barely operationalised.** Only PaperTrail implements it, and only over a fixed four-paper corpus. **For a retrieval-based agent, omission is both more important and much harder** — you must record what you retrieved and discarded to measure what you ignored.
5. **Recovery-oriented provenance is the least developed area of all** (rated `L`/`P` in every benchmark family). In our terms: when the novelty checker says "not novel," or a plan element loses its support, does provenance *drive* the repair? Scideator's facet-swap suggestion is the only mechanism in any of the seven papers that does this, and it is not framed as recovery.
6. **Cross-stage provenance in a pipeline.** Our C2→C3→C4→C5 chain is precisely the *"complete chain from evidence and tool outputs to memory updates, intermediate claims, final answers"* that §6.2 says few benchmarks evaluate. **Building it, and being able to localise an error to a stage, is a genuine contribution — and no paper in this set does it.**
7. **Privacy and governance are entirely unaddressed in our plan.** If the agent ingests user-uploaded papers, the provenance trace contains them. Trace minimisation, retention, and access control are a live concern the survey warns about and our docx does not mention.
