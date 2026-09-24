# PaperTrail — Summary

> *PaperTrail: A Claim-Evidence Interface for Grounding Provenance in LLM-based Scholarly Q&A* — Martin-Boyle, Kaur (Univ. of Minnesota), Leckey, Brown (NASA Langley) · CHI 2026 · arXiv 2602.21045v1 · pre-registered on AsPredicted

## Summary

The only paper in the set that **builds and evaluates with real users** an end-to-end argument-grounded provenance system — and its central negative result is one to internalise before designing our C5 output.

**The system.** Decompose *both* the LLM's answer *and* the source documents into discrete **claims + evidence**, then match them. The interface shows three things: **supported assertions**, **unsupported claims** (no match — hallucination or ungrounded synthesis), and **omitted information** — paper claims relevant to the question that the answer *left out*, an error of omission the authors correctly call a *"difficult-to-detect failure mode"* that flat citation links cannot surface at all.

**The organising engineering idea: a cost/capability gradient allocated per stage.** Cheap similarity (sentence-transformer cosine) for topical filtering and dedup; RAG (similarity filter → LLM over the reduced set) for query-conditioned selection; expensive LLM prompting only where nuanced semantic judgement is required.

- **Stage 1 (offline, per paper):** PyMuPDF extraction with manual validation of math notation, paragraph boundaries and hyphenation; section/paragraph segmentation; **LLM claim extraction per paragraph (10-shot)**; similarity-based evidence retrieval at cosine ≥ 0.75 with neighbouring sentences kept for context. Output is one JSON of claims + evidence + citation + section.
- **Stage 2 (real time):** a separate **answerer LLM** generates the answer unconstrained; an **extraction LLM** decomposes it into claims and evidence using the *same* criteria, then NLTK punkt + programmatic matching maps everything to exact character spans.
- **Stage 3 (real time):** SPECTER similarity filter → LLM selects relevant paper claims and evidence → **LLM claim-to-claim semantic-equivalence matching** → evidence verification flagged below cosine 0.55, a deliberately permissive threshold tuned **to avoid alert fatigue**.

**The five claim criteria** — the best-specified claim definition in the set, externally validated at α = 0.69–0.78: **atomic, verifiable, faithful, decontextualized, declarative** (not a question, hypothesis-as-question, or method description).

**Why pre-extract offline rather than at query time** — four stated benefits that are precisely the argument for our C3 being a persistent artefact: it moves cost off the user-facing path, creates a **single ground truth consistent across users**, gives reproducible results, and lets the claim-evidence corpus be **versioned and audited independently** of the query pipeline.

**The sharpest line in the prompt library**, from claim-to-claim matching: *"Do not match claims that are merely topically related but make different assertions."* Topical relatedness is what embeddings measure; assertional equivalence is what provenance requires. Omitting unmatched claims is what produces the "unsupported" flag — **absence of a match is itself the signal**.

**Backend evaluation.** No gold standard exists for exhaustive atomic claim extraction, so they adapted two datasets (SciClaimHunt, machine-generated; BioClaimDetect, human-annotated, held out as OOD) and matched at a conservative SPECTER cosine **τ = 0.9**. Results: P 0.69 / R 0.62 / F1 0.65 and P 0.63 / **R 0.88** / F1 0.73. Their interpretation is careful and worth copying: **precision is structurally underestimated** because their extraction is more atomic than the reference annotations (manual inspection confirmed most unmatched extractions were valid claims missing from the gold set), so **recall is the meaningful metric**; and the stronger OOD result is what makes the generalisation claim credible.

**User study — the headline.** Within-subjects, N = 26 professional researchers (NASA Langley), two 20-minute tasks (multi-paper synthesis; devil's-advocate defence), reliance measured as normalised edit distance from the LLM draft.

| Measure | Baseline | PaperTrail | |
|---|---|---|---|
| **Trust** | 4.22 | **3.68** | p = .015, d = 0.44 |
| **Reliance** | 0.73 | 0.75 | n.s. |
| **Usability** | 5.05 | **4.48** | p = .026, r = 0.52 |

**Trust down, reliance unchanged, usability down.** Giving experts better provenance made them more sceptical and no more careful. But the correlation analysis is arguably the more important finding: in the baseline, reliance correlated with trust (r = 0.57) and usability (r = 0.60); **under PaperTrail those correlations vanish entirely.** Granular provenance **decoupled generalised trust from behavioural reliance** — plausibly because people were making per-claim rather than one global trust judgement. It changed *the unit of the trust decision*.

Three explanations offered for the trust–behaviour gap: people default to System 1 even when they know deliberation is warranted, and explanations can worsen this by feeling like comprehension; **the feedback loop that builds expertise is missing** (users never learn whether their edits improved accuracy); and design friction is hard to balance — *"our study feedback suggests we erred"* toward too burdensome. **End-to-end latency was ~90 seconds**, called *"excruciatingly slow"*, and over half of participants named time pressure as the main barrier.

## Gaps

**Stated by the authors**

- **Task realism** — unfamiliar papers, 20–30 minute constraints, assigned questions; field studies needed.
- **Reliance-as-edit-distance conflates behaviours**, and participants might edit *less* precisely because they trusted PaperTrail to flag problems.
- Trust may have been measured against the wrong target (interface vs LLM); **output quality was never assessed** — nobody checked whether unsupported claims were actually removed. Blind expert evaluation of the edited text is the missing measurement.
- **Only claims and evidence — no warrants, rebuttals, backings or qualifiers** (a reduced Toulmin model). Single session, one interface paradigm.
- **~90 s latency** from non-parallelisable sequential LLM calls; proposed fixes are caching, parallelism, and federated pre-processing of collections.

**Open for us**

- **The Warrant is missing.** The system matches claim → evidence but never asks *why* the evidence implies the claim. **GAPMAP's Claim/Grounds/Warrant supplies the missing piece, and SciFact's five reasoning types explain what goes wrong in that gap.** Semantic equivalence is not entailment — two claims can be near-identical in embedding space and opposite in polarity. A warrant-checking step over this architecture is a well-defined, unclaimed improvement.
- **Matching is equivalence, not support** — it asks "does the source say the same thing?", not "does the source support this?" A claim can be well-supported by evidence that asserts something different.
- **No REFUTES** — claims are matched or unmatched; there is no representation of a source that *contradicts* the answer, the most alarming provenance state of all.
- **Fixed four-paper corpus, no retrieval.** Our C2 is a genuine addition, and it introduces a failure mode they never face: **the omission view is only trustworthy if the corpus is complete**, which for a retrieval-based agent it never is.
- No ideation, gap detection or experiment planning — a verification layer for our C5, not a competitor. Provenance is not versioned against retractions or supersessions.
- **The negative behavioural result is the biggest open problem, and it is ours too.** Better provenance did not produce better behaviour.

## What We Can Use

| Component | What PaperTrail gives us |
|---|---|
| **C2 — Literature Acquisition** | The PDF pipeline with explicit checks for **math notation, paragraph boundaries and hyphenation across line breaks** — and the sobering note that they needed **manual validation** to trust it. |
| **C3 — Research Knowledge Layer** | **The five claim criteria, adopted verbatim** — externally validated at α = 0.69–0.78. Plus **offline pre-extraction as a design principle**, with their four stated benefits (async cost, single ground truth, reproducibility, **independently versionable and auditable**) as the argument for our C3 being a persistent, versioned artefact. Their per-claim record (`claim + evidence + citation + section`) matches SciGraph-LLM's independently. |
| **C4 — Gap / Novelty Detection** | **The omission view** — "relevant literature the agent found but did not use" is a strong integrity signal for a Novel Evidence Report and a failure mode our design cannot currently detect. And the claim-to-claim prompt's *"topically related but different assertions"* discrimination is sharper than anything in the Scideator/LGNA line. |
| **C5 — Provenance-Aware Experiment Planning** | The whole system is a template, with one valuable architectural property: because generation and extraction are **separate models**, provenance can be computed **over our agent's output without constraining how the agent reasons**. Their **Claim Coverage bar** is a direct implementation of our Provenance Coverage metric, and the **cost triage** (similarity → RAG → LLM per stage) is how we make it affordable. |
| **Provenance Evaluation** | Their §7.2.1 makes our case: the backend *"can be extended into a formal trustworthiness benchmark for scholarly QA… enabling quantitative measure of both faithfulness and completeness"* — our citation correctness and completeness. Reusable: precision/recall over claim sets at a conservative **τ = 0.9**; the reasoning about **why precision is structurally underestimated** when your extraction is more atomic than the reference; and the **OOD holdout** discipline. |
| **Output design (new for us)** | Our plan treats the report and provenance graph as data structures. PaperTrail is evidence that **how provenance is presented determines whether it does any good**, and that "show everything" *reduces usability without changing behaviour*. Their proposed fix — **adaptive provenance** (simple source links for straightforward claims, full claim-evidence structure for contested ones) — is what we should aim at, and **latency is a first-class design constraint**, not an optimisation detail. |

**Steal first:** (1) the five claim criteria; (2) offline pre-extraction as a versioned artefact; (3) the similarity→RAG→LLM cost triage; (4) assertional (not topical) claim matching; (5) the omission view; (6) a coverage indicator as the top-level provenance summary; (7) permissive flagging thresholds tuned against **alert fatigue**.
