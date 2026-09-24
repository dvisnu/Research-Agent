# GAPMAP — Summary

> *GAPMAP: Mapping Scientific Knowledge Gaps in Biomedical Literature Using LLMs* — Salem, White, Bada, Hunter (CU Anschutz / Univ. of Chicago) · NeurIPS 2025 Workshop AI for Science · arXiv 2510.25055v1 · [code](https://github.com/UCDenver-ccp/GAPMAP)

## Summary

The paper that **operationalises "research gap detection" as a measurable NLP task** — our Component 4, and the place our plan flags as the primary area for improvement.

**The core distinction our design doesn't yet make:**

- **Explicit gaps** — the authors say it ("X remains unknown", "no RCT has evaluated Y"). Recoverable from lexical cues: high-uncertainty markers, low-uncertainty hedging, negation.
- **Implicit gaps** — nobody says it; it must be inferred from section-level context. Three sub-types: (i) a claim chain with a missing link, (ii) a **generalisation gap** (result established in a limited scope, out-of-scope applicability unquestioned), (iii) **conflicting findings presented without reconciliation**. Worked example: *"Compound E improves biomarker F in mice"* + *"biomarker F correlates poorly with clinical outcomes in humans"* ⇒ *"it is unknown whether E improves patient outcomes."* Neither premise states a gap; it exists only jointly.

Overlaid on **five semantic categories** from Boguslav et al.: Research Aim, Levels of Evidence, Anomaly/Curious Findings, Barriers, Future Opportunities.

**TABI (Toulmin-Abductive Bucketed Inference)** casts implicit-gap detection as **abductive inference, not entailment** — a gap statement is the *best explanation* of the premises, defeasible rather than deductive. Output schema:

| Field | Role |
|---|---|
| **Claim** | the implied gap — the deliverable |
| **Grounds** | the evidence span(s) — **provenance** |
| **Warrant** | one sentence reasoning Grounds→Claim — **externalises the inferential leap so NLI or a human can check it** |
| **Bucket** | binary self-confidence (`more_probable` / `least_probable`) — a calibration check |

**Data & method.** Explicit: IPBES (657 statements / 286 paragraphs, biodiversity) and a COVID-19 corpus (2,894 statements / 1,786 studies). Implicit: 212 manually-built paragraphs from 137 PubMed articles, plus a 23-paper full-text pilot across 19 STEM domains. The implicit corpus is built by a **cheap ground-truth trick — mask the future-direction claim at the end of each paragraph and have the model recover it.** Chunking: 1000-word chunks aligned at sentence boundaries via Stanza. Models spanned GPT-5/4o/4o-mini, Llama-3.3-70B, Llama-4-Scout, Llama-3.1-8B, gemma-2-9b.

**Evaluation** uses three matching strategies for three situations, all reusable: ROUGE-L F1 with stemming @ **>0.55** for fuzzy explicit matching; **cue-dictionary post-hoc validation** so valid-but-unannotated finds aren't penalised; **bidirectional NLI (RoBERTa-large) @ >0.4** on both (Claim↔gold) and (Warrant↔Premises); and author-in-the-loop validation at document level.

**Results.**

- Explicit, IPBES: Llama-3.3-70B best (F1 0.831), GPT-5 0.795. **Open-weight models are competitive on lexically-signalled gaps.** GPT-4o is a high-precision/low-recall outlier (P 0.845, R 0.422) — relevant to us, since a conservative model under-recalls gaps and biases toward false "novel" verdicts. Chunking is safe and helps small models.
- Explicit, COVID-19: **the ensemble of all models scores 80.4% vs 63.5% for the best single model (GPT-5).** Venn analysis shows genuine complementarity — on IPBES, 843 statements were found by exactly one model. The authors' conclusion: use a mixture of LLMs, not the top performer alone.
- Implicit, paragraph level: GPT-5 84.4%, GPT-4o and 4o-mini 80.7%, Llama-3.3-70B 77.9%; small models collapse (Gemma 22.6%, Llama-3.1-8B 30.7%). **Zero-shot fails even for GPT-5** — outputs degenerate into vague restatement; what rescues it is 3-shot *with a template*. Confidence is poorly calibrated: 10–24% of correct claims land in `least_probable`.
- Implicit, full-document, validated by the papers' own corresponding authors: **83.3% agree the identified gaps are factually true**, ~56% fully agree they remain open, but **only 65% judge the proposed directions valid** — the 35% rejected mostly on **feasibility** (technology limits, budget, "that's another group's problem").

## Gaps

**Stated / evidenced**

- **Feasibility is the weak point** — 35% of directions rejected by the authors themselves. The system has no model of what is doable.
- Benchmarks are small and immature (212 paragraphs, ~23 papers); the project is explicitly ongoing.
- **Gold standards are incomplete** — COVID annotates one gap per section though models legitimately find more, forcing looser metrics.
- Poor confidence calibration; zero-shot collapse for inference-style tasks; small models don't generalise.
- Evaluation must go beyond lexical cues — gaps are often signalled by numeric contrasts and anomalies, which detection under-recalls.

**Open for us**

- **Everything is single-paper, mostly single-paragraph.** No cross-paper reasoning — it cannot see that Paper A and Paper B share a dataset and metric but neither tests condition C. That cross-paper landscape is our C3/C4, and it is the natural home for their two most interesting implicit types (generalisation gaps, unreconciled conflicts), which are *intrinsically* multi-paper but were only evaluated within a single text.
- **No retrieval, no corpus grounding.** The model never searches to check whether the gap has **already been filled**. A gap stated in 2019 may be closed by 2023. Combining GAPMAP-style extraction with a retrieval-based "has this been closed?" check is an obvious unclaimed improvement — and it is what turns their `Claim` into our 5-level rating.
- **Grounds are spans, not citations** — provenance stops at "this text in this paragraph", with no link to which paper or section.
- **The Warrant is generated but barely used** — a failed Warrant check never triggers regeneration. `Warrant fails → retrieve more evidence → regenerate` is a straightforward addition.
- Biomedical/biodiversity-bound; no idea generation, no experiment plan (so it slots into C4 as a component, not a competitor); bucketing is binary and self-assessed.

## What We Can Use

| Component | What GAPMAP gives us |
|---|---|
| **C3 — Research Knowledge Layer** | Our schema has a flat `limitations` field; that is only the **explicit** half. Add a **gap-type tag** (explicit \| implicit) and the five semantic categories as first-class attributes, plus the three implicit sub-types as **cross-paper relations** in the landscape graph — where our graph has the advantage over their paragraph view. |
| **C4 — Gap / Novelty Detection** | **Adopt Claim/Grounds/Warrant directly.** It maps 1:1 onto the Novel Evidence Report: `Claim` → Research Gap, `Grounds` → Evidence with supporting spans, `Warrant` → the reasoning step our traceability chain currently leaves implicit (one sentence, machine-checkable by NLI). Their `Bucket` is a coarse cousin of our 5-level ladder, and the 10–24% miscalibration is the argument for **evidence-defined levels over self-reported confidence**. |
| **C5 — Experiment Planning** | The 65/35 author-validation split is the sharpest empirical warning in the whole reading set: **a correct gap does not imply a feasible direction**, and feasibility fails for mundane reasons. Make Risks/Limitations and resource assumptions a **required, provenance-backed field**, not an afterthought. |
| **Evaluation** | Three transferable tricks: ROUGE-L @ 0.55 with stemming for fuzzy gap matching; **bidirectional NLI @ 0.4 as an automated provenance-consistency metric** ("does the reasoning follow from the cited sources?"); cue-dictionary validation so correct finds outside an incomplete gold set aren't punished. Plus the **masked-future-directions** trick for cheap ground truth. |
| **Architecture** | **Run an ensemble for gap extraction.** 80.4% union vs 63.5% best-single is too large to ignore, and the overlap analysis shows the gain is real complementarity, not noise. |

**Steal first:** (1) CGW+Bucket schema on every asserted gap; (2) NLI Warrant↔Grounds checking as an automated provenance metric; (3) 1000-word sentence-aligned chunking; (4) multi-model ensemble; (5) 3-shot **with a template** for inference tasks — zero-shot is fine for extraction, not for inference.
