# GAPMAP — Extraction Notes

> **Paper:** *GAPMAP: Mapping Scientific Knowledge Gaps in Biomedical Literature Using Large Language Models*
> **Authors:** Nourah M. Salem (Univ. of Colorado Anschutz / Univ. of Chicago), Elizabeth White, Michael Bada, Lawrence Hunter (Univ. of Chicago)
> **Venue:** NeurIPS 2025 Workshop — AI for Science · **arXiv:** 2510.25055v1 [cs.CL], 29 Oct 2025
> **Code/data:** https://github.com/UCDenver-ccp/GAPMAP
> **Funding:** Chan Zuckerberg Initiative, Award AWD105712 (*Rare as One Ignorome* project)
> **Status:** explicitly described as an ongoing project; the implicit-gap benchmarks are small and still being enriched.

---

## 1. TL;DR

GAPMAP is the paper that **operationalises "research gap detection" as a concrete, measurable NLP task** — which is exactly Component 4 of our architecture, and the one place where our plan says "this is a primary area for improvement."

Its central contribution is a distinction our design does not yet make:

- **Explicit gaps** — the authors *say* it: "X remains unknown", "no RCT has evaluated Y in Z", "further research is needed". Recoverable from lexical cues at sentence/short-paragraph level.
- **Implicit gaps** — nobody says it; it must be **inferred from section-level context**. Three sub-types: (i) a chain of claims with a missing link, (ii) a **generalisation gap** (claims established in a limited scope, out-of-scope applicability unquestioned), (iii) **conflicting findings presented without reconciliation**.

For implicit gaps they introduce **TABI (Toulmin-Abductive Bucketed Inference)**, which casts gap inference as **abductive NLI with generation**: the model must emit a `Claim` (the implied gap), the `Grounds` (evidence spans supporting it), a `Warrant` (one sentence connecting Grounds→Claim), and a `Bucket` (self-assessed confidence). The Warrant is the load-bearing design choice — **it externalises the inferential leap so a human or an NLI model can check it.**

---

## 2. Why this matters to us

Our docx (§3.4, Stage 2/3) says the agent should identify "what hasn't been studied, what assumptions existing methods make, what limitations researchers report." GAPMAP is the only one of these papers that **actually defines and measures that**, and it supplies:

1. a **typology** of gap kinds (explicit vs implicit × 5 semantic categories),
2. a **structured output schema** for a gap claim that is inherently auditable (Claim/Grounds/Warrant),
3. **evidence that this works** at 80–84% accuracy with frontier models, and
4. an honest measurement of the failure mode our plan is most exposed to: **the gap may be real but the proposed direction infeasible** (35% of GPT-4o's suggested directions were rejected by the actual paper authors, mostly on feasibility grounds).

---

## 3. Gap Taxonomy

Built on Boguslav et al. (2023)'s taxonomy of author-signalled gap statements, with an **explicit/implicit overlay added on top** (the overlay preserves the existing semantic categories while extending coverage to unstated gaps).

**Five semantic categories** (from Boguslav et al., used as the evaluation axes in Figure 2):
1. Research Aim
2. Levels of Evidence
3. Anomaly / Curious Findings
4. Barriers
5. Future Opportunities

**Explicit cue types:**
- **High-uncertainty lexical cues** — "unknown", "further research is needed"
- **Low-uncertainty hedging** — "may lead to", "could cause"
- **Negation** — "no randomized controlled trial has evaluated…"

**Worked example from Table 1** (the implicit case, which is the interesting one):

```
Grounds:  Compound E improves biomarker F in mice.
          Biomarker F correlates poorly with clinical outcomes in humans.
          [cue: outcome mismatch; translational gap]
      ⇒
Claim:    It is unknown whether E improves patient outcomes.
```

Note that neither premise states a gap. The gap only exists **jointly**, across two sentences, and only if you know that a mouse biomarker result does not transfer. That is the reasoning our C4 "research landscape" is supposed to enable — and this example is a good unit test for it.

---

## 4. TABI — Toulmin-Abductive Bucketed Inference

**Framing.** Implicit gap statements are **defeasible** — they are the *best explanation* of the observed premises, not deductive consequences. So the task is abduction, not entailment.

**Four design requirements** the authors state for the representation, which are worth adopting verbatim as our own design constraints:
1. capture **defeasible** inference,
2. remain **lightweight** enough for small models,
3. follow a **fixed pattern** so outputs can be validated,
4. **transfer across domains and granularity** (paragraph-level and document-level).

A minimal Toulmin scheme — **Claim / Grounds / Warrant (CGW)** — satisfies all four.

**Output schema:**

| Field | Definition | Purpose |
|---|---|---|
| **Claim** | The implied gap. | The deliverable. |
| **Grounds** | The evidence span(s) supporting the Claim. | **Provenance** — ties the claim to specific text. |
| **Warrant** | A *single sentence* reasoning from Grounds to Claim. | Coherence/entailment sanity check (simple NLI or rule-based patterns); **exposes the inferential leap for human validation**. |
| **Bucket** | Binary self-classification: `more_probable` vs `least_probable`. | **Calibration check** — does the `more_probable` bucket actually contain a higher fraction of correct matches? |

**Scoring rule:** a paragraph counts as a success if **any** predicted claim matches the gold gap. They also report *where* the match occurred (which bucket) and the distribution of predicted categories vs ground truth.

---

## 5. Datasets

| Category | Source | Unit | Domain | Size |
|---|---|---|---|---|
| **Explicit** | IPBES | Paragraph-level | Biodiversity & ecosystems | 657 statements from 286 paragraphs |
| **Explicit** | Scientific Challenges & Directions (Lahav et al., AAAI 2022) | Section-level | COVID-19 | 2,894 statements from 1,786 studies |
| **Implicit** | Manual implicit-gap corpus (theirs) | Paragraph-level | Biomedical | 212 paragraphs from 137 PubMed articles |
| **Implicit** | Full-text pilot | Full-paper | STEM (19 domains: immunology, astrophysics, materials science…) | 23–24 studies |

**Manual implicit-gap corpus construction** (the reusable recipe): each paragraph contains a set of premises plus a **masked conclusion statement** — the future-direction claim at the *end* of the paragraph is hidden, and the model must recover it. This is a neat, cheap way to build gap-inference ground truth **without new annotation**: papers already end paragraphs with their own stated future directions; mask them and you have gold labels.

**COVID-19 dataset caveats they hit** (useful warnings for our own eval):
- Only **one** gap statement annotated per section, despite many sections containing multiple latent gaps → models are penalised for correct extra finds.
- Authors often signal gaps via **numeric contrasts, anomalies, or novel findings** rather than uncertainty markers → pure lexical-cue detection under-recalls.
- Non-research general statements had to be filtered out (e.g. *"SARS-CoV-2 has been declared a pandemic that causes COVID-19"*).

---

## 6. Models & Experimental Setup

**Models benchmarked** (deliberately spanning scale, modality, and deployment cost):
- Closed: **GPT-5**, GPT-4o, GPT-4o mini
- Open: **Llama-3.3-70B-Instruct**, Llama-3.1-8B-Instruct, **Llama-4 Scout 17B** (MoE, 17B active / 16 experts), **gemma-2-9b-it**

**Formalisation.** Corpus `C` of sections; each section `S_i = {p_1,…,p_n}`; each paragraph `p_j = {s_j1,…,s_jm}`.

**Chunking strategy** (needed because COVID sections exceed 8K tokens):
- Segment each section into chunks `T_k` with **`|T_k| ≤ 1000 words`**
- Parser: **Stanza**
- **Chunk boundaries aligned at sentence boundaries** to preserve semantic coherence; a sentence that would overflow is deferred to `T_{k+1}`.

**Two implicit-gap experiments:**
- **Exp 1 — paragraph-level**, `D_manual = {p_1,…,p_N}`, each with one gold implicit gap `γ_i`; future-direction claims at paragraph end are masked; model emits Claim/Grounds/Warrant/Bucket.
- **Exp 2 — document-level pilot**, full manuscripts through the **GPT-4o interface** (multimodal, to handle varied input formats and long context). Output: pairs of (implied knowledge gap + supporting textual evidence, suggested future direction). **Validated by surveying the 18 corresponding authors of the papers themselves.**

**Prompting:** explicit-gap extraction used **zero-shot**; implicit-gap inference required **3-shot in-context prompting with a template**. This is a headline finding — see §8.

*(Note: the paper contains no prompt appendix; prompt text is not published. The TABI output schema in §4 is the reconstructable part.)*

---

## 7. Evaluation Protocol (genuinely reusable)

Three different matching strategies for three different situations — worth copying wholesale, because our C4 has exactly the same "the model found something real that isn't in the gold set" problem.

**1. Explicit gaps, IPBES — fuzzy one-to-one matching.**
- **ROUGE-L F1 with stemming**, threshold **> 0.55** → true positive.
- Unmatched predictions = FP; unmatched gold = FN.
- Exact string matches recorded for reference but **not required** for correctness.

**2. Explicit gaps, COVID-19 — dictionary-validated accuracy.**
The models extracted many *valid but unannotated* statements. Rather than penalise plausible finds, predictions were validated post-hoc against **Boguslav et al.'s domain-specific ignorance-cues dictionary** — a prediction counts if it carries at least one knowledge-gap cue. Primary metric becomes **accuracy**.

**3. Implicit gaps — bidirectional entailment.**
- Validate **two pairs**: (Claim ↔ gold) and (Warrant ↔ Premises).
- **Bidirectional entailment probability via RoBERTa-large**, threshold **> 0.4**.
- Rationale, quoted: *"This technique directly captures whether a claim is logically supported by its context rather than relying solely on the gold."*

**4. Document-level — author-in-the-loop.** The actual corresponding authors judge agree/disagree per gap and per direction, with a brief written justification on disagreement (irrelevance, misinterpretation, outdated issue).

---

## 8. Results

### Explicit — IPBES (Table 3; P/R/F1)

| Model | No context limit (F1) | 1000-word chunks (F1) |
|---|---|---|
| **Llama-3.3-70B** | **0.8307** | 0.8138 |
| GPT-5 | 0.7949 | 0.7947 |
| GPT-4o mini | 0.7907 | **0.8143** |
| Llama-4-17B | 0.7756 | 0.7640 |
| Gemma-2-9B | 0.6854 | 0.7401 |
| Llama-3.1-8B | 0.6275 | 0.6524 |
| GPT-4o | 0.5630 | 0.7400 |

Readings:
- **Open-weight models are competitive with closed ones on lexically-signalled gaps.** Llama-3.3-70B tops the no-limit setting on the strength of recall.
- **GPT-4o is the conservative outlier** — highest precision (0.845), worst recall (0.422). A high-precision/low-recall operating point. *(Relevant to us: if our novelty checker uses GPT-4o-class models, expect under-recall of gaps, which biases toward false "novel" verdicts.)*
- **Chunking is safe.** Top systems shift by only ~0.02 F1, and chunking *raises* recall for smaller models (GPT-4o: 0.56→0.74; Gemma: 0.69→0.74).

### Explicit — COVID-19 (Table 4; n=973 per setup, accuracy %)

| Model | 1000-word chunks | No context limit |
|---|---|---|
| **All models (union)** | **80.37** | 75.03 |
| GPT-5 | 63.51 | 60.64 |
| Llama-3.3-70B | 41.73 | 39.57 |
| Llama-4-17B | 37.20 | 38.85 |
| GPT-4o mini | 30.94 | 43.06 |
| GPT-4o | 27.85 | 25.90 |
| Gemma-2-9B | 17.16 | 17.37 |
| Llama-3.1-8B | 14.39 | 17.37 |

**The single most actionable result in the paper: the ensemble of all models (80.4%) massively outperforms the best individual model (GPT-5, 63.5%).** Figure 1's Venn analysis shows ~1.14k items shared by all four top models, but each model — GPT-5 especially — contributes **many unique finds**. On IPBES, ~50% of extracted statements (913) were shared across LLMs and **843 were uniquely identified by a single model**.

Their conclusion, verbatim: *"using a mixture of LLMs instead of relying only on the top performing ones can provide a robust gap recommender system."*

Figure 2 (performance across the 5 gap categories) shows **near-concentric contours** — GPT-5 traces the outermost polygon on every axis, then the Llama models, GPT-4o mini innermost. The consistent ordering across axes means models differ in *overall* gap-retrieval ability rather than specialising in different gap types.

### Implicit — paragraph level (Table 5)

| | GPT-5 | GPT-4o | GPT-4o mini | Gemma-2-9B | Llama-3.3-70B | Llama-4-17B | Llama-3.1-8B |
|---|---|---|---|---|---|---|---|
| FD count | 179 | 171 | 171 | 48 | 163 | 135 | 65 |
| **Accuracy (%)** | **84.43** | 80.66 | 80.66 | 22.64 | 77.89 | 63.68 | 30.67 |

**The 3-shot finding is the key one.** Under zero-shot prompting (the same setup that worked fine for explicit gaps), *"even GPT-5's outputs degenerated into vague restatements or unsupported speculations, resulting in sharp drops in accuracy."* The authors are explicit that what rescued it was not just examples but **a template**: *"it provided not just examples (3-shot) but also a template that translated raw generative ability into precise, valuable future-direction predictions."*

**Calibration is mediocre.** Models bucketed **10–24% of *correct* future-direction claims as `least_probable`**. The self-assessed confidence signal is weak — the authors speculate that other factors (lab/technical feasibility) drive perceived importance in ways the bucket doesn't capture.

### Implicit — full-document, author-validated

| Question | Result |
|---|---|
| Identified knowledge gaps are **factually true** | **83.3% agreed** |
| Gaps **remain open** | 56% fully agreed, 25.9% partially agreed (of the partials, 67% said addressing them could still significantly advance the field) |
| Proposed **future directions are valid** | **only 65%** — 35% judged invalid |

**Reasons for the 35% invalid:** feasibility — technological limits, budget constraints, or relevance to other research groups rather than the authors'.

The authors' own summary: *"although GPT-4o can effectively identify gaps, turning suggestions into actionable research requires attention to context and resources."*

---

## 9. What to take into our Research Agent

| Our component | What GAPMAP gives us |
|---|---|
| **C3 — Research Knowledge Layer** | Our per-paper schema has a flat `limitations` field. GAPMAP says that is only the **explicit** half. Add a **gap-type tag** (`explicit` \| `implicit`) and the 5 semantic categories (Research Aim / Levels of Evidence / Anomaly / Barriers / Future Opportunities) as first-class attributes on extracted gaps. Also add the three implicit sub-types — **missing link in a claim chain, generalisation gap, unreconciled conflicting findings** — as detectable relations *across* papers in our landscape graph, which is precisely where our graph has an advantage over their paragraph-level view. |
| **C4 — Gap / Novelty Detection** | **Adopt the Claim/Grounds/Warrant schema directly.** It maps almost 1:1 onto our Novel Evidence Report: `Claim` → Research Gap, `Grounds` → Evidence (with the supporting spans), `Warrant` → the reasoning step our Traceability chain currently leaves implicit. The Warrant is the missing link in `Recommendation → Reasoning → Evidence → Paper → Section` — GAPMAP shows it should be **one sentence, machine-checkable by NLI**. Also: their **`Bucket`** is a coarser cousin of our 5-level novelty ladder, and their finding that 10–24% of correct claims land in the low-confidence bucket is a caution that **self-reported confidence is not a substitute for evidence-based leveling**. |
| **C5 — Experiment Planning** | The 65%/35% author-validation split is the sharpest empirical warning in any of these papers: **a correct gap does not imply a feasible direction**, and feasibility fails for mundane reasons (budget, equipment, "that's another group's problem"). This is the same feasibility bottleneck CoI reported. It argues for making **Risks/Limitations and resource assumptions a required, provenance-backed field** in our experiment plan, not an afterthought. |
| **C-Eval — Evaluation** | Three transferable tricks: (1) **ROUGE-L F1 @ 0.55 with stemming** for fuzzy matching against gold gap statements; (2) **bidirectional NLI (RoBERTa-large) @ 0.4** to check a claim is supported by its own cited context — this is a *provenance-consistency metric we can use directly for our Provenance Evaluation §4.2*, since it scores "does the reasoning actually follow from the cited sources" automatically; (3) **cue-dictionary post-hoc validation** so the system isn't penalised for correct finds missing from an incomplete gold set. Plus the **masked-future-directions** trick for building ground truth cheaply. |
| **Architecture (system-level)** | **Run an ensemble, not a single best model, for gap extraction.** 80.4% union vs 63.5% best-single is too large to ignore, and the Venn data shows the gain is genuine complementarity, not noise. |

**Cheap wins to steal first:** (1) the CGW+Bucket output schema for every gap our agent asserts; (2) NLI-based Warrant↔Grounds checking as an automated provenance-consistency metric; (3) 1000-word sentence-aligned chunking via Stanza; (4) multi-model ensemble for gap extraction; (5) 3-shot **with a template** for any inference-style task — zero-shot is fine for extraction, not for inference.

---

## 10. Gaps, Limitations & Future Scope

### Stated or evidenced by the paper
1. **Feasibility of proposed directions is the weak point** — 35% rejected by the papers' own authors, mainly on resource/technology grounds. The system has no model of what is *doable*.
2. **Benchmarks are small and immature.** The implicit-gap datasets are 212 paragraphs and ~23 full papers; the authors call the project ongoing and enriching these datasets an "ultimate goal."
3. **Gold standards are incomplete and inconsistent.** COVID-19 annotates only one gap per section; models legitimately find more. Evaluation had to be loosened (cue-dictionary validation) to avoid punishing correct output.
4. **Poor confidence calibration** — 10–24% of correct claims bucketed as less probable.
5. **Zero-shot collapses for implicit inference**, even for GPT-5. Prompt template design is load-bearing, not incidental.
6. **Small models don't generalise** — Llama-3.1-8B and Gemma-2-9B fall off a cliff on implicit gaps (30.7%, 22.6%). Same model-gating effect ResearchAgent observed.
7. **Authors' own stated directions for robust deployment:** domain adaptation, **human-in-the-loop verification**, and benchmarking across open- and closed-weight models.
8. **Evaluation must go beyond lexical cues** — they note gaps are often signalled by numeric contrasts and anomalies, which current detection under-recalls.

### Gaps we can attack (our project's opening)
1. **Everything is single-paper, and mostly single-paragraph.** GAPMAP infers gaps *within* a document. It has **no cross-paper reasoning** — it cannot see that Paper A and Paper B share a dataset and metric but neither tests condition C. **That cross-paper landscape is exactly our C3/C4**, and it is the natural home for their two most interesting implicit types (generalisation gaps and unreconciled conflicting findings), which are *intrinsically* multi-paper phenomena but were only evaluated within a single text.
2. **No retrieval, no corpus grounding.** The model is handed a paragraph or a paper. It never searches to check whether the "gap" has *already been filled* by someone else — the precise failure our docx warns about. A gap statement from a 2019 paper may be closed by 2023. **Combining GAPMAP-style gap extraction with a retrieval-based "has this been closed?" check is an obvious, unclaimed improvement**, and it's the piece that turns their `Claim` into our 5-level potential-novelty rating.
3. **Grounds are spans, not citations.** Provenance stops at "this text in this paragraph." There is no link to *which paper*, which section, or what that paper actually established. Our provenance graph is a strict generalisation.
4. **The Warrant is generated but barely used.** They propose NLI/rule-based checking of it and use bidirectional entailment for scoring, but there is no *loop* — a failed Warrant check never triggers regeneration. **Warrant-failure → retrieve more evidence → regenerate** is a straightforward addition (and mirrors CoI's critique→query→refine).
5. **Biomedical/biodiversity-bound.** The taxonomy, the ignorance-cue dictionary, and the "translational gap" reasoning pattern are life-science-shaped. Transfer to CS/ML topics is assumed, not shown (the full-text pilot spans 19 domains but is only 23 papers).
6. **No idea generation.** GAPMAP finds gaps and suggests directions; it never produces a method or an experiment plan. It is a component, not a system — which makes it clean to slot into our C4 rather than a competitor to it.
7. **Bucketing is binary and uncalibrated.** Two levels, self-assessed, 10–24% miscalibrated. Our 5-level evidence-backed ladder is a direct improvement, and their data is the argument for why *evidence-based* rather than *self-reported* leveling is necessary.
