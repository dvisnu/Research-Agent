# PaperTrail — Extraction Notes

> **Paper:** *PaperTrail: A Claim-Evidence Interface for Grounding Provenance in LLM-based Scholarly Q&A*
> **Authors:** Anna Martin-Boyle (Univ. of Minnesota), Cara A.C. Leckey (NASA Langley), Martha C. Brown (NASA Langley), Harmanpreet Kaur (Univ. of Minnesota)
> **Venue:** **CHI 2026**, April 13–17, Barcelona · 25 pages · DOI 10.1145/3772318.3791101 · **arXiv:** 2602.21045v1 [cs.HC], 24 Feb 2026
> **Pre-registered:** AsPredicted (hypotheses and methods)

---

## 1. TL;DR

**This is the provenance paper for our project.** It is the only one in the set that builds *and evaluates with real users* an end-to-end argument-grounded provenance system, and its central negative result is one we need to internalise before designing our C5 output.

**The system:** decompose *both* the LLM's answer *and* the source documents into discrete **claims + evidence**, then match them. The interface then shows three things:
1. **supported assertions** — answer claims matched to paper claims;
2. **unsupported claims** — answer claims with no match (hallucination / ungrounded synthesis);
3. **omitted information** — paper claims relevant to the question that the answer *left out* — an **error of omission**, which the authors correctly identify as a *"difficult-to-detect failure mode in current systems"* that flat citation links cannot surface at all.

**The finding:** in a within-subjects study with 26 professional researchers, PaperTrail **significantly lowered trust** in the LLM (4.22 → 3.68, *p* = .015, *d* = 0.44) but **did not change reliance behaviour at all** (0.73 → 0.75, n.s.), and **significantly hurt usability** (5.05 → 4.48, *p* = .026, *r* = 0.52).

> **The trust–behaviour gap is the headline.** Giving experts better provenance made them *more sceptical* and *no more careful*. People kept using LLM output because verification was cognitively expensive and they were under time pressure. As one participant put it: *"I depended on the AI more because of the short time constraint and ultimately spent less time revising and checking the results than I otherwise would have."*
>
> Our docx assumes provenance produces trustworthy output. **PaperTrail's evidence is that provenance produces *justified scepticism* — which is valuable and different.** It also means the *presentation* of our provenance is a first-class design problem, not a rendering detail.

---

## 2. Why coarse citations aren't enough

Their framing, drawn from surveys of the attribution literature (Schreieder et al. 2025 over 134 systems; Pang et al.; Li et al.):

- Most systems attach provenance at **document, paragraph, or sentence** granularity, and treat attribution as a **backend / benchmarking** problem.
- Coarse citations produce familiar failure modes: **granularity errors, mistaken synthesis across sources, and hallucinated statements** when complex answers are supported only by flat sentence-level links.
- Source citations *"can improve perceptions of trust, but have been found to sometimes be hallucinated or inaccurate in representing the source material."*
- Rich explanations increase **cognitive load** without giving **actionable paths for evidence validation**, and are vulnerable to an **"illusion of explanatory depth"** — users overestimate their understanding without actually verifying anything.

In Pang et al.'s taxonomy, PaperTrail is a **posterior external-data sourcing system**: it doesn't modify weights or embed watermarks; it **infers and exposes how an answer relates to a fixed corpus at interaction time.** That is exactly the position our C5 provenance graph occupies.

They ground the design in **Toulmin's argumentation model** — claims supported by evidence connected through warrants — on the argument that *"scientific papers inherently follow these argumentation patterns"*, so a claim-evidence interface **leverages domain experts' existing mental models** rather than imposing a new abstraction.

---

## 3. Architecture — the Argument Extraction Engine

The organising idea is a **cost/capability gradient across three extraction methods, deployed strategically per pipeline stage**:

| Method | Cost | Capability | Where used |
|---|---|---|---|
| **Similarity-based** (sentence-transformer + cosine) | **Low** | rapid filtering, dedup, speed, interpretability | Stage 1 evidence retrieval; Stage 3 evidence verification; pre-filtering everywhere |
| **RAG-based** (similarity filter → LLM on reduced set) | Medium | retrieval efficiency + semantic understanding | Stage 3 relevant-claim and relevant-evidence selection |
| **LLM-based** (few-shot prompting) | **High** | nuanced semantic interpretation of scientific discourse | Stage 1 claim extraction (offline); Stage 2 answer decomposition; Stage 3 claim-to-claim matching |

> **This explicit cost/capability triage is the most portable engineering idea in the paper.** Our pipeline will have the same problem — full semantic analysis of every paper against every query is unaffordable — and their answer is a design-time allocation: cheap methods where you only need *topical* filtering, expensive methods where you need *semantic judgement*.

```
┌── STAGE 1: OFFLINE PAPER PROCESSING ──────────────────────────────┐
│ Paper PDFs → PyMuPDF text extraction → manual validation          │
│            → section/paragraph segmentation                       │
│  [LLM]  extract paper claims (per paragraph, 10-shot)             │
│  [SIM]  retrieve evidence for each claim (cosine ≥ 0.75)          │
│       → single JSON: all claims + evidence + citation + section   │
└───────────────────────────────────────────────────────────────────┘
┌── STAGE 2: REAL-TIME ANSWER PROCESSING ───────────────────────────┐
│  answerer LLM  → answer (with citation tags)                      │
│  [LLM] extraction LLM → answer claims + evidence (single pass)    │
│  NLTK punkt → span annotation → exact character positions         │
└───────────────────────────────────────────────────────────────────┘
┌── STAGE 3: REAL-TIME CLAIM MATCHING ──────────────────────────────┐
│  [RAG] SPECTER sim filter → LLM selects relevant paper claims     │
│  [RAG] SPECTER sim filter → LLM selects relevant evidence         │
│  [LLM] claim-to-claim semantic-equivalence matching               │
│  [SIM] evidence verification: flag if cosine < 0.55               │
│       → provenance indicators for the interface                   │
└───────────────────────────────────────────────────────────────────┘
```

### Stage 1 — Offline paper-level extraction

**PDF handling** (worth noting the care taken — and that it required manual work):
- **PyMuPDF v1.23.5** text extraction, followed by **manual validation** against the original PDFs checking: correctness of **mathematical notation**, **paragraph boundary** preservation, and **hyphenated words across line breaks**.
- Programmatic segmentation into **sections and paragraphs** to maintain contextual coherence.

**Claim extraction** — LLM-based, **per paragraph**, using a **10-shot prompt** with examples randomly sampled from **SciClaimHunt** (Kumar et al.) and grounded in Toulmin.

**Five claim criteria** — three from Kumar et al., two added from Toulmin:

| Criterion | Definition |
|---|---|
| **Atomic** | A single, specific, indivisible assertion. No compound statements that can be decomposed further. |
| **Verifiable** *(added)* | States something factual whose truthfulness can be checked against evidence or data. |
| **Faithful** | Accurately and precisely reflects the source paragraph. No outside information, no unsupported inferences. |
| **Decontextualized** | Understandable as a standalone statement, needing minimal surrounding text. |
| **Declarative** *(added)* | A clear statement or assertion — **not a question, a hypothesis phrased as a question, or a description of methods/procedures**. |

Kumar et al. validated these criteria through human annotation of 100 claims: **inter-annotator agreement α = 0.69–0.78**.

**Their own validation approach is a nice cheap technique:** they spot-checked claims from one paragraph each of the **Introduction, Methods, and Results** sections of every paper, *"because these sections represent distinct discourse functions — Introductions contain motivational and background claims, Methods contain procedural claims, and Results contain empirical findings; we wanted to verify that the extraction prompt handled this variation appropriately."*

**Why LLM-based here despite the cost — two stated reasons:**
1. **Without a query there is nothing to filter on.** Similarity and RAG approaches need a query as the reference point for relevance scoring, so comprehensive extraction requires the LLM.
2. **It's offline, so cost is acceptable** — once per document, not per interaction.

**Why pre-extract rather than extract at query time — four stated benefits** (a good argument for our C3 being a persistent store rather than an on-demand computation):
- shifts computational burden from **synchronous user-facing** to **asynchronous preprocessing**;
- **consistent across users** — offline preprocessing creates a **single ground truth**;
- **reproducible results**;
- the claim-evidence corpus can be **versioned and audited independently** of the query pipeline.

**Evidence retrieval** — similarity-based, **cosine ≥ 0.75** (threshold from Kumar et al.). **Preceding and following sentences are included as context** to reconstruct coherent readable snippets. The rationale for using the cheap method here is precise: *"this stage serves only as an initial filtering step… we simply need to establish which text segments have any topical relationship to each claim… deferring the computationally expensive task of determining contextual relevance until it can be informed by the user's actual query."*

**Output:** a single JSON of all paper claims with **evidence, citation, and section name**, loaded at server startup as the ground-truth knowledge base.

### Stage 2 — Real-time answer-level extraction

**Two distinct models with separated roles:**
- **Answerer LLM** — generates the answer from query + conversation history + task context + source documents. Operates **without structural constraints**, deliberately, to mimic the commercial chat experience.
- **Extraction LLM** — decomposes the answer into claims + evidence, using **the same five claim criteria** as paper-level extraction, so the two sides are structurally comparable.

> **This separation is what makes the system model-agnostic**: *"the claim-evidence extraction and matching stages operate on the generated answer regardless of how that answer was produced."* It would work identically over a RAG system's output. For us, that means a provenance layer of this shape can be **bolted onto our agent's output without constraining how the agent reasons** — an attractive architectural property.

Claims and evidence are extracted in **a single pass** (answers are short). Then **span annotation**: NLTK punkt sentence tokenizer + a programmatic matching function that **locates exact character positions** of each claim and evidence piece, mapping structured output back to text spans for highlighting.

The key contrast they draw with RAG interfaces: *"while RAG systems often display retrieved passages as coarse-grained attribution, PaperTrail decomposes both the answer and source documents into discrete claims, which allows for verification of specific assertions rather than entire passages."*

### Stage 3 — Claim-evidence matching

Produces both **global explanations** (overall claim coverage) and **local explanations** (individual claim provenance).

1. **Relevant claim selection** — SPECTER cosine similarity between the question and each paper claim narrows *"hundreds of claims"* to a candidate set; then the LLM selects the most relevant.
2. **Relevant evidence selection** — same two-step pattern, per claim.
3. **Claim-to-claim matching** — the LLM compares answer claims against filtered paper claims and finds **semantically equivalent pairs**. Rationale: this *"benefits from the model's more nuanced language understanding… and avoids spurious connections that can occur when using simple cosine-similarity matching."*
4. **Evidence verification** — cosine similarity, **flag if < 0.55**.

**The 0.55 threshold is deliberately permissive**, tuned on five held-out examples to satisfy two competing requirements: *"first, to identify answers that diverge substantially from the source material; and second, **to avoid alert fatigue** and acknowledging that relevant evidence in an answer may not be present in the filtered set of paper evidence."*

> **Alert fatigue is a real design constraint we should plan for.** A provenance system that flags everything gets ignored. Our Provenance Coverage metric needs a companion policy for *when to surface a warning*, tuned against false-alarm cost.

### Implementation details

| Item | Value |
|---|---|
| LLM (all extraction & matching) | **Gemini 2.5 Pro** (`gemini-2.5-pro-preview-05-06`), **temperature 1.0** |
| Structured output | Gemini's `response_json_schema` parameter — *"guarantees responses conform to our claim-evidence schema"* |
| Embeddings | **SPECTER** (chosen for optimisation on scientific text) |
| PDF extraction | **PyMuPDF v1.23.5** + manual validation |
| Sentence tokenisation | **NLTK punkt** |
| Stage 1 evidence threshold | cosine **≥ 0.75** |
| Stage 3 evidence-flag threshold | cosine **< 0.55** |
| Few-shot examples | **10**, from SciClaimHunt |
| Backend | Flask, RESTful API, session management, interaction logging |
| Frontend | React + Redux Toolkit, three-panel resizable layout |
| **End-to-end latency** | **~90 seconds per query** — *"primarily due to sequential LLM calls in Stages 2 and 3"* |

**The 90-second latency is a cautionary number for us.** Participants called it *"excruciatingly slow"* and *"much slower than others [LLMs] that I have used"*, and the authors identify it as a major contributor to the study's null behavioural result.

---

## 4. Prompt Library (verbatim, Appendix A)

### A.1 — Paper-Level Claim Extraction *(Stage 1, offline)*
```
[Preamble establishing the task]
A claim must satisfy the following criteria:
(1) Atomic: Focus on a single, specific, and indivisible assertion, finding,
    or conclusion. Avoid compound statements that can be decomposed further.
(2) Verifiable: State something factual whose truthfulness can be checked
    against evidence or data, either within this paragraph or the broader
    scientific context.
(3) Faithful: Accurately and precisely reflect the meaning and information
    given in the source paragraph. Do not introduce outside information or
    make inferences not directly supported by the text.
(4) Decontextualized: Be understandable as a standalone statement, requiring
    minimal or no surrounding text from the original paper to grasp its meaning.
(5) Declarative: Be a clear statement or assertion, not a question, hypothesis
    phrased as a question, or a description of methods or procedures.

Based on these principles, transform the following paragraph into zero or more
distinct scientific claims.
{FEW-SHOT EXAMPLES: 10 paragraph-claim pairs randomly sampled from SciClaimHunt}
Paragraph: {PARAGRAPH}
```
```json
JSON Schema:
{ "type": "array",
  "items": { "type": "object",
    "properties": { "claim": { "type": "string",
                               "description": "A single, atomic claim" } },
    "required": ["claim"] } }
```
> *"zero or more"* is a small but important detail — a paragraph with no verifiable claims must be allowed to yield nothing.

### A.2 — Answer Generation *(Stage 2)*
```
You are an advanced AI research assistant designed to assist users with scholarly
literature analysis and question answering. Your primary function is to provide
accurate and insightful answers to questions based on one or more scholarly papers
provided to you.
The user is performing an editing task and may refer to text they are editing. This
text and a description of the editing task will be provided. The conversation history
will also be provided if it exists.
In your response, use tags around each sentence to indicate which paper(s) are being
referenced. These tags will be removed in post-processing before the answer is shown
to the user. Format: <Author et al., year> sentence </Author et al., year>. For
sentences drawing on multiple papers, separate citations with semicolons:
<Author et al., year;Author et al., year>.
Provide your answer in 300 words or fewer. Do not use formatting such as bullet
points or headers.
Papers: {PAPER_CONTENTS}
Task description: {TASK_DESCRIPTION}
Text being edited: {EDITOR_TEXT}
Conversation history: {CONVERSATION_HISTORY}
Question: {USER_QUESTION}
```
> The **inline sentence-level citation tags** stripped in post-processing are how the *baseline* condition gets its source highlights — a cheap, effective attribution mechanism independent of the claim pipeline.

### A.3 — Answer-Level Claim and Evidence Extraction *(Stage 2)*
```
You are an expert research assistant specializing in extracting structured
information from scientific texts. Your task is to carefully read the provided text
and decompose it into discrete claims and their supporting evidence.
A claim must satisfy the following criteria:
(1) Atomic: … (2) Verifiable: … (3) Faithful: … (4) Decontextualized: … (5) Declarative: …
For each claim identified, extract the exact text spans from the input that express
the claim and any text spans that serve as supporting evidence. Text spans must match
the original input exactly to enable precise highlighting in the user interface.
{FEW-SHOT EXAMPLES: text-to-claims-and-evidence pairs}
Text: {ANSWER_TEXT}
```
```json
{ "type": "array", "items": { "type": "object", "properties": {
    "claim":          { "type": "string", "description": "A single atomic answer claim" },
    "claim_texts":    { "type": "array", "items": {"type":"string"},
                        "description": "Verbatim claim texts" },
    "evidence_texts": { "type": "array", "items": {"type":"string"},
                        "description": "Verbatim evidence texts" } },
  "required": ["claim", "claim_texts", "evidence_texts"] } }
```
> Note the separation of the **normalised claim** (`claim`) from its **verbatim spans** (`claim_texts`) — the same split SciGraph-LLM makes between the controlled paraphrase `t` and the verbatim evidence `e`. Two independent papers, same conclusion: **you need both a canonical form for matching and an exact span for grounding.**

### A.4 — Relevant Claims Selection *(Stage 3, RAG)*
```
You are provided with a set of scientific claims extracted from scholarly papers and
a user's question about those papers. Your task is to identify which claims are
relevant to answering the question.
A claim is relevant if it:
 • Directly addresses the question
 • Provides necessary background information
 • Contains factual information that would contribute to a complete answer
Each claim is accompanied by a numerical ID. Return only the IDs of relevant claims.
Question: {USER_QUESTION}
Claims: {CLAIM_LIST_WITH_IDS}
```
```json
{ "type": "array", "items": { "type": "integer",
                              "description": "ID of a relevant claim" } }
```

### A.5 — Relevant Evidence Selection *(Stage 3, RAG)*
```
You are provided with a set of evidence passages associated with a scientific claim,
along with a user's question. Your task is to identify which evidence passages are
most relevant in the context of the question.
Evidence is relevant if it:
 • Directly supports or substantiates the claim
 • Provides data, results, or reasoning that validates the claim
 • Contains contextual information necessary to understand the claim in relation
   to the question
Each evidence passage is accompanied by a numerical ID. Return only the IDs of the
most relevant evidence passages.
Question: {USER_QUESTION}
Claim: {CLAIM_TEXT}
Evidence passages: {EVIDENCE_LIST_WITH_IDS}
```

### A.6 — Claim-to-Claim Matching *(Stage 3)*
```
You are provided with two sets of claims: (1) claims extracted from an LLM-generated
answer, and (2) claims extracted from source scholarly papers. Your task is to identify
which answer claims are semantically equivalent to which paper claims.
Two claims are semantically equivalent if they express the same core assertion, even if
worded differently. Minor differences in phrasing, specificity, or elaboration are
acceptable provided the fundamental meaning is preserved. Do not match claims that are
merely topically related but make different assertions.
For each answer claim that has a match in the paper claims, return the answer claim ID
paired with the ID(s) of the matching paper claim(s). Omit answer claims that lack a
clear match.
Answer claims: {ANSWER_CLAIMS_WITH_IDS}
Paper claims: {PAPER_CLAIMS_WITH_IDS}
```
```json
{ "type": "array", "items": { "type": "object", "properties": {
    "answer_claim_id": { "type": "integer" },
    "paper_claim_ids": { "type": "array", "items": {"type": "integer"},
                         "description": "IDs of semantically equivalent paper claims" } },
  "required": ["answer_claim_id", "paper_claim_ids"] } }
```
> **`"Do not match claims that are merely topically related but make different assertions"`** is the single most important line in the prompt library for our purposes. Topical relatedness is what embedding similarity measures; *assertional equivalence* is what provenance requires. **Omitting unmatched answer claims is what produces the "unsupported" flag** — the absence of a match is itself the signal.

*(Note: Stage 1 evidence retrieval and Stage 3 evidence verification are pure cosine-similarity operations and use no LLM prompt.)*

---

## 5. Interface Design (four design goals)

| Goal | Principle | Instantiation |
|---|---|---|
| **DG1: Support graduated cognitive engagement** | Shneiderman's mantra — *"overview first, zoom and filter, then details-on-demand"*; Pirolli & Card's foraging/sensemaking loops ("shoebox" → "evidence file") | Claim Coverage bar (overview) → claim annotations (zoom) → source text (details) |
| **DG2: Minimise interaction overhead while preserving exploration depth** | Coordinated Multiple Views: optimise space/time, self-evidence via brushing-and-linking, attention management via selective hiding | Three-panel layout, event bus, auto-scroll of linked panels, collapsible per-turn sections, greying out non-evidence text |
| **DG3: Enable flexible verification workflows** | Experts *"find a clue, and follow the trail"*; exploratory search over linear lookup | **Bidirectional navigation**: top-down (answer claim → supporting evidence) and bottom-up (paper claim → gaps in the answer). No single enforced pathway. |
| **DG4: Align with scholarly mental models** | Experts rely on structural/causal features, not superficial ones | Decompose into claims and evidence — the structure of scientific discourse itself |

**Layout:** Left = task context + references + text editor; Middle = QA chat with **Answer Claims** highlighted; Right = provenance panel.

**The provenance panel is the part we should copy conceptually:**
- **Claim Coverage (C1)** — a horizontal bar, coloured red→teal, showing *how many relevant source claims the answer included*.
- **"Claims included in answer"** — teal cards.
- **"Claims omitted from answer"** — **red cards**. Paper claims relevant to the question that the answer never mentioned.
- Clicking an answer claim highlights its matching paper claim with a vertical yellow bar; all other answer text greys out.
- Each card is bound to a specific chat turn in a collapsible section, so prior turns persist and can be revisited.

> **The omission view has no analogue in our current docx.** Our C5 provenance graph answers *"what supports this recommendation?"* It does not answer *"what relevant evidence did the agent ignore?"* — and PaperTrail argues that omission is the harder, more consequential failure. For a system that produces a **Novel Evidence Report** and claims a gap exists, "what did you not look at" is arguably the most important question a reader can ask.

**Baseline condition:** identical except the right panel shows **verbatim source text** when you click a sentence-level source highlight — i.e. exactly what commercial LLM search tools do today.

---

## 6. Offline Backend Evaluation

**The problem they hit is one we will hit too:** *"No gold-standard benchmark exists for evaluating LLM-based claim extraction that generates a complete set of atomic claims (rather than extracting verbatim text spans) from scientific texts."* Existing datasets target either non-exhaustive sentence-level claim *classification* or claim *verification*. So they adapted two:

| Dataset | Nature | Sample |
|---|---|---|
| **SciClaimHunt** (Kumar et al.) | Claims generated from paragraphs by **Llama-2-13B** few-shot; paper-level (not paragraph-level) mappings, so claims were re-matched to paragraphs by exact string match then SPECTER similarity | 50 paragraphs with ≥3 claims (+50 held out for development) |
| **BioClaimDetect** (Achakulvisut et al.) | **Human-annotated** biomedical abstracts, **sentence-level** claim labels (whole sentences, not atomic units) | 50 abstracts from the test set |

**Procedure:** apply the extraction prompt with 10-shot examples from the respective dataset; embed reference and extracted claims with SPECTER; a reference claim counts as **matched** if any extracted claim exceeds **cosine τ = 0.9** (deliberately conservative — *"ensures matched claims are semantically near-equivalent rather than only topically related"*).

- **Recall** = proportion of reference claims matched by ≥1 extracted claim (coverage).
- **Precision** = proportion of extracted claims matched by ≥1 reference claim (accuracy).

**Results:**

| Dataset | Precision | Recall | F1 |
|---|---|---|---|
| SciClaimHunt | 0.69 | 0.62 | 0.65 |
| BioClaimDetect | 0.63 | **0.88** | **0.73** |

**The interpretation is careful and worth copying:**
- **Precision is structurally underestimated** — *"neither dataset provides exhaustive atomic claim annotations… valid extracted claims in our more atomic approach may not match any reference claim simply because the reference set is an output of a less granular approach."* **Manual inspection confirmed most unmatched extracted claims were valid claims absent from the reference annotations rather than extraction errors.**
- **Recall is the meaningful metric** given the intent to provide coverage beyond existing methods.
- **Lower recall on SciClaimHunt (0.62) is attributed to noise in the reference set**, which was *machine*-generated by Llama-2-13B rather than human-annotated.
- **BioClaimDetect was the out-of-distribution test** (human annotations, different domain, not used in development). Stronger performance there suggests genuine generalisation.

> Two transferable lessons: (1) **when no gold standard exists, adapt an adjacent one and reason explicitly about which metric is trustworthy** — here, recall; (2) **hold out an OOD dataset you did not develop against**, because they developed on SciClaimHunt and validating on BioClaimDetect is what makes the generalisation claim credible.

---

## 7. User Study

### Design
- **Within-subjects**, **N = 26** researchers from a research organisation (NASA Langley), pre-registered on AsPredicted, exempt from ethics review.
- **Participants:** 18 with 6–20 years professional experience; all but two held advanced degrees (13 Master's, 11 Doctoral); 21 used LLM tools daily or weekly; domains including chemistry, physics, materials science, aerospace.
- **Two tasks**, fixed order (controlling for complexity, fatigue, learning); **interface order randomised and counterbalanced**; ~20 minutes each.
- **Corpus:** four peer-reviewed papers on Mars exploration and planetary surface operations.

**Task 1 — Multi-Paper Synthesis.** Edit a ~300-word LLM-generated draft synthesising methods and findings from all four papers. Represents literature review.
**Task 2 — Devil's Advocate.** Edit an LLM-generated defence of a paper positioned as the participant's own manuscript, with the other three as prior work. Simulates pre-submission review.

**Question Bank** generated with Gemini using the **QASA** question taxonomy (surface / testing / deep questions), to reduce task burden and standardise engagement.

### Measures

**Primary:**
- **Trust** — TXAI scale, post-task.
- **Reliance** — **normalized Levenshtein edit distance** between the original LLM-generated text and the participant's edited version. (Lemmatised, stop-words removed, word-level.)
- **Confidence** — two items, averaged; Cronbach's α = 0.89.

**Secondary:** NASA-TLX (cognitive load), SUPR-Q (usability), interaction logs (time-on-task, questions asked, provenance clicks), interface preference, open-ended feedback.

### Results (Table 1)

| Measure | Baseline | PaperTrail | Statistic | Effect size |
|---|---|---|---|---|
| **Trust** | 4.22 ± 1.22 | **3.68 ± 1.24** | *t* = 2.61, ***p* = .015*** | *d* = 0.44 |
| **Reliance** | 0.73 ± 0.21 | 0.75 ± 0.29 | *W* = 146.00, *p* = .313 | −0.23 |
| **Confidence** | 4.27 ± 1.51 | 4.05 ± 1.58 | *t* = 0.64, *p* = .525 | 0.14 |
| Cognitive Load | 4.10 ± 0.83 | 4.12 ± 0.78 | *t* = −0.21, *p* = .838 | 0.03 |
| **Usability** | 5.05 ± 1.08 | **4.48 ± 1.17** | *W* = 71.50, ***p* = .026*** | *r* = 0.52 |
| Clicks | 8.26 ± 6.02 | 12.47 ± 14.96 | *W* = 97.50, *p* = .137 | −0.35 |
| Messages | 3.67 ± 2.57 | 4.08 ± 2.84 | *t* = −0.79, *p* = .438 | 0.16 |

**Trust down, reliance unchanged, usability down.** Cognitive load was *not* significantly different — the burden showed up as reduced usability, not measured workload.

### Correlation analysis (Tables 2–3) — a genuinely interesting result

**Baseline condition:** reliance correlated significantly with **trust** (*r* = 0.57, *p* < .01) and **usability** (*r* = 0.60, *p* < .05). People who trusted the LLM more, and found the system more usable, edited less.

**PaperTrail condition: those correlations disappear entirely.** Trust, confidence, and usability remain correlated with each other, but **none of them predicts how much a participant edited.**

> **Claim-evidence provenance decoupled generalised trust from behavioural reliance.** In the baseline, "how much do I trust LLMs" drove "how much do I edit." With PaperTrail, that link broke — plausibly because people were making *per-claim* decisions rather than one global trust judgement. **That is arguably the more important finding than the trust drop**, and it is the one that supports our project's premise: granular provenance changes *the unit of the trust decision*, even when aggregate behaviour looks unchanged.

Also: a significant **negative correlation between education and confidence** in the PaperTrail condition (*r* = −0.41, *p* < .05), absent in baseline — *"more highly educated participants may have been more sensitive to the provenance information, leading to reduced confidence in their outputs."*

### Qualitative themes

**(1) Time pressure and latency.** The most frequently cited barrier, mentioned by **over half** of participants. Combined with ~90s latency:
> *"I feel the time constraints fundamentally changed the way I used and trust the AI output. I depended on the AI more because of the short time constraint and ultimately spent less time revising and checking the results than I otherwise would have."* — P15

**(2) The information-complexity / usability tradeoff.** Participants were dissatisfied with LLM output quality — one compared it to *"an eighth grader"* (P25), another to *"what I would expect from a novice researcher placed in the same bind"* (P4) — and therefore *wanted* rich provenance. But the implementation was *"cluttered…for a fairly small laptop screen"* with nested windows allowing reading *"only a line or two at a time."*

Notably, **DG2 and DG3 both failed on their own terms**: the interface increased rather than minimised overhead, and *"the rigid presentation of claim-evidence cards actually constrained participants' verification strategies."* The one clear success was the question bank (DG1): *"I could not have done this without the suggested questions."* (P18)

**(3) The ethos was valued even when the implementation wasn't.** Participants articulated the need for *"fact checking"*, *"trying to figure out what was missing or possibly incorrect"*, and evaluating *"how credible or trustworthy the application's responses were."* The most on-point feedback:
> *"If more tools could be given to find the location of specific claims within the papers, that would be helpful. I find that errors often occur in LLMs through **stripping context**. To help with overall accuracy, every effort should be made to help the human user track down the original context to verify claims."* — P14

---

## 8. Why the trust–behaviour gap? (three explanations worth internalising)

1. **People default to System 1 thinking** — fast, heuristic — even when they intellectually recognise the need for deliberation. XAI work shows explanations can *worsen* this by *"providing convenient narratives and analogic thinking devices that feel like comprehension without requiring genuine verification."* PaperTrail tried to engage System 2, but *"the cognitive cost of verification remained too high relative to the perceived benefits."*
2. **The feedback loop that builds expertise is missing.** Klein's conditions for genuine intuitive expertise are **high-validity environments with stable regularities** and **opportunities to learn them through feedback**. AI-assisted settings violate both: LLMs err unpredictably, and users get **no feedback on whether their edits actually improved accuracy**. *"Without this feedback loop, people cannot learn when their skepticism should translate into action."*
3. **Design friction is a hard balancing act.** Too burdensome → users circumvent or abandon (*"our study feedback suggests we erred in this direction"*). Too lightweight → System 2 never engages.

**And a scope caveat they raise that applies directly to us:** *"appropriate reliance is conceptually different for these settings. Decision-making offers binary or multi-class constraints, while scholarly QA is **open-ended — lacking ground truth** and depending on question and user context."* Our C4/C5 outputs are open-ended in exactly this sense.

---

## 9. What to take into our Research Agent

| Our component | What PaperTrail gives us |
|---|---|
| **C2 — Literature Acquisition** | The PDF pipeline (PyMuPDF + explicit checks for **math notation, paragraph boundaries, hyphenation across line breaks**) — and the sobering note that they needed **manual validation** to trust it. Section+paragraph segmentation preserved for contextual coherence. |
| **C3 — Research Knowledge Layer** | **The five claim criteria (atomic / verifiable / faithful / decontextualized / declarative)** are the best-specified claim definition in the whole set, and are **externally validated** (α = 0.69–0.78 on 100 human-annotated claims). Adopt them verbatim. Also: **offline pre-extraction as a design principle** — their four stated benefits (async cost, single ground truth across users, reproducibility, **independently versionable and auditable**) are precisely the argument for our C3 being a persistent, versioned artifact rather than an on-demand computation. And their per-claim record (`claim + evidence + citation + section`) matches SciGraph-LLM's independently. |
| **C4 — Gap / Novelty Detection** | **The omission view.** *"Claims omitted from answer"* — relevant source claims the output ignored — is a failure mode our design does not currently detect, and the authors rightly call it *"difficult-to-detect."* For a Novel Evidence Report, "here is relevant literature the agent found but did not use" is a strong integrity signal. Also, the **claim-to-claim semantic-equivalence** prompt — *"do not match claims that are merely topically related but make different assertions"* — is exactly the discrimination our novelty checker needs, and is a sharper formulation than anything in the Scideator/LGNA line. |
| **C5 — Provenance-Aware Experiment Planning** | **The whole system is a template for our provenance layer**, with one crucial architectural property: because answer generation and claim extraction are **separate models**, provenance can be computed **over our agent's output without constraining how the agent reasons**. Their **Claim Coverage** bar is a direct implementation of our **Provenance Coverage (PC) = supported decisions ÷ total decisions**. And the **cost/capability triage** (similarity → RAG → LLM, allocated per stage) is how we make this affordable. |
| **C-Eval — Provenance Evaluation (§4.2)** | Their §7.2.1 makes our case for us: *"The backend architecture of PaperTrail can be extended beyond a user-facing tool into a formal **trustworthiness benchmark** for scholarly QA systems… Comparing answer claims against relevant source claims enables quantitative measure of both **faithfulness and completeness**."* That is our citation correctness and citation completeness. Also reusable: **precision/recall over claim sets with a conservative τ = 0.9 match threshold**; the reasoning about **why precision is structurally underestimated** when your extraction is more atomic than the reference; and the **OOD holdout** discipline. |
| **Output design (new for us)** | Everything in §5 and §8. Our docx treats the Novel Evidence Report and provenance graph as data structures. PaperTrail is evidence that **how provenance is presented determines whether it does any good** — and that a naive "show everything" presentation *reduces usability without changing behaviour*. Their proposed fix, **adaptive provenance** (simple source links for straightforward claims, full claim-evidence structure for complex or contested ones), is the design we should aim at. |

**Cheap wins to steal first:** (1) the five claim criteria; (2) offline pre-extraction as a versioned, auditable artifact; (3) the similarity→RAG→LLM cost triage; (4) claim-to-claim assertional (not topical) matching; (5) the omission view; (6) a coverage indicator as the top-level provenance summary; (7) permissive flagging thresholds tuned against **alert fatigue**.

---

## 10. Gaps, Limitations & Future Scope

### Stated by the authors (§8)
1. **Task realism.** Participants used *unfamiliar* papers under 20–30 minute constraints, rather than their own literature and self-directed questions. *"While this standardization was necessary for measuring reliance consistently, it may have fundamentally altered engagement patterns."* Field studies needed.
2. **Reliance is measured by edit distance**, which *"conflates various behaviors (from wholesale acceptance to strategic delegation)"* and creates a confound — **participants might edit less precisely because they trusted PaperTrail to flag problems for them.**
3. **Trust may have been measured against the wrong target** — *"participants may not even have distinguished between the interface and the LLM."* Component-specific trust measures needed.
4. **Output quality was never assessed.** They did not check whether unsupported claims were removed, omitted information added, or argumentation improved. **Blind expert evaluation of edited text is the missing measurement.**
5. **Design generalisability** — one interface paradigm (three panels, coordinated views); alternatives (inline annotations, progressive disclosure, conversational verification) might behave differently.
6. **Only claims and evidence — no warrants, rebuttals, backings, or qualifiers.** They use a *reduced* Toulmin model. *"Domains within and outside of STEM may have varying argumentation conventions."*
7. **Single session.** *"Trust and reliance patterns likely evolve as users develop mental models."* Longitudinal deployment needed.
8. **Latency (~90s)** from sequential, non-parallelisable LLM calls (answers must exist before claims can be extracted; paper claims must be filtered before matching). Proposed fixes: **aggressive caching** of common query patterns, **parallel processing** of answer chunks, faster inference, and **federated architectures** where institutions pre-process collections offline and share lightweight claim indices.

### Gaps we can attack (our project's opening)
1. **The Warrant is missing.** PaperTrail matches *claim → evidence*, but never asks **why** the evidence implies the claim. **GAPMAP's Claim/Grounds/Warrant supplies exactly the missing piece**, and *Fact or Fiction*'s five reasoning types (directionality, numerical, cause/effect, background, coreference) explain what can go wrong in that gap. **Semantic equivalence is not entailment** — two claims can be equivalent in embedding space and opposite in polarity, as SciGraph-LLM's inverted-claim example demonstrates. **A warrant-checking step over PaperTrail's matching architecture is a well-defined, unclaimed improvement.**
2. **Matching is semantic equivalence, not support.** The system asks *"does the source say the same thing?"*, not *"does the source support this?"* A claim can be well-supported by evidence that asserts something different (a premise, not a paraphrase). This is a real scope limit, and it is the difference between paraphrase-matching and *Fact or Fiction*-style SUPPORTS/REFUTES verification.
3. **No REFUTES.** Claims are matched or unmatched. There is no representation of source claims that **contradict** the answer — the most alarming provenance state of all, and one SciFact treats as first-class.
4. **Fixed four-paper corpus, no retrieval.** Papers are given, pre-processed offline. There is no search, no ranking, no coverage question. **Our C2 is a genuine addition, and it also introduces a failure mode PaperTrail never faces:** the omission view is only trustworthy if the corpus is complete, which for a retrieval-based agent it never is.
5. **No ideation, no gap detection, no experiment planning.** It verifies an answer against a corpus. It is a **verification layer for our C5**, not a competitor.
6. **Provenance is not versioned against a changing corpus.** They note the claim corpus *can* be versioned; nothing handles a paper being retracted, updated, or superseded.
7. **The negative behavioural result is the biggest open problem, and it is ours too.** Better provenance did not produce better behaviour. Our project should plan for this: measure **output quality**, not just provenance coverage; consider **adaptive** rather than exhaustive provenance display; and budget for latency, because 90 seconds per query demonstrably destroyed engagement with an otherwise well-designed system.
