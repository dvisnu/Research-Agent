# SciGraph-LLM — Extraction Notes

> **Paper:** *SciGraph-LLM: Automatic Knowledge Graph Construction from Scientific Papers*
> **Authors:** Ivan P. Malashin, Igor S. Masich, Vadim S. Tynchenko, Aleksei S. Borodulin, Andrei P. Gantimurov, Vladimir A. Nelyub, Dmitry A. Martysyuk — Bauman Moscow State Technical University
> **Venue:** **WSDM Companion '26**, Feb 22–26 2026, Boise, ID · 10 pages · DOI 10.1145/3779211.3793169
> **License:** CC BY-NC-ND 4.0

---

## 1. TL;DR

**This is the paper for Component 3 (Research Knowledge Layer) and the provenance half of Component 5.** It is the only one in the set that treats *provenance as the primary engineering constraint* rather than an afterthought.

Core thesis: conventional KG construction extracts `(head, relation, tail)` triples and loses everything that makes scientific knowledge trustworthy — the argumentative structure and the evidential grounding. SciGraph-LLM instead builds a **claim–evidence knowledge graph** where **every node and every edge is backed by a verbatim span of the source PDF**.

The mechanism is a hard constraint enforced in the prompt:

> **For every extracted field `x`, `x.evidence` must be recoverable as a verbatim substring of the chunk `C_i` it came from.**

Everything except the paraphrased claim text must be a literal substring of the source. Implicit inference is explicitly *prohibited*. The result: **81.3% evidence precision, 74.6% evidence coverage, 59.9% claim F1 under full structural alignment** (73.7% on text alone) over ten annotated papers.

The paper's honest framing of its own core tension is the most useful thing in it:

> **"The tension between extractive fidelity and scientific expressiveness represents a fundamental challenge."**

Strict grounding buys auditability and costs expressiveness. Our provenance-aware architecture will sit on exactly this trade-off.

---

## 2. Formalism (worth adopting — it is precise and maps onto our docx)

### Document structure
```
D = {s₁, s₂, …, sₙ}                       document as sentences
ρ(sᵢ) ∈ {Abstract, Introduction, Methods, Results, Discussion, Conclusion}    rhetorical section
τ(sᵢ) ∈ {Claim, Evidence, Method, Assumption, Limitation, Background}         rhetorical role
```

Note `τ` — sentences carry a **rhetorical role**, and `Limitation` and `Assumption` are first-class. Our docx's C4 wants exactly this (*"what assumptions existing methods make, what limitations researchers report"*), and this gives it a typed home.

### Claim
```
c = ⟨φ, γ, σ⟩
  φ = natural-language statement of a research conclusion or hypothesis
  γ ⊆ D = supporting spans
  σ ∈ [0,1] = system-estimated confidence
```
Binary claims are relational: `φ : x --r--> y`, where `x, y` are scientific entities (models, variables, datasets) and `r` a scientific relation (`outperforms`, `correlates with`, `reduces`, `causes`).

### Evidence & provenance
```
e = ⟨t, ρ(t)⟩,   t ∈ D, t = a contiguous span of one or more sentences
A : C → 2^E                     claim → evidence alignment function
∀ e ∈ A(c),  e ⊆ D              THE PROVENANCE INVARIANT
```

**Note that evidence carries its rhetorical section `ρ(t)`.** That is exactly our Traceability chain's last hop — *"Recommendation → Reasoning → Evidence → Paper → the relevant section of that paper."*

### The knowledge graph
```
G = (V, E_R)
V = C ∪ E ∪ M          claims ∪ evidence ∪ contextual nodes (methods, datasets, metrics, tasks)
E_R = {(u, r, v)}
R = {supports, refutes, uses, reports, evaluated_on, measured_by}
core relation:  (c, supported_by, e) ∈ E_R
```

**Evidence units are nodes in the graph, not attributes.** This is the structural difference from our current docx sketch, where evidence is a field on a report. Making evidence a node is what allows one span to support several claims, and lets you query "what rests on this sentence?"

### Selective human verification
```
H = {(c, e) : σ(c) < τ_σ  ∨  score(e,c) < τ_s}
score(t,c) = sim(emb(t), emb(c))
```
Only low-confidence pairs go to a human; high-confidence extractions stay autonomous.

---

## 3. Pipeline

```
D_PDF  ──►  T_clean  ──►  J_extract  ──►  G_KG
```

Four stages: **document ingestion → LLM semantic processing → JSON structured representation → graph postprocessing.**

### Stage 1 — Document loading & normalisation
- **Layout-aware text parsing** of the PDF.
- Remove structural noise (headers, footers), **repair hyphenated line breaks**, sentence-segment.
- **Overlapping windows**: `C_i = {s_k, …, s_{k+m}}` with **stride `δ < m`**, satisfying `⋃C_i = T_clean` and `C_i ∩ C_{i+1} ≠ ∅` — so entities near window boundaries remain recoverable.

### Stage 2 — GPT-based semantic processing (the constrained extraction)

Framed deliberately as **deterministic semantic mapping, not generation**: `f_LLM : C_i ↦ J_i`.

> *"Unlike generative prompting strategies, the objective here is not to summarize or reinterpret the content but to perform a deterministic semantic mapping."*

**The extracted claim tuple:**
```
q = (t, e, M, Δ, Θ, R)
  t = controlled paraphrase of the claim preserving logical content
  e = set of VERBATIM evidence snippets quoting the source
  M = methodological descriptors (model variants, training regimes, prompting modes)
  Δ = datasets explicitly mentioned
  Θ = evaluation metrics reported
  R = numerical results {(m, v, c)}, m ∈ Θ, v ∈ ℝ ∪ ℝ² (scalar or range),
      c = contextual clause (dataset split, model variant, training configuration)
```

**The constraint set enforced by the prompt** (this is the reusable core; the paper describes rather than prints the prompt):

| Constraint | Formal statement | Purpose |
|---|---|---|
| **Verbatim grounding** | `∀x ∈ J_i, x.evidence ⊆ C_i` | Auditability |
| **Span-only fields** | `e, M, Δ, Θ ⊆ span(C_i)` | Only `t` may be model-generated |
| **Paraphrase fidelity** | `sim(t, t_orig) > τ ≈ 0.90` (sentence embeddings) | The one generated field is still bounded |
| **Atomicity** | one `q` = one indivisible scientific statement; **emit multiple claim objects if a sentence contains multiple propositions** | Decomposability |
| **No implicit inference** | `∄x ∈ J_i : x is derivable only through external knowledge` | No hallucinated implications |
| **Schema conformance** | `schema(J_i) = schema_target` — field presence, allowed entity types, permissible numeric formats | Machine consistency |
| **No default coercion** | missing fields **omitted or returned as empty sets**, never filled with placeholders | Absence stays visible |

Worked illustration of the no-inference rule: *"the model is not allowed to infer that a dataset is 'medical' or that a model is 'transformer-based' unless the text states so explicitly."*

Three claimed benefits of the verbatim constraint: **(i) extraction becomes auditable, (ii) downstream fact-checking is simplified, (iii) human-in-the-loop validation can verify each element by direct alignment with source text.**

**Example JSON output (Listing 1, verbatim from the paper):**
```json
"claims": [
  {
    "text": "Instruct models without fine-tuning perform poorly on strict triple-level KGC; LoRA adaptation is essential to reach strong performance.",
    "evidence": "The results for the instruction models that have not undergone fine-tuning are significantly lower, compared to their fine-tuned counterparts.",
    "methods": [
      "Zero/few-shot prompting of instruction-tuned LLMs (no parameter updates)",
      "Direct comparison to LoRA-adapted versions under identical prompts",
      "Strict micro-F1 computation on exact triples"
    ],
    "datasets": ["ADE", "CoNLL04", "SciERC"],
    "metrics": ["micro-F1"],
    "numerical_results": [
      { "metric": "micro-F1", "value": 0.333,
        "condition": "Mistral-7B-Instruct-v0.3, ADE, best base setting" },
      { "metric": "micro-F1", "value": 0.592,
        "condition": "Llama-3.1-8B-Instruct, CoNLL04, best fine-tuned natural-language setting" }
    ]
  }
]
```

Note the **`condition` field on every numeric result**. A number without its experimental condition is useless for downstream comparison, and this schema makes the condition mandatory. Our C5 experiment plan's "Expected Results" should carry the same shape.

### Stage 3 — JSON aggregation & NLP processing

Per-chunk results are unioned unmodified: `J = ⋃_i J_i`. Since each claim carries its own evidence spans, **every extraction is traceable back to its source chunk `C_i`**, enabling chunk-level provenance and cross-validation.

Then an NLP phase operating **solely on the JSON, without requerying the source**:

**(a) Entity canonicalisation.**
```
sim(e_p, e_q) = 1 − Lev(e_p, e_q) / max(|e_p|, |e_q|)      Levenshtein-normalised
sim(e_p, e_q) > τ  ⇒  e_p ≡ e_q
ambiguous cases in [τ₁, τ₂] → GPT-5 as semantic disambiguator
```
**A two-threshold design with an LLM adjudicating the grey zone** — cheap string matching for the easy cases, expensive LLM calls only for genuinely ambiguous ones. Directly reusable.

**(b) Normalisation & typing.**
```
N(x) = MWU(lemma(clean(x)))
```
Unicode normalisation, lowercasing, scientific-domain lemmatisation, determiner removal, and **MWU** = multi-word-expression collapsing (e.g. *"Mistral-7B-Instruct-v0.3 fine-tuned with LoRA"* → a single canonical unit).

Typed into `T ∈ {Model, Dataset, Metric, Method, Claim}` by a **hybrid classifier**: rule-based dataset-name patterns + transformer embedding similarity for model families + metric-recognition regexes + heuristic fallbacks for methodological phrases.

**(c) Relation extraction.** A **dependency parser with semantic role labeling** converts claim sentences into directed tuples `R = E(s) = {(h, r, t)}`.

Worked example: *"Fine-tuned lightweight LLMs outperform CodeKGC by 15–20 F1 points on ADE"* →
```
(LLM_light, outperforms, CodeKGC)
(LLM_light, improvement, F1: 15–20)
```

**(d) Numeric normalisation.** Absolute values, deltas, and ranges `(v_min, v_max)` normalised into `⟨m, v, c⟩` (metric, value, condition).

**(e) Claim deduplication** via sentence-embedding cosine similarity **> 0.88**.

### Stage 4 — Graph construction & export

Proto-graph `G = (V, E)`, with each relation stored as a **quadruple**:
```
(h, r, t, e_src)          e_src = the supporting quotation
```

**The admission rule — the paper's key invariant:**
```
(h, r, t) ∈ E  ⟺  e_src ⊨ (h, r, t)
```
An edge exists **iff** its evidence textually entails it. Uncertain cases go to **GPT-5 for a binary "supported / unsupported" verification**.

Export: **GraphML, JSON-LD, Neo4j import sets, CSV triples.**

---

## 4. Implementation Details

| Item | Value |
|---|---|
| Extraction LLM | **GPT-5**, schema-enforced prompts |
| Chunking | Overlapping windows, stride `δ < m`, `C_i ∩ C_{i+1} ≠ ∅` |
| Paraphrase fidelity threshold | `sim(t, t_orig) > τ ≈ **0.90**` (sentence embeddings) |
| Claim-match threshold (eval) | `sim(t_pred, t_gold) > **0.88**` |
| Claim dedup threshold | cosine **> 0.88** |
| Entity canonicalisation | **Levenshtein-normalised similarity**, two thresholds + LLM disambiguation in `[τ₁,τ₂]` |
| Entity types | `{Model, Dataset, Metric, Method, Claim}` |
| Relations | `{supports, refutes, uses, reports, evaluated_on, measured_by}` |
| Relation extraction | Dependency parsing + **semantic role labeling** |
| Export | GraphML / JSON-LD / Neo4j / CSV |
| Evaluation corpus | **10 LLM–KG papers**, manually annotated |
| Gold set | **143 gold claims, 367 fact-level relations** |
| Annotators | 2 with information-extraction background; **Cohen's κ = 0.82** before adjudication |

**Gold-annotation criteria** (a good, tight definition we can reuse for our own eval): a claim is admissible iff
1. it expresses a **verifiable scientific finding involving at least one model, dataset, metric or methodological setup**, **and**
2. it is **explicitly present in the PDF text, such that its supporting span can be quoted verbatim**.

Claims relying on inference, prior knowledge, or implicit interpretation were excluded. Multi-proposition sentences were **split into separate atomic claims**. Candidates were consolidated via **sentence-level embedding clustering** to remove near-duplicates, then disagreements adjudicated.

---

## 5. Results

### RQ1 — Extraction quality

| Matching criterion | Claim F1 |
|---|---|
| Textual assertion only (`sim > 0.88`) | **73.7%** |
| **Full structural alignment** (text **+** entities aligned after canonicalisation) | **59.9%** |

**The 14-point gap between these two numbers is the paper's most instructive result.** Getting the *sentence* right is much easier than getting the *sentence plus its model, dataset, metric and value* right — and only the latter is usable for a knowledge graph.

**Table 1 — four GPT interpretations of one evidence span.** Source: *"Training lightweight LLMs to generate chain-of-thought rationale before triples generally degrades KGC performance after fine-tuning."*

| Type | Generated claim | Sim | Matched |
|---|---|---|---|
| **Correct** | "Chain-of-thought prompting harms triple-level KGC accuracy when applied to small LLMs." | 0.91 | ✅ |
| Partially correct | "Chain-of-thought improves interpretability but does not always improve KGC performance." | 0.84 | ❌ |
| **Incorrect (inverted)** | "Generating explanations before triples **boosts** KGC performance on lightweight models." | 0.59 | ❌ |
| Overly general | "Reasoning strategies often affect KGC quality in unpredictable ways." | 0.51 | ❌ |

> **An *inverted* claim was generated from correctly-retrieved evidence.** This is the single most important warning in the paper for our project: **correct provenance does not guarantee a correct claim.** A recommendation can cite exactly the right sentence and still assert its opposite. Our Provenance Evaluation §4.2 metrics — *citation correctness* ("does the cited paper actually support the claim?") and *provenance consistency* ("does the reasoning actually follow from the cited sources?") — are separate from *coverage* for precisely this reason, and this table is the empirical justification for keeping them separate.

**Two dominant failure categories:** (a) no explicit metric/value present, making the claim incomplete under the matching criterion; (b) the predicted sentence expressed a *general observation* rather than an *experimentally supported finding*. Claims reported only in **tables** were often missed when numbers were abbreviated or split across LaTeX formatting.

### RQ2 — Evidence grounding

| Metric | Value |
|---|---|
| **Evidence precision** (correctly localised spans / all predicted spans) | **81.3%** |
| **Evidence coverage** (gold annotations recovered with ≥1 valid span) | **74.6%** |

**Table 2 — error taxonomy** (this is a ready-made error taxonomy for our provenance evaluation):

| Error type | Example | Share |
|---|---|---|
| **Boundary mismatch** | *"…their fine-tuned counterparts also show improved…"* (gold span excluded the last clause) | **41.2%** |
| **Overly broad span** | evidence includes adjacent unrelated clauses | **33.5%** |
| **Wrong sentence** | mentioned the dataset name but no experimental result | **17.8%** |
| **Unsupported (hallucinated)** | mentioned a metric not present in the paper | **7.5%** |

**Fully incorrect evidence was rare (7.5%); three-quarters of errors are boundary problems.** Encouraging for feasibility — span boundaries are a tractable engineering problem in a way that hallucination is not.

**Successful grounding example:**
> **Claim:** *"The experimental dataset consists of 480 multimodal equipment fault tickets and five Excel sheets with component-level numerical attributes for a bridge crane, from which 4,585 triples and multimodal hypernodes are constructed."*
> **Matched evidence:** *"Our dataset contains 480 multimodal equipment fault tickets and five XLS files with detailed numeric attributes for each bridge crane component, resulting in a total of 4,585 triples and multimodal hypernodes."*

**Failed grounding example — the non-contiguous evidence failure:**
> **Claim (valid):** *"Computer Science Knowledge Graph (CS-KG) is a large scientific KG generated from 6.7M publications, containing about 41M statements represented as 350M triples and about 10M entities typed into five categories and 179 relations."*
> **Evidence (invalid):** two separate, **non-adjacent** sentences concatenated to look like one span.

The structured claim was *correct*; the evidence was *fabricated as a span* by stitching. This violates `∀e ∈ A(c), e ⊆ D` and is called out as the pipeline's main constraint.

### RQ3 — Graph quality

Measured on three dimensions: **strict triple-level F1**, **claim coverage** (proportion of annotated atomic claims appearing as ≥1 triple), and **query accuracy** on templated questions like *"Which model achieves the highest F1 on dataset d?"* and *"Which prompting strategy was compared to LoRA on SciERC?"*.

Qualitative finding: **model–dataset–metric subgraphs emerged naturally**, revealing experimental comparisons across methods and configurations.

> **This is directly our C3's stated goal** — *"noticing that Paper A and Paper B share a dataset and evaluation metric but propose different methods."* SciGraph-LLM shows that structure emerges from the extraction schema rather than needing to be imposed, and gives templated queries as the way to test it.

---

## 6. What to take into our Research Agent

| Our component | What SciGraph-LLM gives us |
|---|---|
| **C2 — Literature Acquisition** | The **PDF→clean-text pipeline** is spelled out and non-trivial: layout-aware parsing, header/footer removal, **hyphenation repair**, sentence segmentation, **overlapping windows with stride**. Also a warning: **tables and LaTeX-split numerics are a systematic failure point** — if our schema wants `experiments` and results, we need a table-aware path (they suggest vision-language models). |
| **C3 — Research Knowledge Layer** | **This is the reference design for our C3.** Adopt: (a) **evidence as graph nodes**, not fields; (b) the **claim tuple `q = (t, e, M, Δ, Θ, R)`**, which is a strict superset of our `methodology / datasets / experiments / claims` fields and adds mandatory evidence and condition-tagged numeric results; (c) **`τ(s)` rhetorical roles** including `Limitation` and `Assumption` as first-class types — our C4 needs exactly these; (d) **`ρ(t)` section labels on evidence**, which completes our Traceability chain's final hop; (e) **atomicity** — one claim per proposition, split compound sentences; (f) the **two-threshold canonicalisation with LLM disambiguation in the grey zone**; (g) **no default coercion** — absent fields stay absent, so we never mistake a placeholder for a finding. |
| **C4 — Gap / Novelty Detection** | Typed `Limitation` and `Assumption` claims with verbatim evidence make our docx's *"what limitations researchers report, what assumptions existing methods make"* mechanically queryable rather than aspirational. And **`refutes` is in their relation set** — contradictions between papers become explicit edges, which is precisely GAPMAP's "conflicting findings presented without reconciliation" implicit-gap type, made computable across papers. |
| **C5 — Provenance-Aware Experiment Planning** | The closest thing in the literature to what we want. **The admission rule `(h,r,t) ∈ E ⟺ e_src ⊨ (h,r,t)`** is exactly the invariant our provenance graph needs: *no element enters the plan unless its evidence entails it.* Their `(h, r, t, e_src)` quadruple is our provenance edge. Their **selective-verification rule `H = {(c,e) : σ(c) < τ_σ ∨ score(e,c) < τ_s}`** gives us a principled way to decide when a human must check a plan element. And the mandatory **`condition`** on every numeric value is what makes an "Expected Results" field meaningful. |
| **C-Eval — Provenance Evaluation (§4.2)** | Their metrics map almost 1:1 onto ours and give us **concrete numbers to beat**: *evidence precision* ≈ our **citation correctness** (81.3%); *evidence coverage* ≈ our **citation completeness** (74.6%); *claim F1 under full structural alignment* ≈ our **provenance consistency** (59.9%). Their **four-category error taxonomy** (boundary mismatch / overly broad / wrong sentence / hallucinated) is a ready-made annotation scheme. Their **templated graph queries** are how to test Traceability. And **κ = 0.82** inter-annotator agreement sets the bar for our own annotation quality. |

**Cheap wins to steal first:** (1) the verbatim-substring constraint on every non-paraphrase field; (2) atomicity — split compound sentences into separate claims; (3) `condition` mandatory on every numeric result; (4) evidence as nodes with section labels; (5) the entailment admission rule for graph edges; (6) the four-way evidence-error taxonomy for evaluation; (7) no-default-coercion so missing data stays visibly missing.

---

## 7. Gaps, Limitations & Future Scope

### Stated by the authors (§6.1)
1. **Evidence fragmentation is the main constraint.** Despite overlapping windows, **33.5% of spans were overly broad** and **7.5% hallucinated**. The pipeline **cannot handle non-contiguous evidence** — demonstrated by the CS-KG case where GPT concatenated two separate sentences, violating `∀e ∈ A(c), e ⊆ D`.
2. **PDF formatting breaks numeric extraction.** Table-based results and split LaTeX expressions cause systematic failures, *"particularly when value–condition alignment is required."*
3. **Structural rigidity limits expressiveness.** Full structural alignment scores only 59.9% vs 73.7% on text alone, because *"scientific discourse frequently embeds multiple propositions within single sentences"* — e.g. *"LoRA fine-tuning improves performance but increases inference latency by 15%."*
4. **The no-inference prohibition excludes valid scientific implications**, and the model still produced both correct and *inverted* interpretations from identical evidence.
5. **Three GPT-5 dependency bottlenecks:** proprietary-model dependency limits reproducibility; context limits force aggressive chunking that **disrupts discourse flow across rhetorical sections**; evidence-constraint templates need **re-engineering per domain** with different argumentative structures.
6. **Levenshtein canonicalisation fails on coreferential entities with divergent surface forms** — e.g. *"BERT"* vs *"Bidirectional Encoder Representations from Transformers"*.
7. **Domain-bound** — evaluated on ten LLM–KG papers only.

### Authors' proposed future work (§6.2) — several are worth adopting directly
1. **Rhetorically-aware segmentation** replacing fixed windowing: use section headers `ρ(sᵢ)` and discourse markers to make **variable-length segments preserving logical units**.
2. **Table-specific extraction modules using vision-language models** to recover structured results currently lost in PDF normalisation.
3. **Hybrid extraction architectures** — fine-tune open-weight models (e.g. Mistral-7B) with explicit **evidence-grounding objectives** for comparable precision at lower cost.
4. **Verifiable inference** — the most interesting proposal: *"rather than prohibiting all implicit reasoning, they could flag derived claims with explicit provenance trails like `(c, derived_from, {e₁, e₂})`."* **This is the resolution of the fidelity/expressiveness tension, and it is exactly our "derivation provenance" type from docx §3.5.**
5. **Dynamic schema evolution** — meta-learning to adapt claim structures to domain conventions rather than a static `schema_target`; semi-supervised relation extraction guided by domain ontologies to expand `R` while preserving grounding.
6. **Impact-weighted human verification** — currently verification triggers only on low confidence; instead **flag claims with high centrality in `G_KG`, or claims containing critical numerical results.**

### Gaps we can attack (our project's opening)
1. **Single-paper only.** Every graph is a *document-level* graph. There is **no cross-paper merging, no shared entity space across papers, no comparison between papers**. Our C3 explicitly wants the cross-paper view (*"Paper A and Paper B share a dataset and metric but propose different methods"*), and the fact that their model–dataset–metric subgraphs emerge naturally *within* one paper is strong evidence it will work *across* papers. **Cross-paper canonicalisation and contradiction detection is the obvious extension, and they have not done it.**
2. **No retrieval, no ideation, no gap detection.** SciGraph-LLM is purely an extraction pipeline. It never searches for papers, never proposes anything, never assesses novelty. It is a **component for our C3**, not a competitor — which makes it unusually easy to adopt.
3. **`refutes` is in the relation vocabulary but never exercised.** Contradiction detection between claims is declared and unused. Cross-paper contradiction is both GAPMAP's third implicit-gap type and our C4's "conflicting findings" case.
4. **Confidence `σ` is declared but never calibrated or evaluated.** The selective-verification rule depends entirely on `σ(c) < τ_σ`, yet no calibration numbers are reported anywhere. (GAPMAP found LLM self-confidence miscalibrated 10–24% of the time — assume the same here.)
5. **Only 10 papers, one narrow domain (LLM–KG research), 143 claims.** Very thin evidence for a pipeline claiming generality; the authors admit domain dependence.
6. **The paraphrase is the weak link.** `t` is the only generated field, bounded only by `sim(t, t_orig) > 0.90` — and Table 1 shows an *inverted* claim scoring 0.59 while a *partially correct* one scored 0.84. **Embedding similarity does not detect negation or polarity inversion.** Any system of ours that paraphrases a claim needs an explicit **entailment/polarity check**, not just a similarity threshold.
7. **No cost, latency, or throughput numbers** — for a GPT-5 pipeline over overlapping windows of full PDFs, this is a significant omission and a real risk for our own scaling.
8. **Human-in-the-loop is formalised but never actually evaluated.** `H` is defined; no study measures whether it helps, how much expert time it costs, or what it recovers.
