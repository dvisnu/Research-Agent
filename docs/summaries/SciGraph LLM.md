# SciGraph-LLM — Summary

> *SciGraph-LLM: Automatic Knowledge Graph Construction from Scientific Papers* — Malashin, Masich, Tynchenko, Borodulin, Gantimurov, Nelyub, Martysyuk (Bauman Moscow State Technical Univ.) · WSDM Companion '26

## Summary

The reference design for Component 3, and the only paper in the set that treats **provenance as the primary engineering constraint** rather than an afterthought.

**Core thesis:** conventional KG construction extracts `(head, relation, tail)` triples and loses everything that makes scientific knowledge trustworthy — argumentative structure and evidential grounding. SciGraph-LLM instead builds a **claim–evidence knowledge graph where every node and edge is backed by a verbatim span of the source PDF.**

**Formalism** (precise, and it maps onto our design):

- Sentences carry a **rhetorical section** `ρ(s) ∈ {Abstract, Introduction, Methods, Results, Discussion, Conclusion}` and a **rhetorical role** `τ(s) ∈ {Claim, Evidence, Method, Assumption, Limitation, Background}` — `Limitation` and `Assumption` are first-class types, which is exactly what our C4 needs.
- A claim is `c = ⟨φ, γ, σ⟩` — statement, supporting spans, confidence — with binary claims relational (`x --outperforms--> y`).
- Evidence carries its section label, and the **provenance invariant** is `∀e ∈ A(c), e ⊆ D`: every evidence span must be a literal substring of the document.
- The graph is `V = claims ∪ evidence ∪ contextual nodes`, with **evidence as nodes, not attributes** — which is what lets one span support several claims and lets you ask "what rests on this sentence?"
- Selective human verification: `H = {(c,e) : σ(c) < τ_σ ∨ score(e,c) < τ_s}` — only low-confidence pairs go to a human.

**Pipeline.** Layout-aware PDF parsing → header/footer removal, **hyphenation repair**, sentence segmentation, **overlapping windows with stride** → constrained LLM extraction → JSON aggregation → graph.

The extraction is framed as **deterministic semantic mapping, not generation**, and enforces a constraint set that is the reusable core: **verbatim grounding** (every evidence field must be a substring of its chunk), **span-only fields** (only the paraphrased claim text may be generated, and it must satisfy `sim > ≈0.90`), **atomicity** (one claim per proposition; split compound sentences), **no implicit inference** (the model may not infer that a dataset is "medical" unless the text says so), schema conformance, and **no default coercion** — missing fields stay empty rather than being filled with placeholders, so absence stays visible. The claim tuple is `q = (t, e, M, Δ, Θ, R)` — text, verbatim evidence, methods, datasets, metrics, and **numerical results with a mandatory `condition` field** (a number without its experimental condition is useless downstream).

Post-processing works on the JSON alone: entity canonicalisation by normalised Levenshtein with **two thresholds and an LLM adjudicating only the grey zone**; normalisation and typing into `{Model, Dataset, Metric, Method, Claim}`; relation extraction by dependency parsing + semantic role labeling; claim dedup at cosine > 0.88. Every relation is stored as a **quadruple `(h, r, t, e_src)`**, and the admission rule is the paper's key invariant: **an edge exists iff its evidence textually entails it.**

**Results** (10 annotated LLM–KG papers, 143 gold claims, 367 relations, annotator κ = 0.82):

- Claim F1 **73.7%** on textual assertion alone, **59.9%** under full structural alignment (text + entities). **The 14-point gap is the instructive result:** getting the sentence right is much easier than getting the sentence plus its model, dataset, metric and value — and only the latter is usable in a graph.
- Evidence **precision 81.3%**, **coverage 74.6%**. Error taxonomy: boundary mismatch **41.2%**, overly broad span **33.5%**, wrong sentence 17.8%, hallucinated **7.5%**. Three-quarters of errors are boundary problems — a tractable engineering issue, unlike hallucination.
- Model–dataset–metric subgraphs **emerged naturally**, revealing experimental comparisons across methods — exactly our C3's stated goal, tested via templated queries ("which model achieves the highest F1 on dataset d?").

**The single most important warning in the paper:** from one correctly-retrieved evidence span, GPT produced four claims — one correct (sim 0.91), one partially correct (0.84), one **inverted** ("boosts" instead of "degrades", sim 0.59), and one overly general (0.51). **Correct provenance does not guarantee a correct claim**, and **embedding similarity cannot detect polarity inversion.** This is the empirical justification for keeping our *citation correctness* and *provenance consistency* metrics separate from *coverage*.

The paper's honest framing of its own tension: *"the tension between extractive fidelity and scientific expressiveness represents a fundamental challenge."* Strict grounding buys auditability and costs expressiveness.

## Gaps

**Stated by the authors**

- **Evidence fragmentation is the main constraint** — the pipeline **cannot handle non-contiguous evidence**, demonstrated by a case where GPT stitched two non-adjacent sentences into a fake single span.
- **PDF formatting breaks numeric extraction** — table results and split LaTeX cause systematic failures, especially where value–condition alignment is required.
- **Structural rigidity limits expressiveness** (59.9% vs 73.7%), because scientific sentences routinely embed multiple propositions. The no-inference prohibition also excludes valid scientific implications.
- GPT-5 dependency limits reproducibility; context limits force chunking that disrupts discourse flow; evidence templates need re-engineering per domain. Levenshtein canonicalisation fails on coreferential entities with divergent surface forms ("BERT" vs the full expansion). Evaluated on ten papers in one narrow domain.

**Their proposed future work, worth adopting**: rhetorically-aware variable-length segmentation; table extraction via vision-language models; **verifiable inference** — rather than prohibiting implicit reasoning, flag derived claims with explicit trails like `(c, derived_from, {e₁, e₂})`, which is exactly our *derivation provenance* type; and **impact-weighted human verification** (flag high-centrality or numerically critical claims, not just low-confidence ones).

**Open for us**

- **Single-paper only.** Every graph is document-level: no cross-paper merging, no shared entity space, no comparison between papers. **Cross-paper canonicalisation and contradiction detection is the obvious extension and they have not done it** — and the fact that model–dataset–metric subgraphs emerge *within* a paper is good evidence it will work *across* papers.
- **No retrieval, no ideation, no gap detection** — it is a component for our C3, not a competitor, which makes it unusually easy to adopt.
- **`refutes` is in the relation vocabulary but never exercised** — cross-paper contradiction is declared and unused.
- **Confidence `σ` is declared but never calibrated**, though the entire selective-verification rule depends on it.
- **The paraphrase is the weak link** — bounded only by a similarity threshold, and their own Table 1 shows an inverted claim scoring 0.59 while a partially-correct one scored 0.84. Any paraphrase step of ours needs an explicit **entailment/polarity check**, not a similarity threshold.
- No cost, latency or throughput numbers for a GPT-5 pipeline over full PDFs. Human-in-the-loop is formalised but never evaluated.

## What We Can Use

| Component | What SciGraph-LLM gives us |
|---|---|
| **C2 — Literature Acquisition** | The PDF→clean-text pipeline spelled out: layout-aware parsing, header/footer removal, **hyphenation repair**, sentence segmentation, **overlapping windows with stride**. Plus a warning — **tables and LaTeX-split numerics are a systematic failure point**, so if our schema wants experiments and results we need a table-aware path. |
| **C3 — Research Knowledge Layer** | **The reference design.** Adopt: (a) **evidence as graph nodes**; (b) the claim tuple `q = (t, e, M, Δ, Θ, R)`, a strict superset of our fields adding mandatory evidence and condition-tagged numerics; (c) **rhetorical roles** with `Limitation` and `Assumption` first-class; (d) **section labels on evidence**, completing our traceability chain's last hop; (e) **atomicity**; (f) two-threshold canonicalisation with LLM adjudication in the grey zone; (g) **no default coercion**. |
| **C4 — Gap / Novelty Detection** | Typed `Limitation` and `Assumption` claims with verbatim evidence make "what limitations researchers report, what assumptions methods make" mechanically queryable rather than aspirational. `refutes` makes cross-paper contradiction an explicit edge — GAPMAP's third implicit-gap type, made computable. |
| **C5 — Provenance-Aware Experiment Planning** | **The admission rule `(h,r,t) ∈ E ⟺ e_src ⊨ (h,r,t)`** is exactly the invariant our provenance graph needs: *no element enters the plan unless its evidence entails it.* Their `(h, r, t, e_src)` quadruple is our provenance edge; their selective-verification rule tells us when a human must check a plan element; and the mandatory `condition` on numerics is what makes an "Expected Results" field meaningful. |
| **Provenance Evaluation** | Their metrics map onto ours with **concrete numbers to beat**: evidence precision ≈ citation correctness (**81.3%**), evidence coverage ≈ citation completeness (**74.6%**), claim F1 under structural alignment ≈ provenance consistency (**59.9%**). The four-category error taxonomy is a ready-made annotation scheme; templated graph queries are how to test traceability; **κ = 0.82** sets the annotation bar. |

**Steal first:** (1) the verbatim-substring constraint on every non-paraphrase field; (2) atomicity; (3) mandatory `condition` on every numeric result; (4) evidence as nodes with section labels; (5) the entailment admission rule; (6) the four-way evidence-error taxonomy; (7) no default coercion.
