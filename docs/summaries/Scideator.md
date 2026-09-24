# Scideator — Summary

> *Scideator: Human-LLM Compound System for Scientific Ideation through Facet Recombination and Novelty Evaluation* — Radensky\*, Shahid\*, Fok, Siangliulue, Hope†, Weld† (UW / Microsoft / AI2) · ACM CAIS '26 · arXiv 2409.14634v7 (May 2026)
> Its **Module 3 is the Idea Novelty Checker** from *Literature-Grounded Novelty Assessment*. This file covers the ideation half.

## Summary

**Central design commitment:** maintain **one structured representation — facets — across retrieval, generation, novelty evaluation and suggestion.** Every module speaks facets; every user action *is* a facet operation. The payoff is that user intent arrives as a structured signal rather than free text the LLM must reinterpret at each stage, and the same vocabulary describes papers and ideas so the user can trace which facet produced which idea.

**Facets:** *purpose* (problem addressed), *mechanism* (proposed solution), *evaluation* (how you'd know it worked — Scideator's addition to the prior purpose/mechanism analogy literature). Extraction constraints are unusually well specified and directly reusable: **≤7-word phrases**, specific enough to inspire but not tied to one paper, **no numbers or acronyms**, multiple facets of a type combined into one, **the evaluation facet must not reference the purpose**, and each facet carries a 1–2 sentence definition that **must not reuse its own words**.

**Module 1 — Analogous Paper Facet Finder.** Not just relevant papers: **distance-controlled retrieval** at four conceptual distances — *very near* (Semantic Scholar similarity endpoint), *near* (same topic, different approach), *far* (same subarea, different topic), *very far* (different subarea, connected only by high-level analogy). 4 pairs per distance = 12, plus 4 very-near = 16 papers. Each generated (purpose, mechanism) pair is **grounded in literature** via a ≤5-word search query, iteratively shortened if fewer than 4 results return. The key generation constraint: **"P is to M as P′ is to M′ because both involve…"** — the structural relationship must be preserved, which is what makes it an analogy rather than a random facet pair.

**Module 2 — Faceted Idea Generator.** A shared 5-step chain: ground in prior work → deduplicate against previous ideas → brainstorm N analogies → select the K best → **elaborate with self-critique** (initial idea → self-identified issues → plan to address them → revised idea). Five quality requirement categories with in-prompt negative examples: understandability, relevance, specificity, **feasibility ("achievable by a lab with moderate resources" — encoded as a generation-time constraint, not a post-hoc review)**, novelty. Ideas use *mirrored* facet combinations (A.purpose+B.mechanism and the reverse), with a distance-mixing constraint so purpose and mechanism come from papers at different distances. Temperature raised to 0.75 for variety.

**Module 3 — Idea Novelty Checker.** Steps 1–3 as in the LGNA paper (three retrieval channels → SPECTER top-100 → facet-based RankGPT top-10 → classify with expert in-context examples). Two Scideator-specific additions: the prompt is a **simulated multi-turn dialogue injecting each retrieved paper as a separate user message**, so the model attends to each one individually (a direct fix for AI Scientist's 100-papers-in-one-prompt failure); and **Step 4 — facet-swap suggestions**: when an idea is judged not novel, generate three alternatives each swapping **exactly one facet**, with justifications of novelty and usefulness. *"By constraining edits to a single facet swap, suggestions remain close to the user's original intent while systematically exploring the facet space for novelty."* This is the loop-closer CoI and ResearchAgent both lack.

**Evaluation.** User study N=22 CS researchers, within-subjects, 20-minute sessions, same backbone LLM in both conditions so the comparison isolates structured faceted interaction vs. free-text prompting. **CSI median 70.5 vs 61.0 (p<.01)**, gains concentrated in *exploration* and *expressiveness*. The most telling qualitative result: in the baseline, 6 of 8 participants who found new concepts attributed them to the **input papers**; with Scideator, **6 of 6 credited the tool's facets or generated ideas**. Participants used custom text instructions for a median of 0 of their 2 favourite ideas (vs 1.5 in baseline) — *"these [facets] served like custom instructions."* Two ablations, one varying retrieval with the classifier fixed (89.66% → 13.79% when facet reranking is swapped for generic relevance) and one varying the classifier with the papers fixed (expert ICL with reasoning 0.81 vs zero-shot 0.68 vs AI Scientist 0.47). Together they show **the structured representation independently improves both retrieval and classification** — the strongest evidence in the set that a knowledge layer is load-bearing rather than decorative.

**The most transferable finding — the verifiability asymmetry:** a **"not novel"** verdict is verifiable (here are the overlapping papers, go look); a **"novel"** verdict is not. Participants trusted the first and discounted the second, and one explicitly distrusted generated rationales — *"reasons can be kind of made up… I relied more on the references that it retrieved."*

## Gaps

**Stated by the authors**

- **Retrieval coverage too narrow** — only 10 papers per novelty check, limited by latency; participants complained important works were missed and that ideas were called novel too often. Their fix: adaptive or broader retrieval.
- **Users avoided distant facets** because their relevance wasn't apparent; proposed remedies are in-situ QA about unfamiliar facets or surfacing the motivating analogy.
- Some participants preferred paper-level interaction ("a map of literature") — tools should offer both granularities. The pipeline is fixed-order; modules could act proactively.
- The novelty checker **did not significantly improve overall confidence** (it only shifted judgments on "not novel"). Distance labels are subjective.
- Risks acknowledged: researchers may offload thinking or over-rely on generated facets and verdicts; unconstrained LLM generation is known to reduce idea diversity.

**Open for us**

- **Title + abstract only, everywhere** — facet extraction, analogy generation and novelty assessment all read only titles and abstracts, so a method buried in §4 is invisible to the entire system. Our C2 schema reads deeper, and CoI showed full-paper reading is affordable with a cheap model. **Our clearest technical advantage over this line of work.**
- **No provenance beyond a source link** — no claim→span grounding, no derivation trace, nothing verifying that a cited paper supports the asserted overlap. Their own users named this trust deficit.
- **No experiment plan.** Scideator produces a 100–250 word idea and stops; the evaluation *facet* is a ≤7-word phrase, not a plan. Our C5 is entirely unoccupied by this system.
- **Binary novelty with no gradient** — facet-swap suggestions imply a gradient exists but never expose it.
- **No retrieval-confidence signal**, despite the ablation proving the verdict flips on 2–3 marginal papers.
- **No cross-paper landscape** — papers are retrieved and faceted independently; nothing models shared datasets, contradictions or temporal progression between them.
- Human-in-the-loop by design ("the system proposes, the user decides"). Our architecture is more autonomous, so **we inherit the burden their design offloads to the human** — which is why provenance and calibrated levels matter more for us. CS-only, N=22, 20-minute sessions, no cost/latency numbers.

## What We Can Use

| Component | What Scideator gives us |
|---|---|
| **C1 — Research Input** | User-supplied ideas are **decomposed into facets and added to the pool**, so user contributions enter the same representation as everything else. Our "optionally upload papers" should do the same rather than treating uploads as a separate channel. |
| **C2 — Literature Acquisition** | **Distance-controlled retrieval** (very near / near / far / very far) is the mechanism that actually surfaces "where different research areas haven't been combined." Plus the ≤5-word query rule and the query-shortening fallback when fewer than 4 results return. |
| **C3 — Research Knowledge Layer** | The strongest lesson here: **one representation persisting across every module** beats per-stage ad-hoc structures, proven by two independent ablations. Add a **facet layer on top of our per-paper record** — keep concrete fields for provenance, add facets for recombination and comparison — and copy their extraction constraints verbatim. |
| **C4 — Gap / Novelty Detection** | (a) facet-priority reranking and the facet-based novelty rule; (b) the **one-paper-per-message** prompt structure; (c) **facet-swap suggestions** — when we land at novelty level 1–2, don't just report the bad news, emit minimal single-facet edits that would move the idea up the ladder. That converts our ladder from a verdict into a **navigable gradient**. |
| **C5 — Experiment Planning** | The **evaluation facet** is the abstract form of our Baselines/Metrics/Conditions fields, so retrieving by evaluation-facet similarity is a principled way to make methodological choices provenance-backed. And since CoI and GAPMAP both found feasibility to be the failure axis, **encoding feasibility as a generation-time constraint** is probably the right move. |
| **Evaluation** | CSI as a validated instrument; **within-subjects, same backbone, randomised order** to isolate the architectural contribution (the design for our Baseline 3 vs 4 comparison); **two separate ablations** (vary retrieval / hold classifier, and the reverse) to attribute credit in a compound system; ablate on "not novel" only. |

**Steal first:** (1) the facet layer with its extraction constraints; (2) distance-controlled retrieval; (3) the "P is to M as P′ is to M′" analogy template; (4) facet-swap suggestions on a low-novelty verdict; (5) one-paper-per-message prompting; (6) feasibility as a generation-time constraint. And **lead the output with the evidence, treat the rationale as secondary** — users trust retrieved references over generated prose.
