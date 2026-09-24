# ResearchAgent — Summary

> *ResearchAgent: Iterative Research Idea Generation over Scientific Literature with LLMs* — Baek (KAIST), Jauhar & Cucerzan (MSR), Hwang · arXiv 2404.07738v2 (Feb 2025) · [code](https://github.com/JinheonBaek/ResearchAgent)
> Prior work that Chain-of-Ideas uses as a baseline. Read the two as **entity-centric breadth (this) vs. temporal-chain depth (CoI)**.

## Summary

**Formalisation** (clean, worth copying): an idea `o = [p, m, d]` — problem, method, experiment design — generated submodularly, `p = f(L)`, `m = f(p, L)`, `d = f(p, m, L)`, each instantiated as an LLM call over the core paper, its citation neighbours, and retrieved entities. Framing premise: *"a new idea is more often than not just a new combination of old elements."*

**Three knowledge sources modelled on how researchers work:**

1. **Citation-graph survey (depth).** Anchor on a high-impact core paper (>20 citations in the experiments), pull its citing/cited neighbours (~87 on average), narrow by abstract similarity. Limitation the authors name themselves: this confines the system to one local neighbourhood.
2. **Entity-centric knowledge store (breadth — the signature contribution).** BLINK entity-links titles+abstracts of **50,091 papers** (May–Dec 2023 + references) into a sparse `m×m` co-occurrence matrix. Retrieval deliberately returns the top-k entities **absent from** the current context, via a Bayes-factorised co-occurrence score — the opposite objective of prior entity-based query suggestion. This is the cross-domain cross-pollination channel. An embedding-based variant performed about the same, so pick on engineering cost.
3. **15 ReviewingAgents (peer review).** 3 artefacts × 5 criteria, each emitting `Review / Feedback / Rating (1-5)`; the system regenerates from the reviews. Improvement **saturates after ~3 iterations**.

**Human-preference-aligned criteria induction** (the cheap, clever trick): take 10 human-scored idea pairs, prompt GPT-4 to *induce* prose rubrics for each 1–5 level, then use those induced rubrics inside both the ReviewingAgents and the automatic evaluation. This visibly pulls the model's score distribution toward the human one. The 15 criteria: Problem (clarity, relevance, originality, feasibility, significance), Method (clarity, validity, rigorousness, innovativeness, generalizability), Experiment (clarity, validity, robustness, feasibility, reproducibility).

**Results.** The full system beats both ablations on all 15 criteria, with the largest gains on *originality* and *innovativeness* — exactly the entity store's purpose. Knowledge-source ablation (model scores, Problem/Method/Experiment): full **4.52 / 4.28 / 4.18**; w/o entities 4.35 / 4.13 / 4.02; **w/ random entities 4.41 / 4.19 / 4.13**; w/o references 4.26 / 4.08 / 3.97; w/o both 4.20 / 4.03 / 3.92. Two readings: **references matter more than entities**, and **random context beats no context** — so an ablation must compare against random-context, not no-context, or the improvement is overstated. Agreement: human–human Spearman 0.83 / 0.76 / **0.67**, pairwise κ 0.62 / 0.62 / **0.41** — **experiment designs are the hardest thing to evaluate reliably**. The knowledge-augmentation gain is **model-gated**: large for GPT-4, Llama3-8B and Qwen1.5-32B; nil for GPT-3.5 and Mixtral-8x7B. Higher-cited core papers yield higher-quality ideas, justifying the impact gate.

## Gaps

**Stated by the authors**

- Entity store built from **titles + abstracts only**, over a limited window (May–Dec 2023), so most entities and their links are never captured; BLINK yields only ~3 entities/paper and is an open-domain linker, so published entity lists contain visible noise ("LMS color space", "Adhesive" for NLP papers).
- **Hallucination remains**: knowledge augmentation only partially grounds generation, and the authors state that experimental validation of generated ideas is essential and absent.
- Review scope is a fixed 15 criteria chosen by presumed importance; poor fit for theoretical sciences where proof, not experiment design, is the deliverable; the store must be continually expanded and kept current.
- Ethics statement flags misuse and **unintentional plagiarism** from training-data regurgitation.

**Open for us**

- **No provenance whatsoever.** Ideas are conditioned on ~87 abstracts and a bag of entities; nothing links a claim back to a supporting paper. No citation correctness, completeness or traceability. The clearest open space.
- **Abstract-only grounding** — the system never reads methods, results or limitations sections, so it cannot know what a method actually requires or whether a proposed baseline is runnable.
- **Novelty is never checked at all.** Unlike CoI, nothing retrieves prior work to test whether the generated problem already exists; "originality" is only a reviewer opinion score.
- Entity retrieval assumes independence over raw counts — **no typing** (dataset? metric? method?), no ontology linking, no noise filtering.
- **Requires a paper as the entry point, not a research question** — our C1 takes natural language, so a problem→anchor-paper resolution step is needed (CoI's multi-query retrieval supplies it).
- The review loop cannot **seek new information** when a critique exposes a knowledge gap (contrast CoI's critique→query→retrieve step). Combining the two is an obvious unclaimed improvement.
- Score ceiling effects (everything clusters 3.9–4.5; the prompt has to beg the model not to rate uniformly high) — pairwise/ELO is the better headline metric. No cost, latency or scaling numbers anywhere.

## What We Can Use

| Component | What ResearchAgent gives us |
|---|---|
| **C2 — Literature Acquisition** | Defensible heuristics: **impact-gated core paper** (citation threshold, empirically justified by the citation↔quality correlation), abstract-similarity narrowing of the reference set, and a **post-LLM-cutoff paper window** so the model can't regurgitate a memorised answer. |
| **C3 — Research Knowledge Layer** | The **entity-centric co-occurrence store** is the piece our design is missing. Our graph is intra-corpus; theirs is a global store whose purpose is surfacing entities *absent from* the current context. These compose: keep our typed per-paper graph, add a global entity layer as a cross-domain suggestion channel. |
| **C4 — Gap / Novelty Detection** | The `problem → method → experiment` decomposition with per-stage rationale is a clean skeleton for the Novel Evidence Report. Their **originality rubric (5 induced prose levels)** is structurally close to our 5-level novelty ladder and can be adapted rather than written from scratch. |
| **C5 — Experiment Planning** | The five experiment axes (clear, robust, reproducible, valid, feasible) map onto our Risks/Limitations and Ablation fields, and their `Experiment: / Rationale:` output format makes the **rationale field the natural place to hang derivation provenance**. |
| **Evaluation** | (a) **Human-induced rubrics** from ~10 scored pairs before trusting any LLM-judge number; (b) **random-context ablations**, not just no-context; (c) report agreement (Spearman for scores, κ for pairwise); (d) budget extra annotation for experiment designs, the noisiest axis; (e) reusable logistics — Label Studio, written guideline, domain experts judging ideas built from their own papers, ~9 ideas/hour. |

**Combining with CoI:** the two mechanisms are orthogonal and nobody has combined them — a CoI-style temporal chain, each step augmented with ResearchAgent-style out-of-context entities, with provenance attached to every element, is a defensible position for this project.
