# Literature-Grounded Novelty Assessment — Summary

> *Literature-Grounded Novelty Assessment of Scientific Ideas* — Shahid (Microsoft), Radensky, Fok (UW), Siangliulue, Weld, Hope (AI2) · arXiv 2506.22026v1 (Jun 2025) · system: **Idea Novelty Checker** · [code](https://github.com/simra-shahid/idea_novelty_checker)
> Same group as **Scideator**; this is Scideator's Module 3 studied in depth. Read the two together.

## Summary

The most directly relevant paper in the set for Component 4. It attacks precisely the risk our plan names: *searching 50 papers and finding nothing does not mean the idea is novel — the search may simply have failed.* Their answer: **novelty assessment is a retrieval problem first and a judgement problem second.**

**Formative study (why the design looks like this).** Two expert annotators judged 51 ideas. Round 1, three categories (novel / moderately novel / not novel): **Cohen's κ = 0.64**, with disagreement traced to annotators falling back on background knowledge because the retrieved papers were insufficient. Round 2 — judge **solely on the provided papers**, collapse to **binary** — **κ = 0.68**. Two challenges crystallised: high-quality retrieval, and a fixed criterion for what counts as novel.

**Their operational novelty rule** (conjunctive over papers, disjunctive over facets): an idea is novel if it **differs from all retrieved papers in ≥1 core facet** — *purpose* (objective), *mechanism* (technical approach), *evaluation* (validation method) — **or uniquely combines** those facets, **or applies them to a new application domain**.

**Pipeline.**

1. **Broad candidate retrieval, three parallel channels** — LLM-generated keyword *and candidate-title* queries → Semantic Scholar Search; **whole-idea snippet search** (the whole idea as the query, catching papers with no keyword overlap); seed-paper → Recommendations API.
2. **Two-stage reranking** — SPECTER-2 cosine → top **N = 100**, then **facet-based RankGPT** → top **k = 10**. RankGPT's generic relevance criterion is replaced by an explicit facet-priority ladder (all facets match > domain+purpose > any single facet > partial). *This substitution is the highest-value idea in the paper.*
3. **Novelty judgement** — an LLM classifies novel / not-novel with literature-grounded reasoning, steered by **15 expert-labeled in-context examples** whose format is `Idea → top-10 papers → expert reasoning naming the closest paper by index and the differing facet`.

**Results.** Expert examples with reasoning: **acc 0.81, F1 0.79, κ 0.59** — beating TextGRAD (0.78), DSPy (0.68), zero-shot (0.68), AI Researcher/GPT-4o (0.78) and AI Scientist (**0.47, κ = 0.05**, essentially chance). Reasoning in the examples adds ~3 points over labels alone. OpenReview reviews are a poor substitute for expert labels (0.59) because real reviews don't reference the compared papers. Same prompt, different model swings hugely (AI Researcher: 0.78 GPT-4o vs 0.56 Claude-3.5-Sonnet). Prompt sensitivity is unexplained and severe — two near-identically-worded prompts scored 0 and 0.6.

**The ablation is the striking part** (accuracy at detecting "not novel"): complete system **89.66%**; swapping only the reranking criterion to generic relevance → **13.79%**; embedding-only → 10.34%; snippet-only → 8.62%; keyword-only → 5.17%. The overlap analysis explains why: relevance-RankGPT still retrieves **~8 of the same 10 papers**, yet accuracy falls 76 points. **The 2–3 papers at the margin decide the verdict.** They also deliberately ablate on **"not novel" cases only**, because a "novel" label isn't stable across retrieval variants.

**Their critique of AI Scientist** is worth internalising: keyword-only retrieval; >100 papers dumped into one prompt so instructions get lost; a decision made by string-matching `"decision made: novel"`, and **if no decision is reached the idea is automatically counted novel**. Failing *open* toward novel is exactly our stated failure mode, implemented as a literal code path.

## Gaps

**Stated by the authors**

- **Only top-10 papers are considered**, forced by context limits, and those 10 "may disproportionately influence" the verdict.
- **Annotator circularity** — the same two people wrote the in-context examples and labeled the test set, so the +13% over AI Researcher is likely an upper bound. **Idea-source circularity** too: most test ideas came from Scideator.
- **Binary classification is too coarse** — their stated future work is treating novelty as a **continuum**.
- Unexplained prompt sensitivity; tiny dataset (67 labeled examples, 32 test, 2 annotators).

**Open for us**

- **No retrieval-confidence signal.** The system never reports how well it searched, though its own overlap table proves the verdict flips on 2–3 papers. **A calibrated coverage/confidence estimate attached to every novelty claim is unclaimed space, and it is the whole justification for "potential novelty, never proven novelty."**
- **Title + abstract only** — a paper whose *method section* contains the overlapping idea is invisible.
- **No provenance beyond bracketed indices** — no span grounding, no claim→section link, nothing verifying the cited paper actually supports the asserted similarity. Citation correctness and provenance consistency are entirely unmeasured.
- **It judges ideas, it doesn't generate them, and it does no gap reasoning** — it never says what is *missing* or what *would make* the idea novel. GAPMAP infers gaps but never checks whether they're closed. **Nobody has combined the two; that combination is our C4.**
- **Static and one-shot** — a "not novel" verdict triggers no refinement loop. Expensive per assessment with no cost/latency numbers reported. CS/NLP-bound.

## What We Can Use

| Component | What this paper gives us |
|---|---|
| **C2 — Literature Acquisition** | **Three retrieval channels, not one**: keyword/title-query generation + **whole-idea snippet search** + seed-paper recommendations. The snippet channel catches semantically-matching papers with no keyword overlap — the exact failure our plan names. All three are free Semantic Scholar APIs. |
| **C3 — Research Knowledge Layer** | The **facet decomposition** (purpose / mechanism / evaluation / application domain) is a better organising schema for novelty comparison than our flat `methodology / datasets / experiments`. It composes with our graph: store facets per paper and gap detection becomes "which facet combination is unoccupied." |
| **C4 — Gap / Novelty Detection** | (a) the operational novelty rule as our definition; (b) **facet-priority reranking** as the retrieval-quality mechanism; (c) **expert-labeled in-context examples that include reasoning** — they beat every prompt optimiser; (d) the output format — classification plus reasoning that names the closest papers and the differing facet, which maps directly onto our *Closest Existing Work* and *Difference from Existing Work* fields. **And fail closed:** when the system can't conclude, lower the novelty level and say the search was inconclusive — never default to "novel". |
| **C4 — the 5-level ladder** | Their stated future work is *"novelty as a continuum rather than binary"* — literally our ladder. But their formative study warns that three categories dropped human κ from 0.68 to 0.64, so our five levels must be **evidence-defined** (level 4 = literature explicitly states the limitation; level 5 = consistent gap across multiple papers), and we must report inter-annotator agreement. |
| **Evaluation** | (a) Cohen's κ against expert labels as headline metric, with κ≈0.59 system / κ≈0.68 human as realistic expectations; (b) **ablate on "not novel" cases only**; (c) fix the paper set when comparing prompts, vary it only when comparing retrieval; (d) report **retrieval overlap and rank shift** to explain *why* an ablation moved; (e) measure prompt sensitivity — don't assume an optimiser helps. |

**Steal first:** (1) whole-idea snippet search; (2) facet-priority RankGPT reranking; (3) the facet-based novelty rule; (4) in-context examples with expert *reasoning*; (5) fail closed.
