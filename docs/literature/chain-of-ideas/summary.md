# Chain of Ideas — Summary

> *Chain of Ideas: Revolutionizing Research via Novel Idea Development with LLM Agents* — Li et al., DAMO Academy / Zhejiang Univ. · arXiv 2410.13185v5 (Oct 2024) · [code](https://github.com/DAMO-NLP-SG/CoI-Agent) · ~$0.50 per idea + experiment design

## Summary

**Thesis:** how retrieved literature is *organised* matters more than how much of it there is. Instead of dumping N papers into a prompt (vanilla RAG), CoI arranges a small set into a **linear chain mirroring a subfield's actual evolution** (`CoT → SC → ToT → GoT`), makes the LLM state *what changed between each adjacent pair*, then extrapolates the next step. Ideation becomes trend extrapolation rather than blind recombination. The failure it attacks: flat context produces category errors (grafting a fine-tuning method onto a prompting method).

**Three stages.**

1. **CoI construction.** Topic → K search queries (K perspectives). Per query, retrieve an anchor paper, then grow the chain **forward** (citing papers, ranked by `text-embedding-3-large` cosine against *topic + anchor abstract* so the chain can't drift) and **backward** (LLM nominates references the paper built on / used as baselines / on the same topic; stops at a **>1,000-citation milestone paper**). Each paper yields a structured record: `Entities`, `Idea{Background, Novelty, Contribution, Detail Reason, Limitation}`, `Experiment{Process, Technical Details, Plan, Baseline, Verification}`, top-3 `References`. Then a **trend transition** is written for every adjacent pair ("Paper 0 to Paper 1: …") — the paper's highest-value artefact.
2. **Idea generation.** Future-trend prediction under three mandated modes — **Reflection** (where does the method break), **Analogy** (transplant a solution from another field), **Deep Dive** (which sub-component could be improved) — then step-by-step consolidation `Motivation → Novelty → Method → Final Idea`. A **novelty-check agent** compares the draft against retrieved papers and outputs `Similar: 0/1`; on a hit, the idea is appended to a `[Bad case]` list fed back into the next generation round. K branch ideas compete pairwise; the winner proceeds.
3. **Experiment design.** Generate → review → **derive a search query from the critique** → retrieve → refine. The query-derivation step is the clever bit: refinement is grounded in *new* literature rather than re-reasoning over the same context.

**Idea Arena** replaces Likert scoring: round-robin pairwise judging, each pair judged **twice with order swapped**, converted to ELO, with explicit anti-position/anti-length instructions in the judge prompt. Human↔GPT-4o agreement 70.8%; LLM↔LLM agreement ~90%.

**Key numbers.** Human ELO: Real Paper 1100 > **CoI 1085** > RAG 1029 > GPT-Researcher 992 > ResearchAgent 980 > AI-Scientist 812. CoI **beats real papers on novelty and significance** but loses on clarity, feasibility, effectiveness. Ablations (out of 50): removing the chain **42.4** (worst), removing future-trend prediction 46.2, removing entities 45.4 — the chain structure is the load-bearing component. Chain length saturates at **5**; branch width `K = 3` suffices. Operating config: GPT-4o main, GPT-4o-mini for full-paper reads, Semantic Scholar, 1 refinement iteration. Benchmark = 50 topics from post-cutoff HF Daily Papers, with the real papers' extracted ideas as an upper-bound baseline.

## Gaps

**Stated / evidenced by the paper**

- **Feasibility is the standing bottleneck** for all automated methods, in both ideation and experiment design. Named as the open problem; no mechanism offered.
- **No execution.** Ideas are never run, so "expected effectiveness" is a judged guess. The gap to real papers is attributed exactly to real papers having survived experiments.
- Diminishing returns past length 5 / width 3; refinement fixed at 1 iteration purely for cost.
- Evaluation is LLM-judged at scale with only ~70% human agreement, while LLM–LLM agreement is ~90% — a systematic, self-consistent bias humans don't share.

**Open for us**

- **A single linear chain is a lossy model of a field.** Real progress is a DAG with merges, forks and dead ends; the K branches are independent and never reconciled. A landscape graph with cross-branch edges generalises this.
- **Greedy and unbacktracked** — forward extension always takes the top-1 ranked citing paper; one bad hop corrupts the rest. No beam search, no whole-chain re-ranking.
- **Novelty is a binary similarity verdict** over a handful of papers, with no notion of retrieval coverage or confidence — precisely the "no prior work found ≠ novel" trap.
- **No provenance.** Ideas cite paper titles inside free text; nothing traces recommendation → reasoning → evidence → paper → section.
- Entities are extracted per paper but never canonicalised or pooled across papers.
- Domain-bound to AI/ML (the >1,000-citation milestone rule and dense fast citation graphs are AI-specific); no user-uploaded documents; no researcher-in-the-loop steering, despite feasibility being the axis a human could cheapest correct.

## What We Can Use

| Component | What CoI gives us |
|---|---|
| **C2 — Literature Acquisition** | Multi-query decomposition (K perspectives) instead of one query; anchor paper + **bidirectional citation walk**; embedding re-rank against *topic + anchor abstract* to stop drift; stop rules (max length, LLM relevance gate, citation milestone). |
| **C3 — Research Knowledge Layer** | Their per-paper schema is richer than ours in places and directly compatible. The key addition: a **temporal-derivation edge** (`paper_i → paper_{i+1}` recording *what changed and why*) — our graph has `proposes/uses/evaluates/limitation` but no such edge, and the ablation shows it is the most valuable one. |
| **C4 — Gap / Novelty Detection** | The `[Bad case]` memory that feeds rejected ideas back into regeneration; the Reflection / Analogy / Deep-Dive triad as a promptable operationalisation of "where hasn't this been combined"; the trend narrative as a readable form of our research landscape. |
| **C5 — Experiment Planning** | The generate → review → **query-derivation** → refine loop, reusable as-is. Their design rubric already demands ablation design, feasibility assessment with risk mitigation, and pseudocode/flowchart — a good skeleton for our plan structure. |
| **Evaluation** | Adopt **Idea Arena** over Likert: round-robin, both orders, ELO, anti-bias judge instructions, and a reported human↔judge agreement figure to justify the automated numbers. Their baseline set maps onto our Baselines 1–4, and the **Real Paper upper bound** (post-cutoff paper, topic+idea+design extracted) is an addition our plan lacks. |

**Steal first:** (1) trend-transition edges; (2) `[Bad case]` feedback in the novelty loop; (3) critique → search query → refine; (4) ELO arena + Real-Paper upper bound; (5) post-cutoff paper selection so the model can't have memorised the answer.
