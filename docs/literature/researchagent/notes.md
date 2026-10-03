# ResearchAgent — Extraction Notes

> **Paper:** *ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models*
> **Authors:** Jinheon Baek (KAIST), Sujay Kumar Jauhar (Microsoft Research), Silviu Cucerzan (Microsoft Research), Sung Ju Hwang (KAIST / DeepAuto.ai)
> **arXiv:** 2404.07738v2 [cs.CL], 9 Feb 2025 (v1 Apr 2024)
> **Code:** https://github.com/JinheonBaek/ResearchAgent
> **Relationship to CoI:** ResearchAgent is *prior* work that Chain-of-Ideas uses as a baseline (CoI beats it). Conversely, this v2 explicitly cites CoI as work building on an earlier version of this paper. The two are complementary, not redundant — read them as **entity-centric breadth (this) vs. temporal-chain depth (CoI)**.

---

## 1. TL;DR

ResearchAgent decomposes a research idea into **three sequential artefacts — Problem → Method → Experiment Design** — and generates each with an LLM conditioned on three knowledge sources modelled on how human researchers work:

1. **Citation-graph literature survey** (a high-impact core paper + its citing/cited neighbours) — *depth*,
2. **An entity-centric knowledge store** built from entity co-occurrences across ~50k papers — *breadth and cross-pollination*,
3. **A community of LLM ReviewingAgents** that critique and drive iterative refinement — *peer review*.

Its distinctive move is **aligning the evaluation criteria to human judgements**: 10 human-scored idea pairs are fed to GPT-4, which *induces* detailed 5-point rubrics; those induced rubrics then drive both the ReviewingAgents and the automatic evaluation. This visibly recalibrates the model score distribution toward the human one.

---

## 2. Problem Formalisation (worth copying — it's clean)

Let `L` be the literature and `o` an idea composed of problem `p`, method `m`, and experiment design `d`:

```
o = [p, m, d]
o = f(L)
```

decomposed into three **submodular** steps:

```
p = f(L)          # problem identification
m = f(p, L)       # method development
d = f(p, m, L)    # experiment design
```

instantiated with an LLM and per-step templates:

```
p = LLM(T_p(L))
m = LLM(T_m(p, L))
d = LLM(T_e(p, m, L))
```

and finally, knowledge-augmented:

```
o = LLM( T( {l_0,…,l_n}, Ret({l_0,…,l_n}; K) ) )
```

where `{l_0,…,l_n}` is the core paper + its citation-graph neighbours and `Ret(·; K)` retrieves entities from the knowledge store `K`.

Framing premise (footnoted in the paper, from Young 2003): *"a new idea is more often than not just a new combination of old elements."* That premise is what justifies the entity store.

---

## 3. Architecture

```
                 (A) SCIENTIFIC KNOWLEDGE SOURCES
   core paper l_0 ──citation graph──► {l_1 … l_n} related papers
                                   │
   ~50k papers ──entity linking──► Entity-Centric Knowledge Store K (m×m sparse
                                   co-occurrence matrix)
                                   │  Ret(·;K) → top-k external entities
                                   ▼
                 (B) SYSTEMATIC IDEA GENERATION
   Problem Identification ──► Method Development ──► Experiment Design
             ▲                        ▲                      ▲
             └────── reviews & feedback ──────────────────────┘
                 15 ReviewingAgents (3 artefacts × 5 criteria)
                 criteria induced from real human judgements
```

### 3.1 Citation-graph literature survey

Motivation: the whole of `L` cannot fit in a context window, and LLMs degrade over very long contexts ("lost in the middle"). So mirror the human habit of expanding outward from one paper.

Two design choices, both explicit:
1. **Core-paper selection by impact** — citation count as a proxy (e.g. *>100 citations over 3 months*; in the experiments, *>20 citations*), mirroring how researchers anchor on influential work.
2. **Neighbour narrowing by abstract similarity** — the (potentially ~87) related papers are filtered by abstract similarity with the core paper, keeping the set focused.

Stated limitation of this alone: it confines the system to one local neighbourhood of the citation graph, whereas scientific knowledge accumulates *across* fields. Hence the entity store.

### 3.2 Entity-centric knowledge store (the paper's signature contribution)

**Premise.** Entities are the *atomic units of knowledge*, uniform across disciplines, so they can be extracted and accumulated over a whole corpus. If "database" is frequent in medical science but rare in hematology, the store captures that affinity and can suggest "database" when ideating about hematology — i.e. **entities are the vehicle for cross-domain idea cross-pollination**.

**Construction.**
- Entity linker: **BLINK** (Wu et al., 2020) — an off-the-shelf, open-domain (not science-specific) linker. It tags and canonicalises entities: `E_l = EL(l)`, a *multiset* (repetitions allowed).
- Extraction target is restricted to **titles and abstracts only** (publications are too long to process wholesale).
- Corpus: papers from **May 1 2023 – Dec 31 2023** from Semantic Scholar, plus their references — **50,091 papers total**.
- Store: a sparse **`K ∈ R^{m×m}`** matrix over `m` unique entities, counting **pairwise co-occurrences** within papers *and* per-entity totals. Built from all pairs `{e_i, e_j}` for `(i,j) ∈ C(|E|,2)`.

**Retrieval.** Retrieve the top-k entities *not already present* in the given papers:

```
Ret({l_0,…,l_n}; K) = argmax_{I⊂[m], |I|=k}  Π P(e_i | E_{l_0,…,l_n}),   e_i ∉ E_{l_0,…,l_n}
```

which, by Bayes' rule and an independence assumption, approximates to:

```
argmax_{I⊂[m], |I|=k}  ( Π_{e_j ∈ E} P(e_j | e_i) ) × P(e_i)
```

with `P(e_j|e_i)` and `P(e_i)` read off normalised entries of `K`.

**Crucially:** the retrieval deliberately returns entities **outside** the current context. The authors contrast this with prior entity work (including Baek et al. 2024 on query suggestion) which *narrows* the LLM to entities already present — the opposite objective.

**Alternative tried (Appendix B.3):** contextual **embedding-based** retrieval using the entity linker's latent space instead of co-occurrence. Result: essentially a wash (Problem 4.52 vs 4.49, Method 4.28 vs **4.34**, Experiment 4.18 vs 4.16). An interesting asymmetry they note: co-occurrence retrieval can surface entities *opposite* to the core idea (because papers mention prior work's limitations alongside their proposal), whereas embedding retrieval returns mostly similar concepts.

### 3.3 ReviewingAgents & iterative refinement

- **15 ReviewingAgents** = 3 artefacts (problem, method, experiment) × 5 criteria each. Each agent evaluates one artefact along one criterion, producing `Review / Feedback / Rating (1-5)`.
- ResearchAgent consumes all reviews and regenerates the artefacts.
- Refinement improves quality but **saturates after ~3 iterations** (Figure 4), matching the pattern in multi-agent debate work (Du et al. 2023).
- ResearchAgent still beats its ablations **even with zero refinement steps** (Appendix B.1) — the knowledge augmentation is doing independent work.

### 3.4 Human-preference-aligned criteria induction

The clever, cheap trick:

1. Collect **10 pairs** of generated research ideas scored on 5-point Likert per criterion by human researchers (each with ≥3 published papers).
2. **Prompt GPT-4 with those human-annotated pairs to *induce* detailed textual descriptions of each rubric level** (following Lin et al. 2024).
3. Use the induced rubrics as `{criteria}` inside both the ReviewingAgents and the model-based evaluation.

Effect (Figure 5): unaligned model scoring is skewed and unlike human scoring; after alignment the distribution closely resembles the human one. Sanity check: 5 human annotators judged the *induced criteria themselves* — 2 agreed strongly, 3 moderately.

---

## 4. Implementation Details

| Item | Value |
|---|---|
| Main LLM | **GPT-4** (release 2023-11-06; training data to Apr 2023) |
| Also evaluated | GPT-3.5, Llama3-8B, Mixtral-8x7B, Qwen1.5-32B |
| Literature source | Semantic Scholar Academic Graph API |
| Paper window | **after 2023-05-01** (post-dates GPT-4's training cutoff) |
| Core-paper filter | **>20 citations** (high-impact proxy) |
| Benchmark size | **300 core papers** sampled |
| Avg. references per core paper | **87** |
| Avg. entities per abstract | **2.17** (BLINK reports ~3 per paper) |
| Entity linker | **BLINK** (Wu et al. 2020), open-domain |
| Entity-store corpus | **50,091 papers** (May–Dec 2023 + references) |
| Entity extraction scope | titles + abstracts only |
| Refinement steps | up to 4 tested; **saturates at ~3** |
| ReviewingAgents | **15** (3 artefacts × 5 criteria) |
| Human evaluators | 10 researchers (US + South Korea; CS, medicine, biology; ≥3 papers each), paid **$22.20/hr**, 6-page guideline, **Label Studio**, ~9 ideas evaluated per hour, **150 ideas fully evaluated** |
| Domain mix of core papers | top 3 = computer science, medicine, engineering |

---

## 5. Prompt Library (verbatim)

Placeholders as in the paper: `{paper['title']}`, `{relatedPaper['titles']}`, `{Entities}`, `{researchProblem}`, `{scientificMethod}`, `{metric}`, `{criteria}`, etc.

### Table 6 — Problem Identification

**System message**
```
You are an AI assistant whose primary goal is to identify promising, new, and key scientific
problems based on existing scientific literature, in order to aid researchers in discovering novel
and significant research opportunities that can advance the field.
```

**User message**
```
You are going to generate a research problem that should be original, clear, feasible, relevant, and
significant to its field. This will be based on the title and abstract of the target paper, those of
{len(references)} related papers in the existing literature, and {len(entities)} entities potentially
connected to the research area.

Understanding of the target paper, related papers, and entities is essential:
- The target paper is the primary research study you aim to enhance or build upon through future
research, serving as the central source and focus for identifying and developing the specific
research problem.
- The related papers are studies that have cited the target paper, indicating their direct relevance
and connection to the primary research topic you are focusing on, and providing additional context
and insights that are essential for understanding and expanding upon the target paper.
- The entities can include topics, keywords, individuals, events, or any subjects with possible
direct or indirect connections to the target paper or the related studies, serving as auxiliary sources
of inspiration or information that may be instrumental in formulating the research problem.

Your approach should be systematic:
- Start by thoroughly reading the title and abstract of the target paper to understand its core focus.
- Next, proceed to read the titles and abstracts of the related papers to gain a broader perspective
and insights relevant to the primary research topic.
- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of
inspiration and information, while keeping in mind that not all may be relevant.

I am going to provide the target paper, related papers, and entities, as follows:
Target paper title: {paper['title']}
Target paper abstract: {paper['abstract']}
Related paper titles: {relatedPaper['titles']}
Related paper abstracts: {relatedPaper['abstracts']}
Entities: {Entities}

With the provided target paper, related papers, and entities, your objective now is to formulate a
research problem that not only builds upon these existing studies but also strives to be original,
clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title
and abstract of the target paper, to ensure it remains the focal point of your research problem
identification process.

Target paper title: {paper['title']}
Target paper abstract: {paper['abstract']}

Then, following your review of the above content, please proceed to generate one research
problem with the rationale, in the format of
Problem:
Rationale:
```

> **Prompt-engineering pattern worth noting:** the core paper's title+abstract is repeated *immediately before the instruction* to counteract "lost in the middle" — the anchor is restated after the long context so it stays the focal point. The same re-statement pattern recurs in the method and experiment prompts.

### Table 7 — Method Development

**System message**
```
You are an AI assistant whose primary goal is to propose innovative, rigorous, and valid method-
ologies to solve newly identified scientific problems derived from existing scientific literature, in
order to empower researchers to pioneer groundbreaking solutions that catalyze breakthroughs in
their fields.
```

**User message**
```
You are going to propose a scientific method to address a specific research problem. Your method
should be clear, innovative, rigorous, valid, and generalizable. This will be based on a deep
understanding of the research problem, its rationale, existing studies, and various entities.

Understanding of the research problem, existing studies, and entities is essential:
- The research problem has been formulated based on an in-depth review of existing studies
and a potential exploration of relevant entities, which should be the cornerstone of your method
development.
- The existing studies refer to the target paper that has been pivotal in identifying the problem, as
well as the related papers that have been additionally referenced in the problem discovery phase,
all serving as foundational material for developing the method.
- The entities can include topics, keywords, individuals, events, or any subjects with possible
direct or indirect connections to the existing studies, serving as auxiliary sources of inspiration or
information that may be instrumental in method development.

Your approach should be systematic:
- Start by thoroughly reading the research problem and its rationale, to understand your primary
focus.
- Next, proceed to review the titles and abstracts of existing studies, to gain a broader perspective
and insights relevant to the primary research topic.
- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of
inspiration and information, while keeping in mind that not all may be relevant.

I am going to provide the research problem, existing studies (target paper & related papers), and
entities, as follows:
Research problem: {researchProblem}
Rationale: {researchProblemRationale}
Target paper title: {paper['title']}
Target paper abstract: {paper['abstract']}
Related paper titles: {relatedPaper['titles']}
Related paper abstracts: {relatedPaper['abstracts']}
Entities: {Entities}

With the provided research problem, existing studies, and entities, your objective now is to
formulate a method that not only leverages these resources but also strives to be clear, innovative,
rigorous, valid, and generalizable. Before crafting the method, revisit the research problem, to
ensure it remains the focal point of your method development process.

Research problem: {researchProblem}
Rationale: {researchProblemRationale}

Then, following your review of the above content, please proceed to propose your method with
its rationale, in the format of
Method:
Rationale:
```

### Table 8 — Experiment Design

**System message**
```
You are an AI assistant whose primary goal is to design robust, feasible, and impactful ex-
periments based on identified scientific problems and proposed methodologies from existing
scientific literature, in order to enable researchers to systematically test hypotheses and validate
groundbreaking discoveries that can transform their respective fields.
```

**User message**
```
You are going to design an experiment, aimed at validating a proposed method to address a
specific research problem. Your experiment design should be clear, robust, reproducible, valid,
and feasible. This will be based on a deep understanding of the research problem, scientific
method, existing studies, and various entities.

Understanding of the research problem, scientific method, existing studies, and entities is essential:
- The research problem has been formulated based on an in-depth review of existing studies and a
potential exploration of relevant entities.
- The scientific method has been proposed to tackle the research problem, which has been
informed by insights gained from existing studies and relevant entities.
- The existing studies refer to the target paper that has been pivotal in identifying the problem and
method, as well as the related papers that have been additionally referenced in the discovery phase
of the problem and method, all serving as foundational material for designing the experiment.
- The entities can include topics, keywords, individuals, events, or any subjects with possible
direct or indirect connections to the existing studies, serving as auxiliary sources of inspiration or
information that may be instrumental in your experiment design.

Your approach should be systematic:
- Start by thoroughly reading the research problem and its rationale followed by the proposed
method and its rationale, to pinpoint your primary focus.
- Next, proceed to review the titles and abstracts of existing studies, to gain a broader perspective
and insights relevant to the primary research topic.
- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of
inspiration and information, while keeping in mind that not all may be relevant.

I am going to provide the research problem, scientific method, existing studies (target paper &
related papers), and entities, as follows:
Research problem: {researchProblem}
Rationale: {researchProblemRationale}
Scientific method: {scientificMethod}
Rationale: {scientificMethodRationale}
Target paper title: {paper['title']}
Target paper abstract: {paper['abstract']}
Related paper titles: {relatedPaper['titles']}
Related paper abstracts: {relatedPaper['abstracts']}
Entities: {Entities}

With the provided research problem, scientific method, existing studies, and entities, your
objective now is to design an experiment that not only leverages these resources but also strives to
be clear, robust, reproducible, valid, and feasible. Before crafting the experiment design, revisit
the research problem and proposed method, to ensure they remain at the center of your experiment
design process.

Research problem: {researchProblem}
Rationale: {researchProblemRationale}
Scientific method: {scientificMethod}
Rationale: {scientificMethodRationale}

Then, following your review of the above content, please proceed to outline your experiment with
its rationale, in the format of
Experiment:
Rationale:
```

### Table 9 — ReviewingAgent: Problem Validation

**System message**
```
You are an AI assistant whose primary goal is to assess the quality and validity of scientific
problems across diverse dimensions, in order to aid researchers in refining their problems based
on your evaluations and feedback, thereby enhancing the impact and reach of their work.
```

**User message**
```
You are going to evaluate a research problem for its {metric}, focusing on how well it is defined
in a clear, precise, and understandable manner.

As part of your evaluation, you can refer to the existing studies that may be related to the problem,
which will help in understanding the context of the problem for a more comprehensive assessment.
- The existing studies refer to the target paper that has been pivotal in identifying the problem, as
well as the related papers that have been additionally referenced in the discovery phase of the
problem.

The existing studies (target paper & related papers) are as follows:
Target paper title: {paper['title']}
Target paper abstract: {paper['abstract']}
Related paper titles: {relatedPaper['titles']}
Related paper abstracts: {relatedPaper['abstracts']}

Now, proceed with your {metric} evaluation approach that should be systematic:
- Start by thoroughly reading the research problem and its rationale, keeping in mind the context
provided by the existing studies mentioned above.
- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing
on the {metric} of the problem.
- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a
discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless
fully justified:
{criteria}

I am going to provide the research problem with its rationale, as follows:
Research problem: {researchProblem}
Rationale: {researchProblemRationale}

After your evaluation of the above content, please provide your review, feedback, and rating, in
the format of
Review:
Feedback:
Rating (1-5):
```

### Table 10 — ReviewingAgent: Method Validation

**System message**
```
You are an AI assistant whose primary goal is to assess the quality and soundness of scientific
methods across diverse dimensions, in order to aid researchers in refining their methods based on
your evaluations and feedback, thereby enhancing the impact and reach of their work.
```

**User message** (same shape; context = research problem + existing studies)
```
You are going to evaluate a scientific method for its {metric} in addressing a research problem,
focusing on how well it is described in a clear, precise, and understandable manner that allows for
replication and comprehension of the approach.
[...context block: Research problem / Rationale / Target paper title & abstract /
   Related paper titles & abstracts...]
Now, proceed with your {metric} evaluation approach that should be systematic:
- Start by thoroughly reading the proposed method and its rationale, keeping in mind the context
provided by the research problem, and existing studies mentioned above.
- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing
on the {metric} of the method.
- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a
discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless
fully justified:
{criteria}

I am going to provide the proposed method with its rationale, as follows:
Scientific method: {scientificMethod}
Rationale: {scientificMethodRationale}

After your evaluation of the above content, please provide your review, feedback, and rating, in
the format of
Review:
Feedback:
Rating (1-5):
```

### Table 11 — ReviewingAgent: Experiment Design Validation

**System message**
```
You are an AI assistant whose primary goal is to meticulously evaluate the experimental designs
of scientific papers across diverse dimensions, in order to aid researchers in refining their experi-
mental approaches based on your evaluations and feedback, thereby amplifying the quality and
impact of their scientific contributions.
```

**User message** (context = problem + method + existing studies; same Review/Feedback/Rating output)
```
You are going to evaluate an experiment design for its {metric} in validating a scientific method
to address a research problem, focusing on how well it is described in a clear, precise, and
understandable manner, enabling others to grasp the setup, procedure, and expected outcomes.
[...context block: Research problem / Rationale / Scientific method / Rationale /
   Target paper title & abstract / Related paper titles & abstracts...]
Now, proceed with your {metric} evaluation approach that should be systematic:
- Start by thoroughly reading the experiment design and its rationale, keeping in mind the context
provided by the research problem, scientific method, and existing studies mentioned above.
- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing
on the {metric} of the experiment.
- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a
discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless
fully justified:
{criteria}

I am going to provide the designed experiment with its rationale, as follows:
Experiment design: {experimentDesign}
Rationale: {experimentDesignRationale}

After your evaluation of the above content, please provide your review, feedback, and rating, in
the format of
Review:
Feedback:
Rating (1-5):
```

---

## 6. Evaluation Criteria (Table 12) — 15 criteria, 5 per artefact

### Problem
| Criterion | Definition |
|---|---|
| **Clarity** | Whether the problem is defined in a clear, precise, and understandable manner. |
| **Relevance** | Whether the problem is pertinent and applicable to the current field or context of study. |
| **Originality** | Whether the problem presents a novel challenge or unique perspective not extensively explored before. |
| **Feasibility** | Whether the problem can realistically be investigated or solved with available resources and within reasonable constraints. |
| **Significance** | The importance and potential impact of solving the problem, including contribution to the field and broader implications. |

### Method
| Criterion | Definition |
|---|---|
| **Clarity** | Whether the method is described precisely enough to allow replication and comprehension. |
| **Validity** | Accuracy, relevance and soundness of the method in addressing the research problem. |
| **Rigorousness** | Thoroughness, precision and consistency; whether the approach is systematic, well-structured, and meets high research-quality standards. |
| **Innovativeness** | Whether the method introduces new techniques, approaches or perspectives that differ from standard practice. |
| **Generalizability** | Extent to which the method applies to other contexts, populations or settings beyond the study. |

### Experiment Design
| Criterion | Definition |
|---|---|
| **Clarity** | Whether the design is described precisely enough for others to grasp setup, procedure, expected outcomes. |
| **Validity** | Appropriateness and soundness of the design in addressing the research questions / validating the method. |
| **Robustness** | Durability across a wide range of conditions and variables; outcomes not reliant on a few specific cases. |
| **Feasibility** | Whether it can realistically be implemented with available resources, time and technological/methodological constraints. |
| **Reproducibility** | Whether the information is sufficient and detailed enough for others to reproduce the experiment under identical conditions. |

### Human-induced rubrics (Tables 13–15) — the alignment artefact

Each of the 15 criteria gets a 5-level textual rubric induced from human judgements. Representative examples (verbatim):

**Problem · Originality**
```
1. The problem exhibits no discernible originality, closely mirroring existing studies without introducing any
   novel perspectives or challenges.
2. The problem shows minimal originality, with slight variations from known studies, lacking significant new
   insights or innovative approaches.
3. The problem demonstrates moderate originality, offering some new insights or angles, but these are not
   sufficiently groundbreaking or distinct from existing work.
4. The problem is notably original, presenting a unique challenge or perspective that is well-differentiated from
   existing studies, contributing valuable new understanding to the field.
5. The problem is highly original, introducing a pioneering challenge or perspective that has not been explored
   before, setting a new direction for future research.
```

**Problem · Feasibility**
```
1. The problem is fundamentally infeasible due to insurmountable resource constraints, lack of foundational
   research, or critical methodological flaws.
2. The problem faces significant feasibility challenges related to resource availability, existing knowledge gaps,
   or technical limitations, making progress unlikely.
3. The problem is feasible to some extent but faces notable obstacles in resources, existing research support, or
   technical implementation, which could hinder significant advancements.
4. The problem is mostly feasible with manageable challenges in resources, supported by adequate existing
   research, and has a clear, achievable methodology, though minor issues may persist.
5. The problem is highly feasible with minimal barriers, well-supported by existing research, ample resources,
   and a robust, clear methodology, promising significant advancements.
```

**Method · Innovativeness**
```
1. The method introduces no novel elements, fully relying on existing techniques without any attempt to modify
   or adapt them for the specific research problem, showing a lack of innovativeness.
2. The method shows minimal innovation, with only slight modifications to existing techniques that do not
   substantially change or improve the approach to the research problem.
3. The method demonstrates moderate innovativeness, incorporating known techniques with some new elements
   or combinations that offer a somewhat fresh approach to the research problem but fall short of a significant
   breakthrough.
4. The method is highly innovative, introducing new techniques or novel combinations of existing methods that
   significantly differ from standard practices, offering a new perspective or solution to the research problem.
5. The method represents a groundbreaking innovation, fundamentally transforming the approach to the
   research problem with novel techniques or methodologies that redefine the field's standard practices.
```

**Experiment · Reproducibility**
```
1. The experiment design lacks critical details, making it virtually impossible for other researchers to replicate
   the study under the same conditions or methodologies.
2. The experiment provides some essential information but omits significant details needed for replication,
   leading to considerable ambiguity in methodology or conditions.
3. The experiment design includes sufficient details for replication, but lacks clarity or completeness in certain
   areas, posing challenges for seamless reproducibility.
4. The experiment is well-documented with clear, detailed instructions and methodologies that allow for
   consistent replication, albeit with minor areas for improvement.
5. The experiment design is exemplary in its clarity, detail, and comprehensiveness, ensuring that other
   researchers can precisely and effortlessly replicate the study under identical conditions and methodologies.
```

*(The full set covers Problem: Clarity/Relevance/Originality/Feasibility/Significance; Method: Clarity/Validity/Rigorousness/Innovativeness/Generalizability; Experiment: Clarity/Validity/Robustness/Feasibility/Reproducibility — all in the same 1–5 prose-rubric form.)*

---

## 7. Results & Ablations

**Baselines** are ablations of itself (there is no prior system generating problem+method+experiment jointly):
1. **Naive ResearchAgent** — core paper only.
2. **ResearchAgent w/o Entity Retrieval** — core paper + references, no entities.
3. **ResearchAgent (full)** — core paper + references + retrieved entities.

**Main result.** The full model beats both ablations on *every* one of the 15 criteria, in both human and model evaluation, and wins the pairwise comparisons. Gains are **largest on creativity metrics** — Problem *Originality*, Method *Innovativeness* — consistent with the entity store's purpose of injecting concepts absent from the local citation neighbourhood.

**Ablation on knowledge sources (Table 2, model-based scores):**

| Method | Problem | Method | Experiment |
|---|---|---|---|
| **ResearchAgent (full)** | **4.52** | **4.28** | **4.18** |
| – w/o Entities | 4.35 | 4.13 | 4.02 |
| – w/ Random Entities | 4.41 | 4.19 | 4.13 |
| – w/o References | 4.26 | 4.08 | 3.97 |
| – w/ Random References | 4.35 | 4.16 | 4.02 |
| – w/o Entities & References | 4.20 | 4.03 | 3.92 |

Two findings worth carrying forward:
- **References matter more than entities** (removing references costs more).
- **Random elements beat no elements.** Their hypothesis: the LLM filters noise while still extracting incidental value. This is a caution for our evaluation design — *any* context injection lifts scores, so an ablation must compare against **random-context**, not just **no-context**, or the improvement is overstated.

**Refinement steps (Figure 4).** Scores rise then **saturate after ~3 iterations**.

**Agreements (Table 1):**

| | Problem | Method | Experiment |
|---|---|---|---|
| Human–Human, scoring (Spearman) | 0.83 | 0.76 | 0.67 |
| Human–Human, pairwise (Cohen's κ) | 0.62 | 0.62 | 0.41 |
| Human–Model, scoring | 0.64 | 0.58 | 0.49 |
| Human–Model, pairwise | 0.71 | 0.62 | 0.52 |

**Experiment designs have the lowest agreement everywhere** (0.41 human–human pairwise). The authors read this as inherent subjectivity in judging designs, not as a defect in the designs. Either way: **experiment-plan quality is the hardest thing to evaluate reliably** — directly relevant to us, since it's half our output.

**vs. hypothesis-generation baselines (Table 3):**

| Method | Clarity | Relevance | Originality | Feasibility | Significance |
|---|---|---|---|---|---|
| SciMON | 4.04 | 4.37 | 4.56 | 3.98 | 4.15 |
| Hypothesis Proposer | 3.97 | 4.14 | 4.07 | 4.01 | 4.11 |
| **ResearchAgent** | 4.11 | **4.88** | **4.77** | **4.05** | **4.81** |

**Across LLMs (Table 4).** The knowledge-augmentation gain **only materialises in capable models**:

| LLM | Naive → Full (Problem) | Verdict |
|---|---|---|
| GPT-4 | 4.20 → **4.52** | large gain |
| Llama3-8B | 3.76 → **4.18** | large gain |
| Qwen1.5-32B | 3.64 → **4.02** | large gain |
| GPT-3.5 | 3.56 → 3.58 | **no gain** |
| Mixtral-8x7B | 3.31 → 3.28 | **no gain / slight loss** |

Attributed to emergent complex-reasoning ability. **Implication for us: architecture gains are model-gated — a weak model will show a flat ablation and hide a real improvement.**

**By domain (Figure 9).** Ideas in high-resource domains (CS, medicine, engineering) score higher than low-resource ones (physics, chemistry, mathematics) — attributed to the LLM's training-data distribution.

**Citation-count correlation (Figure 6).** Higher-cited core papers → higher-quality generated ideas, in both CS-only and all-domain slices. Justifies the high-impact core-paper heuristic *and* suggests the induced criteria generalise beyond CS.

**Qualitative highlight (Table 16).** The flagship example of cross-pollination: for a *Drosophila* whole-brain connectome paper, two retrieved entities — **Drosophila Genetic Reference Panel** and **CRISPR** — enabled a genuinely non-obvious idea bridging genetic variability and CRISPR applications. That is the entity store paying off exactly as designed. (The table also honestly shows the failure mode: entity lists for an NLP paper include noise like *'LMS color space'*, *'Adhesive'*, *'London, Midland and Scottish Railway'*, *'Tomas Mikolov'*.)

---

## 8. What to take into our Research Agent

Mapped to the components in *Architecture Overview and Project Plan*:

| Our component | What ResearchAgent gives us |
|---|---|
| **C2 — Literature Acquisition** | Concrete, defensible heuristics: **impact-gated core paper** (citation threshold), **abstract-similarity narrowing** of the ~87 references, and a **post-LLM-cutoff paper window** so the model cannot regurgitate a memorised answer. The citation-count→quality correlation (Fig. 6) is empirical justification for the impact gate. |
| **C3 — Research Knowledge Layer** | The **entity-centric knowledge store** is the piece our current design is missing. Our C3 graph is *intra-corpus* (`Paper → proposes → Method`); theirs is a **global co-occurrence matrix over ~50k papers** whose whole purpose is retrieving entities *absent from* the current context. These compose cleanly: keep our typed per-paper graph, add a global entity co-occurrence layer as a **cross-domain suggestion channel**. Note both retrieval variants (co-occurrence vs embedding) work about equally — pick on engineering cost. |
| **C4 — Gap / Novelty Detection** | The `problem → method → experiment` decomposition with per-stage rationale is a clean structure for our Novel Evidence Report. Their **Originality** criterion + the induced 5-level rubric is a ready-made calibrated scale — and it is structurally close to our 5-level potential-novelty ladder, so the rubrics can be adapted rather than written from scratch. |
| **C5 — Experiment Planning** | Their experiment prompt targets **clear, robust, reproducible, valid, feasible** — five axes that map onto our plan's Risks/Limitations and Ablation sections. Their `Experiment: / Rationale:` output format is the seed of provenance: **the rationale field is the natural place to attach derivation provenance**. |
| **C-Eval — Evaluation** | (a) The **human-induced criteria** trick — 10 human-scored pairs → LLM-induced rubrics → measurably better human/model score-distribution match — is cheap and should be adopted before we trust any LLM-judge number. (b) **Random-context ablations**, not just no-context. (c) Report **inter-annotator agreement** (Spearman for scores, Cohen's κ for pairwise). (d) Expect experiment-design evaluation to be the noisiest axis and budget more annotations there. (e) Human-eval logistics we can reuse: Label Studio, 6-page guideline, domain-expert annotators judging ideas generated *from their own papers*, ~9 ideas/hour. |

**Combining the two papers.** CoI beats ResearchAgent in CoI's own arena — but the two mechanisms are orthogonal, and nobody has combined them:
- ResearchAgent = **breadth** (cross-domain entities, structured problem/method/experiment decomposition, human-aligned reviewing).
- CoI = **depth** (temporal chain, trend transitions, future-trend extrapolation, ELO arena).
A system that walks a **CoI-style temporal chain** *and* augments each step with **ResearchAgent-style out-of-context entities**, then attaches **provenance** to every element, is a defensible position for our project — and is precisely the space our C3 + C5 already point at.

---

## 9. Gaps, Limitations & Future Scope

### Stated by the authors (Limitations §)
1. **Entity store coverage is thin.** Built only from **titles + abstracts** of a **limited window** of papers (May–Dec 2023), so most entities and their interconnections are never captured.
2. **BLINK yields ~3 entities per paper** — low coverage, and it is an **open-domain** linker with no scientific-domain training. (The published entity lists visibly contain noise like *'LMS color space'* and *'Adhesive'* for NLP papers.)
3. **Hallucination.** Knowledge augmentation only partially grounds generation; the authors state plainly that **experimental validation of generated ideas is essential** and absent here.
4. **Review scope is narrow.** 15 ReviewingAgents × 5 fixed criteria may not span the perspectives needed across all research domains; criteria were chosen by presumed importance, not exhaustively.
5. **Poor fit for theoretical sciences** (mathematics, theoretical physics) where proof generation, not experiment design, is the deliverable. They suggest omitting the experiment stage and using reasoning models there.
6. **Scaling & currency** (Conclusion): the knowledge store must be expanded across domains and kept up to date with new publications for the system to work in emerging fields.
7. **Ethics** (their own statement): misuse potential (explosives, malware, surveillance), and **unintentional plagiarism** from training-data regurgitation. Their proposed mitigations — a comprehensive knowledge base to inform users of prior work, and recording/tracking all generated ideas to detect repetition — are acknowledged as complex and incomplete.

### Gaps we can attack (our project's opening)
1. **No provenance whatsoever.** Ideas are conditioned on titles+abstracts of ~87 papers and a bag of entities; the output carries **no link back to which paper supports which claim**. There is no notion of citation correctness, citation completeness, or traceability. **This is the clearest open space and it is exactly our C5.**
2. **Abstract-only grounding.** The system never reads a paper's methods, results, or limitations sections. It therefore cannot know what a method *actually* requires, what its reported limitation *actually* is, or whether a proposed baseline is even runnable. **Our C2 schema (`methodology / datasets / experiments / limitations / claims`) is a strict improvement**, and the CoI paper shows full-paper reading is affordable with a cheap model.
3. **Novelty is never checked at all.** Unlike CoI, there is **no novelty-verification step** — nothing retrieves prior work to test whether the generated problem already exists. "Originality" is only a *reviewer opinion score*, never an evidence-backed verdict. Our 5-level potential-novelty ladder with explicit closest-existing-work and "no matching approach found in searched corpus" is a substantial advance over both papers.
4. **Entity independence assumption is crude.** Retrieval assumes entities are independent and works off raw co-occurrence counts; no typing (is this entity a dataset? a metric? a method?), no canonical linking to a scientific ontology, no filtering of the obvious noise. **A typed, science-specific entity layer is a straightforward, high-yield improvement.**
5. **Single core paper as the entry point.** The system requires a *paper* to start from, not a *research question*. **Our C1 takes a natural-language problem** — so we need a problem→anchor-paper resolution step that ResearchAgent simply does not have. (CoI's multi-query + anchor-retrieval solves exactly this and can be dropped in.)
6. **No user-uploaded documents**, no interaction, no researcher-in-the-loop steering — same gap as CoI.
7. **Refinement saturates at ~3 rounds** with a fixed criteria set; the reviewing loop has no way to *seek new information* when a critique reveals a knowledge gap. (Contrast CoI's T19, which turns a critique into a literature-search query. **Combining the two — critique-driven retrieval inside a multi-criteria review loop — is an obvious unclaimed improvement.**)
8. **Evaluation ceiling effects.** Scores cluster in 3.9–4.5 on a 5-point scale; the reviewing prompt has to *explicitly beg* the model to avoid uniformly high ratings. Pairwise/ELO (CoI's approach) sidesteps this and should be preferred for our headline numbers.
9. **Low-resource-domain degradation** is measured but not addressed — an open problem if our agent is meant to serve fields outside CS/medicine.
10. **No cost, latency, or scaling numbers.** 15 ReviewingAgents × up to 3 refinement rounds × GPT-4 is expensive, and the paper never quantifies it (CoI, by contrast, reports ~$0.50/idea). We should instrument cost from day one.

