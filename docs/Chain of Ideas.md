# Chain of Ideas (CoI) — Extraction Notes

> **Paper:** *Chain of Ideas: Revolutionizing Research via Novel Idea Development with LLM Agents*
> **Authors:** Long Li, Weiwen Xu, Jiayan Guo, Ruochen Zhao, Xingxuan Li, Yuqian Yuan, Boqiang Zhang, Yuming Jiang, Yifei Xin, Ronghao Dang, Yu Rong, Deli Zhao, Tian Feng, Lidong Bing
> **Affiliations:** DAMO Academy (Alibaba Group), Zhejiang University, USTC
> **arXiv:** 2410.13185v5 [cs.AI], 30 Oct 2024 — Preprint
> **Code:** https://github.com/DAMO-NLP-SG/CoI-Agent · **Demo:** https://huggingface.co/spaces/DAMO-NLP-SG/CoI_Agent
> **Cost claim:** ~$0.50 per candidate idea + experiment design

---

## 1. TL;DR

CoI's thesis is that **the *organisation* of retrieved literature matters more than its quantity**. Instead of dumping N retrieved papers into a prompt (vanilla RAG), CoI arranges a small set of papers into a **linear chain that mirrors the actual chronological/derivational evolution of a subfield** (e.g. `CoT → SC → ToT → GoT`), makes the LLM explicitly articulate *what changed between each adjacent pair*, and then asks it to **extrapolate the next step in the chain**. This turns ideation into trend extrapolation rather than blind recombination.

Three stages: **(1) CoI Construction → (2) Idea Generation → (3) Experiment Design**, plus **Idea Arena**, a pairwise/ELO evaluation protocol that replaces unreliable Likert scoring.

---

## 2. Problem & Motivation (why this design)

**Failure mode they attack.** When an LLM is given a big undifferentiated pile of retrieved papers, it produces *logically incoherent* ideas. Their worked example: the model borrows from **GraphGPT** (a *fine-tuning* method using GNNs) and grafts it onto **GoT** (a *prompting* method) — a category error, because nothing in the flat context told the model these live at different levels of the stack.

**Human analogy they encode.** Real researchers trace a field from foundational work → most recent advances, understand *why* each step was taken, then identify what's missing. Three claimed advantages of the chain structure:

1. **Noise reduction** — careful selection avoids interference from marginally-relevant literature.
2. **Implicit few-shot demonstration** — each `I_{k} → I_{k+1}` transition is a worked example of "how a novel idea is derived from a prior one" (SC derived from CoT). This is few-shot prompting of the *ideation act itself*.
3. **Global progression signal** — the model sees the *direction and velocity* of a field, not just its contents, which is what enables forward extrapolation.

---

## 3. Architecture

```
Topic
  │
  ├─ LLM generates K queries (K perspectives of the topic)
  │
  ▼
STAGE 1 — CoI CONSTRUCTION  (×K branches, in parallel)
  ├─ retrieve anchor paper P_k^0 (top-ranked for query q_k)
  ├─ FORWARD extension:  P^0 → P^1 → … → P^N   (via citing papers)
  ├─ BACKWARD extension: P^0 → P^-1 → … → P^-M (via references)
  ├─ extract per paper: Idea, Experiment, Entities, References
  └─ summarise Trends between every adjacent pair
  │
  ▼
STAGE 2 — IDEA GENERATION  (per branch)
  ├─ Future Trend Prediction  (Reflection / Analogy / Deep-Dive modes)
  ├─ Idea Consolidation → Motivation, Novelty, Method, Final Idea
  ├─ Novelty-check agent → if too similar, regenerate (with bad case fed back)
  └─ cross-branch pairwise tournament → pick highest win-rate idea
  │
  ▼
STAGE 3 — EXPERIMENT DESIGN
  ├─ generate design (few-shot on extracted experiments from the chain)
  ├─ review agent critiques (clarity / completeness / feasibility)
  ├─ derive search query from critique → retrieve papers
  └─ refine design → final Experiment Design
```

### 3.1 Stage 1 — CoI Construction

A CoI is a sequence `{I_{-M} → … → I_0 → … → I_N}` of `M+N+1` ideas extracted from `M+N+1` papers, ordered so they display the evolution of the field.

**Step 1 — Multi-query decomposition.** The LLM converts the topic into `K` search queries representing `K` different *perspectives* on the topic (Prompt T7). Explicit instruction: "output one query representing the whole and other queries representing different aspects of the whole." Each query seeds an independent CoI branch — this hedges against a single chain landing on a poor trajectory.

**Step 2 — Anchor paper.** For each `q_k`, retrieve the top-ranked paper `P_k^0` (Semantic Scholar). The anchor is the pivot from which the chain grows in both directions.

**Step 3 — Forward extension (tracing progression).**
- Get papers that **directly cite** `P_k^i` via the Semantic Scholar API.
- Rank candidates by **cosine similarity of `text-embedding-3-large` embeddings** against `concat(initial research topic, abstract of the anchor paper)`. *Note: the query vector is anchored on the original anchor abstract + topic, not on the current chain tip — this keeps the chain from drifting.*
- Take the top-ranked paper as `P_k^{i+1}`. Repeat.
- **Termination:** chain reaches preset max length, **or** the LLM judges there is no valuable follow-up work (relevance judge, Prompt T8).

**Step 4 — Backward extension (tracing origins).** More elaborate than forward. The LLM reads the **full paper** and nominates candidate references under three criteria:
1. references the paper **directly built upon**,
2. references used as **baselines**,
3. references tackling the **same topic**.

Then a second LLM call picks the single **most relevant** candidate as `P_k^{-1}` (Prompts T9/T10). Iterate backwards.
- **Termination:** preset max length, **or** a **milestone paper is reached — defined as >1,000 citations** (a natural, well-understood starting point for the chain), **or** no topic-relevant reference is found.

**Step 5 — Idea extraction & inheritance.** For each paper in each chain, extract a structured record (Prompts T9/T10):
- **Entities** — model names, datasets, metrics, specialised terminology, each `name + brief description`.
- **Idea** — decomposed into five fields: `Background`, `Novelty`, `Contribution`, `Detail Reason`, `Limitation`.
- **Experiment** — five fields: `Experimental Process`, `Technical Details`, `Clarity of Plan`, `Baseline`, `Verification`.
- **References** — the three most relevant onward references, titles only, no authors/years, sorted by relevance, ties broken toward newer.

The paper chain's ordering is inherited by the idea chain.

**Step 6 — Trend summarisation.** For **every adjacent pair** in the chain, the LLM writes an explicit transition statement: how Paper *i* leads to Paper *i+1*, what module/design was added, and *why it addresses the previous challenge* (Prompt T11). Output format is literally `Paper 0 to Paper 1: …`. **This is the paper's highest-value artefact** — the ablation shows removing the chain (and therefore these transitions) is the single most damaging change.

### 3.2 Stage 2 — Idea Generation

**Step 1 — Future Trend Prediction** (Prompts T12/T13). Inputs: the CoI, the trend transitions, and the extracted entities. The prompt does two unusual things:

- It asks the model to **model the researchers' reasoning process**, not just the content: *"analyze how human experts think and transition from previous methods to subsequent approaches. Focus on their reasoning logic and the sources of their thought processes. Learn to emulate their reasoning patterns."*
- It mandates **three explicit modes of thinking**:
  1. **Reflection** — where does a method break down; what solution would fix it.
  2. **Analogy** — find a similar problem solved *in another field* and transplant its principles. Explicitly encourages cross-domain borrowing.
  3. **Deep Dive** — take an existing approach and ask which sub-component could be modified to be more rational/effective.

- Guardrails in the prompt: *"Each article's limitations are specific to that particular piece and should not be applied to others"*, and *"avoid introducing unfamiliar information, ensuring that the trends you present are both authentic and reasonable."*

**Step 2 — Idea Consolidation** (Prompts T14/T15). Step-by-step build rather than one-shot: `Motivation → Novelty → Method → Final Idea`. Notable requirements:
- Motivation must state prior work **and** the specific unsolved problem.
- Novelty must **name specific competing approaches** and list **at least three concrete contributions**.
- Method must give step-by-step methodology, per-module function, and rationale for *why* it fixes the prior challenge, **citing papers by title**.
- A `[Bad case]` slot feeds back previously-generated ideas that turned out too close to real papers: *"The following are examples of ideas you have proposed in the past that are similar to real papers. Please avoid this situation."* — this is the memory channel of the novelty loop.
- Style constraints: write for a reader with limited background; explain jargon; don't import concepts from uncertain fields.

**Step 3 — Novelty check** (Prompt T16). A separate agent retrieves related papers and compares them to the draft idea, outputting `Think / Similar: 0|1 / Summary of the idea / Similar paper id`. If `Similar = 1` (framed bluntly as plagiarism), another generation round is triggered, with the offending idea appended to `[Bad case]`.

**Step 4 — Cross-branch selection.** All `K` branch ideas compete in a **pairwise round-robin** using the same judging machinery as Idea Arena; highest win-rate idea proceeds to Stage 3.

### 3.3 Stage 3 — Experiment Design

A **generate → review → retrieve → refine** loop.

1. **Generate** (Prompt T17): few-shot on the `Experiment` records extracted from the chain's papers, plus the final idea and entities. Six required dimensions: experimental design (baselines, comparative methods, **ablation study design**, data-analysis criteria), implementation of technologies, **feasibility assessment** (technology availability, timelines, resources, personnel, plus risk mitigation), references with titles, **visual aids (pseudocode or flowchart)**, and plain language. Output is a numbered `Step1: / Step2: …` plan.
2. **Review** (Prompt T18): a review agent asks whether the experiment can actually *validate the idea*, flags confusing/infeasible procedures, dataset-construction ambiguity, and unexplained implementations. Constrained to `Suggestion: …` and explicitly forbidden from altering the idea itself.
3. **Query derivation** (Prompt T19): the agent decides whether it needs more literature to fix the critique, and if so emits **one search query** (else the empty string). The paper's own example: a critique about an unclear "dynamic knowledge-graph update" step yields the query `"dynamic knowledge graph update"`.
4. **Refine** (Prompt T20): regenerate the plan with retrieved papers + suggestions.

Iteration count set to **1** in the main experiments, purely for cost.

---

## 4. Implementation Details (directly reusable)

| Item | Value |
|---|---|
| Main LLM | GPT-4o (2024-05-13) |
| Full-paper reading / summarisation | GPT-4o-mini (2024-07-18) — chosen for price + good summarisation |
| Search engine | Semantic Scholar API |
| Embeddings for ranking citing papers | OpenAI `text-embedding-3-large` |
| Max CoI length | **5** (saturation point; see §7) |
| Number of CoI branches (width `K`) | **3** |
| Experiment-design self-refinement iterations | **1** (cost-saving) |
| Milestone-paper stop rule (backward) | **>1,000 citations** |
| Cost | **~$0.50** per idea + experiment design |
| Judge LLM | GPT-4o (highest human agreement of those tested) |

**Benchmark data construction.** 50 research topics extracted from **Hugging Face Daily Papers**, papers submitted **1 Aug – 15 Sep 2024** (deliberately after the LLM's data cutoff). 10 researchers picked papers matching their interests; GPT-4o extracted `topic / idea / experiment design` (Prompts T21–T23); researchers then validated the extracted topics. The extracted ideas/designs become the **"Real Paper"** upper-bound baseline.

---

## 5. Prompt Library (verbatim)

> Placeholders are shown as in the paper: `[Topic]`, `[Entities]`, `[Chain of ideas]`, `[Trend]`, `[Idea]`, `[Experiment]`, `[Suggestions]`, `[Bad case]`, etc.

### T7 — Topic → search queries (CoI construction)
```
You are a master of literature searching, tasked with finding
relevant research literature based on a specific topic.

Currently, we would like to study the following topic: [Topic]
Please provide the literature search queries you would use to
search for papers related to the topic and idea.
Each query should be a string and should be enclosed in double
quotes. It is best to output one query representing the whole and
other queries representing different aspects of the whole.

Output strictly in the following format:
Queries: ...
```

### T8 — Relevance judge (chain-extension gate)
```
You are an expert researcher tasked with evaluating whether a given
paper is relevant to our research topic based on its title and
abstract.

Below are the details of the paper you need to assess:
Title: [Title]
Abstract: [Abstract]
The topic is: [Topic]
If the paper title and abstract are related to the topic, output
1; otherwise, output 0. As long as you feel that this article has
reference value for your question, you can use it to help you study
the topic, it does not need to be completely consistent in topic.

Please follow the strict format below:
Think: ...
Relevant: 0/1
```

### T9 + T10 — Extract idea / experiment / entities / references from a paper
```
You are a scientific research expert, tasked with extracting and
summarizing information from provided paper content relevant to
the topic: [Topic]. Your deliverables will include pertinent
references, extracted entities, a detailed summary, and the
experimental design.

The topic you are studying is: [Topic] (Ensure that the references
are pertinent to this topic.)
Extraction Requirements:
Entities:
1. Identify unique entities mentioned in the paper, such as model
names, datasets, metrics, and specialized terminology.
2. Format the entities with a name followed by a brief
description.
3. Ensure all entities are relevant to the specified topic
([Topic]).
Summary Idea:
1. Background: Elaborate on the task's context and previous work,
outlining the starting point of this paper.
2. Novelty: Describe the main innovations and contributions of
this paper in comparison to prior work.
3. Contribution: Explain the primary methods used, detailing the
theory and functions of each core component.
4. Detail Reason: Provide a thorough explanation of why the
chosen methods are effective, including implementation details for
further research.
5. Limitation: Discuss current shortcomings of the approach.

Experimental Content:
1. Experimental Process: Detail the entire experimental
procedure, from dataset construction to specific steps, ensuring
clarity and thoroughness.
2. Technical Details: Describe any specific technologies
involved, providing detailed implementation processes.
3. Clarity of Plan: State your experimental plan concisely to
facilitate understanding without unnecessary complexity.
4. Baseline: Elaborate on the baseline used, comparative methods,
and experimental design, illustrating how these support and
validate the conclusions drawn.
5. Verification: Explain how your experimental design assists in
verifying the core idea and ensure it is detailed and feasible.
Relevance Criteria:
1. Method Relevance: References must directly correlate with the
paper's methodology, indicating improvements or modifications.
2. Task Relevance: References should address the same task, even
if methods differ, better have the same topic [Topic]
3. Baseline Relevance: References should serve as baselines for
the methods discussed in the paper.
4. Output Format: Provide references without author names or
publication years, formatted as titles only.
The paper content is as follows: [Paper content]
Please provide the entities, summary idea, experimental design,
and the three most relevant references (Sort by relevance, with
priority given to new ones with the same level of relevance, do not
reference the original paper.) based on the paper's content.
Note: Ensure the references are pertinent to the topic you are
studying: [Topic]. If there are no relevant references, output
[].

Now please output strictly in the following format:
Entities: ...
Idea: ...
Experiment: ...
References: ...
```

### T11 — Trend summarisation across the chain
```
You are a scientific research expert tasked with summarizing the
historical progression of research related to our current topic,
based on the literature we have reviewed.

Here are the entities you need to know : [Entities]
The topic you are studying is: : [Topic]
The literature from early to late: [Idea chain]
Your objective is to outline the historical evolution of the
research in light of current trends. Please follow these
requirements:
Analysis of Published Viewpoints: Examine the progression of ideas
across the identified papers. Detail how each paper transitions
to the next--for instance, how Paper 0 leads to Paper 1, and so
forth. Focus on understanding how Paper 1 builds upon the concepts
in Paper 0. Elaborate on specific advancements made, including
proposed modules, their designs, and the rationale behind their
effectiveness in addressing previous challenges. Apply this
analytical approach to each paper in the sequence.

Please present your findings in the following format:
Trends:
Paper 0 to Paper 1: ...
Paper 1 to Paper 2: ...
...
```

### T12 + T13 — Future trend prediction (the "three modes of thinking" prompt)
```
You are a scientific expert tasked with formulating a novel and
innovative research idea based on your comprehensive literature
review. Your objective is to propose a feasible approach that
could significantly advance the field.

Here are the entities you need to know : [Entities]
The literature you have studied is as follows: [Chain of ideas]
The following section delineates the progressive relationships
among the previously summarized research papers: [Trend]
Based on previous research, analyze how human experts think
and transition from previous methods to subsequent approaches.
Focus on their reasoning logic and the sources of their thought
processes. Learn to emulate their reasoning patterns to further
develop and guide your own research direction in a natural and
coherent manner.
Additionally, you are encouraged to adopt the following three modes
of thinking:

1. Reflection: Reflect on scenarios where a specific method
encounters significant challenges. Consider potential solutions
that could effectively address these issues, make the solutions
sounds reasonable, novel and amazing.
2. Analogy: Identify a specific problem you are currently
facing and research existing solutions that have successfully
tackled similar challenges. Explore these solutions and adapt
key principles and strategies to your situation. Think creatively
about how tools and approaches from other domains can be
re-imagined to devise a novel strategy for your issue. Encourage
you to actively explore methods in other fields to solve your
current problems.
3. Deep Dive: Some methods may present specific approaches to
addressing a particular problem. Consider whether there are
aspects that could be modified to enhance their rationale and
effectiveness.
Note:Each article's limitations are specific to that particular
piece and should not be applied to others. Carefully consider the
task at hand and analyze the potential issues you might encounter
if you proceed with your original approach, reflecting on the
challenges previously faced. Then, think critically about how to
address these issues effectively.
You are encouraged to apply human reasoning strategies to identify
future research directions based on prior studies. Aim for
in-depth analysis rather than mere integration of existing ideas.
Please avoid introducing unfamiliar information, ensuring that
the trends you present are both authentic and reasonable. Before
proposing any trends, take a moment to reflect on the principles
underlying the methods you're employing and assess their relevance
to your research area.
The future research direction should be related to the topic:
[Topic]
Please present the future research direction in the following
format:
Future direction: ...
```

### T14 + T15 — Idea generation / consolidation
```
You are a scientific expert tasked with formulating a novel and
innovative research idea based on your comprehensive literature
review. Your objective is to propose a feasible approach that
could significantly advance the field.
The following are examples of ideas you have proposed in the past
that are similar to real papers. Please avoid this situation as
much as possible. You can continue to make in-depth innovations,
but avoid plagiarism: [Bad case]
Here are the entities you need to know: [Entities]
The topic you are studying is: [Topic]
The literature you have studied is as follows: [Chain of ideas]
Your idea is composed of the following components:
Motivation:
1. Provide a background for your idea, summarizing relevant work.
2. Identify shortcomings in previous research and highlight the
specific problems that remain unsolved and that you aim to address.
Novelty:
1. Distinguish your proposed method from existing methods
(preferably by naming specific approaches).
2. Detail the improvements of your method compared to past work.
3. Clearly outline at least three contributions your idea offers
to the field, including the problems it resolves and the benefits
it delivers.
Method:
1. Present a detailed description of your idea, focusing on the
core method, the specific problem it solves, and enhancements over
earlier research (citing relevant literature with titles).
2. Explain the step-by-step methodology, including the functions
of each module and the rationale for why this approach effectively
addresses previous challenges.
Please adhere to the following guidelines:
1. Your research idea should be innovative, feasible, and
contribute meaningfully to the field. Please carefully examine
the idea you have proposed, avoid immediate perception, and try to
be different from the previous methods as much as possible.
2. Ensure your proposal is solid, clearly defined, and practical
to implement. Logic should underpin your reasoning.
3. Write in clear, concise language aimed at an audience with
limited background knowledge in the subject. Avoid complex
technical jargon, but when professional terms are necessary,
provide thorough explanations.
4. Refrain from introducing concepts from uncertain fields to
prevent proposing ideas that may be incorrect or impractical.
5. When referencing other research, please include the titles of
the cited papers.
6. Please avoid introducing unfamiliar information, ensuring that
the trends you present are both authentic and reasonable. Before
proposing any trends, take a moment to reflect on the principles
underlying the methods you're employing and assess their relevance
to your research area.
7. Each article's limitations are specific to that particular
piece and should not be applied to others. Carefully consider the
task at hand and analyze the potential issues you might encounter
if you proceed with your original approach, reflecting on the
challenges previously faced. Then, think critically about how to
address these issues effectively.
The following section delineates the progressive relationships
among the previously summarized research papers: [Trend]
The following section outlines the potential future research
directions based on the literature you have studied: [Future
direction]
Please output your motivation,novelty,method firstly and then
output your final idea.The final idea should clearly explain the
origins, motivation, and challenges of your idea, detailing how you
overcame these hurdles.

Please present the final idea in the following format:
Motivation: ...
Novelty: ...
Method: ...
Final idea: ...
```

### T16 — Novelty check
```
You are a scientific research expert tasked with evaluating the
similarity between a specified idea and existing research. Your
objective is to determine if the target idea closely resembles any
findings in the provided papers.
The target idea you need to check is as follows: [Idea]
The relevant papers you need to refer to are as follows:[Content of
retrieved papers]
Here are your guidelines:
1. Comparison Process: Begin by thoroughly comparing each
paper's ideas with the target idea. Consider the methodologies,
conclusions, and underlying concepts in each paper in your
analysis.
2. Similarity Assessment: If the target idea shares fundamental
similarities with any existing research to the extent that they can
be considered identical, classify this as plagiarism.
3. Output: Your output should provide a clear thought process,
the similarity assessment, a summary of the target idea, and the ID
of the most relevant similar paper.
Please output strictly in the following format:
Think: ...
Similar: 0/1
Summary of the idea: ...
Similar paper id: 0 to n
```

### T17 — Experiment design generation
```
You are a scientific expert tasked with designing rigorous,
feasible experiments based on specified scientific questions
and the methodologies derived from the idea I provide, along
with relevant past research. Your goal is to assist researchers
in systematically testing hypotheses and validating innovative
discoveries that could significantly advance their fields.

Past Related Research Experiments: [Past experiments]
Here are the entities you need to know: [Entities]
Here is the idea you need to design an experiment for: [Idea]
Please propose a detailed experimental plan addressing the
following points:
1. Experimental Design: Develop rigorous experiments to
ensure the reliability and validity of your results. Provide
a comprehensive explanation of the baseline used, comparative
methods, ablation study design, and criteria for data analysis
and result evaluation. Clarify how these components collectively
reinforce and validate the conclusions of your research. Structure
your experimental design in a clear, logical, and step-by-step
manner, ensuring each step is well-defined and easy to understand.
2. Implementation of Technologies/Methods: If your experimental
design involves specific technologies or methodologies, describe
the implementation process in detail, including key technical
aspects. For any critical concepts utilized, provide thorough
explanations. For instance, if you propose a modular approach,
detail its construction, components, and functionality.
3. Feasibility Assessment: Ensure your experimental plan is
realistic, considering technological availability, timelines,
resources, and personnel. Identify potential challenges and
propose strategies for addressing them.
4. References to Previous Studies: When citing related
literature, include titles and pertinent details of the original
papers. Strive to use as many references as necessary to support
your experimental design.
5. Visual Aids: If useful, provide pseudo code or a flowchart to
illustrate the implementation process. For example, you can use
pseudo code to detail the core algorithm or the model architecture,
or employ a flowchart to map out the experimental procedure and
data flow.
6. Clarity of Language: Use straightforward language to describe
your methods, assuming the reader may have limited knowledge of
the subject matter. Avoid complex jargon and utilize accessible
terminology. If professional terms are necessary, please provide
clear and detailed explanations.

Please output strictly in the following format:
Experiment:
Step1: ...
Step2: ...
...
```

### T18 — Experiment review agent
```
You are an expert in paper review. Your task is to analyze whether
a given experiment can effectively verify a specific idea, as well
as assess the detail and feasibility of the experiment.

Here are the related entities you need to know: [Entities]
The idea presented is: [Idea]
The corresponding experiment designed for this idea is:
[Experiment]
Please conduct your analysis based on the following criteria:
1. Can the experiment validate the idea? If not, identify
the issues and suggest improvements to enhance its verification
capability and feasibility.
2. Are there specific experimental procedures that are confusing
or poorly designed? Discuss any methods that may not be feasible,
uncertainties in constructing the dataset, or a lack of explanation
regarding the implementation of certain methods.
3. Evaluate the clarity, detail, reasonableness, and feasibility
of the experimental design.
4. Provide suggestions for improving the experiment based on the
shortcomings identified in your analysis.
5. Focus solely on the experiment design; please refrain from
altering the original idea.
6. Ensure that your suggestions are constructive, concise, and
specific.

Please strictly follow the following format for output:
Suggestion: ...
```

### T19 — Derive a retrieval query from the critique
```
You are a research expert tasked with refining and improving an
experimental plan based on the feedback received.

The experimental plan you proposed is as follows: [Experiment]
You have received the following suggestions for improvement:
[Suggestions]
Please decide whether you need to search for relevant papers to
obtain relevant knowledge to improve your experiment.
If you need to search for relevant papers, please provide a search
query for literature search, else provide "".
For example: if suggestions say that the dynamic query additional
information and update knowledge graph described in the experiment
is not clearly described, so you need to output "dynamic knowledge
graph update".

Please output strictly in the following format:
Query:...
```

### T20 — Experiment refinement
```
You are a research expert tasked with refining and improving an
experimental plan based on the feedback received.

The information of the literature you maybe need to refer to are
as follows: [Searched paper information]
The experimental plan you proposed is as follows: [Experiment]
Please propose a detailed experimental plan addressing the
following points:
  [ ... identical six-point rubric as T17: Experimental Design,
    Implementation of Technologies/Methods, Feasibility Assessment,
    References to Previous Studies, Visual Aids, Clarity of Language ... ]
You have received the following suggestions for
improvement:[Suggestions]
Please refine your experimental plan based on the feedback
provided. Ensure your refined plan is feasible, clearly defined,
and addresses the feedback you received.

Please output strictly in the following format:
Experiment: ...
```

### T21 — Extract topic from a real paper (benchmark construction)
```
You are a research expert tasked with extracting the main topic
from the provided paper information.

The main topic should encompass broad fields such as "Retrieve
augment generation" or "using diffusion models for video
generation". However, it should also include a relevant task to
the topic, formatted as "topic:... task:...".
Please read the provided paper and extract only the topic, which
should follow this structure.
The paper's title is [Title]
The paper's abstract is as follows: [Abstract]
The paper's introduction is as follows: [Introduction]

Please output strictly in the following format:
topic: ...
```

### T22 — Extract idea from a real paper
```
You are a research expert tasked with extracting the main idea from
the provided paper information.

The main idea should encompass the motivation, solved problem,
novelty, method of the paper.
Please read the provided paper and extract the main idea from the
paper.
The paper content is as follows: [Content]
Idea is composed of the following components:
Motivation: Explain the background of the idea and past related
work, identify the shortcomings of past work, identify the problems
that need improvement, and identify the issues the paper want to
address.
Novelty: Explain the differences between the method and the
current method (preferably list specific methods), explain what
improvements the paper have made to the previous method, and then
identify the problems that can be solved and the benefits that can
be gained from these improvements.
Method: Provide a detailed description of your idea, including the
core method, the problem it solves, and the improvement compared
with previous work(Cite the previous work with the title of the
paper). Explain the specific steps of the method, the specific
functions of each module, and the specific reasons why this method
can solve the previous problem.
Here are some tips for extracting the main idea:
1. Make idea easy to understand, use clear and concise language
to describe, assuming the reader is someone who has few knowledge
of the subject, avoid using complex technical terms, and try to
use easy-to-understand terms to explain.If the paper use some
professional terms, please explain them in detail.
2. When the paper cite other papers, please indicate the title of
the original paper.
The final idea should be detailed and specific, clearly explain
the origins, motivation, novelty, challenge, solved problem and
method of the paper, and detail how the overcame these hurdles.
Ensure your approach is innovative, specifying how this innovation
is reflected in your experimental design.
The final idea should be double-blind, i.e. no experimental
results or codes should be shown.

Please output strictly in the following format:
Final idea: ...
```

### T23 — Extract experiment from a real paper
```
You are a research expert tasked with extracting the specific
experiment steps from the provided paper information.

The specific experiment steps should include the specific methods
for each step.
Please read the provided paper and extract specific experiment
steps from the paper.
The paper content is as follows: [Content]
There are some tips for extracting the experiment steps:
1. Detail the Experimental Process: Describe the entire
experimental process, including how to construct the dataset and
each specific experimental step. Ensure that each experimental
method is clearly and thoroughly detailed.
2. If specific technologies are involved in the experimental
design, describe the implementation process in as much detail as
possible (i.e., technical details)
3. Make sure your experimental plan is concise and clear, and can
be easily understood by others,should not be too complicated.
4. Please provide a detailed explanation of the baseline used in
the paper, the comparative methods, the ablation design and the
experimental design. Specifically, elaborate on how these elements
collectively support and validate the conclusions drawn in your
research.
5. Explain how your experimental design can help you verify the
idea and how the experiment is detailed and feasible.

Now please output strictly in the following format:
Experiment:
Step1: ...
Step2: ...
...
```

### T24 — Idea Arena judge (pairwise idea comparison)
```
You are a judge in a competition. You have to decide which idea is
better.

The idea0 is: [idea0]
The idea1 is: [idea1]
The topic is: [topic]
Which idea do you think is better? Please write a short paragraph
to explain your choice.
Here are your evaluation criteria:
1. Novelty: Are the problems or approaches new? Is this a novel
combination of familiar techniques? Is it clear how this work
differs from previous contributions? Is related work adequately
referenced?
2. Significance: Are the idea important? Are other people
(practitioners or researchers) likely to use these ideas or build
on them? Does the idea address a difficult problem in a better way
than previous research? Does it provide a unique theoretical or
pragmatic approach?
3. Feasibility: Can the idea be realized with existing technology
or methods? Are there any technical difficulties or bottlenecks?
Is the idea clear and logical? Is there any obvious error or
unreasonable part in the idea, and can the experiment be designed
normally according to this idea.
4. Clarity: Is the paper clearly written? Is it well-organized?
Does it adequately inform the reader?
5. Effectiveness: How likely the proposed idea is going to work
well (e.g., better than existing baselines).
Note:
Avoid any position biases and ensure that the order in which
the responses were presented does not influence your decision.
DO NOT allow the LENGTH of the responses to influence your
evaluation, choose the one that is straight-to-the-point instead
of unnecessarily verbose. Be as objective as possible. (very
important!!!)
If you think idea0 is better than idea1, you should output 0. If
you think idea1 is better than idea0, you should output 1. If you
think idea0 and idea1 are equally good, you should output 2.

Your output should be strictly in following format:
Your thinking process: ...
Your choice:
Novelty: 0/1/2
Significance: 0/1/2
Feasibility: 0/1/2
Clarity: 0/1/2
Effectiveness: 0/1/2
```

### T25 — Idea Arena judge (pairwise experiment comparison)
```
You are a judge in a competition. You have to decide which
experiment is better.

The idea of experiment0 is: [idea0]
The experiment0 is: [experiment0]
The idea of experiment1 is: [idea1]
The experiment1 is: [experiment1]
Which experiment do you think is better? Please write a short
paragraph to explain your choice.
Here are your evaluation criteria:
1. Feasibility: Can the experiment be realized with existing
technology or methods? Are there any technical difficulties or
bottlenecks? Is the experimental plan detailed and feasible? Are
the experimental steps clear and logical? Is there any obvious
error or unreasonable part in the experiment. Consider the
rationality of its steps and the possibility that the idea can be
successfully implemented.
2. Quality: Is there a clear rationale for each step of the
experimental design? Are the baseline and evaluation metrics
chosen appropriately? Has the design taken into account the
potential advantages and limitations of the methods used? Can
this experimental design effectively support the claims made in the
idea.
3. Clarity: Is the experimental plan clearly written? Dose it
provide enough information for the expert reader to understand the
experiment? Is it well organized? Does it adequately inform the
reader?
Note: Avoid any position biases and ensure that the order in which
the responses were presented does not influence your decision.
DO NOT allow the LENGTH of the responses to influence your
evaluation, choose the one that is straight-to-the-point instead
of unnecessarily verbose. Be as objective as possible. (very
important!!!)
If you think experiment0 is better than experiment1, you should
output 0. If you think experiment1 is better than experiment0,
you should output 1. If you think experiment0 and experiment1 are
equally good, you should output 2.

Your output should be strictly in following format:
Your thinking process: ...
Your choice:
Feasibility: 0/1/2
Quality: 0/1/2
Clarity: 0/1/2
```

---

## 6. Idea Arena — the evaluation protocol

**Why not Likert.** Prior work (ResearchAgent, AI-Scientist) scores ideas on a 5-point Likert scale. Si et al. (2024) showed this aligns poorly with human preference; LLMs are much better at **ranking** than at **absolute scoring**. CoI therefore builds a tournament.

**Mechanics.**
- **Round-robin, pairwise**: every method competes against every other method on every topic.
- **Order-reversal**: each pair is judged **twice with order swapped** to cancel position bias.
- Win/loss/tie records → **ELO scores** (following Zheng et al. / Zhao et al.).
- The judge prompt explicitly warns against position bias *and* length bias.
- The **same machinery is reused internally** for selecting among the `K` branch ideas.

**Idea criteria** (from ICML 2020 reviewer guidelines + Si et al.): Novelty, Significance, Clarity, Feasibility, Expected Effectiveness.
**Experiment-design criteria**: Feasibility, Technical Quality, Clarity.

**Metric definitions (Table 5 / 6):**

| Metric | Definition |
|---|---|
| Novelty | Are the problems or approaches new? Is this a novel combination of familiar techniques? Is it clear how this work differs from previous contributions? Is related work adequately referenced? |
| Significance | Are the ideas important? Are others likely to use or build on them? Does the idea address a difficult problem in a better way than previous research? Does it provide a unique theoretical or pragmatic approach? |
| Clarity | Is it clearly written? Well-organized? Does it adequately inform the reader? |
| Feasibility | Can the idea be realized with existing technology or methods? Any bottlenecks? Any obvious error or unreasonable part, and can the experiment be designed normally from this idea? |
| Expected Effectiveness | How likely the proposed idea is to work well (e.g. better than existing baselines). |
| Exp. Feasibility | Can the experiment be realized with existing technology? Is the plan detailed and feasible? Are steps clear and logical? |
| Exp. Quality | Clear rationale for each step? Baselines and metrics appropriate? Does the design account for advantages/limitations? Can it support the idea's claims? |
| Exp. Clarity | Is the plan clearly written, sufficiently informative for an expert reader, well organized? |

**Judge validation.** Human↔GPT-4o agreement = **70.8%** average (Novelty 66.5, Significance 71.0, Clarity 76.3, Feasibility 70.2, Effectiveness 71.0). GPT-4o had the highest human agreement among GPT-4o / Gemini-1.5-Pro / Claude-3.5-Sonnet, so GPT-4o was used as judge. Model-model agreement is much higher (~90–93%) than model-human (~70%) — worth remembering: **LLM judges agree with each other far more than with humans.**

---

## 7. Results & Ablations (the empirically-grounded design lessons)

**Main ranking (human evaluation, ELO):** Real Paper 1100 > **CoI Agent 1085** > RAG 1029 > GPT-Researcher 992 > ResearchAgent 980 > AI-Scientist 812.
**GPT-4o judge:** Real Paper 1115 > **CoI 1107** > GPT-Researcher 999 > ResearchAgent 995 > RAG 950 > AI-Scientist 833.

- CoI **beats Real Paper on Novelty and Significance** in human eval (1100 vs 1075; 1103 vs 1071) but **loses on Clarity, Feasibility, Effectiveness** — the human-written ideas have been through actual experimental validation.
- **Feasibility is the standing bottleneck** for all automated methods, in both idea and experiment-design phases. The paper names this explicitly as the open problem.
- AI-Scientist scores poorly because it was designed to go from executable code → full paper; given only a topic, its thin ideation stage underperforms.

**Ablation (each variant scored against full CoI over 50 matches, 2/1/0 for win/tie/loss; full agent = 50 by definition):**

| Variant | Novelty | Significance | Clarity | Feasibility | Effectiveness | Avg |
|---|---|---|---|---|---|---|
| **CoI Agent (full)** | 50 | 50 | 50 | 50 | 50 | **50** |
| – CoI (flat retrieval, no chain) | 41 | 39 | 44 | 49 | 39 | **42.4** |
| – Future Trend prediction | 40 | 43 | 51 | 53 | 44 | **46.2** |
| – Entities | 46 | 49 | 42 | 47 | 43 | **45.4** |

Readings:
- **Removing the chain hurts most** → the progressive organisation, not the volume of papers, is the load-bearing component.
- **Removing Future Trend prediction costs novelty** (40) and significance while slightly *raising* clarity/feasibility — i.e. extrapolation buys novelty at some cost in groundedness. A tunable knob.
- **Removing entities costs clarity and effectiveness** (42, 43) — without concrete datasets/models/metrics named, ideas drift abstract. Entities are the *specificity* channel.

**Length of chain (§4.5).** ELO rises steeply from length 0 → 3, keeps improving to **5**, then **plateaus**. Quote worth keeping: *"a clear developmental trend analysis is more pivotal than the quantity of related literature."* Length 5 is the recommended operating point.

**Width / branches (§4.6).** More branches correlate positively with quality, but the effect is **small** — branching mainly insures against a single bad chain, and bad chains are rare. `K = 3` is enough.

**Case study (§4.3).** On the topic "using LLM agents to generate novel research ideas", CoI built the chain `Kim 2021 (molecular generation) → Boiko 2023 (autonomous experiments) → Yang 2024 (MOOSE hypothesis generation) → Baek 2024 (ResearchAgent) → Si 2024 (LLM vs 100 NLP researchers)`, identified from the trends that LLM ideation is limited in *diversity and novelty*, and proposed **EvoResearchAgent** — an evolutionary multi-agent system with Idea Initialize → automatic topic-diversity/novelty metrics → Selection → Crossover → Mutation → Iteration.

---

## 8. What to take into our Research Agent

Mapped to the components in *Architecture Overview and Project Plan*:

| Our component | What CoI gives us |
|---|---|
| **C2 — Literature Acquisition** | Multi-query decomposition (K perspectives) instead of one query. Anchor-paper + **bidirectional citation-graph walk** (forward = citing papers, backward = LLM-selected references under 3 criteria). Embedding re-rank against `topic + anchor abstract` to prevent drift. Stop rules: max length, LLM relevance gate, **>1,000-citation milestone**. |
| **C3 — Research Knowledge Layer** | Their per-paper schema is directly compatible with ours and richer in places: `Entities` (name + description), `Idea{Background, Novelty, Contribution, Detail Reason, Limitation}`, `Experiment{Process, Technical Details, Plan, Baseline, Verification}`, `References` (titles only, top-3, relevance-sorted). **Add the chain/trend edge type**: an explicit `paper_i → paper_{i+1}` transition record saying *what changed and why*. Our docx graph has `proposes/uses/evaluates/limitation` edges but **no temporal-derivation edge** — that is precisely the edge CoI shows to be the most valuable one. |
| **C4 — Gap / Novelty Detection** | The **novelty-check agent** (T16) with a `[Bad case]` memory that feeds rejected ideas back into the next generation round — a cheap, effective regeneration loop. Their *Trends* output is essentially a narrative form of our "research landscape". Their Future-Trend module (Reflection / Analogy / Deep-Dive) is a concrete, promptable operationalisation of "where hasn't this been combined". |
| **C5 — Experiment Planning** | The generate → review → **query-derivation** → refine loop (T17–T20) is directly reusable; the query-derivation step (T19) is the clever bit — the critique itself becomes a retrieval query, so the refinement is grounded in *new* literature rather than re-reasoning over the same context. Their T17 rubric already demands ablation design, feasibility assessment with risk mitigation, and pseudocode/flowchart — a good skeleton for our Experiment Plan structure. |
| **C-Eval — Evaluation** | **Adopt Idea Arena wholesale** rather than Likert scoring: round-robin pairwise, both orders, ELO, explicit anti-position/anti-length instructions in the judge prompt, and report human↔judge agreement to justify the automated numbers. Their baseline set (RAG / ResearchAgent / GPT-Researcher / AI-Scientist / **Real Paper**) maps almost 1:1 onto our Baselines 1–4 — and the **Real Paper upper bound** (extract topic+idea+experiment from a held-out post-cutoff paper) is a strong addition our plan does not currently have. |

**Cheap wins to steal first:** (1) the trend-transition edge; (2) `[Bad case]` feedback in the novelty loop; (3) critique → search-query → refine for the experiment plan; (4) ELO arena + Real-Paper upper bound; (5) post-cutoff paper selection so the LLM cannot have memorised the answer.

---

## 9. Gaps, Limitations & Future Scope

### Stated or evidenced by the paper
1. **Feasibility is the bottleneck.** Explicitly named: automated ideas and designs lag human ones on feasibility in both phases. The paper offers no mechanism to fix it and flags it as future work.
2. **No execution, no empirical validation.** CoI stops at a design on paper. Ideas are never run, so "Expected Effectiveness" is a judged guess. The gap to Real Paper on Clarity/Feasibility/Effectiveness is attributed precisely to real papers having survived experiments.
3. **Diminishing returns past length 5 and width 3** — the architecture saturates; more literature does not help further.
4. **Cost/latency trade-offs are hard-coded**: experiment-design refinement fixed at 1 iteration purely for budget.
5. **Evaluation is LLM-judged at scale.** Human agreement is only ~70%, while LLM-LLM agreement is ~90%+ — the automated arena has a systematic, self-consistent bias humans do not share.

### Gaps we can attack (our project's opening)
1. **A single linear chain is a lossy model of a research field.** Real progress is a DAG with merges (two lines converging), forks, and dead ends. CoI's `K` branches are *independent* chains — never reconciled or intersected. **A landscape graph with cross-branch edges (our C3/C4) is a direct generalisation of CoI**, and one they never attempted.
2. **The chain is greedy and unbacktracked.** Forward extension always takes the single top-1 embedding-ranked citing paper; one bad hop corrupts the rest of the chain. No beam search, no backtracking, no re-ranking of the chain as a whole.
3. **Novelty is a binary similarity verdict.** T16 outputs `Similar: 0/1` from a handful of retrieved papers. That is exactly the trap our docx warns about — *"if the agent finds no prior work, that does not mean the idea is novel; the search may simply have failed."* CoI has **no notion of retrieval coverage or confidence**. Our **5-level potential-novelty scale + explicit "no matching approach found in searched corpus" evidence line** is a genuine improvement over the state of this paper.
4. **No provenance.** Ideas cite paper *titles* inside free text at best. There is no traceable link from a recommendation → reasoning → evidence → paper → section, and no way to compute anything like Provenance Coverage. **This is the largest unclaimed space, and it is the core of our C5.**
5. **Entities are extracted per paper but never unified.** No canonicalisation, no cross-paper entity store, no co-occurrence structure (contrast ResearchAgent, which builds exactly that). CoI gets specificity but not cross-domain reach.
6. **Backward extension needs full-paper reads** (GPT-4o-mini over whole PDFs), which is the expensive, fragile part of the pipeline — a place where a structured knowledge layer like our C3 would pay for itself by reading each paper once.
7. **Domain-bound.** Everything is validated on AI/ML topics from HF Daily Papers. The milestone heuristic (>1,000 citations), the citation-graph density, and even "papers cite forward within months" are all AI-specific assumptions.
8. **No user documents.** Purely retrieval-driven; there is no path for a researcher's own uploaded papers or unpublished context to enter the chain — our C1 explicitly wants this.
9. **Single-shot topics only.** No interaction, no clarification, no researcher-in-the-loop steering between stages, despite the fact that Feasibility — the weakest axis — is exactly what a human could correct cheaply.

