# Fact or Fiction (SciFact / VeriSci) — Extraction Notes

> **Paper:** *Fact or Fiction: Verifying Scientific Claims*
> **Authors:** David Wadden, Shanchuan Lin (UW), Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan (AI2), Hannaneh Hajishirzi (UW/AI2)
> **arXiv:** 2004.14974v6 [cs.CL], 3 Oct 2020 (EMNLP 2020)
> **Dataset:** **SciFact** — 1,409 claims / 5,183 abstracts · **Model:** **VeriSci**
> **Code/data:** https://github.com/allenai/scifact · **Leaderboard + COVID demo:** https://scifact.apps.allenai.org
> **Note:** the oldest paper in this set (2020, pre-LLM). Its value to us is **not** the model — a BERT pipeline superseded many times over — but the **task formalisation, the rationale annotation protocol, the two-granularity evaluation metrics, and the taxonomy of scientific reasoning failures.** Those are exactly what our Provenance Evaluation (§4.2) needs and they have aged well.

---

## 1. TL;DR

This paper defines **scientific claim verification**: given a claim and a corpus of abstracts, (a) retrieve the abstracts containing evidence, (b) label each as **SUPPORTS / REFUTES / NOINFO**, and (c) **identify the specific sentences (rationales) that justify each decision.**

That third requirement is the whole point, and it is the thing our project needs. Our docx §4.2 lists *citation correctness* ("does the cited paper actually support the claim?") and *traceability* ("can a researcher trace a recommendation back through reasoning, evidence, source paper, and the relevant section?"). **SciFact is the task formulation, dataset, and metric suite for exactly that question**, and it is what makes those metrics measurable rather than aspirational.

The motivating example (Figure 1) shows why scientific verification is harder than political fact-checking:

```
Claim:     Cardiac injury is common in critical cases of COVID-19.
Rationale: More severe COVID-19 infection is associated with higher mean troponin
           (SMD 0.53, 95% CI 0.30 to 0.75, p < 0.001)
Verdict:   SUPPORTS
```

To get this right the system must (1) know **troponin is a cardiac protein and elevated troponin marks cardiac injury** — background knowledge never stated in the text; (2) reason about **direction** — swapping *higher* for *lower* flips SUPPORTS to REFUTES; (3) interpret **p < 0.001** as statistical significance.

---

## 2. Task Definition (the formalism, worth adopting)

### Definitions

**Scientific claim:** *"an atomic verifiable statement expressing a finding about one aspect of a scientific entity or process, which can be verified from a single source."*
- Valid: *"The R0 of the novel coronavirus is 2.5"*
- Invalid (opinion): *"The government should require people to stand six feet apart"*
- **Compound claims must be split** — *"Aerosolized coronavirus droplets can travel at least 6 feet and can remain in the air for 3 hours"* → two atomic claims.

> This is the same **atomicity** requirement SciGraph-LLM enforces, arrived at independently. Two papers, six years apart, both concluded that a checkable unit of scientific knowledge must be a single proposition. Worth treating as settled.

**Rationale:** *"a minimal collection of sentences which, taken together as premises in the context of the abstract, can reasonably be judged by a domain expert as implying the claim."*
- Purpose: *"facilitate the development of interpretable models which not only make label predictions, but can also identify the exact sentences necessary for their decisions."*

**A deliberate scoping decision we should copy:** they do *not* assign a global truth label to a claim, because *"accurately assigning a global truth label to a scientific claim requires a systematic review by a team of experts."* Instead they label **individual claim–abstract pairs**.

> This is the same epistemics as our docx's *"the final output is always framed as potential novelty, never proven novelty."* A per-pair relation is defensible; a global verdict is not. And note that **the same claim can be both supported and refuted by different abstracts** — observed on real COVID-19 claims (§6.3, Table 1). Our C4's "Evidence" field should be able to represent contradictory evidence, not just supporting evidence.

### Formal task

```
Inputs:   claim c, corpus of abstracts A
Labels:   y(c,a) ∈ {SUPPORTS, REFUTES, NOINFO}
E(c)      = evidence abstracts (those that SUPPORT or REFUTE c)
R(c,a)    = {R₁(c,a), …, Rₙ(c,a)}    set of rationales
Rᵢ(c,a)   = {r₁(c,a), …, r_m(c,a)}   one rationale = a set of sentences

System must predict:  Ê(c),  ŷ(c,a),  Ŝ(c,a) = {ŝ₁, …, ŝ_ℓ}
```
Note the simplification: gold annotations may contain multiple separate rationales, but the model predicts **one flat collection of rationale sentences** which may span several gold rationales.

---

## 3. Evaluation Metrics (the most reusable part of the paper)

Two granularities × two strictness levels. **This 2×2 is a ready-made design for our provenance metrics.**

### Abstract-level (inspired by the FEVER score)

A predicted evidence abstract `a ∈ Ê(c)` is:
- **correctly labeled** if (1) `a` is a gold evidence abstract for `c`, and (2) `ŷ(c,a) = y(c,a)`.
- **correctly rationalized** if, in addition, the predicted rationale sentences **contain a gold rationale**: `∃ Rᵢ(c,a) ⊆ Ŝ(c,a)`.

Reported as micro-F1 → **Abstract_Label-Only** and **Abstract_Label+Rationale**.
Predicted rationale sentences are **capped at 3** (FEVER caps at 5).

### Sentence-level

A predicted rationale sentence `ŝ(c,a)` is:
- **correctly selected** if (1) it belongs to some gold rationale `Rᵢ(c,a)`, (2) **all other sentences from that same gold rationale are also predicted**, and (3) `ŷ(c,a) ≠ NOINFO`.
- **correctly labeled** if, in addition, `ŷ(c,a) = y(c,a)`.

Reported as micro-F1 → **Sentence_Selection-Only** and **Sentence_Selection+Label**.
**No cap** here, because *"the evaluation penalizes models that over-predict."*

Condition (3) is subtle and smart: a rationale sentence identified by the selector but which **proved insufficient to justify a SUPPORTS/REFUTES decision** does not count. Finding text is not the same as finding evidence.

### Two retrieval settings
- **"Oracle abstract"** (ERASER-style) — gold evidence abstracts provided; isolates reasoning from retrieval.
- **"Open"** (FEVER-style) — abstracts must be retrieved from the corpus; the realistic setting.

> **Direct mapping to our §4.2 metrics:**
> - *Citation correctness* ≈ **Abstract_Label+Rationale** — the cited source is right *and* the specific sentences support the stated relation.
> - *Citation completeness* ≈ **recall** at both levels.
> - *Provenance consistency* ≈ the **Selection-Only vs Selection+Label gap** — the model found the right text but got the relationship wrong.
> - *Traceability* ≈ **sentence-level** evaluation, since it is the only one that checks you can point at the exact sentences.
>
> And the **oracle-vs-open split is how to attribute our own errors**: is the agent wrong because it retrieved the wrong papers, or because it reasoned wrongly over the right ones?

---

## 4. Dataset Construction (a genuinely clever protocol)

**1,409 claims verified against 5,183 abstracts.** Comparable in size to PubMedQA (1,000 questions) and SciERC (500 abstracts).

### The core trick: claims from **citances**

Claims are derived from **citation sentences (citances)** occurring naturally in scientific articles. Two benefits stated:
1. **Naturalness** — *"guarantees that the topics discussed in SciFact are representative of the research literature"*, unlike FEVER whose claims are *synthetic* (annotators mutate Wikipedia sentences).
2. **Free supervision on where the evidence is** — *"citation links indicate the exact documents likely to contain evidence necessary to verify a given claim."*

> **This is a cheap ground-truth generator we can reuse directly.** A citance is an assertion; the cited paper is the labelled evidence source. For our project it gives a way to build a provenance benchmark **without new annotation**: harvest citances, treat the cited paper as gold provenance, and test whether our agent can recover it. (Compare GAPMAP's masked-future-directions trick — same family of idea.)

### Corpus construction (Figure 2)

```
Seed set        N = 140    manually curated well-regarded journals (Cell, Nature, JAMA, BMJ),
                           sampled from S2ORC, restricted to ≥10 citations
Citing          N = 601    articles citing the seed papers; source citances taken from these
Co-cited                   other articles cited by the same citance, added to the corpus
Distractor      N = 4,259  five papers cited in the SAME paper but in a DIFFERENT paragraph
```

**The distractor design is worth stealing.** Distractors *"often discuss similar topics to the evidence documents, increasing the difficulty of abstract retrieval and making our metrics more accurately reflect the system's performance on a large research corpus."* A retrieval benchmark without hard negatives flatters the system.

Most citances used cite **only** the seed article (no co-cited papers) — initial experiments found these *"tended to yield specific, easy-to-verify claims."*

### Claim writing

- Annotators see a **source citance in article context** and write up to 3 claims from it.
- **Critically, the annotator does not see the cited abstract at claim-writing time** — this is what makes the claims natural rather than reverse-engineered from the evidence.
- Citances not making statements about specific scientific findings are skipped.
- Writers: 4 scientific-NLP experts, 15 life-science undergraduates, 4 graduate students (doctoral/medical).

### Claim negation (how REFUTES examples are made)

An NLP expert wrote negations of existing claims, *"taking precautions not to bias the negations by using obvious keywords like 'not'"* — guarding against the known annotation-artifact problem (Gururangan et al. 2018; Schuster et al. 2019).

**Validated by an ablation:** a **claim-only** model (no evidence access) reaches just **44.5% accuracy** — near the 3-class chance floor — confirming no exploitable statistical artifacts were introduced.

> **Adopt this control.** If our novelty checker can predict a label without reading the retrieved papers, our benchmark is broken. A claim-only / idea-only baseline is a cheap and necessary sanity check.

### Verification annotation

- Annotators see one claim–abstract pair, label SUPPORTS / REFUTES / NOINFO.
- On SUPPORTS/REFUTES they must **identify all rationales**.
- Verifiers: 3 NLP experts, 5 life-science undergrads, 5 grad students. **Annotators verified claims they did not write themselves.**
- Evidence found in **63% of cited abstracts**.

**Why abstracts, not full articles** (three stated reasons):
1. abstracts can be annotated more scalably,
2. evidence is in the abstract in **>60%** of cases,
3. **previous attempts at full-document annotation suffered from low annotator agreement** — they cite BioMedSumm, where expert annotators achieved **21.7%** agreement over 314 citations.

> Point 3 is a real warning for our C2/C3 ambition to read full papers. Full-text grounding is more informative *and* much noisier to validate. Expect to pay for it in annotation cost and agreement.

### Quality

| Measure | Value | Comparison |
|---|---|---|
| Label agreement (232 re-annotated pairs) | **Cohen's κ = 0.75** | FEVER 0.68 Fleiss' κ; UKP Snopes 0.70 Cohen's κ |
| Rationale agreement (per-sentence in/out) | **Cohen's κ = 0.71** | — |

### Statistics

| Label | Train | Dev | Test | All |
|---|---|---|---|---|
| SUPPORTS | 332 | 124 | 100 | 556 |
| NOINFO | 304 | 112 | 100 | 516 |
| REFUTES | 173 | 64 | 100 | 337 |
| **Total** | 809 | 300 | 300 | **1409** |

| Granularity | 0 | 1 | 2 | 3+ |
|---|---|---|---|---|
| Cited abstracts per claim | – | 1278 | 86 | 45 |
| Evidence abstracts per claim | 516 | 830 | 37 | 26 |
| Rationales per abstract | – | 552 | 290 | 153 |
| Sentences per rationale | – | 1542 | 92 | 11 |

**Most rationales are a single sentence (1542 of 1645).** No abstract has more than 3 rationales for a claim; no rationale exceeds 3 sentences. Rationales are mutually exclusive. **28 rationales contain non-contiguous sentences** — the same non-contiguity problem SciGraph-LLM identified as its main failure mode, here present but rare.

---

## 5. VeriSci — the baseline model

A three-stage "BERT-to-BERT" pipeline (following DeYoung et al. 2020a; Soleimani et al. 2019):

**1. AbstractRetrieval** — top **k = 3** abstracts by **TF-IDF similarity** (unigram + bigram features).

**2. RationaleSelection** — for each sentence `aᵢ` in abstract `a`, encode `wᵢ = [aᵢ, SEP, c]` with a BERT-style LM and predict `z̃ᵢ = σ[f(CLS(wᵢ))]`. Cross-entropy loss against `zᵢ = 1[aᵢ is a rationale]`. Select all sentences with `z̃ᵢ > t`, `t` tuned on dev.
- Negatives: sentences from NOINFO-labeled cited abstracts, plus non-rationale sentences from SUPPORTS/REFUTES abstracts.

**3. LabelPrediction** — concatenate `u = [ŝ₁, …, ŝ_ℓ, SEP, c]`, predict `ỹ(c,a) = φ[f(CLS(u))]` over 3 classes. Trained on gold rationales; for NOINFO abstracts, the `k` most TF-IDF-similar sentences serve as pseudo-rationales. **NOINFO is predicted whenever no rationale sentences are selected.**

**A negative result worth noting:** they tried a **Longformer** label predictor encoding whole abstracts with a document-level CLS token. *"Performance was not competitive with our pipeline setup, likely because the label predictor struggles to identify relevant information when given full abstracts."*

> Long-context ≠ better. The pipeline that **first narrows to rationale sentences, then judges** beat the one that read everything at once. This is the same lesson Scideator encodes by injecting one paper per message, and the same one the LGNA paper draws from AI Scientist's 100-paper prompts. **Narrow before you judge.**

### Hyperparameters (Appendix A)

| Item | Value |
|---|---|
| AbstractRetrieval | TF-IDF, unigram+bigram, **k = 3** |
| Rationale threshold | 0.5 for SciFact-trained; **t = 0.025** (FEVER-trained), **t = 0.75** (UKP Snopes-trained) |
| SciFact training | LR 1e-5 (transformer) / 1e-3 (linear), **batch 256** via gradient accumulation, cosine decay, **20 epochs** |
| FEVER training | LR 5e-6 / 5e-5, 3 epochs |
| UKP Snopes | LR 1e-5 / 1e-4, 5 epochs |
| Hyperparameter search | LR ∈ {1e-3, 1e-4, 5e-5, 1e-5, 5e-6}; batch ∈ {64, 256} |
| Hardware | single Nvidia P100 (Colab Pro); ~150 min for 20 epochs on SciFact |
| Framework | HuggingFace Transformers |

---

## 6. Results

### 6.1 Component analysis (Table 3, dev set)

| Training data | RationaleSelection F1 | LabelPrediction Acc |
|---|---|---|
| FEVER | 48.4 | 67.6 |
| UKP Snopes | 50.5 | 71.3 |
| **SciFact** | **72.1** | 75.7 |
| **FEVER + SciFact** | 69.7 | **81.9** |

**The asymmetry is the finding:**
- **LabelPrediction benefits from out-of-domain pretraining** (FEVER→SciFact = 81.9%). Analysis: of 44 cases where FEVER+SciFact was right and SciFact-only wrong, **36 were cases where the SciFact-only model predicted NOINFO**. FEVER pretraining teaches **general textual entailment** — cues like *"is associated with"*, *"has an important role in"*.
- **RationaleSelection does not** — SciFact alone is best. Rationales the SciFact model found and FEVER missed *"generally contain science-specific vocabulary."*

> **Two different skills:** *finding* scientific evidence needs domain vocabulary; *judging* the entailment relation is a general capability. Worth remembering when choosing models per stage in our pipeline — the same split may justify different models for retrieval/extraction vs. relation judgement.

| Sentence encoder | Rationale F1 | Label Acc |
|---|---|---|
| SciBERT | **74.4** | 69.2 |
| BioMedRoBERTa | 72.5 | 71.7 |
| RoBERTa-base | 70.8 | 62.9 |
| RoBERTa-large | 72.1 | **75.7** |

Domain-pretrained SciBERT wins on rationale selection; general RoBERTa-large wins on labelling — the same split again.

| Model inputs | Rationale F1 | Label Acc |
|---|---|---|
| **Claim-only** | – | **44.5** |
| **Abstract-only** | 60.5 | 53.3 |

Both ablations near-chance → **no exploitable artifacts in the dataset.**

### 6.2 Full task (Table 4, test set, with bootstrap std devs)

| Setting | Model | Sent Sel-Only | Sent Sel+Label | Abs Label-Only | Abs Label+Rat |
|---|---|---|---|---|---|
| **Oracle abstract** | Oracle rationale | 89.2 | 79.9 | 83.3 | 83.3 |
| | Zero-shot (FEVER) | 43.8 | 37.2 | 66.3 | 51.8 |
| | **VeriSci** | **69.4** | **60.6** | **74.7** | **72.7** |
| **Open** | Oracle rationale | 72.2 | 63.2 | 67.2 | 67.2 |
| | Zero-shot (FEVER) | 32.5 | 26.9 | 48.2 | 36.4 |
| | **VeriSci** | **46.1** | **39.5** | **47.4** | **46.5** |

Key readings:
- **In-domain training matters a lot** — VeriSci vs zero-shot FEVER: **+47% relative** on open Sentence_Selection+Label, **+28%** on open Abstract_Label+Rationale.
- **All three pipeline stages contribute roughly equally to error.** Oracle rationale adds ~20 F1 points; oracle abstract adds ~20 more.
- **Retrieval roughly halves end-to-end performance** (72.7 → 46.5 on Abstract_Label+Rationale).
- **Nearly all correctly-labeled abstracts have at least one correct rationale** — only 2 points between Label-Only and Label+Rationale in the oracle setting, 1 point in open. **If the model gets the label right, it usually got the evidence right too.** The gap is much larger at sentence level: models *find* the evidence but *misjudge its relation to the claim*.
- Statistical robustness: **10,000 bootstrap resamples**, std devs reported on every metric.

### 6.3 COVID-19 case study

A medical student wrote **36 COVID-related claims**; VeriSci retrieved evidence for each. A response counted as *plausible* if `Ê(c)` was non-empty and **≥ half** the retrieved evidence abstracts had reasonable rationales and labels.

**Result: 23 / 36 plausible.** In both showcased examples (Table 1) the system surfaced **both supporting and refuting evidence for the same claim** — e.g. on lopinavir/ritonavir for coronavirus, and on whether coronavirus thrives in warm climates.

*(The authors add a footnote we should note in spirit: "our model is a research prototype and should not be used to make any medical decisions whatsoever.")*

### 6.4 Error analysis — five reasoning capabilities (Table 5)

The most transferable content in the paper. A **taxonomy of scientific reasoning failures**, each with a worked example:

| Capability | Example claim / evidence | Why it's hard |
|---|---|---|
| **Science background** | Claim: *"Rapamycin slows aging in fruit flies."* Evidence: *"…feeding rapamycin to adult **Drosophila** produces life span extension…"* → SUPPORTS | Requires knowing **Drosophila is a fruit fly** — domain-specific lexical relationships never stated. |
| **Directionality** | Claim: *"Inhibiting G6PD **impairs** lipogenesis."* Evidence: *"suppression of 6PGD **increased** lipogenesis"* → REFUTES | Requires reasoning about **increases/decreases** in scientific quantities. |
| **Numerical reasoning** | Claim: *"Bariatric surgery improves resolution of diabetes."* Evidence: *"…**HR of 9.29 (95% CI 6.84–12.62)**…"* → SUPPORTS | Requires knowing **HR > 1 with 95% CI indicates improvement.** |
| **Cause and effect** | Claim: *"MVP functions to **decrease** tumor aggression."* Evidence: *"**Knockout** of MVP…**inhibiting** tumor progression"* → REFUTES | Counterfactual reasoning: removing MVP inhibits tumours ⇒ MVP *increases* aggression. |
| **Coreference** | Claim: *"Low saturated fat diets have adverse effects on infant development."* Evidence: *"Neurological development of children in **the intervention group** was at least as good as…the control group"* → REFUTES | Requires context **outside the rationale sentence** — that the intervention group was the low-fat-diet group. |

> **This taxonomy is the answer to "how can provenance be correct while the claim is wrong?"** SciGraph-LLM observed the phenomenon (an *inverted* claim generated from correct evidence); this table explains the mechanism and names the categories. **Directionality** and **cause-and-effect** are precisely the failure modes that produce polarity inversion — and note that embedding similarity, the check SciGraph-LLM relies on, is blind to all five.
>
> For our C5, **numerical reasoning** deserves special attention: an experiment plan that cites a reported effect size but misreads its direction or significance is worse than one with no citation at all.

---

## 7. What to take into our Research Agent

| Our component | What SciFact/VeriSci gives us |
|---|---|
| **C2 — Literature Acquisition** | The **distractor-abstract design** — papers cited in the same source paper but a different paragraph — is a cheap, principled way to build **hard negatives** for retrieval evaluation. Without them, retrieval metrics flatter the system. Also, the **open vs oracle** split is how we attribute error between our retrieval layer and our reasoning layer. |
| **C3 — Research Knowledge Layer** | **Atomicity** — a claim is one proposition, compound statements get split — confirmed independently here and in SciGraph-LLM. **Rationale = a minimal sentence set that a domain expert would judge as implying the claim**, and the empirical finding that **most rationales are a single sentence** (1542/1645) is a useful default granularity for our evidence spans. Also: **rationales can be non-contiguous** (28 cases) — rare, but our schema shouldn't forbid it, since forbidding it is exactly what broke SciGraph-LLM's CS-KG example. |
| **C4 — Gap / Novelty Detection** | **REFUTES is a first-class label.** Our docx's Evidence field is framed around support; SciFact shows a claim can be **simultaneously supported and refuted by different papers** — observed in the real COVID study. Conflicting evidence is a *finding* (and, per GAPMAP, an implicit-gap signal), not an error to be resolved away. Our Novel Evidence Report should be able to represent it. **NOINFO also matters**: "this paper is retrieved but says nothing about the claim" is different from "no paper was found," and only the first is verifiable. |
| **C5 — Provenance-Aware Experiment Planning** | The **five reasoning types** are a checklist for what can go wrong between a cited paper and a plan element. If our plan asserts "Method X improves metric M on dataset D, per Paper A," we are making a **directionality + numerical-reasoning** claim, and those are exactly the two the models here failed most often. This argues for an explicit **verification pass over the finished plan**, treating each plan element as a claim and each cited paper as an abstract to verify against. |
| **C-Eval — Provenance Evaluation (§4.2)** | **The single richest source in the set for our §4.2.** (a) The **2×2 metric design** (abstract/sentence × label-only/+rationale) maps onto citation correctness, completeness, consistency and traceability. (b) **Cohen's κ = 0.75 (labels) / 0.71 (rationales)** are realistic targets for our own annotation. (c) The **claim-only and abstract-only ablations** are mandatory sanity checks — if our checker works without reading the papers, the benchmark is broken. (d) **Bootstrap resampling (n=10,000)** for std devs on every metric. (e) **Citances as free ground truth** for building a provenance benchmark without new annotation. (f) The finding that **full-document annotation yields low agreement (21.7% on BioMedSumm)** is a cost warning for our full-text ambitions. |

**Cheap wins to steal first:** (1) the 2×2 evaluation metrics, adapted to provenance; (2) citance-derived ground truth; (3) distractor documents as hard negatives; (4) the claim-only ablation as an artifact check; (5) the five-reasoning-type taxonomy as an error-analysis scheme; (6) **narrow-then-judge** rather than long-context-then-judge.

---

## 8. Gaps, Limitations & Future Scope

### Stated or evidenced by the paper
1. **Abstracts only, not full text.** Justified by scalability and the >60% coverage figure, but it means evidence in a Methods or Results section is out of reach. They cite low full-document agreement (21.7% on BioMedSumm) as the reason.
2. **No global truth labels** — only per-claim-abstract relations, because a global verdict *"requires a systematic review by a team of experts."*
3. **Retrieval roughly halves performance** (72.7 → 46.5 F1 open vs oracle); TF-IDF over 5,183 abstracts is weak, and the real corpus is millions.
4. **Five unmet reasoning capabilities** (§6.4) — background knowledge, directionality, numerical reasoning, cause/effect, coreference.
5. **Small dataset** — 1,409 claims, 809 train. Acknowledged as comparable to peers but thin.
6. **Their own stated future directions:** *"models capable of incorporating background information, reasoning about scientific processes, and assessing the strength and provenance of various evidence sources"*, and extending verification to **sources beyond the research literature** (social media, news). They also position claim verification as a component in future **automated evidence synthesis** (systematic-review automation) frameworks.

### Gaps we can attack (our project's opening)
1. **Pre-LLM baselines.** VeriSci is TF-IDF + BERT. Every model number here is superseded — but **the task, dataset, and metrics are not**, and SciFact remains a live leaderboard. **The right use of this paper is as an evaluation harness, not an architecture.** Running a modern LLM pipeline on SciFact would give our provenance layer a calibrated, externally-comparable number, which none of the other papers in this set can offer.
2. **Verification only — no generation, no ideation, no gap detection.** It checks claims someone else wrote. It is a component for our C4/C5 verification pass, not a competitor.
3. **No evidence *strength*.** SUPPORTS is binary — a single mouse study and a large RCT are the same label. Our C4's five-level novelty ladder has an obvious analogue here (an "evidence strength" ladder), and GAPMAP's *Levels of Evidence* gap category points the same way. **Nobody in this set grades evidence strength, and a provenance-aware experiment plan arguably should.**
4. **No aggregation across evidence.** The system reports per-abstract relations; it never synthesises "3 papers support, 1 refutes, therefore…". Our Novel Evidence Report needs exactly that aggregation, and this paper deliberately stops short of it.
5. **Biomedical-only.** S2ORC-sampled from Cell/Nature/JAMA/BMJ. Four of the five reasoning types (directionality, numerical, cause/effect, background) generalise readily to CS/ML; the domain vocabulary does not.
6. **Rationale is a flat sentence set.** No structure — no distinction between a premise, a warrant, and a conclusion. **GAPMAP's Claim/Grounds/Warrant is the natural upgrade**: SciFact's rationale ≈ Grounds, with the *Warrant* left implicit — and §6.4 shows that the Warrant (the background knowledge, the direction, the statistical interpretation) is exactly where the models fail. **Making the warrant explicit and checkable is an unclaimed improvement that both papers point at from different sides.**
7. **Claim-abstract pairs are judged independently.** No cross-abstract reasoning, no notion that two abstracts contradict each other — even though the COVID study surfaced exactly that. Our cross-paper landscape graph (C3) is where that would live.
