# Fact or Fiction (SciFact / VeriSci) — Summary

> *Fact or Fiction: Verifying Scientific Claims* — Wadden, Lin (UW), Lo, Wang, van Zuylen, Cohan (AI2), Hajishirzi · EMNLP 2020 · arXiv 2004.14974v6 · [code/data](https://github.com/allenai/scifact)
> The oldest paper in the set (pre-LLM). Its value is **not** the model — a BERT pipeline superseded many times — but the **task formalisation, annotation protocol, two-granularity metrics, and taxonomy of scientific reasoning failures**, which are exactly what our Provenance Evaluation needs.

## Summary

**The task.** Given a claim and a corpus of abstracts: (a) retrieve the abstracts containing evidence, (b) label each **SUPPORTS / REFUTES / NOINFO**, and (c) **identify the specific sentences (rationales) that justify the decision.** That third requirement is the whole point.

Two definitions worth adopting. A **scientific claim** is *"an atomic verifiable statement expressing a finding about one aspect of a scientific entity or process, which can be verified from a single source"* — compound claims must be split (the same atomicity conclusion SciGraph-LLM reached independently six years later). A **rationale** is *"a minimal collection of sentences which, taken together as premises in the context of the abstract, can reasonably be judged by a domain expert as implying the claim."*

**A deliberate scoping decision worth copying:** they do **not** assign a global truth label to a claim, because that *"requires a systematic review by a team of experts."* They label individual claim–abstract *pairs*. Same epistemics as our "always potential novelty, never proven novelty" rule — and note that the same claim can be **both supported and refuted by different abstracts**, observed on real COVID-19 claims.

**Metrics — the most reusable part, a ready-made 2×2 for our provenance metrics.**

- **Abstract-level:** *label-only* (right abstract, right label) and *label+rationale* (also, the predicted sentences contain a gold rationale). Predicted rationales capped at 3.
- **Sentence-level:** a rationale sentence counts only if it belongs to a gold rationale, **all other sentences of that same rationale were also predicted**, and the abstract label isn't NOINFO. No cap, because the metric must penalise over-prediction. That third condition is subtle and smart: **finding text is not the same as finding evidence.**
- **Two retrieval settings** — *oracle abstract* (gold evidence given, isolating reasoning) vs *open* (retrieve from the corpus). This split is how to attribute our own errors between retrieval and reasoning.

**Dataset construction — a genuinely clever protocol.** 1,409 claims over 5,183 abstracts. Claims are derived from **citances** (citation sentences), which guarantees topical naturalness *and* gives free supervision on where the evidence lives. Annotators write claims from a citance **without seeing the cited abstract**, so claims aren't reverse-engineered from the evidence. **4,259 distractor abstracts** — papers cited in the same source paper but a *different paragraph* — provide hard negatives, without which retrieval metrics flatter the system. REFUTES examples come from expert-written negations that deliberately avoid obvious keywords. Quality: label κ = 0.75, rationale κ = 0.71. Most rationales are a **single sentence** (1542 of 1645); 28 are non-contiguous.

**Why abstracts, not full text:** scalability, evidence is in the abstract >60% of the time, and **previous full-document annotation attempts had very low agreement (21.7% on BioMedSumm)** — a real cost warning for our own full-text ambitions.

**Results.** VeriSci (TF-IDF retrieval → BERT rationale selection → BERT label prediction) reaches 72.7 abstract label+rationale F1 in the oracle setting but **46.5 in the open setting — retrieval roughly halves end-to-end performance.** Two findings transfer: **label prediction benefits from out-of-domain pretraining** (FEVER→SciFact, 81.9% acc) while **rationale selection does not** (SciFact alone best; SciBERT wins there, RoBERTa-large wins at labelling) — *finding* scientific evidence needs domain vocabulary, *judging* entailment is a general capability. And a negative result: a **Longformer over whole abstracts underperformed the pipeline** that narrows to rationale sentences first. **Narrow before you judge.** Ablations confirm no dataset artifacts: claim-only accuracy 44.5%, near chance.

**Error analysis — five reasoning capabilities** (the most transferable content): **science background** (Drosophila = fruit fly), **directionality** (increase vs decrease), **numerical reasoning** (HR > 1 with a 95% CI means improvement), **cause and effect** (knockout inhibits tumours ⇒ the protein increases aggression), **coreference** (which arm was the intervention group). **This taxonomy explains how provenance can be correct while the claim is wrong** — directionality and cause/effect are exactly what produce polarity inversion, and embedding similarity is blind to all five.

## Gaps

**Stated / evidenced**

- **Abstracts only** — evidence in a Methods or Results section is out of reach, justified by scalability and the low agreement on full-document annotation.
- **No global truth labels**, only per-pair relations.
- **Retrieval halves performance**; TF-IDF over 5,183 abstracts is weak and the real corpus is millions of papers.
- Five unmet reasoning capabilities; small dataset (1,409 claims, 809 train).
- Their own future directions: models that incorporate background knowledge, reason about scientific processes, and **assess the strength and provenance of evidence sources**; and claim verification as a component in automated evidence synthesis.

**Open for us**

- **Pre-LLM baselines.** Every model number is superseded, but **the task, dataset and metrics are not** — SciFact is still a live leaderboard. The right use of this paper is as an **evaluation harness, not an architecture**: running a modern pipeline on SciFact would give our provenance layer a calibrated, externally-comparable number no other paper in the set can offer.
- **Verification only** — no generation, ideation or gap detection. A component for our C4/C5 verification pass.
- **No evidence *strength*.** SUPPORTS is binary — a single mouse study and a large RCT get the same label. Nobody in the set grades evidence strength, and a provenance-aware experiment plan arguably should.
- **No aggregation across evidence** — it reports per-abstract relations and never synthesises "3 support, 1 refutes, therefore…". Our Novel Evidence Report needs exactly that.
- **Rationale is a flat sentence set** — no premise/warrant/conclusion structure. **GAPMAP's Claim/Grounds/Warrant is the natural upgrade**: SciFact's rationale ≈ Grounds, with the Warrant left implicit — and their error analysis shows the Warrant is exactly where models fail. Making it explicit and checkable is an unclaimed improvement both papers point at from different sides.
- **Pairs are judged independently** — no cross-abstract reasoning, even though the COVID study surfaced contradictions. That belongs in our cross-paper graph. Biomedical-only.

## What We Can Use

| Component | What SciFact/VeriSci gives us |
|---|---|
| **C2 — Literature Acquisition** | The **distractor design** (papers cited in the same source but a different paragraph) is a cheap, principled way to build hard negatives for retrieval evaluation. The **open vs oracle split** is how we attribute error between our retrieval and our reasoning. |
| **C3 — Research Knowledge Layer** | **Atomicity**, confirmed independently here and in SciGraph-LLM — treat as settled. The rationale definition, and the finding that **most rationales are a single sentence**, gives a default granularity for our evidence spans. Rationales can be **non-contiguous** (rare) — our schema shouldn't forbid it, since forbidding it is what broke SciGraph-LLM's hardest case. |
| **C4 — Gap / Novelty Detection** | **REFUTES is a first-class label.** Our Evidence field is framed around support, but a claim can be simultaneously supported and refuted by different papers — conflicting evidence is a *finding* (and per GAPMAP an implicit-gap signal), not an error to resolve away. **NOINFO matters too**: "retrieved but says nothing" differs from "nothing found", and only the first is verifiable. |
| **C5 — Experiment Planning** | The **five reasoning types are a checklist** for what can go wrong between a cited paper and a plan element. "Method X improves metric M on dataset D, per Paper A" is a directionality + numerical-reasoning claim — the two the models here failed most. This argues for an explicit **verification pass over the finished plan**, treating each plan element as a claim and each cited paper as an abstract to verify against. |
| **Provenance Evaluation** | The richest source in the set. (a) The **2×2 metric design** maps onto citation correctness, completeness, consistency and traceability. (b) **κ = 0.75 / 0.71** are realistic annotation targets. (c) **Claim-only and abstract-only ablations are mandatory sanity checks** — if our checker works without reading the papers, the benchmark is broken. (d) **Bootstrap resampling (n=10,000)** for std devs on every metric. (e) **Citances as free ground truth** for a provenance benchmark without new annotation. (f) The 21.7% full-document agreement figure as a cost warning. |

**Steal first:** (1) the 2×2 metrics adapted to provenance; (2) citance-derived ground truth; (3) distractor hard negatives; (4) the claim-only artifact check; (5) the five-reasoning-type error taxonomy; (6) **narrow-then-judge** rather than long-context-then-judge.
