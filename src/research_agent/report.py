"""The two linked outputs: the Novel Evidence Report and the experiment plan.

Phase 4 fills the Novel Evidence Report from the run's best idea and lists every
idea checked; Phase 5 fills the plan. The report also says what the run built on
the way: its gaps, related work and landscape (Phase 3) and its knowledge graph
(Phase 2).
"""

import re
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field

from research_agent.costs import RunCost


class NovelEvidenceReport(BaseModel):
    proposed_idea: str | None = Field(None, title="Proposed Idea")
    novelty_assessment: str | None = Field(None, title="Novelty Assessment")
    evidence: list[str] = Field([], title="Evidence")
    research_gap: str | None = Field(None, title="Research Gap")
    closest_existing_work: list[str] = Field([], title="Closest Existing Work")
    difference_from_existing_work: str | None = Field(None, title="Difference from Existing Work")


class PlanSource(BaseModel):
    """A sentence plan elements cite, numbered in the plan's list of sources."""

    number: int
    paper: str
    sentence_id: str
    quote: str
    section: str | None = None


class PlanElementReport(BaseModel):
    """One element of the experiment plan, with its provenance in short."""

    element_id: str
    label: str
    text: str
    name: str | None = None
    basis: str | None = None
    sources: list[int] = []  # numbers in the plan's sources
    decision: str | None = None
    alternatives: list[str] = []
    derivation: str | None = None
    verdict: str | None = None  # supported, not_supported, contradicted, no_source, unverified
    verdict_reason: str | None = None
    status: str = "kept"  # kept, flagged, removed

    def line(self) -> str:
        cites = f" [{', '.join(map(str, self.sources))}]" if self.sources else ""
        head = f"**{self.name}**: " if self.name and not self.text.lower().startswith(self.name.lower()) else ""
        flag = ""
        if self.status == "flagged":
            reason = f": {self.verdict_reason}" if self.verdict_reason else ""
            flag = f" *(flagged, {VERDICT_TEXT.get(self.verdict or '', self.verdict)}{reason})*"
        return f"{head}{self.text}{cites}{flag}"

    def provenance_lines(self, sources: dict[int, "PlanSource"]) -> list[str]:
        """The element's provenance in full, for a contested element: what it rests on, the
        sentences it cites and how it was derived."""
        lines = [f"  - *Rests on:* {self.basis}"] if self.basis else []
        for number in self.sources:
            source = sources.get(number)
            if source is not None:
                where = f", {source.section}" if source.section else ""
                lines.append(f"  - *[{number}]* {source.paper}{where}: \"{source.quote}\" (`{source.sentence_id}`)")
        if self.derivation:
            lines.append(f"  - *Derived:* {self.derivation}")
        return lines


VERDICT_TEXT = {"supported": "supported by its sources", "not_supported": "not supported by its sources",
                "contradicted": "contradicted by its sources", "no_source": "no source",
                "unverified": "not verified"}


class ExperimentPlan(BaseModel):
    idea_id: str | None = None
    idea_title: str | None = None
    hypothesis: PlanElementReport | None = Field(None, title="Hypothesis")
    research_question: PlanElementReport | None = Field(None, title="Research Question")
    method: list[PlanElementReport] = Field([], title="Method")
    datasets: list[PlanElementReport] = Field([], title="Datasets")
    baselines: list[PlanElementReport] = Field([], title="Baselines")
    metrics: list[PlanElementReport] = Field([], title="Metrics")
    conditions: list[PlanElementReport] = Field([], title="Experimental Conditions")
    ablations: list[PlanElementReport] = Field([], title="Ablation Studies")
    expected_results: list[PlanElementReport] = Field([], title="Expected Results")
    risks_and_limitations: list[PlanElementReport] = Field([], title="Risks / Limitations")
    provenance_coverage: dict[str, Any] = {}
    sources: list[PlanSource] = []
    removed: list[PlanElementReport] = []  # removed by verification: contradicted by their sources
    open_points: list[str] = []  # critique points of the final version, not revised for
    revisions: list[str] = []
    note: str | None = None  # why there is no plan

    def elements(self) -> list[tuple[str, list[PlanElementReport]]]:
        out = []
        for name, info in type(self).model_fields.items():
            if info.title:
                value = getattr(self, name)
                out.append((info.title, [value] if isinstance(value, PlanElementReport) else (value or [])))
        return out

    def markdown(self) -> list[str]:
        if not any(items for _, items in self.elements()):
            return ["", f"_Not produced: {self.note or 'the plan stages have not run'}._"]
        c, numbered = self.provenance_coverage, {source.number: source for source in self.sources}
        lines = ["", f"Plan for the idea in the Novel Evidence Report above: **{self.idea_title}** (`{self.idea_id}`). "
                     "Numbers in brackets are the sources listed at the end of the plan; a flagged element shows "
                     "its provenance in full."]
        if c.get("elements"):
            lines += ["", f"**Provenance coverage:** {c['supported']} of {c['elements']} elements "
                          f"({c['coverage']:.0%}) are supported by the sentences they cite, as the verifier judged; "
                          f"{c.get('flagged', 0)} flagged, {c.get('removed', 0)} removed as contradicted by their sources."]
        for title, items in self.elements():
            lines += ["", f"### {title}", ""]
            if not items:
                lines.append("_None._")
            for item in items:
                lines.append(f"- {item.line()}")
                if item.decision:
                    others = f" Alternatives: {', '.join(item.alternatives)}." if item.alternatives else ""
                    lines.append(f"  - *Why:* {item.decision}{others}")
                if item.status == "flagged":
                    lines += item.provenance_lines(numbered)
        if self.removed:
            lines += ["", "### Removed by verification", ""]
            for e in self.removed:
                lines.append(f"- ({e.label}) {e.text}: {e.verdict_reason or 'contradicted by its sources'}")
                lines += e.provenance_lines(numbered)
        if self.open_points:
            lines += ["", "### Open critique points", "", *(f"- {point}" for point in self.open_points)]
        if self.revisions:
            lines += ["", "### Revisions", "", *(f"- {entry}" for entry in self.revisions)]
        if self.sources:
            lines += ["", "### Sources", ""]
            for source in self.sources:
                where = f", {source.section}" if source.section else ""
                lines.append(f"{source.number}. {source.paper}{where} (`{source.sentence_id}`): \"{source.quote}\"")
        lines += ["", "Each element's full provenance (its sources, how it was derived and why it was chosen) is in "
                      "`provenance.json` (PROV-JSON) and shown by `agent plan <run_id>`."]
        return lines


class KnowledgeGraphSummary(BaseModel):
    """What the extract and build_graph stages produced."""

    papers_in_corpus: int = 0
    papers_extracted: int = 0
    extraction_calls: int = 0
    extractions_reused: int = 0
    extraction_failures: int = 0
    quote_check: dict[str, Any] = {}
    nodes: dict[str, int] = {}
    edges: dict[str, int] = {}
    names_merged: int = 0
    links: dict[str, Any] = {}

    @classmethod
    def from_stages(cls, extraction: dict[str, Any], graph: dict[str, Any]) -> Self:
        names = graph.get("names", {}).values()
        return cls(
            papers_in_corpus=graph.get("papers", {}).get("in_corpus", 0),
            papers_extracted=graph.get("papers", {}).get("extracted", 0),
            extraction_calls=extraction.get("calls", 0),
            extractions_reused=extraction.get("reused", 0),
            extraction_failures=extraction.get("failed", 0),
            quote_check=graph.get("quote_check", {}),
            nodes=graph.get("nodes", {}),
            edges=graph.get("edges", {}),
            names_merged=sum(n.get("by_key", 0) + n.get("by_model", 0) for n in names),
            links=graph.get("links", {}),
        )

    def markdown(self) -> list[str]:
        q, links = self.quote_check, self.links
        rate = f"{q['pass_rate']:.1%}" if q.get("pass_rate") is not None else "n/a"
        embedding = links.get("embedding", {})
        embedded = embedding.get("status") or (
            f"{embedding.get('reused', 0)} reused, {embedding.get('sent', 0)} sent"
            + (f", {embedding['missing']} missing ({embedding.get('unavailable', '')})" if embedding.get("missing") else "")
        )
        return [
            f"- **Papers:** {self.papers_extracted} of {self.papers_in_corpus} extracted "
            f"({self.extraction_calls} LLM calls, {self.extractions_reused} extractions reused, "
            f"{self.extraction_failures} failed)",
            f"- **Quote check:** {rate} of {q.get('quotes', 0)} quotes found "
            f"({q.get('exact', 0)} in the cited sentence, {q.get('readdressed', 0)} in another); "
            f"{q.get('items_kept', 0)} of {q.get('items', 0)} items kept",
            "- **Nodes:** " + (", ".join(f"{n} {kind}" for kind, n in sorted(self.nodes.items())) or "none"),
            "- **Edges:** " + (", ".join(f"{n} {kind}" for kind, n in sorted(self.edges.items())) or "none"),
            f"- **Names merged:** {self.names_merged}",
            f"- **Links:** {links.get('candidates', 0)} candidate pairs "
            f"({links.get('similar_text', 0)} by similar text, {links.get('shared_setting', 0)} by shared setting): "
            f"{links.get('supports', 0)} supports, {links.get('contradicts', 0)} contradicts "
            f"({links.get('contradicts_other_setting', 0)} more in different settings, not kept)",
            f"- **Claim embeddings:** {embedded}",
        ]


class RelatedPaper(BaseModel):
    title: str
    year: int | None = None
    similarity: float
    shares: list[str] = []  # entities it shares with the other close papers
    purpose: str | None = None


class Landscape(BaseModel):
    """The most-studied methods (rows) x datasets (columns). A cell is "R<n>" for n results,
    "p<n>" when n papers use both but report no result for the pair, "-" when nothing."""

    methods: list[str]
    datasets: list[str]
    cells: list[list[str]]


class GapEvidence(BaseModel):
    sentence_id: str
    quote: str
    paper: str


class ResearchGap(BaseModel):
    rank: int
    kind: str  # stated, structural, conflict
    gap: str
    warrant: str | None = None
    relevance: float
    papers: int  # independent papers its evidence comes from
    evidence: list[GapEvidence] = []


def _words(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


class Quote(BaseModel):
    sentence_id: str
    quote: str
    paper: str | None = None


class ClosestWork(BaseModel):
    """One of an idea's closest papers, and how it compares with the idea."""

    position: int
    title: str
    year: int | None = None
    read: str  # "abstract", "full text", or "not compared"
    purpose: str | None = None  # same, similar, different, unknown
    mechanism: str | None = None
    evaluation: str | None = None
    paper_mechanism: str | None = None  # what the paper itself does
    solves_gap: str | None = None
    difference: str | None = None
    evidence: list[Quote] = []

    def line(self) -> str:
        year = f" ({self.year})" if self.year else ""
        if self.read == "not compared":
            return f"{self.title}{year}: not compared"
        return (f"{self.title}{year}, read by {self.read}: purpose {self.purpose}, mechanism {self.mechanism}, "
                f"evaluation {self.evaluation}")


class IdeaReport(BaseModel):
    """An idea and its novelty check (Phase 4)."""

    rank: int | None = None
    idea_id: str
    origin: str  # generated, revised, user, planted
    title: str
    purpose: str
    mechanism: str
    evaluation: str
    kind: str | None = None
    builds_on: list[str] = []
    feasibility: str | None = None
    level: int | None = None
    label: str | None = None
    rating: str | None = None
    inconclusive: bool = False
    reasons: list[str] = []
    coverage: dict[str, Any] = {}
    gap: str | None = None
    gap_kind: str | None = None
    gap_evidence: list[Quote] = []
    addressed_by: list[str] = []
    closest: list[ClosestWork] = []
    history: list[str] = []  # earlier versions of the idea

    def assessment(self) -> str:
        if self.level is None:
            return "Not assessed."
        flag = ", inconclusive" if self.inconclusive else ""
        return (f"Potential novelty: {self.rating} (level {self.level} of 5, {self.label}{flag}). "
                + " ".join(self.reasons) + " " + self.coverage_text()).strip()

    def coverage_text(self) -> str:
        c = self.coverage
        if not c:
            return ""
        compared = sum(1 for w in self.closest if w.read != "not compared")
        full = sum(1 for w in self.closest if w.read == "full text")
        channels = ", ".join(c.get("channels_answered", [])) or "no channel"
        text = (f"Search coverage: {c.get('candidates', 0)} candidate papers from {channels}; "
                f"{compared} closest papers compared ({full} read in full); "
                f"{c.get('closest_found_by_several_channels', 0)} of them found by more than one channel, "
                f"{c.get('closest_from_recommendations_only', 0)} only through recommendations.")
        if c.get("weak"):
            text += " Weak: " + "; ".join(c["weak"]) + "."
        return text

    def evidence_lines(self) -> list[str]:
        lines = [f'{q.paper or "a paper"} states the gap: "{q.quote}" (`{q.sentence_id}`)' for q in self.gap_evidence[:4]]
        for work in self.closest[:3]:
            if work.paper_mechanism and work.evidence:
                q = work.evidence[0]
                year = f" ({work.year})" if work.year else ""
                said = "" if _words(q.quote) in _words(work.paper_mechanism) else f"{work.paper_mechanism} "
                lines.append(f'{work.title}{year} establishes: {said}"{q.quote}" (`{q.sentence_id}`)')
        if self.level is not None and self.level >= 3:
            compared = sum(1 for w in self.closest if w.read != "not compared")
            lines.append(f"No directly matching approach found: none of the {compared} closest papers compared, out of "
                         f"{self.coverage.get('candidates', 0)} searched, matches both the purpose and the mechanism.")
        return lines

    def novel_evidence_report(self) -> NovelEvidenceReport:
        facets = [f"**{self.title}**", "", f"- Purpose: {self.purpose}", f"- Mechanism: {self.mechanism}",
                  f"- Evaluation: {self.evaluation}"]
        if self.kind:
            facets.append(f"- Kind: {self.kind.replace('_', ' ')}")
        if self.builds_on:
            facets.append(f"- Builds on: {', '.join(self.builds_on)}")
        if self.feasibility:
            facets.append(f"- Feasibility: {self.feasibility}")
        if self.origin == "user":
            facets.append("- Origin: your own idea, as the research brief read it")
        gap = None
        if self.gap:
            gap = f"{self.gap} ({self.gap_kind} gap)" if self.gap_kind else self.gap
            if self.addressed_by:
                gap += f" Possibly addressed by: {', '.join(self.addressed_by)}."
        differences = [f"vs {w.title}: {w.difference}" for w in self.closest[:3] if w.difference]
        return NovelEvidenceReport(
            proposed_idea="\n".join(facets),
            novelty_assessment=self.assessment(),
            evidence=self.evidence_lines(),
            research_gap=gap,
            closest_existing_work=[w.line() for w in self.closest[:5]],
            difference_from_existing_work=" ".join(differences) or None,
        )


class RunReport(BaseModel):
    run_id: str
    question: str
    generated_at: str
    novel_evidence_report: NovelEvidenceReport = NovelEvidenceReport()
    experiment_plan: ExperimentPlan = ExperimentPlan()
    related_work: list[RelatedPaper] = []
    landscape: Landscape | None = None
    research_gaps: list[ResearchGap] = []
    ideas: list[IdeaReport] = []  # the checked ideas, best first
    knowledge_graph: KnowledgeGraphSummary | None = None
    cost: RunCost | None = None  # time, model calls, API requests and reuse, from the trace

    def add_phase5(self, plan: ExperimentPlan) -> None:
        self.experiment_plan = plan

    def add_phase4(self, ideas: list[IdeaReport]) -> None:
        self.ideas = ideas
        if ideas:
            self.novel_evidence_report = ideas[0].novel_evidence_report()

    def phase4_markdown(self) -> list[str]:
        if not self.ideas:
            return []
        lines = ["", "## Ideas checked", "",
                 "Each idea's best version, best first. The first is the one in the Novel Evidence Report above."]
        for idea in self.ideas:
            level = (f"level {idea.level} ({idea.label}{', inconclusive' if idea.inconclusive else ''})"
                     if idea.level else "not assessed")
            own = " (your idea)" if idea.origin == "user" else ""
            lines += ["", f"### {idea.rank}. {idea.title}{own}: {level}", "",
                      f"- Purpose: {idea.purpose}", f"- Mechanism: {idea.mechanism}", f"- Evaluation: {idea.evaluation}"]
            if idea.gap:
                lines.append(f"- Gap: {idea.gap}")
            lines += [f"- {reason}" for reason in idea.reasons]
            if idea.closest:
                lines.append(f"- Closest: {idea.closest[0].line()}")
            lines += [f"- Earlier version: {entry}" for entry in idea.history]
        return lines

    def add_phase3(
        self, related: list[dict[str, Any]], landscape: dict[str, Any] | None, gaps: list[dict[str, Any]]
    ) -> None:
        self.related_work = [RelatedPaper(**paper) for paper in related]
        if landscape and landscape["methods"] and landscape["datasets"]:
            label = {"result": "R{results}", "same_paper": "p{papers}", "empty": "-"}
            self.landscape = Landscape(
                methods=[m["name"] for m in landscape["methods"]],
                datasets=[d["name"] for d in landscape["datasets"]],
                cells=[[label[cell["status"]].format(**cell) for cell in row] for row in landscape["cells"]],
            )
        self.research_gaps = [ResearchGap(**gap) for gap in gaps]

    def phase3_markdown(self) -> list[str]:
        lines: list[str] = []
        if self.research_gaps:
            lines += ["", "## Research gaps", ""]
            for gap in self.research_gaps:
                lines += [f"### {gap.rank}. {gap.gap}", "",
                          f"*{gap.kind}; relevance {gap.relevance:.3f}; evidence from {gap.papers} paper(s)*", ""]
                if gap.warrant:
                    lines += [f"**Warrant:** {gap.warrant}", ""]
                lines += [f"- \"{e.quote}\" ({e.paper}, `{e.sentence_id}`)" for e in gap.evidence[:4]]
                lines.append("")
        if self.related_work:
            lines += ["", "## Related work", ""]
            for n, paper in enumerate(self.related_work, start=1):
                year = f" ({paper.year})" if paper.year else ""
                shares = f"; shares {', '.join(paper.shares)}" if paper.shares else ""
                lines.append(f"{n}. **{paper.title}**{year}: similarity {paper.similarity:.3f}{shares}")
                if paper.purpose:
                    lines.append(f"   - {paper.purpose}")
        if self.landscape:
            grid = self.landscape
            lines += ["", "## Research landscape", "",
                      "Most-studied methods (rows) x datasets (columns): `R<n>` n results, "
                      "`p<n>` n papers use both without a result for the pair, `-` nothing.", "",
                      "| method | " + " | ".join(grid.datasets) + " |",
                      "|---|" + "---|" * len(grid.datasets)]
            lines += [f"| {method} | " + " | ".join(row) + " |" for method, row in zip(grid.methods, grid.cells)]
        return lines


def to_markdown(report: RunReport) -> str:
    lines = [
        "# Research report",
        "",
        f"**Question:** {report.question}",
        "",
        f"Run `{report.run_id}`, generated {report.generated_at}",
    ]
    part = report.novel_evidence_report
    lines += ["", "## Novel Evidence Report"]
    for name, info in type(part).model_fields.items():
        value = getattr(part, name)
        lines += ["", f"### {info.title}", ""]
        if not value:
            lines.append("_Not produced yet._")
        elif isinstance(value, list):
            lines += [f"- {item}" for item in value]
        else:
            lines.append(str(value))
    lines += ["", "## Experiment Plan", *report.experiment_plan.markdown()]
    lines += report.phase4_markdown()
    lines += report.phase3_markdown()
    if report.knowledge_graph is not None:
        lines += ["", "## Knowledge graph", "", *report.knowledge_graph.markdown()]
    if report.cost is not None:
        lines += ["", "## Run cost", "", *report.cost.markdown()]
    return "\n".join(lines) + "\n"


def write_report(report: RunReport, out_dir: Path) -> tuple[Path, Path]:
    """Write ``report.md`` and ``report.json``; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, json_path = out_dir / "report.md", out_dir / "report.json"
    md_path.write_text(to_markdown(report), encoding="utf-8")
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return md_path, json_path
