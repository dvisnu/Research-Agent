"""The two linked outputs: the Novel Evidence Report and the experiment plan.

Phase 0 writes them empty; Phases 4 and 5 fill them in.
"""

from pathlib import Path

from pydantic import BaseModel, Field


class NovelEvidenceReport(BaseModel):
    proposed_idea: str | None = Field(None, title="Proposed Idea")
    novelty_assessment: str | None = Field(None, title="Novelty Assessment")
    evidence: list[str] = Field([], title="Evidence")
    research_gap: str | None = Field(None, title="Research Gap")
    closest_existing_work: list[str] = Field([], title="Closest Existing Work")
    difference_from_existing_work: str | None = Field(None, title="Difference from Existing Work")


class ExperimentPlan(BaseModel):
    hypothesis: str | None = Field(None, title="Hypothesis")
    research_question: str | None = Field(None, title="Research Question")
    method: str | None = Field(None, title="Method")
    datasets: list[str] = Field([], title="Datasets")
    baselines: list[str] = Field([], title="Baselines")
    metrics: list[str] = Field([], title="Metrics")
    conditions: list[str] = Field([], title="Experimental Conditions")
    ablations: list[str] = Field([], title="Ablation Studies")
    expected_results: list[str] = Field([], title="Expected Results")
    risks_and_limitations: list[str] = Field([], title="Risks / Limitations")


class RunReport(BaseModel):
    run_id: str
    question: str
    generated_at: str
    novel_evidence_report: NovelEvidenceReport = NovelEvidenceReport()
    experiment_plan: ExperimentPlan = ExperimentPlan()


def to_markdown(report: RunReport) -> str:
    lines = [
        "# Research report",
        "",
        f"**Question:** {report.question}",
        "",
        f"Run `{report.run_id}`, generated {report.generated_at}",
    ]
    for heading, part in (
        ("Novel Evidence Report", report.novel_evidence_report),
        ("Experiment Plan", report.experiment_plan),
    ):
        lines += ["", f"## {heading}"]
        for name, info in type(part).model_fields.items():
            value = getattr(part, name)
            lines += ["", f"### {info.title}", ""]
            if not value:
                lines.append("_Not produced yet._")
            elif isinstance(value, list):
                lines += [f"- {item}" for item in value]
            else:
                lines.append(str(value))
    return "\n".join(lines) + "\n"


def write_report(report: RunReport, out_dir: Path) -> tuple[Path, Path]:
    """Write ``report.md`` and ``report.json``; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, json_path = out_dir / "report.md", out_dir / "report.json"
    md_path.write_text(to_markdown(report), encoding="utf-8")
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return md_path, json_path
