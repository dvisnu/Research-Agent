"""The pipeline's stages in order, with the phase that builds each one."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    phase: int
    summary: str


STAGES = (
    Stage("brief", 1, "Turn the question or idea into a research brief"),
    Stage("search", 1, "Keyword, snippet and arXiv search"),
    Stage("expand", 1, "Follow references, citations and recommendations"),
    Stage("rank", 1, "Rank papers; split them into abstract-only and full-read sets"),
    Stage("fulltext", 1, "Fetch and parse full texts into sections and sentences"),
    Stage("extract", 2, "Extract each paper's facts, with quoted evidence"),
    Stage("build_graph", 2, "Merge names, build the graph, link supporting and contradicting claims"),
    Stage("related_work", 3, "Find the work closest to the question"),
    Stage("landscape", 3, "Build the methods x datasets x metrics grid"),
    Stage("gaps", 3, "Find and rank evidence-backed gaps"),
    Stage("ideate", 4, "Generate ideas from the top gaps"),
    Stage("novelty_search", 4, "Search independently for work close to the idea"),
    Stage("novelty_assess", 4, "Compare the idea with the closest work and assign a level 1-5"),
    Stage("plan", 5, "Draft the experiment plan from the graph"),
    Stage("critique", 5, "Review the plan's feasibility and missing baselines"),
    Stage("targeted_search", 5, "Search for what the critique found missing"),
    Stage("verify", 5, "Check every cited claim against its source"),
    Stage("report", 0, "Write the Novel Evidence Report and the experiment plan"),
)

STAGE_NAMES = tuple(stage.name for stage in STAGES)
