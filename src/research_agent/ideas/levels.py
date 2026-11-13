"""Novelty levels 1-5, assigned by rules from the per-paper comparisons.

| Level | Rule |
| 1 Direct duplication | some paper has the same purpose and the same mechanism |
| 2 Minor variation | some paper has a same or similar purpose and a same or similar mechanism |
| 3 Combination | no paper matches purpose and mechanism together |
| 4 Research gap | as 3, the idea answers a stated or conflict gap, and no paper solves it |
| 5 New research direction | as 4, the gap rests on 2+ independent papers, and every paper's mechanism differs |

The rules fail closed: when the result is unclear, the idea gets the lower level.
A "different" verdict nothing supports counts as "similar" (see ``compare``), a
mechanism the text doesn't state blocks level 5, and the level is capped at 3
and marked inconclusive when the search's coverage is weak, when a comparison
with one of the CHECKED_FIRST closest papers failed, or when fewer than
MIN_COMPARED comparisons succeeded. A paper that already solves the gap caps the
level at 3 too, and the gap is reported as possibly addressed by it. The result
is always "potential" novelty: a search can miss work.
"""

from dataclasses import dataclass, field
from typing import Any

LABELS = {1: "Direct duplication", 2: "Minor variation", 3: "Combination", 4: "Research gap",
          5: "New research direction"}
RATINGS = {1: "Very low", 2: "Low", 3: "Moderate", 4: "Potentially high", 5: "Potentially high"}
CHECKED_FIRST = 3
MIN_COMPARED = 5
EXPLICIT_GAPS = ("stated", "conflict")  # gaps the literature itself states


@dataclass
class Verdict:
    """One closest paper's comparison, as the rules see it."""

    paper_id: str
    title: str
    year: int | None
    position: int
    ok: bool  # the comparison succeeded
    purpose: str = "unknown"  # effective match: same, similar, different, unknown
    mechanism: str = "unknown"
    evaluation: str = "unknown"
    solves_gap: str = "no"  # yes, partly, no
    verified: bool = True  # the matches that count are backed by a passing quote

    @classmethod
    def from_comparison(cls, paper: dict[str, Any], verdict: dict[str, Any]) -> "Verdict":
        if verdict.get("status") != "ok":
            return cls(paper["paper_id"], paper["title"], paper.get("year"), paper["position"], ok=False)
        return cls(
            paper["paper_id"], paper["title"], paper.get("year"), paper["position"], ok=True,
            purpose=verdict["purpose"]["effective"], mechanism=verdict["mechanism"]["effective"],
            evaluation=verdict["evaluation"]["effective"], solves_gap=verdict["solves_gap"],
            verified=all(verdict[f]["verified"] for f in ("purpose", "mechanism") if verdict[f]["effective"] in ("same", "similar")),
        )

    def label(self) -> str:
        return f"{self.title}" + (f" ({self.year})" if self.year else "")


@dataclass
class GapInfo:
    kind: str | None  # stated, structural, conflict; None when the idea answers no stored gap
    papers: int = 0  # independent papers its evidence comes from
    newest: int | None = None  # the newest evidence paper's year


@dataclass
class Assessment:
    level: int
    inconclusive: bool
    reasons: list[str] = field(default_factory=list)
    caps: list[str] = field(default_factory=list)
    deciding: list[str] = field(default_factory=list)  # the papers that set the level
    addressed_by: list[str] = field(default_factory=list)  # papers judged to solve the gap

    @property
    def label(self) -> str:
        return LABELS[self.level]

    @property
    def rating(self) -> str:
        return RATINGS[self.level]

    def as_dict(self) -> dict[str, Any]:
        return {"level": self.level, "label": self.label, "rating": self.rating, "inconclusive": self.inconclusive,
                "reasons": self.reasons, "caps": self.caps, "deciding": self.deciding, "addressed_by": self.addressed_by}


def _close(match: str) -> bool:
    return match in ("same", "similar")


def base_level(verdicts: list[Verdict], gap: GapInfo) -> Assessment:
    """The level the comparisons support, before the coverage caps."""
    done = [v for v in verdicts if v.ok]
    duplicates = [v for v in done if v.purpose == "same" and v.mechanism == "same"]
    if duplicates:
        note = "" if all(v.verified for v in duplicates) else " (not every match is backed by a quote)"
        return Assessment(1, False, [f"{duplicates[0].label()} has the same purpose and the same mechanism{note}."],
                          deciding=[v.paper_id for v in duplicates])
    variants = [v for v in done if _close(v.purpose) and _close(v.mechanism)]
    if variants:
        v = variants[0]
        return Assessment(2, False, [f"{v.label()} has a {v.purpose} purpose and a {v.mechanism} mechanism."],
                          deciding=[x.paper_id for x in variants])

    reasons = [f"None of the {len(done)} closest papers compared matches both the purpose and the mechanism."]
    solvers = [v for v in done if v.solves_gap == "yes"]
    if gap.kind not in EXPLICIT_GAPS:
        why = "the idea answers no stated gap" if gap.kind is None else f"its gap is {gap.kind}, not stated by the papers"
        return Assessment(3, False, reasons + [f"Level 3 at most: {why}."], addressed_by=[v.paper_id for v in solvers])
    if solvers:
        names = ", ".join(v.label() for v in solvers)
        newer = gap.newest is not None and any(v.year and v.year > gap.newest for v in solvers)
        since = f", newer than the gap's evidence ({gap.newest})" if newer else ""
        return Assessment(3, False, reasons + [f"Level 3 at most: the gap is possibly addressed by {names}{since}."],
                          caps=["gap possibly addressed"], addressed_by=[v.paper_id for v in solvers])
    reasons.append(f"It answers a {gap.kind} gap that no compared paper solves.")
    if gap.papers >= 2 and done and all(v.mechanism == "different" for v in done):
        return Assessment(5, False, reasons + [
            f"The gap rests on {gap.papers} independent papers, and every compared paper uses a different mechanism."])
    if gap.papers < 2:
        reasons.append("Level 4, not 5: the gap rests on one paper.")
    else:
        unclear = sum(1 for v in done if v.mechanism != "different")
        reasons.append(f"Level 4, not 5: {unclear} compared paper(s) use a similar or unstated mechanism.")
    return Assessment(4, False, reasons)


def assess(verdicts: list[Verdict], gap: GapInfo, weak_coverage: list[str]) -> Assessment:
    """The level, with the fail-closed caps applied."""
    result = base_level(verdicts, gap)
    if result.level <= 2:
        return result  # a close match was found: weak coverage can't make it less close
    problems = list(weak_coverage)
    failed_first = [v for v in verdicts if not v.ok and v.position <= CHECKED_FIRST]
    if failed_first:
        problems.append(f"the comparison with {failed_first[0].label()}, one of the {CHECKED_FIRST} closest, failed")
    compared = sum(1 for v in verdicts if v.ok)
    if compared < MIN_COMPARED:
        problems.append(f"only {compared} closest papers could be compared")
    if problems:
        result.inconclusive = True
        result.caps += problems
        if result.level > 3:
            result.reasons.append(f"Capped at level 3 (was {result.level}): " + "; ".join(problems) + ".")
            result.level = 3
        else:
            result.reasons.append("Inconclusive: " + "; ".join(problems) + ".")
    return result
