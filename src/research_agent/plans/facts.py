"""The facts an experiment plan can cite: what the literature around the idea reports.

The plan papers are the idea's closest papers (from its novelty search), the papers its
gap's evidence comes from, the run's related work, and those a targeted search adds for
the critique. Each is extracted as in Phase 2 (kept extractions are reused) and every
item goes through the quote check. From them and from the run's knowledge graph, code
builds a numbered menu:

- the idea's gap, with its evidence;
- the closest papers' own approach, as the novelty comparison quoted it;
- datasets, methods (possible baselines) and metrics, merged by name with the graph's
  names, each with the papers that use it and the sentences that show it;
- results with their method, dataset, metric and conditions: the plan papers' own, and
  those any corpus paper reports on a menu dataset (from the graph);
- limitations and other findings the plan papers report.

Each fact keeps its evidence (sentence addresses with quotes) and its origin: the
extraction or graph query it came from, and why its papers are plan papers. Code ranks
the menu, so the same inputs give the same menu: entities the idea itself names first,
then by how many plan papers use them, the closest papers counting most. A revision's
menu keeps every earlier fact under its number and adds what is new.
"""

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from research_agent.graph.build import PaperFacts, check_extractions, load_graph
from research_agent.graph.names import name_key, vague_name
from research_agent.ideas.ideas import Idea
from research_agent.storage import Database

ROLE_WEIGHTS = {"closest": 2.0, "gap": 1.5, "related": 1.0, "search": 2.0}
TOP_CLOSEST = 3  # the closest papers count CLOSEST_BONUS more
CLOSEST_BONUS = 1.0
CORPUS_WEIGHT = 0.5  # a corpus paper that is no plan paper but reports a result on a menu dataset
NAMED = 10.0  # an entity the idea names comes first
CAPS = {"approach": 5, "dataset": 12, "method": 15, "metric": 10, "result": 24, "limitation": 10, "finding": 8}
NEW_FACTS = 20  # at most this many facts a revision's menu adds
MAX_EVIDENCE = 3  # quotes kept per fact
KIND_ORDER = ("gap", "approach", "dataset", "method", "metric", "result", "limitation", "finding")
ENTITY_KINDS = ("dataset", "method", "metric")
KIND_TITLES = {
    "gap": "Research gap",
    "approach": "Closest existing work (from the novelty check), each paper's own approach",
    "dataset": "Datasets and benchmarks",
    "method": "Methods (possible baselines)",
    "metric": "Metrics",
    "result": "Reported results",
    "limitation": "Limitations the papers report",
    "finding": "Other findings of the closest papers",
}
TITLE_CHARS = 90


# Plan papers -------------------------------------------------------------------------------

@dataclass
class PlanPaper:
    paper_id: str
    title: str
    year: int | None
    roles: dict[str, dict[str, Any]] = field(default_factory=dict)  # role -> {"position", **detail}

    @property
    def weight(self) -> float:
        best = 0.0
        for role, info in self.roles.items():
            weight = ROLE_WEIGHTS[role]
            if role == "closest" and (info.get("position") or 99) <= TOP_CLOSEST:
                weight += CLOSEST_BONUS
            best = max(best, weight)
        return best

    def label(self) -> str:
        title = self.title if len(self.title) <= TITLE_CHARS else self.title[: TITLE_CHARS - 3] + "..."
        return f"{title} ({self.year})" if self.year else title

    def why(self) -> list[str]:
        reasons = []
        for role, info in self.roles.items():
            if role == "closest":
                reasons.append(f"closest paper {info.get('position')} to the idea (novelty search)")
            elif role == "gap":
                reasons.append("evidence for the idea's gap")
            elif role == "related":
                reasons.append(f"related work {info.get('position')} (closest corpus papers to the question)")
            else:
                reasons.append(f"targeted search for {info.get('query')!r} (round {info.get('round')}, {info.get('channel')})")
        return reasons

    def extractions(self) -> list[str]:
        return list(dict.fromkeys(x for info in self.roles.values() for x in info.get("extractions", [])))


def load_plan_papers(db: Database, run_id: str) -> dict[str, PlanPaper]:
    papers: dict[str, PlanPaper] = {}
    for row in db.query(
        "SELECT r.paper_id, r.role, r.position, r.detail, p.title, p.year FROM plan_papers r JOIN papers p USING (paper_id)"
        " WHERE r.run_id = ? ORDER BY r.role, r.position", (run_id,),
    ):
        paper = papers.setdefault(row["paper_id"], PlanPaper(row["paper_id"], row["title"], row["year"]))
        paper.roles[row["role"]] = {"position": row["position"], **json.loads(row["detail"])}
    return dict(sorted(papers.items(), key=lambda item: (-item[1].weight, item[0])))


def add_plan_paper(db: Database, run_id: str, paper_id: str, role: str, position: int | None,
                   detail: dict[str, Any]) -> None:
    db.execute(
        "INSERT OR REPLACE INTO plan_papers (run_id, paper_id, role, position, detail) VALUES (?, ?, ?, ?, ?)",
        (run_id, paper_id, role, position, json.dumps(detail, ensure_ascii=False)),
    )


# Facts and the menu ----------------------------------------------------------------------------

@dataclass
class Fact:
    fact_id: str
    kind: str
    key: str  # what makes two facts the same, so a later menu keeps their number
    text: str
    paper_ids: list[str]
    evidence: list[dict[str, str]]  # {"sentence_id", "quote"}
    origin: dict[str, Any]
    attrs: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    added: int = 1  # the plan round whose menu first had it


@dataclass
class Menu:
    facts: dict[str, Fact]  # by ID, in menu order
    papers: dict[str, PlanPaper]

    def of_kind(self, kind: str) -> list[Fact]:
        return [fact for fact in self.facts.values() if fact.kind == kind]

    def get(self, cite: str) -> Fact | None:
        """The fact a citation names: "F7", "f7", "[F7]" or "7"."""
        digits = re.sub(r"[^0-9]", "", cite)
        return self.facts.get(f"F{int(digits)}") if digits else None

    def as_json(self) -> list[dict[str, Any]]:
        return [asdict(fact) for fact in self.facts.values()]

    @classmethod
    def from_json(cls, items: list[dict[str, Any]], papers: dict[str, PlanPaper]) -> "Menu":
        return cls({item["fact_id"]: Fact(**item) for item in items}, papers)

    def label(self, paper_id: str) -> str:
        paper = self.papers.get(paper_id)
        return paper.label() if paper else paper_id

    def render(self, new_since: int | None = None) -> list[str]:
        """The menu as the plan writer sees it, grouped by kind. Facts first added after
        round ``new_since`` are marked (new)."""
        lines: list[str] = []
        for kind in KIND_ORDER:
            facts = self.of_kind(kind)
            if not facts:
                continue
            lines += ["", f"{KIND_TITLES[kind]}:"]
            for fact in facts:
                new = " (new)" if new_since is not None and fact.added > new_since else ""
                lines.append(f"[{fact.fact_id}]{new} {self.line(fact)}")
        return lines

    def line(self, fact: Fact) -> str:
        papers = [self.label(p) for p in fact.paper_ids]
        if fact.kind == "gap":
            quotes = "; ".join(f'"{e["quote"]}" ({self.label(e["paper_id"])})' for e in fact.attrs.get("quoted", []))
            return f"({fact.attrs.get('gap_kind')} gap) {fact.text} Evidence: {quotes}"
        if fact.kind == "approach":
            a = fact.attrs
            return (f"{papers[0]}, closest paper {a['position']}: {fact.text} (the novelty check found its purpose "
                    f"{a['purpose']} and its mechanism {a['mechanism']} to the idea's)")
        if fact.kind in ENTITY_KINDS:
            parts = []
            proposed = [self.label(p) for p in fact.attrs.get("proposed_by", [])]
            if proposed:
                parts.append("proposed by " + ", ".join(proposed[:2]))
            users = [p for p in papers if p not in proposed]
            if users:
                more = f" and {len(users) - 3} more" if len(users) > 3 else ""
                parts.append("used by " + ", ".join(users[:3]) + more)
            if fact.attrs.get("results"):
                parts.append(f"{fact.attrs['results']} reported results")
            if fact.attrs.get("named"):
                parts.append("named in the idea")
            return f"{fact.text}: " + "; ".join(parts)
        setting = ""
        if fact.kind == "result":
            fields = [f"{k} {fact.attrs[k]}" for k in ("method", "dataset", "metric") if fact.attrs.get(k)]
            if fact.attrs.get("conditions"):
                fields.append(f"setting: {fact.attrs['conditions']}")
            setting = f" [{'; '.join(fields)}]" if fields else ""
        return f"{fact.text} ({papers[0] if papers else 'corpus'}){setting}"


def idea_text(idea: Idea) -> str:
    return " ".join([idea.title, idea.purpose, idea.mechanism, idea.evaluation, *idea.detail.get("builds_on", [])])


def mentioned_keys(text: str) -> set[str]:
    """The name keys of every run of 1-5 words in ``text``: a name the text mentions has its key here."""
    tokens = re.findall(r"\w+(?:[-.+']\w+)*", text)
    keys = set()
    for size in range(1, 6):
        for start in range(len(tokens) - size + 1):
            key = name_key(" ".join(tokens[start : start + size]))
            if len(key) >= 2:
                keys.add(key)
    return keys


def _evidence(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extraction evidence ({"sentence", "quote", "check"}) as stored evidence."""
    return [{"sentence_id": e["sentence"], "quote": e["quote"]} for e in items]


@dataclass
class _Entity:
    kind: str
    key: str
    name: str
    node_id: str | None
    keys: set[str] = field(default_factory=set)
    papers: dict[str, list[dict[str, str]]] = field(default_factory=dict)  # paper -> evidence
    proposed_by: set[str] = field(default_factory=set)
    results: int = 0


class _Names:
    """Entity names merged the way the run's graph merged them."""

    def __init__(self, graph: Any) -> None:
        self.by_key: dict[tuple[str, str], tuple[str, str, set[str]]] = {}  # (kind, key) -> (name, node, keys)
        for node, data in graph.nodes(data=True):
            if data["node_type"] in ENTITY_KINDS:
                keys = {name_key(data["name"]), *(name_key(a) for a in data["aliases"])}
                for key in keys:
                    self.by_key[(data["node_type"], key)] = (data["name"], node, keys)
        self.entities: dict[tuple[str, str], _Entity] = {}

    def entity(self, kind: str, name: str) -> _Entity | None:
        if not name or vague_name(kind, name):
            return None
        key = name_key(name)
        known = self.by_key.get((kind, key))
        canonical, node, keys = known if known else (name, None, {key})
        ckey = name_key(canonical)
        if (kind, ckey) not in self.entities:
            self.entities[(kind, ckey)] = _Entity(kind, ckey, canonical, node, set(keys))
        entity = self.entities[(kind, ckey)]
        entity.keys.add(key)
        return entity

    def mention(self, kind: str, name: str | None, paper_id: str, evidence: list[dict[str, str]]) -> _Entity | None:
        entity = self.entity(kind, name or "")
        if entity is not None:
            held = entity.papers.setdefault(paper_id, [])
            held += [e for e in evidence if e not in held]
        return entity


def _weight(papers: dict[str, PlanPaper], paper_id: str) -> float:
    paper = papers.get(paper_id)
    return paper.weight if paper else CORPUS_WEIGHT


def _top(entities: list[_Entity], score: Any, cap: int) -> list[tuple[_Entity, float]]:
    scored = [(entity, score(entity)) for entity in entities if entity.papers]
    return sorted(scored, key=lambda item: (-item[1], item[0].name.casefold()))[:cap]


def _pick_evidence(papers: dict[str, PlanPaper], by_paper: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    """Up to MAX_EVIDENCE quotes, one per paper first, the heaviest papers first."""
    order = sorted(by_paper, key=lambda p: (-_weight(papers, p), p))
    picked = [by_paper[p][0] for p in order if by_paper[p]][:MAX_EVIDENCE]
    for p in order:
        for e in by_paper[p][1:]:
            if len(picked) < MAX_EVIDENCE and e not in picked:
                picked.append(e)
    return picked


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def _mentions(text: str, names: list[str]) -> bool:
    words = f" {_norm(text)} "
    return any(f" {_norm(name)} " in words for name in names if len(_norm(name)) >= 3)


def build_menu(
    db: Database, run_id: str, idea: Idea, gap: dict[str, Any] | None, comparisons: list[dict[str, Any]],
    previous: Menu | None = None, round_: int = 1,
) -> Menu:
    """The facts the plan can cite. ``gap`` is the idea's gap as ``load_gaps`` gives it;
    ``comparisons`` its closest papers' rows (paper_id, position, title, year, comparison_id,
    verdict). A revision passes the previous menu: its facts keep their numbers."""
    papers = load_plan_papers(db, run_id)
    ids = [x for paper in papers.values() for x in paper.extractions()]
    rows = []
    for start in range(0, len(ids), 500):
        part = ids[start : start + 500]
        rows += db.query(
            f"SELECT e.*, p.title FROM extractions e JOIN papers p USING (paper_id)"
            f" WHERE e.extraction_id IN ({','.join('?' * len(part))}) ORDER BY e.paper_id, e.chunk", tuple(part))
    paper_facts, _ = check_extractions(db, rows)
    graph = load_graph(db, run_id)
    names = _Names(graph)
    named = mentioned_keys(idea_text(idea))
    why = {p.paper_id: p.why() for p in papers.values()}

    results: list[tuple[str, dict[str, Any], dict[str, Any]]] = []  # (paper, item, origin)
    statements: list[tuple[str, str, dict[str, Any]]] = []  # (kind, paper, item)
    for paper_id, facts in paper_facts.items():
        if paper_id not in papers:
            continue
        _collect(names, paper_id, facts, results, statements)

    def origin(query: str, paper_ids: list[str], **more: Any) -> dict[str, Any]:
        return {"from": "extraction", "query": query, **more,
                "papers": {p: {"why": why.get(p, ["corpus paper of the run"]),
                               "extractions": papers[p].extractions() if p in papers else []} for p in paper_ids}}

    built: list[Fact] = []

    # The gap, and the closest papers' own approach.
    if gap is not None:
        quoted = [{"paper_id": e["paper_id"], "quote": e["quote"]} for e in gap["evidence"][:MAX_EVIDENCE]]
        built.append(Fact("", "gap", f"gap:{gap['node_id']}", gap["gap"],
                          list(dict.fromkeys(e["paper_id"] for e in gap["evidence"])),
                          [{"sentence_id": e["sentence_id"], "quote": e["quote"]} for e in gap["evidence"][:MAX_EVIDENCE]],
                          {"from": "gap", "query": "the gap the idea answers (Phase 3)", "node_id": gap["node_id"]},
                          {"gap_kind": gap["kind"], "warrant": gap.get("warrant"), "quoted": quoted}))
    approaches = 0
    for row in comparisons:
        verdict = row["verdict"]
        if verdict.get("status") != "ok" or approaches >= CAPS["approach"]:
            continue
        mechanism = verdict["mechanism"]
        if not mechanism.get("paper") or not mechanism.get("evidence"):
            continue
        approaches += 1
        built.append(Fact("", "approach", f"approach:{row['paper_id']}", mechanism["paper"], [row["paper_id"]],
                          mechanism["evidence"][:MAX_EVIDENCE],
                          {"from": "novelty comparison", "query": "the idea compared with its closest papers (Phase 4)",
                           "comparison_id": row["comparison_id"], "depth": verdict.get("depth"),
                           "papers": {row["paper_id"]: {"why": why.get(row["paper_id"], [])}}},
                          {"position": row["position"], "purpose": verdict["purpose"]["effective"],
                           "mechanism": mechanism["effective"], "solves_gap": verdict.get("solves_gap")},
                          score=10.0 - row["position"]))

    # Datasets first: the corpus results on them, and so the methods and metrics, depend on them.
    def entity_fact(entity: _Entity, score: float, query: str) -> Fact:
        paper_ids = sorted(entity.papers, key=lambda p: (-_weight(papers, p), p))
        attrs = {"proposed_by": sorted(entity.proposed_by & set(paper_ids)), "results": entity.results,
                 "named": bool(entity.keys & named), "node_id": entity.node_id, "keys": sorted(entity.keys)}
        return Fact("", entity.kind, f"{entity.kind}:{entity.key}", entity.name, paper_ids,
                    _pick_evidence(papers, entity.papers), origin(query, paper_ids, graph_node=entity.node_id), attrs,
                    score=round(score, 3))

    def plain_score(entity: _Entity) -> float:
        return NAMED * bool(entity.keys & named) + sum(_weight(papers, p) for p in entity.papers)

    datasets = _top([e for e in names.entities.values() if e.kind == "dataset"], plain_score, CAPS["dataset"])
    top_datasets = {e.key for e, _ in datasets}
    dataset_nodes = {e.node_id for e, _ in datasets if e.node_id}
    built += [entity_fact(e, s, "datasets and benchmarks the plan papers evaluate on") for e, s in datasets]

    seen_results = {(paper, _norm(item["text"])) for paper, item, _ in results}
    for node, data in graph.nodes(data=True):
        if data["node_type"] != "result":
            continue
        on = {dst for _, dst, d in graph.out_edges(node, data=True) if d["edge_type"] == "evaluates_on"}
        paper_id = data["attrs"].get("paper_id")
        if not on & dataset_nodes or paper_id in papers or (paper_id, _norm(data["name"])) in seen_results:
            continue
        evidence = [dict(r) for r in db.query(
            "SELECT sentence_id, quote FROM evidence WHERE owner_kind = 'node' AND owner_id = ? ORDER BY sentence_id",
            (node,))]
        if not evidence:
            continue
        item = {"text": data["name"], "evidence": [{"sentence": e["sentence_id"], "quote": e["quote"]} for e in evidence],
                **{k: data["attrs"].get(k) for k in ("method", "dataset", "metric", "conditions")}}
        for kind in ("method", "metric"):
            names.mention(kind, item.get(kind), paper_id, evidence)
        results.append((paper_id, item, {"from": "graph", "node_id": node,
                                         "query": "results reported on the menu's datasets in the run's knowledge graph"}))
        seen_results.add((paper_id, _norm(data["name"])))

    def on_top_dataset(item: dict[str, Any]) -> bool:
        entity = names.entity("dataset", item.get("dataset") or "")
        return entity is not None and entity.key in top_datasets

    for _, item, _ in results:
        if on_top_dataset(item):
            for kind in ("method", "metric"):
                entity = names.entity(kind, item.get(kind) or "")
                if entity is not None:
                    entity.results += 1

    def method_score(entity: _Entity) -> float:
        return (NAMED * bool(entity.keys & named) + 0.5 * entity.results
                + sum(_weight(papers, p) * (1.5 if p in entity.proposed_by else 1.0) for p in entity.papers))

    def metric_score(entity: _Entity) -> float:
        return plain_score(entity) + 0.5 * entity.results

    methods = _top([e for e in names.entities.values() if e.kind == "method"], method_score, CAPS["method"])
    metrics = _top([e for e in names.entities.values() if e.kind == "metric"], metric_score, CAPS["metric"])
    built += [entity_fact(e, s, "methods the plan papers propose or use, and methods with results on the menu's datasets")
              for e, s in methods]
    built += [entity_fact(e, s, "metrics the plan papers report") for e, s in metrics]
    top = {kind: {e.key for e, _ in chosen} for kind, chosen in (("method", methods), ("metric", metrics))}
    top_names = [e.name for e, _ in datasets + methods]

    scored_results = []
    for paper_id, item, where in results:
        keys = {kind: (names.entity(kind, item.get(kind) or "") or _Entity(kind, "", "", None)).key
                for kind in ("dataset", "method", "metric")}
        score = (4.0 * (keys["dataset"] in top_datasets) + 2.0 * (keys["method"] in top["method"])
                 + 1.0 * (keys["metric"] in top["metric"]) + _weight(papers, paper_id))
        source = where if where else {}
        fact_origin = ({**source, "papers": {paper_id: {"why": why.get(paper_id, ["corpus paper of the run"])}}}
                       if source.get("from") == "graph" else origin("results the plan papers report", [paper_id]))
        scored_results.append(Fact(
            "", "result", f"result:{paper_id}:{_norm(item['text'])[:120]}", item["text"], [paper_id],
            _evidence(item["evidence"])[:MAX_EVIDENCE], fact_origin,
            {k: item.get(k) for k in ("method", "dataset", "metric", "conditions")}, score=round(score, 3)))
    built += sorted(scored_results, key=lambda f: (-f.score, f.key))[: CAPS["result"]]

    for kind, query in (("limitation", "limitations the plan papers report"),
                        ("finding", "claims the closest papers and the gap's papers make")):
        facts = []
        for statement_kind, paper_id, item in statements:
            if statement_kind != kind:
                continue
            score = _weight(papers, paper_id) + 2.0 * _mentions(item["text"], top_names)
            facts.append(Fact("", kind, f"{kind}:{paper_id}:{_norm(item['text'])[:120]}", item["text"], [paper_id],
                              _evidence(item["evidence"])[:MAX_EVIDENCE], origin(query, [paper_id]), score=round(score, 3)))
        built += sorted(facts, key=lambda f: (-f.score, f.key))[: CAPS[kind]]

    return _number(built, papers, previous, round_)


def _collect(names: _Names, paper_id: str, facts: PaperFacts, results: list, statements: list) -> None:
    for kind in ENTITY_KINDS:
        for item in facts.lists.get(f"{kind}s", []):
            entity = names.mention(kind, item["name"], paper_id, _evidence(item["evidence"]))
            if entity is not None and kind == "method" and item.get("role") == "proposed":
                entity.proposed_by.add(paper_id)
    for item in facts.lists.get("results", []):
        for kind in ENTITY_KINDS:
            names.mention(kind, item.get(kind), paper_id, _evidence(item["evidence"]))
        results.append((paper_id, item, None))
    for item in facts.lists.get("limitations", []):
        statements.append(("limitation", paper_id, item))
    for item in facts.lists.get("claims", []):
        statements.append(("finding", paper_id, item))


def _number(built: list[Fact], papers: dict[str, PlanPaper], previous: Menu | None, round_: int) -> Menu:
    """Give the facts their numbers, in menu order. A revision keeps the previous menu as it
    was and appends up to NEW_FACTS new facts: from search papers first."""
    order = {kind: i for i, kind in enumerate(KIND_ORDER)}
    if previous is None:
        facts = sorted(built, key=lambda f: order[f.kind])  # stable: each kind keeps its ranking
        for number, fact in enumerate(facts, start=1):
            fact.fact_id, fact.added = f"F{number}", round_
        return Menu({fact.fact_id: fact for fact in facts}, papers)
    held = {fact.key for fact in previous.facts.values()}

    def from_search(fact: Fact) -> bool:
        return any("search" in papers[p].roles for p in fact.paper_ids if p in papers)

    new = sorted((f for f in built if f.key not in held), key=lambda f: (not from_search(f), order[f.kind], -f.score))
    facts = dict(previous.facts)
    for number, fact in enumerate(new[:NEW_FACTS], start=len(facts) + 1):
        fact.fact_id, fact.added = f"F{number}", round_
        facts[fact.fact_id] = fact
    return Menu(facts, papers)
