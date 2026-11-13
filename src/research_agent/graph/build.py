"""Build a run's knowledge graph from the checked extractions of its papers.

Nodes: paper; task, method, dataset and metric, merged across papers; and, per
paper, result, claim, limitation, assumption, future_work and facet. Edges:

    paper -addresses-> task            paper -proposes / uses-> method
    paper -evaluates_on-> dataset      paper -measured_by-> metric
    paper -reports-> result, claim, limitation, assumption, future_work
    paper -has_facet-> facet           (attrs.facet: purpose, mechanism, evaluation)
    result -about-> method, result -evaluates_on-> dataset, result -measured_by-> metric
    result or claim -supports / contradicts-> result or claim, across papers

Every node and edge except paper nodes and the result -> entity edges (which
rest on the result's own evidence) has evidence rows: the sentences it rests
on, with the quotes. The whole graph is built in memory and written in one
transaction under the run's ID, replacing any earlier build of the same run.
"""

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from research_agent.embeddings import EmbeddingCache, TextEmbedder
from research_agent.graph.links import Finding, candidate_pairs, label_pairs
from research_agent.graph.names import Entity, merge_names, name_key, vague_name
from research_agent.graph.quotes import QuoteChecker, QuoteStats
from research_agent.graph.store import JudgmentStore
from research_agent.llm import LLM, Role
from research_agent.storage import Database, new_id, utc_now
from research_agent.trace import Event, ref

ENTITY_KINDS = ("task", "method", "dataset", "metric")
STATEMENT_KINDS = {  # extraction list -> node type
    "results": "result", "claims": "claim", "limitations": "limitation",
    "assumptions": "assumption", "future_work": "future_work",
}
_PAPER_EDGE = {"task": "addresses", "dataset": "evaluates_on", "metric": "measured_by"}  # methods: proposes / uses


@dataclass
class PaperFacts:
    paper_id: str
    title: str
    lists: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    facets: dict[str, dict[str, Any] | None] = field(default_factory=dict)


def checked_facts(db: Database, run_id: str, event: Event) -> tuple[dict[str, PaperFacts], QuoteStats]:
    """The run's extractions after the quote check, merged per paper."""
    rows = db.query(
        "SELECT e.*, p.title FROM run_extractions r JOIN extractions e USING (extraction_id)"
        " JOIN papers p USING (paper_id) WHERE r.run_id = ? ORDER BY e.paper_id, e.chunk",
        (run_id,),
    )
    return check_extractions(db, rows, event)


def check_extractions(
    db: Database, rows: list[sqlite3.Row], event: Event | None = None
) -> tuple[dict[str, PaperFacts], QuoteStats]:
    """Extraction rows (with their paper's title) after the quote check, merged per paper."""
    facts: dict[str, PaperFacts] = {}
    total = QuoteStats()
    for row in rows:
        checker = QuoteChecker(_shown_sentences(db, row["paper_id"], json.loads(row["sections"])))
        result = checker.check(json.loads(row["output"]))
        total.add(result.stats)
        for dropped in result.dropped if event is not None else []:
            event.drop(f"{dropped['ref']}@{row['extraction_id']}", dropped["reason"])
        paper = facts.setdefault(row["paper_id"], PaperFacts(row["paper_id"], row["title"]))
        for key, value in result.output.items():
            if key == "facets":
                for name, facet in (value or {}).items():
                    paper.facets[name] = paper.facets.get(name) or facet
            elif isinstance(value, list):
                seen = {_statement_key(item) for item in paper.lists.get(key, [])}
                paper.lists.setdefault(key, []).extend(
                    item for item in value if key not in STATEMENT_KINDS or _statement_key(item) not in seen
                )
    return facts, total


def _shown_sentences(db: Database, paper_id: str, sections: list[int]) -> dict[str, str]:
    rows = db.query(
        f"SELECT t.sentence_id, t.text FROM sentences t JOIN sections s ON s.section_id = t.section_id"
        f" WHERE s.paper_id = ? AND s.idx IN ({','.join('?' * len(sections))}) ORDER BY s.idx, t.idx",
        (paper_id, *sections),
    )
    return {row["sentence_id"]: row["text"] for row in rows}


def _statement_key(item: dict[str, Any]) -> str:
    return " ".join(str(item.get("text", "")).casefold().split())


class GraphDraft:
    """Nodes, edges, aliases and evidence, collected before they are written."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.nodes: list[tuple[str, str, str, dict[str, Any]]] = []  # (id, type, name, attrs)
        self.edges: dict[tuple[str, str, str], tuple[str, dict[str, Any]]] = {}  # (type, src, dst) -> (id, attrs)
        self.aliases: list[tuple[str, str]] = []
        self.evidence: dict[tuple[str, str, str], str] = {}  # (owner kind, owner id, sentence) -> quote
        self.node_evidence: dict[str, list[dict]] = {}

    def node(self, node_type: str, name: str, attrs: dict[str, Any] | None = None, evidence: list[dict] = ()) -> str:
        node_id = new_id("n")
        self.nodes.append((node_id, node_type, name, attrs or {}))
        self.node_evidence[node_id] = list(evidence)
        self._evidence("node", node_id, evidence)
        return node_id

    def edge(
        self, edge_type: str, src: str, dst: str, attrs: dict[str, Any] | None = None, evidence: list[dict] = ()
    ) -> str:
        """One edge per (type, source, target); adding it again adds evidence and attributes."""
        key = (edge_type, src, dst)
        if key not in self.edges:
            self.edges[key] = (new_id("e"), {})
        edge_id, existing = self.edges[key]
        existing.update(attrs or {})
        self._evidence("edge", edge_id, evidence)
        return edge_id

    def _evidence(self, kind: str, owner: str, evidence: list[dict]) -> None:
        for item in evidence:
            self.evidence.setdefault((kind, owner, item["sentence"]), item["quote"])

    def counts(self) -> tuple[Counter[str], Counter[str]]:
        return Counter(n[1] for n in self.nodes), Counter(key[0] for key in self.edges)

    def write(self, db: Database) -> None:
        now = utc_now()
        with db.transaction() as conn:
            clear_run_graph(conn, self.run_id)
            conn.executemany(
                "INSERT INTO nodes (node_id, node_type, name, attrs, created_at, run_id) VALUES (?, ?, ?, ?, ?, ?)",
                [(i, t, name, json.dumps(attrs, ensure_ascii=False), now, self.run_id) for i, t, name, attrs in self.nodes],
            )
            conn.executemany(
                "INSERT INTO edges (edge_id, edge_type, src_id, dst_id, attrs, created_at, run_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(i, t, src, dst, json.dumps(attrs, ensure_ascii=False), now, self.run_id)
                 for (t, src, dst), (i, attrs) in self.edges.items()],
            )
            conn.executemany("INSERT OR IGNORE INTO node_aliases (node_id, alias) VALUES (?, ?)", self.aliases)
            conn.executemany(
                "INSERT INTO evidence (owner_kind, owner_id, sentence_id, quote) VALUES (?, ?, ?, ?)",
                [(*key, quote) for key, quote in self.evidence.items()],
            )


def clear_run_graph(conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute(
        "DELETE FROM evidence WHERE owner_kind = 'edge' AND owner_id IN (SELECT edge_id FROM edges WHERE run_id = ?)",
        (run_id,),
    )
    conn.execute(
        "DELETE FROM evidence WHERE owner_kind = 'node' AND owner_id IN (SELECT node_id FROM nodes WHERE run_id = ?)",
        (run_id,),
    )
    conn.execute("DELETE FROM edges WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM nodes WHERE run_id = ?", (run_id,))  # aliases go with them


def build_run_graph(
    db: Database, llm: LLM, embedder: TextEmbedder | None, run_id: str, event: Event
) -> dict[str, Any]:
    """Build and store the run's graph; returns its summary for the report and trace."""
    facts, quotes = checked_facts(db, run_id, event)
    judgments = JudgmentStore(db, llm.models(Role.EXTRACTION))
    draft = GraphDraft(run_id)

    papers = db.query(
        "SELECT p.paper_id, p.title, p.year, r.read_level FROM run_papers r JOIN papers p USING (paper_id)"
        " WHERE r.run_id = ? AND r.read_level IN ('full', 'abstract') ORDER BY r.rank IS NULL, r.rank",
        (run_id,),
    )
    paper_nodes = {
        row["paper_id"]: draft.node("paper", row["title"], {
            "paper_id": row["paper_id"], "year": row["year"], "read_level": row["read_level"],
            "extracted": row["paper_id"] in facts,
        })
        for row in papers
    }

    # Names: every mention of a task, method, dataset or metric, merged across papers.
    # Vague names ("several benchmarks", "large language models") get no node.
    mentions: dict[str, Counter[str]] = {kind: Counter() for kind in ENTITY_KINDS}
    vague: Counter[str] = Counter()
    for paper in facts.values():
        named = [(kind, item["name"]) for kind in ENTITY_KINDS for item in paper.lists.get(f"{kind}s", [])]
        named += [(kind, result[kind]) for result in paper.lists.get("results", [])
                  for kind in ("method", "dataset", "metric") if result.get(kind)]
        for kind, name in named:
            if vague_name(kind, name):
                vague[kind] += 1
            else:
                mentions[kind][name] += 1
    entity_nodes: dict[str, dict[str, str]] = {}  # kind -> name key -> node ID
    name_stats: dict[str, dict[str, int]] = {}
    for kind in ENTITY_KINDS:
        entities, name_stats[kind] = merge_names(llm, judgments, kind, mentions[kind])
        entity_nodes[kind] = {}
        made: dict[int, str] = {}
        for key, entity in entities.items():
            if id(entity) not in made:
                made[id(entity)] = _entity_node(draft, kind, entity)
            entity_nodes[kind][key] = made[id(entity)]

    def entity(kind: str, name: str | None) -> str | None:
        return entity_nodes[kind].get(name_key(name)) if name and not vague_name(kind, name) else None

    findings: list[Finding] = []
    for paper in facts.values():
        source = paper_nodes.get(paper.paper_id)
        if source is None:  # extracted, but no longer in the run's corpus
            continue
        for kind in ENTITY_KINDS:
            for item in paper.lists.get(f"{kind}s", []):
                target = entity(kind, item["name"])
                if target is None:
                    continue
                edge_type = ("proposes" if item.get("role") == "proposed" else "uses") if kind == "method" else _PAPER_EDGE[kind]
                draft.edge(edge_type, source, target, evidence=item["evidence"])
        for key, node_type in STATEMENT_KINDS.items():
            for item in paper.lists.get(key, []):
                attrs = {"paper_id": paper.paper_id}
                if node_type == "result":
                    attrs |= {k: item.get(k) for k in ("method", "dataset", "metric", "conditions")}
                node = draft.node(node_type, item["text"], attrs, item["evidence"])
                draft.edge("reports", source, node, evidence=item["evidence"])
                if node_type == "result":
                    _link_result(draft, source, node, item, entity)
                if node_type in ("result", "claim"):
                    findings.append(Finding(
                        node, paper.paper_id, paper.title, node_type, item["text"], item["evidence"][0]["quote"],
                        dataset=entity("dataset", item.get("dataset")) if node_type == "result" else None,
                        metric=entity("metric", item.get("metric")) if node_type == "result" else None,
                        conditions=item.get("conditions"),
                    ))
        for facet, item in paper.facets.items():
            if item:
                node = draft.node("facet", item["text"], {"paper_id": paper.paper_id, "facet": facet}, item["evidence"])
                draft.edge("has_facet", source, node, {"facet": facet}, item["evidence"])

    links = _link_findings(db, llm, judgments, embedder, draft, findings, event)
    draft.write(db)
    event.generated.append(ref("graph", run_id))
    nodes, edges = draft.counts()
    return {
        "papers": {"in_corpus": len(paper_nodes), "extracted": sum(p in paper_nodes for p in facts)},
        "quote_check": quotes.as_dict(),
        "nodes": dict(nodes),
        "edges": dict(edges),
        "names": {kind: {**stats, "vague_dropped": vague[kind]} for kind, stats in name_stats.items()},
        "links": links,
    }


def _entity_node(draft: GraphDraft, kind: str, entity: Entity) -> str:
    node = draft.node(kind, entity.name, {"mentions": sum(entity.aliases.values())})
    draft.aliases += [(node, alias) for alias in entity.aliases]
    return node


def _link_result(draft: GraphDraft, paper: str, result: str, item: dict[str, Any], entity: Any) -> None:
    """A result's method, dataset and metric: linked from the result, and from its paper."""
    for kind, edge_type in (("method", "about"), ("dataset", "evaluates_on"), ("metric", "measured_by")):
        target = entity(kind, item.get(kind))
        if target is None:
            continue
        draft.edge(edge_type, result, target, {"from": "result"})
        if kind != "method":
            draft.edge(edge_type, paper, target, evidence=item["evidence"])


def _link_findings(
    db: Database,
    llm: LLM,
    judgments: JudgmentStore,
    embedder: TextEmbedder | None,
    draft: GraphDraft,
    findings: list[Finding],
    event: Event,
) -> dict[str, Any]:
    vectors: dict[str, Any] = {}
    if embedder is None:
        embedding = {"status": "no embedding model set (RA_EMBEDDING): similar-text pairs skipped"}
    elif findings:
        vectors, embedding = EmbeddingCache(db, embedder).vectors([f.text for f in findings])
    else:
        embedding = {"status": "nothing to embed"}
    pairs = candidate_pairs(findings, vectors)
    labelled, stats = label_pairs(llm, judgments, pairs)
    counts = Counter()
    for pair, label in labelled:
        relation = label["relation"]
        if relation == "contradicts" and not label["same_setting"]:
            counts["contradicts_other_setting"] += 1
            event.drop(ref("pair", f"{pair.a.node_id}|{pair.b.node_id}"), f"contradicts, but not in the same setting: {label['reason']}")
            continue
        if relation == "unrelated":
            counts["unrelated"] += 1
            continue
        counts[relation] += 1
        attrs = {"similarity": None if pair.similarity is None else round(pair.similarity, 4),
                 "found_by": pair.found_by, "same_setting": label["same_setting"], "reason": label["reason"]}
        evidence = draft.node_evidence[pair.a.node_id] + draft.node_evidence[pair.b.node_id]
        draft.edge(relation, pair.a.node_id, pair.b.node_id, attrs, evidence)
    found_by = Counter(source for pair in pairs for source in pair.found_by)
    return {"findings": len(findings), "candidates": len(pairs), **dict(found_by), **stats, **dict(counts), "embedding": embedding}


def load_graph(db: Database, run_id: str) -> nx.MultiDiGraph:
    """The run's graph in NetworkX. Node data: node_type, name, aliases, attrs; edge data
    (keyed by edge ID): edge_type, attrs."""
    graph = nx.MultiDiGraph()
    aliases: dict[str, list[str]] = {}
    for row in db.query(
        "SELECT a.node_id, a.alias FROM node_aliases a JOIN nodes n USING (node_id) WHERE n.run_id = ?", (run_id,)
    ):
        aliases.setdefault(row["node_id"], []).append(row["alias"])
    for row in db.query("SELECT * FROM nodes WHERE run_id = ?", (run_id,)):
        graph.add_node(row["node_id"], node_type=row["node_type"], name=row["name"],
                       aliases=aliases.get(row["node_id"], []), attrs=json.loads(row["attrs"]))
    for row in db.query("SELECT * FROM edges WHERE run_id = ?", (run_id,)):
        graph.add_edge(row["src_id"], row["dst_id"], key=row["edge_id"], edge_type=row["edge_type"],
                       attrs=json.loads(row["attrs"]))
    return graph
