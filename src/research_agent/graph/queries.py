"""Questions asked of a run's graph. Names are matched the way names were merged
(``name_key``), against every alias; failing that, by containment."""

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from research_agent.graph.names import name_key
from research_agent.storage import Database


@dataclass
class Answer:
    """One row of a query's answer: what was found, how, and the sentences behind it."""

    subject: str
    detail: str
    via: str  # the node or edge the row rests on
    evidence: list[dict[str, str]] = field(default_factory=list)  # {"sentence_id", "quote"}


def find_nodes(graph: nx.MultiDiGraph, node_type: str, name: str) -> list[str]:
    wanted = name_key(name)
    nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == node_type]

    def keys(node: str) -> set[str]:
        data = graph.nodes[node]
        return {name_key(data["name"]), *(name_key(alias) for alias in data["aliases"])}

    exact = [n for n in nodes if wanted in keys(n)]
    return exact or [n for n in nodes if any(wanted in key or key in wanted for key in keys(n) if len(key) > 2)]


def _out(graph: nx.MultiDiGraph, node: str, edge_type: str) -> list[tuple[str, str]]:
    """(target, edge ID) for the node's outgoing edges of one type."""
    return [(dst, key) for _, dst, key, d in graph.out_edges(node, keys=True, data=True) if d["edge_type"] == edge_type]


def _in(graph: nx.MultiDiGraph, node: str, edge_type: str) -> list[tuple[str, str]]:
    return [(src, key) for src, _, key, d in graph.in_edges(node, keys=True, data=True) if d["edge_type"] == edge_type]


def _name(graph: nx.MultiDiGraph, node: str) -> str:
    return graph.nodes[node]["name"]


def methods_on_dataset(graph: nx.MultiDiGraph, db: Database, dataset: str) -> list[Answer]:
    """Methods evaluated on a dataset: from results that report them on it, then from
    papers that use the dataset and propose or use the method."""
    answers: list[Answer] = []
    for node in find_nodes(graph, "dataset", dataset):
        for result, _ in _in(graph, node, "evaluates_on"):
            if graph.nodes[result]["node_type"] != "result":
                continue
            for method, _ in _out(graph, result, "about"):
                answers.append(Answer(_name(graph, method), _name(graph, result), result, node_evidence(db, result)))
        reported = {a.subject for a in answers}
        for paper, _ in _in(graph, node, "evaluates_on"):
            if graph.nodes[paper]["node_type"] != "paper":
                continue
            for edge_type in ("proposes", "uses"):
                for method, edge in _out(graph, paper, edge_type):
                    if _name(graph, method) not in reported:
                        detail = f"{edge_type} it in \"{_name(graph, paper)}\", which evaluates on {_name(graph, node)}"
                        answers.append(Answer(_name(graph, method), detail, edge, edge_evidence(db, edge)))
    return answers


def limitations_for_task(graph: nx.MultiDiGraph, db: Database, task: str) -> list[Answer]:
    """Limitations reported by the papers that address a task."""
    answers: list[Answer] = []
    for node in find_nodes(graph, "task", task):
        for paper, _ in _in(graph, node, "addresses"):
            for statement, _ in _out(graph, paper, "reports"):
                if graph.nodes[statement]["node_type"] == "limitation":
                    answers.append(Answer(_name(graph, paper), _name(graph, statement), statement, node_evidence(db, statement)))
    return answers


def findings_about_method(graph: nx.MultiDiGraph, db: Database, method: str) -> list[Answer]:
    """Results about a method, each followed by the findings that support or contradict it."""
    answers: list[Answer] = []
    for node in find_nodes(graph, "method", method):
        for result, _ in _in(graph, node, "about"):
            answers.append(Answer(_name(graph, node), _name(graph, result), result, node_evidence(db, result)))
            for relation in ("supports", "contradicts"):
                for other, edge in [*_out(graph, result, relation), *_in(graph, result, relation)]:
                    answers.append(Answer(f"  {relation}", _name(graph, other), edge, node_evidence(db, other)))
    return answers


def conflicts(graph: nx.MultiDiGraph, db: Database) -> list[Answer]:
    """Every pair of findings that contradict each other in the same setting."""
    return [
        Answer(_name(graph, a), f"contradicts: {_name(graph, b)} ({d['attrs'].get('reason', '')})", key, edge_evidence(db, key))
        for a, b, key, d in graph.edges(keys=True, data=True)
        if d["edge_type"] == "contradicts"
    ]


def gaps(graph: nx.MultiDiGraph, db: Database) -> list[Answer]:
    """The run's ranked research gaps (Phase 3), each with its warrant and evidence."""
    nodes = sorted(
        ((n, d) for n, d in graph.nodes(data=True) if d["node_type"] == "gap"), key=lambda item: item[1]["attrs"]["rank"]
    )
    return [
        Answer(f"{d['attrs']['rank']}. {d['attrs']['kind']}",
               d["name"] + (f" | warrant: {d['attrs']['warrant']}" if d["attrs"].get("warrant") else ""),
               n, node_evidence(db, n))
        for n, d in nodes
    ]


def resting_on(graph: nx.MultiDiGraph, db: Database, sentence_id: str) -> list[Answer]:
    """Every node and edge of the run whose evidence includes a sentence."""
    answers: list[Answer] = []
    rows = db.query("SELECT owner_kind, owner_id, quote FROM evidence WHERE sentence_id = ?", (sentence_id,))
    edges = {key: (src, dst, d) for src, dst, key, d in graph.edges(keys=True, data=True)}
    for row in rows:
        evidence = [{"sentence_id": sentence_id, "quote": row["quote"]}]
        if row["owner_kind"] == "node" and row["owner_id"] in graph:
            data = graph.nodes[row["owner_id"]]
            answers.append(Answer(data["node_type"], data["name"], row["owner_id"], evidence))
        elif row["owner_kind"] == "edge" and row["owner_id"] in edges:
            src, dst, data = edges[row["owner_id"]]
            answers.append(Answer(data["edge_type"], f"{_name(graph, src)} -> {_name(graph, dst)}", row["owner_id"], evidence))
    return answers


def node_evidence(db: Database, node_id: str) -> list[dict[str, Any]]:
    return _evidence(db, "node", node_id)


def edge_evidence(db: Database, edge_id: str) -> list[dict[str, Any]]:
    return _evidence(db, "edge", edge_id)


def _evidence(db: Database, kind: str, owner: str) -> list[dict[str, Any]]:
    rows = db.query(
        "SELECT sentence_id, quote FROM evidence WHERE owner_kind = ? AND owner_id = ? ORDER BY sentence_id",
        (kind, owner),
    )
    return [dict(row) for row in rows]
