"""Related work: the corpus papers closest to the question.

Two signals. SPECTER2 similarity between each paper (title and abstract) and
the question, and graph neighbourhood: a personalized PageRank over the
paper-entity graph, seeded by that similarity, which lifts papers sharing tasks,
methods and datasets with the most similar ones. Score =
z(similarity) + GRAPH_WEIGHT * z(log PageRank).
"""

import math
from collections import Counter
from typing import Any

import networkx as nx
import numpy as np

GRAPH_WEIGHT = 0.5
RELATED = 10
ENTITY_EDGES = frozenset({"addresses", "proposes", "uses", "evaluates_on", "measured_by"})


def _zscores(values: np.ndarray) -> np.ndarray:
    spread = values.std()
    return (values - values.mean()) / spread if spread else np.zeros_like(values)


def personalized_pagerank(
    graph: nx.Graph, seed: dict[str, float], alpha: float = 0.85, iterations: int = 200, tolerance: float = 1e-10
) -> dict[str, float]:
    """PageRank whose random jumps land on ``seed`` nodes in proportion to their weight;
    a node without edges sends its rank back to the seed. (networkx's own needs scipy.)"""
    nodes = list(graph)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes)
    degree = adjacency.sum(axis=1)
    linked = degree > 0
    jump = np.array([seed.get(node, 0.0) for node in nodes])
    jump = jump / jump.sum() if jump.sum() else np.full(len(nodes), 1 / len(nodes))
    rank = jump.copy()
    for _ in range(iterations):
        spread = (rank[linked] / degree[linked]) @ adjacency[linked]
        updated = alpha * spread + (alpha * rank[~linked].sum() + 1 - alpha) * jump
        if np.abs(updated - rank).sum() < tolerance:
            rank = updated
            break
        rank = updated
    return dict(zip(nodes, rank.tolist()))


def related_papers(graph: nx.MultiDiGraph, similarity: dict[str, float], count: int = RELATED) -> list[dict[str, Any]]:
    """The ``count`` closest papers, best first: title, year, similarity, score, the
    entities it shares with the other close papers, and its purpose facet."""
    paper_nodes = {
        data["attrs"]["paper_id"]: node
        for node, data in graph.nodes(data=True)
        if data["node_type"] == "paper" and data["attrs"]["paper_id"] in similarity
    }
    if not paper_nodes:
        return []
    ids = list(paper_nodes)
    sims = np.array([similarity[paper_id] for paper_id in ids])
    z_sim = _zscores(sims)

    links = nx.Graph()
    links.add_nodes_from(paper_nodes.values())
    for src, dst, data in graph.edges(data=True):
        if src in links and data["edge_type"] in ENTITY_EDGES:
            links.add_edge(src, dst)
    seed = {paper_nodes[paper_id]: math.exp(z) for paper_id, z in zip(ids, z_sim)}
    rank = personalized_pagerank(links, seed)
    z_graph = _zscores(np.log(np.array([rank[paper_nodes[paper_id]] for paper_id in ids]) + 1e-12))
    scores = z_sim + GRAPH_WEIGHT * z_graph

    order = [int(i) for i in np.argsort(-scores)[:count]]
    chosen = [paper_nodes[ids[i]] for i in order]
    reach = Counter(entity for node in chosen for entity in set(links.neighbors(node)))
    related = []
    for i in order:
        node = paper_nodes[ids[i]]
        shared = sorted((e for e in links.neighbors(node) if reach[e] > 1), key=lambda e: (-reach[e], graph.nodes[e]["name"]))
        purpose = next(
            (graph.nodes[dst]["name"] for _, dst, d in graph.out_edges(node, data=True)
             if d["edge_type"] == "has_facet" and d["attrs"].get("facet") == "purpose"),
            None,
        )
        related.append({
            "paper_id": ids[i],
            "title": graph.nodes[node]["name"],
            "year": graph.nodes[node]["attrs"].get("year"),
            "similarity": round(float(sims[i]), 4),
            "score": round(float(scores[i]), 3),
            "shares": [graph.nodes[e]["name"] for e in shared[:4]],
            "purpose": purpose,
        })
    return related
