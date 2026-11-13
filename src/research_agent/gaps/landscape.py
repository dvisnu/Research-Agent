"""The landscape grid: the most-studied methods against the most-studied datasets.

A method's papers are those that propose or use it, or report a result about
it; a dataset's papers are those that evaluate on it. The core is the
CORE_METHODS methods and CORE_DATASETS datasets with the most papers. Each cell
says what the corpus holds for that pair:

- "result": a result about the method on the dataset;
- "same_paper": no such result, but a paper uses both (it may have tested the pair);
- "empty": nothing. Only these can become structural gaps.

The metrics of a cell are those its results are measured with. A name that is also a
dataset or metric elsewhere in the graph is not a method of the grid (extraction
sometimes types "MT-Bench" or "BLEU" as a method), nor a metric one of its datasets.
"""

from typing import Any

import networkx as nx

from research_agent.graph.names import name_key

CORE_METHODS = 12
CORE_DATASETS = 10
# Classic text-evaluation metrics, never methods of the grid even when no paper of the run
# types them as metrics (name keys; a key starting with one counts, e.g. "rougel").
KNOWN_METRICS = ("bleu", "sacrebleu", "rouge", "meteor", "bertscore", "bartscore", "bleurt", "moverscore",
                 "chrf", "cider", "spice", "comet", "perplexity")


def _keys(data: dict[str, Any]) -> set[str]:
    """The name keys of a node: its name and every alias."""
    return {name_key(data["name"]), *(name_key(alias) for alias in data["aliases"])}


def _out(graph: nx.MultiDiGraph, node: str, *edge_types: str) -> set[str]:
    return {dst for _, dst, d in graph.out_edges(node, data=True) if d["edge_type"] in edge_types}


def build_landscape(
    graph: nx.MultiDiGraph, methods: int = CORE_METHODS, datasets: int = CORE_DATASETS
) -> dict[str, Any]:
    nodes = graph.nodes
    papers = [n for n, d in graph.nodes(data=True) if d["node_type"] == "paper"]
    method_papers: dict[str, set[str]] = {}
    dataset_papers: dict[str, set[str]] = {}
    for paper in papers:
        for method in _out(graph, paper, "proposes", "uses"):
            method_papers.setdefault(method, set()).add(paper)
        for dataset in _out(graph, paper, "evaluates_on"):
            dataset_papers.setdefault(dataset, set()).add(paper)
        for result in _out(graph, paper, "reports"):
            if nodes[result]["node_type"] == "result":
                for method in _out(graph, result, "about"):
                    method_papers.setdefault(method, set()).add(paper)

    # Extraction sometimes types a benchmark or a metric as a method ("MT-Bench", "BLEU"). When the
    # same name is also a dataset or metric node in the graph, keep it out of the grid's methods,
    # and a metric out of its datasets.
    names = {kind: {key for n, d in graph.nodes(data=True) if d["node_type"] == kind for key in _keys(d)}
             for kind in ("dataset", "metric")}

    def core(found: dict[str, set[str]], size: int, exclude: set[str]) -> list[str]:
        kept = [n for n in found if not _keys(nodes[n]) & exclude]
        return sorted(kept, key=lambda n: (-len(found[n]), nodes[n]["name"]))[:size]

    known = {n for n in method_papers if any(key.startswith(KNOWN_METRICS) for key in _keys(nodes[n]))}
    core_methods = core({n: p for n, p in method_papers.items() if n not in known}, methods,
                        names["dataset"] | names["metric"])
    core_datasets = core(dataset_papers, datasets, names["metric"])
    results = [n for n, d in graph.nodes(data=True) if d["node_type"] == "result"]
    cells = []
    for method in core_methods:
        row = []
        for dataset in core_datasets:
            found = [r for r in results if method in _out(graph, r, "about") and dataset in _out(graph, r, "evaluates_on")]
            both = method_papers[method] & dataset_papers[dataset]
            metrics = sorted({nodes[m]["name"] for r in found for m in _out(graph, r, "measured_by")})
            status = "result" if found else "same_paper" if both else "empty"
            row.append({"status": status, "results": len(found), "papers": len(both), "metrics": metrics[:3]})
        cells.append(row)
    return {
        "methods": [{"node_id": m, "name": nodes[m]["name"], "papers": len(method_papers[m])} for m in core_methods],
        "datasets": [{"node_id": d, "name": nodes[d]["name"], "papers": len(dataset_papers[d])} for d in core_datasets],
        "cells": cells,
    }
