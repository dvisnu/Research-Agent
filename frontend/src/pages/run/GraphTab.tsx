import { useQuery } from "@tanstack/react-query";
import cytoscape, { type Core } from "cytoscape";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../../api";
import { useDrawer } from "../../components/Drawer";
import { Card, Empty, Spinner } from "../../components/ui";
import { words } from "../../format";

const COLORS: Record<string, string> = {
  paper: "#334155",
  task: "#0ea5e9",
  method: "#4f46e5",
  dataset: "#10b981",
  metric: "#f59e0b",
  claim: "#a855f7",
  result: "#ec4899",
  limitation: "#ef4444",
  future_work: "#14b8a6",
  assumption: "#84cc16",
  gap: "#dc2626",
  facet: "#94a3b8",
};
// At first, the field's entities and the gaps; findings (claims, results, their facets)
// make a dense ball, so they are a checkbox away.
const HIDDEN_AT_FIRST = new Set(["facet", "claim", "result", "future_work", "assumption"]);

export function GraphTab({ runId }: { runId: string }) {
  const { data, error, isLoading } = useQuery({ queryKey: ["graph", runId], queryFn: () => api.graph(runId) });
  const show = useDrawer();
  const box = useRef<HTMLDivElement>(null);
  const cy = useRef<Core | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(HIDDEN_AT_FIRST);
  const [search, setSearch] = useState("");

  const types = useMemo(() => {
    const counts = new Map<string, number>();
    data?.nodes.forEach((n) => counts.set(n.type, (counts.get(n.type) ?? 0) + 1));
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  }, [data]);

  const focus = useCallback((nodeId: string) => {
    const node = cy.current?.getElementById(nodeId);
    if (!node || node.empty()) return;
    cy.current!.elements().unselect();
    node.select();
    cy.current!.animate({ center: { eles: node }, zoom: 1.6 }, { duration: 400 });
  }, []);

  useEffect(() => {
    if (!data || !box.current) return;
    const nodes = data.nodes.filter((n) => !hidden.has(n.type));
    const ids = new Set(nodes.map((n) => n.id));
    const instance = cytoscape({
      container: box.current,
      elements: [
        ...nodes.map((n) => ({ data: { id: n.id, label: n.name, type: n.type, size: 14 + Math.min(n.evidence, 8) * 2 } })),
        ...data.edges
          .filter((e) => ids.has(e.source) && ids.has(e.target))
          .map((e) => ({ data: { id: e.id, source: e.source, target: e.target, type: e.type } })),
      ],
      style: [
        {
          selector: "node",
          style: {
            "background-color": (el: cytoscape.NodeSingular) => COLORS[el.data("type")] ?? "#64748b",
            width: "data(size)",
            height: "data(size)",
            label: "data(label)",
            "font-size": 9,
            color: "#334155",
            "text-valign": "bottom",
            "text-margin-y": 3,
            "text-max-width": "120px",
            "text-wrap": "ellipsis",
            "min-zoomed-font-size": 7,
          },
        },
        {
          selector: "edge",
          style: {
            width: 1,
            "line-color": "#94a3b8",
            "curve-style": "haystack",
            opacity: 0.6,
          },
        },
        { selector: 'edge[type = "supports"]', style: { "line-color": "#10b981", width: 2 } },
        { selector: 'edge[type = "contradicts"]', style: { "line-color": "#ef4444", width: 2.5 } },
        { selector: "node:selected", style: { "border-width": 3, "border-color": "#111827" } },
        { selector: ".faded", style: { opacity: 0.12 } },
        { selector: ".match", style: { "border-width": 3, "border-color": "#f59e0b" } },
      ],
      layout: { name: "cose", animate: false, nodeRepulsion: () => 9000, idealEdgeLength: () => 60, numIter: 600 },
      wheelSensitivity: 0.25,
    });
    instance.on("tap", "node", (event) =>
      show({ kind: "graph", item: "node", id: event.target.id(), onFocus: focus }),
    );
    instance.on("tap", "edge", (event) =>
      show({ kind: "graph", item: "edge", id: event.target.id(), onFocus: focus }),
    );
    cy.current = instance;
    return () => {
      instance.destroy();
      cy.current = null;
    };
  }, [data, hidden, show, focus]);

  useEffect(() => {
    const instance = cy.current;
    if (!instance) return;
    instance.elements().removeClass("faded match");
    const term = search.trim().toLowerCase();
    if (!term) return;
    const matches = instance.nodes().filter((n) => String(n.data("label")).toLowerCase().includes(term));
    if (matches.empty()) return;
    instance.elements().addClass("faded");
    const around = matches.closedNeighborhood();
    around.removeClass("faded");
    matches.addClass("match");
    instance.animate({ fit: { eles: around, padding: 40 } }, { duration: 300 });
  }, [search, data, hidden]);

  if (isLoading) return <Spinner label="Loading the graph" />;
  if (error) return <Empty>{error.message}</Empty>;
  if (!data?.nodes.length) return <Empty>No knowledge graph yet: the build_graph stage hasn't run.</Empty>;

  const toggle = (type: string) =>
    setHidden((h) => {
      const next = new Set(h);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[14rem_minmax(0,1fr)]">
      <div className="space-y-4">
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Find a node…"
          className="w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm shadow-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 focus:outline-none"
        />
        <Card className="p-3">
          <div className="mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">Node types</div>
          <ul className="space-y-1">
            {types.map(([type, n]) => (
              <li key={type}>
                <label className="flex cursor-pointer items-center gap-2 text-sm">
                  <input type="checkbox" checked={!hidden.has(type)} onChange={() => toggle(type)} className="rounded" />
                  <span className="h-3 w-3 rounded-full" style={{ background: COLORS[type] ?? "#64748b" }} />
                  <span className="flex-1 text-slate-700">{words(type)}</span>
                  <span className="text-xs text-slate-400">{n}</span>
                </label>
              </li>
            ))}
          </ul>
        </Card>
        <p className="text-xs leading-relaxed text-slate-500">
          Click a node or an edge to see the sentences it rests on. Green edges: findings that support each other; red:
          that contradict. Bigger nodes have more evidence.
        </p>
      </div>
      <Card className="relative h-[70vh] min-h-[28rem] overflow-hidden">
        {/* Cytoscape makes its container position: relative, so it needs a height of its own. */}
        <div ref={box} className="h-full w-full" />
        <button
          onClick={() => cy.current?.fit(undefined, 30)}
          className="absolute top-3 right-3 rounded-lg bg-white/90 px-2.5 py-1 text-xs text-slate-600 shadow ring-1 ring-slate-200 hover:bg-white"
        >
          Fit
        </button>
      </Card>
    </div>
  );
}
