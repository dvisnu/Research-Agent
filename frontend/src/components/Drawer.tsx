// The side panel that shows where something comes from: a sentence in its section, a plan
// element's chain down to the quoted sentences, or a node or edge of the knowledge graph.
import { useQuery } from "@tanstack/react-query";
import { createContext, type ReactNode, useCallback, useContext, useEffect, useState } from "react";
import { api } from "../api";
import { words } from "../format";
import type { Sentence } from "../types";
import { Badge, ErrorBox, Spinner, VerdictBadge } from "./ui";

export type DrawerView =
  | { kind: "sentence"; sentenceId: string; quote?: string }
  | { kind: "element"; runId: string; elementId: string }
  | { kind: "graph"; item: "node" | "edge"; id: string; onFocus?: (nodeId: string) => void };

const DrawerContext = createContext<(view: DrawerView | null) => void>(() => {});

export const useDrawer = () => useContext(DrawerContext);

export function DrawerProvider({ children }: { children: ReactNode }) {
  const [stack, setStack] = useState<DrawerView[]>([]);
  const show = useCallback((view: DrawerView | null) => setStack(view ? [view] : []), []);
  const push = useCallback((view: DrawerView) => setStack((s) => [...s, view]), []);
  const back = useCallback(() => setStack((s) => s.slice(0, -1)), []);
  const view = stack[stack.length - 1];

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setStack([]);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <DrawerContext.Provider value={show}>
      {children}
      {view && (
        <div className="fixed inset-0 z-40 flex justify-end">
          <button aria-label="Close" className="absolute inset-0 bg-slate-900/20" onClick={() => setStack([])} />
          <aside className="relative flex h-full w-full max-w-2xl flex-col bg-white shadow-2xl">
            <header className="flex items-center gap-2 border-b border-slate-200 px-5 py-3">
              {stack.length > 1 && (
                <button onClick={back} className="rounded px-2 py-1 text-sm text-slate-600 hover:bg-slate-100">
                  ← Back
                </button>
              )}
              <div className="flex-1 text-sm font-medium text-slate-500">
                {{ sentence: "Source sentence", element: "Plan element provenance", graph: "Knowledge graph" }[view.kind]}
              </div>
              <button
                onClick={() => setStack([])}
                className="rounded px-2 py-1 text-sm text-slate-600 hover:bg-slate-100"
              >
                Close ✕
              </button>
            </header>
            <div className="flex-1 overflow-y-auto px-5 py-4">
              {view.kind === "sentence" && <SentenceView sentenceId={view.sentenceId} quote={view.quote} />}
              {view.kind === "element" && <ElementView runId={view.runId} elementId={view.elementId} push={push} />}
              {view.kind === "graph" && <GraphItemView view={view} push={push} />}
            </div>
          </aside>
        </div>
      )}
    </DrawerContext.Provider>
  );
}

// Marks the quoted fragments (split at ellipses) inside a sentence.
export function Highlight({ text, quote }: { text: string; quote?: string }) {
  if (!quote) return <>{text}</>;
  const fragments = quote
    .split(/\s*(?:\.\.\.|…)\s*/)
    .map((f) => f.trim())
    .filter((f) => f.length > 3);
  const marks: [number, number][] = [];
  const lower = text.toLowerCase();
  let from = 0;
  for (const fragment of fragments) {
    const at = lower.indexOf(fragment.toLowerCase(), from);
    if (at >= 0) {
      marks.push([at, at + fragment.length]);
      from = at + fragment.length;
    }
  }
  if (!marks.length) return <>{text}</>;
  const parts: ReactNode[] = [];
  let cursor = 0;
  marks.forEach(([start, end], i) => {
    parts.push(text.slice(cursor, start));
    parts.push(<mark key={i}>{text.slice(start, end)}</mark>);
    cursor = end;
  });
  parts.push(text.slice(cursor));
  return <>{parts}</>;
}

export function QuoteLink({ sentenceId, quote, paper }: { sentenceId: string; quote: string; paper?: string | null }) {
  const show = useDrawer();
  return (
    <button
      onClick={() => show({ kind: "sentence", sentenceId, quote })}
      className="group block w-full rounded-lg border-l-2 border-indigo-200 bg-slate-50 px-3 py-2 text-left text-sm text-slate-700 transition hover:border-indigo-500 hover:bg-indigo-50/50"
    >
      <span className="italic">“{quote}”</span>
      {paper && <span className="mt-1 block text-xs text-slate-500 not-italic">{paper}</span>}
      <span className="mt-1 block font-mono text-[11px] text-slate-400 group-hover:text-indigo-600">
        {sentenceId} · show in context →
      </span>
    </button>
  );
}

function PaperHeader({ paper, section }: { paper: Sentence["paper"]; section: string | null }) {
  return (
    <div>
      <div className="font-medium text-slate-900">
        {paper.title}
        {paper.year && <span className="font-normal text-slate-500"> ({paper.year})</span>}
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
        <span>Section: {section || "untitled"}</span>
        {Object.entries(paper.links).map(([name, url]) => (
          <a key={name} href={url} target="_blank" rel="noreferrer" className="text-indigo-600 hover:underline">
            {name} ↗
          </a>
        ))}
      </div>
    </div>
  );
}

function SentenceView({ sentenceId, quote }: { sentenceId: string; quote?: string }) {
  const { data, error, isLoading } = useQuery({ queryKey: ["sentence", sentenceId], queryFn: () => api.sentence(sentenceId) });
  if (isLoading) return <Spinner />;
  if (error) return <ErrorBox error={error} />;
  if (!data) return null;
  return (
    <div className="space-y-4">
      <PaperHeader paper={data.paper} section={data.section} />
      {data.retired && (
        <div className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">
          This sentence belongs to an earlier text of the paper (it was parsed again since); the evidence still points
          at it.
        </div>
      )}
      <div className="space-y-2 rounded-xl bg-slate-50 p-4 text-sm leading-relaxed">
        {data.context.map((s) =>
          s.sentence_id === data.sentence_id ? (
            <p key={s.sentence_id} className="rounded-lg bg-white p-2 text-slate-900 shadow-sm ring-1 ring-indigo-200">
              <Highlight text={s.text} quote={quote} />
            </p>
          ) : (
            <p key={s.sentence_id} className="px-2 text-slate-500">
              {s.text}
            </p>
          ),
        )}
      </div>
      <div className="font-mono text-xs text-slate-400">{data.sentence_id}</div>
    </div>
  );
}

function ElementView({ runId, elementId, push }: { runId: string; elementId: string; push: (v: DrawerView) => void }) {
  const { data, error, isLoading } = useQuery({
    queryKey: ["element", runId, elementId],
    queryFn: () => api.element(runId, elementId),
  });
  if (isLoading) return <Spinner />;
  if (error) return <ErrorBox error={error} />;
  if (!data) return null;
  const d = data.derivation;
  return (
    <div className="space-y-5">
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-xs text-slate-500">{data.label}</span>
          <VerdictBadge verdict={data.verdict} />
          {data.status !== "kept" && <Badge tone={data.status === "removed" ? "red" : "amber"}>{data.status}</Badge>}
        </div>
        <p className="mt-2 text-sm leading-relaxed text-slate-900">{data.text}</p>
      </div>

      <Step title="Verdict">
        {data.verification?.reason ? <p>{data.verification.reason}</p> : <p className="text-slate-500">Not checked.</p>}
        {data.verification?.model && <p className="mt-1 text-xs text-slate-500">by {data.verification.model}</p>}
      </Step>

      <Step title="Decision">
        <p>{data.decision?.reason || "No reason given."}</p>
        {!!data.decision?.alternatives?.length && (
          <p className="mt-1 text-xs text-slate-500">Chosen over: {data.decision.alternatives.join(", ")}</p>
        )}
        {data.decision?.answers?.map((a) => (
          <p key={a.point} className="mt-1 text-xs text-slate-500">
            Answers critique point {a.point}: {a.problem}
          </p>
        ))}
      </Step>

      <Step title="Derivation">
        <p>
          Plan round {d.round ?? "?"} ({d.change ?? "—"}), written by {d.model ?? "?"}, prompt {d.prompt ?? "?"}
        </p>
        {!!d.checks?.length && <p className="mt-1 text-xs text-slate-500">Checks: {d.checks.join("; ")}</p>}
        {data.basis && <p className="mt-2 text-slate-600">Basis: {data.basis}</p>}
      </Step>

      <div>
        <h3 className="mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">
          Cited facts ({data.facts.length})
        </h3>
        {!data.facts.length && <p className="text-sm text-slate-500">This element cites no facts.</p>}
        <div className="space-y-3">
          {data.facts.map((fact) => (
            <div key={fact.fact_id} className="rounded-xl border border-slate-200 p-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone="indigo">{fact.fact_id}</Badge>
                <Badge>{words(fact.kind)}</Badge>
                <span className="text-sm text-slate-900">{fact.text}</span>
              </div>
              {fact.origin.from && (
                <p className="mt-1 text-xs text-slate-500">
                  From {words(fact.origin.from)}
                  {fact.origin.query ? `: ${fact.origin.query}` : ""}
                </p>
              )}
              {fact.papers.map((p) => (
                <p key={p.paper} className="mt-1 text-xs text-slate-500">
                  {p.paper}
                  {p.why.length ? ` — ${p.why.join("; ")}` : ""}
                </p>
              ))}
              <div className="mt-2 space-y-2">
                {fact.evidence.map((e) => (
                  <button
                    key={e.sentence_id}
                    onClick={() => push({ kind: "sentence", sentenceId: e.sentence_id, quote: e.quote })}
                    className={`block w-full rounded-lg border-l-2 px-3 py-2 text-left text-sm transition hover:bg-indigo-50/50 ${e.kept ? "border-indigo-400 bg-slate-50" : "border-slate-200 bg-white"}`}
                  >
                    {e.sentence ? (
                      <>
                        <span className="text-slate-800">
                          <Highlight text={e.sentence.text} quote={e.quote} />
                        </span>
                        <span className="mt-1 block text-xs text-slate-500">
                          {e.sentence.paper.title}
                          {e.sentence.paper.year ? ` (${e.sentence.paper.year})` : ""} · {e.sentence.section || "untitled"}
                        </span>
                      </>
                    ) : (
                      <span className="text-slate-500">“{e.quote}” — no longer stored</span>
                    )}
                    <span className="mt-1 block font-mono text-[11px] text-slate-400">
                      {e.sentence_id}
                      {e.kept ? " · kept as this element's evidence" : ""}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Step({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="border-l-2 border-slate-200 pl-3 text-sm text-slate-700">
      <h3 className="mb-1 text-xs font-semibold tracking-wide text-slate-500 uppercase">{title}</h3>
      {children}
    </div>
  );
}

function GraphItemView({
  view,
  push,
}: {
  view: Extract<DrawerView, { kind: "graph" }>;
  push: (v: DrawerView) => void;
}) {
  const { data, error, isLoading } = useQuery({
    queryKey: ["graph-item", view.item, view.id],
    queryFn: () => api.graphItem(view.item, view.id),
  });
  if (isLoading) return <Spinner />;
  if (error) return <ErrorBox error={error} />;
  if (!data) return null;
  const attrs = Object.entries(data.attrs).filter(([, v]) => v != null && v !== "" && typeof v !== "object");
  return (
    <div className="space-y-5">
      <div>
        <div className="flex items-center gap-2">
          <Badge tone="indigo">{words(data.node_type ?? data.edge_type ?? "")}</Badge>
          {view.onFocus && data.kind === "node" && (
            <button onClick={() => view.onFocus?.(view.id)} className="text-xs text-indigo-600 hover:underline">
              Show in graph
            </button>
          )}
        </div>
        {data.kind === "node" ? (
          <h3 className="mt-2 text-lg font-semibold text-slate-900">{data.name}</h3>
        ) : (
          <h3 className="mt-2 text-sm text-slate-900">
            <b>{data.source?.name}</b> <span className="text-slate-500">— {words(data.edge_type ?? "")} →</span>{" "}
            <b>{data.target?.name}</b>
          </h3>
        )}
        {!!data.aliases?.length && <p className="mt-1 text-xs text-slate-500">Also called: {data.aliases.join(", ")}</p>}
        {!!attrs.length && (
          <dl className="mt-2 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-xs">
            {attrs.map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-slate-500">{words(k)}</dt>
                <dd className="text-slate-800">{String(v)}</dd>
              </div>
            ))}
          </dl>
        )}
      </div>
      <div>
        <h3 className="mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">
          Evidence ({data.evidence.length})
        </h3>
        {!data.evidence.length && (
          <p className="text-sm text-slate-500">No sentence of its own; its links below carry the evidence.</p>
        )}
        <div className="space-y-2">
          {data.evidence.map((e) => (
            <button
              key={e.sentence_id}
              onClick={() => push({ kind: "sentence", sentenceId: e.sentence_id, quote: e.quote })}
              className="block w-full rounded-lg border-l-2 border-indigo-300 bg-slate-50 px-3 py-2 text-left text-sm hover:bg-indigo-50/50"
            >
              {e.sentence ? <Highlight text={e.sentence.text} quote={e.quote} /> : `“${e.quote}”`}
              {e.sentence && (
                <span className="mt-1 block text-xs text-slate-500">
                  {e.sentence.paper.title} · {e.sentence.section || "untitled"}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>
      {!!data.links?.length && (
        <div>
          <h3 className="mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">
            Links ({data.links.length})
          </h3>
          <ul className="divide-y divide-slate-100 text-sm">
            {data.links.map((l) => (
              <li key={l.edge_id + l.direction} className="flex items-center gap-2 py-1.5">
                <span className="w-28 shrink-0 text-xs text-slate-500">
                  {l.direction === "out" ? `${words(l.type)} →` : `← ${words(l.type)}`}
                </span>
                <button
                  onClick={() => push({ kind: "graph", item: "node", id: l.node_id, onFocus: view.onFocus })}
                  className="truncate text-left text-slate-800 hover:text-indigo-600 hover:underline"
                >
                  {l.name}
                </button>
                <Badge>{words(l.node_type)}</Badge>
                <button
                  onClick={() => push({ kind: "graph", item: "edge", id: l.edge_id, onFocus: view.onFocus })}
                  className="ml-auto shrink-0 text-xs text-slate-400 hover:text-indigo-600"
                >
                  edge evidence
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
