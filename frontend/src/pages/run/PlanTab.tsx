import { useState } from "react";
import { QuoteLink, useDrawer } from "../../components/Drawer";
import { Badge, Bar, Card, Empty, Spinner, Stat, VerdictBadge } from "../../components/ui";
import { percent, words } from "../../format";
import { PLAN_FIELDS, type PlanElement, type PlanSource } from "../../types";
import { useReport } from "./useReport";

type Filter = "all" | "supported" | "flagged";

export function PlanTab({ runId, live }: { runId: string; live: boolean }) {
  const { data, error, isLoading } = useReport(runId, live);
  const [filter, setFilter] = useState<Filter>("all");
  if (isLoading) return <Spinner />;
  if (error) return <Empty>{error.message}</Empty>;
  const plan = data?.experiment_plan;
  if (!plan || !plan.idea_title) return <Empty>{plan?.note ?? "No experiment plan yet: the plan stage hasn't run."}</Empty>;
  const cov = plan.provenance_coverage;
  const sources = new Map(plan.sources.map((s) => [s.number, s]));
  const keep = (e: PlanElement) =>
    filter === "all" || (filter === "supported" ? e.verdict === "supported" : e.verdict !== "supported");

  return (
    <div className="space-y-6">
      <div>
        <div className="text-xs font-semibold tracking-wide text-slate-500 uppercase">Plan for the idea</div>
        <h2 className="mt-1 text-lg font-semibold text-slate-900">{plan.idea_title}</h2>
      </div>

      {"coverage" in cov && (
        <div className="grid gap-3 sm:grid-cols-4">
          <Stat
            label="Provenance coverage"
            value={percent(cov.coverage)}
            hint={`${cov.supported} of ${cov.elements} elements supported by their sources`}
          />
          <Stat label="Flagged" value={cov.flagged ?? 0} hint="kept, but their sources don't fully back them" />
          <Stat label="Removed" value={cov.removed ?? 0} hint="contradicted by their own sources" />
          <Stat label="Sources" value={plan.sources.length} hint="quoted sentences the plan cites" />
        </div>
      )}

      <div className="flex items-center gap-2 text-sm">
        <span className="text-slate-500">Show</span>
        {(["all", "supported", "flagged"] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`rounded-full px-3 py-1 ${filter === f ? "bg-slate-900 text-white" : "bg-white text-slate-600 ring-1 ring-slate-300 hover:bg-slate-50"}`}
          >
            {f === "flagged" ? "not supported" : f}
          </button>
        ))}
        <span className="ml-2 text-xs text-slate-500">Click an element to trace it to its sources.</span>
      </div>

      <div className="space-y-4">
        {PLAN_FIELDS.map(([field, title]) => {
          const value = plan[field];
          const elements = (Array.isArray(value) ? value : value ? [value] : []).filter(keep);
          if (!elements.length) return null;
          const stats = "fields" in cov ? cov.fields?.[field === "risks_and_limitations" ? "risks" : field] : undefined;
          return (
            <Card key={field} className="overflow-hidden">
              <div className="flex items-center gap-3 border-b border-slate-100 bg-slate-50 px-4 py-2">
                <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
                {stats && (
                  <div className="ml-auto flex w-40 items-center gap-2 text-xs text-slate-500">
                    <Bar value={stats.elements ? stats.supported / stats.elements : 0} tone="green" />
                    <span className="whitespace-nowrap">
                      {stats.supported}/{stats.elements}
                    </span>
                  </div>
                )}
              </div>
              <ul className="divide-y divide-slate-100">
                {elements.map((e) => (
                  <ElementRow key={e.element_id} runId={runId} element={e} sources={sources} />
                ))}
              </ul>
            </Card>
          );
        })}
      </div>

      {plan.removed.length > 0 && (
        <Card className="overflow-hidden border-red-200">
          <div className="border-b border-red-100 bg-red-50 px-4 py-2 text-sm font-semibold text-red-900">
            Removed by verification: contradicted by their own sources
          </div>
          <ul className="divide-y divide-slate-100">
            {plan.removed.map((e) => (
              <ElementRow key={e.element_id} runId={runId} element={e} sources={sources} />
            ))}
          </ul>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {plan.open_points.length > 0 && (
          <Card className="p-5">
            <h3 className="mb-2 text-sm font-semibold text-slate-900">Open critique points</h3>
            <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-700">
              {plan.open_points.map((p, i) => (
                <li key={i}>{p}</li>
              ))}
            </ul>
          </Card>
        )}
        {plan.revisions.length > 0 && (
          <Card className="p-5">
            <h3 className="mb-2 text-sm font-semibold text-slate-900">Revisions</h3>
            <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-700">
              {plan.revisions.map((p, i) => (
                <li key={i}>{p}</li>
              ))}
            </ul>
          </Card>
        )}
      </div>

      <Card className="p-5">
        <h3 className="mb-3 text-sm font-semibold text-slate-900">Sources</h3>
        <ol className="grid gap-2 md:grid-cols-2">
          {plan.sources.map((s) => (
            <li key={s.number} className="flex gap-2">
              <span className="mt-2 w-6 shrink-0 text-right font-mono text-xs text-slate-400">[{s.number}]</span>
              <div className="min-w-0 flex-1">
                <QuoteLink sentenceId={s.sentence_id} quote={s.quote} paper={`${s.paper}${s.section ? ` · ${s.section}` : ""}`} />
              </div>
            </li>
          ))}
        </ol>
      </Card>
    </div>
  );
}

function ElementRow({
  runId,
  element,
  sources,
}: {
  runId: string;
  element: PlanElement;
  sources: Map<number, PlanSource>;
}) {
  const show = useDrawer();
  return (
    <li>
      <button
        onClick={() => show({ kind: "element", runId, elementId: element.element_id })}
        className="flex w-full gap-4 px-4 py-3 text-left hover:bg-indigo-50/40"
      >
        <div className="w-32 shrink-0 space-y-1">
          <VerdictBadge verdict={element.verdict} />
          {element.status !== "kept" && (
            <div>
              <Badge tone={element.status === "removed" ? "red" : "amber"}>{element.status}</Badge>
            </div>
          )}
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm leading-relaxed text-slate-900">
            {element.name && <span className="font-semibold">{element.name}: </span>}
            {element.text}
          </p>
          {element.verdict !== "supported" && element.verdict_reason && (
            <p className="mt-1 text-xs text-amber-800">{element.verdict_reason}</p>
          )}
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500">
            <span className="font-mono">{words(element.label)}</span>
            {element.sources.map((n) => (
              <span key={n} title={sources.get(n) ? `${sources.get(n)!.paper}: “${sources.get(n)!.quote}”` : undefined}>
                [{n}]
              </span>
            ))}
            {!element.sources.length && <span>no sources</span>}
          </div>
        </div>
        <span className="self-center text-xs text-indigo-600">trace →</span>
      </button>
    </li>
  );
}
