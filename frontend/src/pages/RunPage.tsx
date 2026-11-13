import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { lazy, Suspense, useEffect } from "react";
import { api } from "../api";
import { Badge, Button, ErrorBox, Spinner, StateBadge } from "../components/ui";
import { ago, when, words } from "../format";
import { Link, navigate, type Tab, TABS } from "../router";
import type { Run } from "../types";
import { CostTab } from "./run/CostTab";
import { GapsTab } from "./run/GapsTab";
import { IdeasTab } from "./run/IdeasTab";
import { OverviewTab } from "./run/OverviewTab";
import { PapersTab } from "./run/PapersTab";
import { PlanTab } from "./run/PlanTab";
import { ProgressTab } from "./run/ProgressTab";

// The graph view brings Cytoscape, most of the app's size: load it when first opened.
const GraphTab = lazy(() => import("./run/GraphTab").then((m) => ({ default: m.GraphTab })));

const LABELS: Record<Tab, string> = {
  progress: "Progress",
  overview: "Overview",
  gaps: "Gaps",
  ideas: "Ideas",
  plan: "Experiment plan",
  graph: "Knowledge graph",
  papers: "Papers",
  cost: "Cost",
};

export function RunPage({ runId, tab }: { runId: string; tab: Tab | null }) {
  const client = useQueryClient();
  const { data: run, error, isLoading } = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.run(runId),
    refetchInterval: (query) => (query.state.data?.state === "running" ? 3_000 : 15_000),
  });
  const live = run?.state === "running";

  // When a run finishes, everything it wrote is new: drop what was cached while it ran.
  const state = run?.state;
  useEffect(() => {
    if (state && state !== "running")
      client.invalidateQueries({ predicate: (q) => q.queryKey[1] === runId && q.queryKey[0] !== "run" });
  }, [state, runId, client]);

  const act = useMutation({
    mutationFn: (action: "resume" | "stop") => (action === "resume" ? api.resume(runId) : api.stop(runId)),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["run", runId] });
      client.invalidateQueries({ queryKey: ["status"] });
      navigate(`/runs/${runId}/progress`);
    },
  });

  if (isLoading) return <Spinner />;
  if (error || !run) return <ErrorBox error={error ?? "No such run."} />;

  const current: Tab = tab ?? (run.state === "completed" ? "overview" : "progress");
  const canResume = ["failed", "interrupted", "stale"].includes(run.state);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <Link to="/" className="text-xs text-slate-500 hover:text-slate-700">
            ← All runs
          </Link>
          <h1 className="mt-1 text-xl leading-snug font-semibold text-slate-900">{run.question}</h1>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500">
            <StateBadge state={run.state} />
            <span className="font-mono">{run.run_id}</span>
            <span>started {when(run.created_at)}</span>
            {run.last_activity && <span>last activity {ago(run.last_activity)}</span>}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {canResume && (
            <Button kind="primary" onClick={() => act.mutate("resume")} disabled={act.isPending}>
              Resume
            </Button>
          )}
          {live && run.started_here && (
            <Button kind="danger" onClick={() => act.mutate("stop")} disabled={act.isPending} title="It can be resumed">
              Stop
            </Button>
          )}
          {(run.files ?? [])
            .filter((f) => f !== "console.log")
            .map((name) => (
              <a
                key={name}
                href={api.fileUrl(runId, name)}
                className="rounded-lg px-2.5 py-1.5 text-xs font-medium text-slate-600 ring-1 ring-slate-300 ring-inset hover:bg-slate-50"
              >
                {name} ↓
              </a>
            ))}
        </div>
      </div>

      {act.error && <ErrorBox error={act.error} />}
      <StopBanner run={run} />

      <nav className="-mb-px flex gap-1 overflow-x-auto border-b border-slate-200">
        {TABS.map((t) => (
          <Link
            key={t}
            to={`/runs/${runId}/${t}`}
            className={`border-b-2 px-3 py-2 text-sm whitespace-nowrap ${t === current ? "border-indigo-600 font-medium text-indigo-700" : "border-transparent text-slate-600 hover:border-slate-300 hover:text-slate-900"}`}
          >
            {LABELS[t]}
          </Link>
        ))}
      </nav>

      <div>
        {current === "progress" && <ProgressTab runId={runId} live={live} />}
        {current === "overview" && <OverviewTab runId={runId} live={live} />}
        {current === "gaps" && <GapsTab runId={runId} live={live} />}
        {current === "ideas" && <IdeasTab runId={runId} live={live} />}
        {current === "plan" && <PlanTab runId={runId} live={live} />}
        {current === "graph" && (
          <Suspense fallback={<Spinner label="Loading the graph view" />}>
            <GraphTab runId={runId} />
          </Suspense>
        )}
        {current === "papers" && <PapersTab runId={runId} />}
        {current === "cost" && <CostTab runId={runId} live={live} />}
      </div>
    </div>
  );
}

function StopBanner({ run }: { run: Run }) {
  const d = run.diagnosis;
  if (run.state === "failed" && d?.quota) {
    return (
      <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-950">
        <div className="font-semibold">
          A quota ran out{d.role ? ` for the ${words(d.role)} role` : ""}
          {d.providers.length ? ` (${d.providers.map(words).join(", ")})` : ""}.
        </div>
        <p className="mt-1">
          Switch {d.keys.map((k) => <code key={k} className="mx-0.5 rounded bg-amber-100 px-1">{k}</code>)} in{" "}
          <code className="rounded bg-amber-100 px-1">.env</code>, then press <b>Resume</b>. The run continues from its
          last finished stage and reuses every reply it already has.
        </p>
        <Details error={run.error} tail={run.console_tail} />
      </div>
    );
  }
  if (run.state === "failed") {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-900">
        <div className="font-semibold">The run stopped.</div>
        <p className="mt-1 break-words">{run.error}</p>
        <p className="mt-1 text-red-800">Resume continues from its last finished stage.</p>
        <Details tail={run.console_tail} />
      </div>
    );
  }
  if (run.state === "interrupted" || run.state === "stale") {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
        <div className="font-semibold">
          {run.state === "interrupted"
            ? "The run's process ended without finishing (it may have been killed)."
            : `No activity for ${ago(run.last_activity).replace(" ago", "")}: the run may have been interrupted.`}
        </div>
        <p className="mt-1">
          {run.state === "stale" && "If it isn't still going in another terminal, "}Resume continues from its last
          finished stage.
        </p>
        <Details tail={run.console_tail} />
      </div>
    );
  }
  return null;
}

function Details({ error, tail }: { error?: string | null; tail?: string[] }) {
  if (!error && !tail?.length) return null;
  return (
    <details className="mt-2">
      <summary className="cursor-pointer text-xs opacity-80">Details</summary>
      {error && <p className="mt-2 font-mono text-xs break-words opacity-80">{error}</p>}
      {!!tail?.length && (
        <pre className="mt-2 max-h-64 overflow-auto rounded-lg bg-slate-900 p-3 text-[11px] text-slate-100">
          {tail.join("\n")}
        </pre>
      )}
      <Badge tone="slate">console.log in the run's folder</Badge>
    </details>
  );
}
