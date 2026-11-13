import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "../../api";
import { Badge, Card, ErrorBox, SectionTitle, Spinner } from "../../components/ui";
import { duration, since, words } from "../../format";
import type { StageProgress } from "../../types";

const PHASES: Record<number, string> = {
  0: "Report",
  1: "Literature",
  2: "Knowledge graph",
  3: "Gaps",
  4: "Ideas and novelty",
  5: "Experiment plan",
};

export function ProgressTab({ runId, live }: { runId: string; live: boolean }) {
  const { data, error, isLoading } = useQuery({
    queryKey: ["progress", runId],
    queryFn: () => api.progress(runId),
    refetchInterval: live ? 3_000 : false,
  });
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!live) return;
    const timer = setInterval(() => setNow(Date.now()), 1_000);
    return () => clearInterval(timer);
  }, [live]);

  if (isLoading) return <Spinner />;
  if (error) return <ErrorBox error={error} />;
  if (!data) return null;

  const done = data.stages.filter((s) => s.status === "done").length;
  const current = live ? data.current : null;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_24rem]">
      <div>
        <SectionTitle aside={`${done} of ${data.stages.length} stages done`}>Stages</SectionTitle>
        {current && (
          <Card className="mb-4 border-indigo-200 bg-indigo-50/40 p-4">
            <div className="text-xs font-medium tracking-wide text-indigo-700 uppercase">Now</div>
            <div className="mt-1 text-lg font-semibold text-slate-900">{words(current.stage)}</div>
            <div className="mt-1 text-sm text-slate-600">
              {data.stages.find((s) => s.name === current.stage)?.summary} · running for{" "}
              {duration(since(current.since, now))} · {current.calls} calls so far
            </div>
          </Card>
        )}
        <ol className="space-y-1">
          {data.stages.map((stage, i) => {
            const newPhase = i === 0 || data.stages[i - 1].phase !== stage.phase;
            return (
              <li key={stage.name}>
                {newPhase && (
                  <div className="mt-3 mb-1 text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
                    {PHASES[stage.phase]}
                  </div>
                )}
                <StageRow stage={stage} running={live && current?.stage === stage.name} />
              </li>
            );
          })}
        </ol>
      </div>
      <div>
        <SectionTitle aside={live ? "updates every 3 s" : undefined}>Latest calls</SectionTitle>
        <Card className="divide-y divide-slate-100">
          {!data.recent.length && <div className="p-4 text-sm text-slate-500">No calls yet.</div>}
          {data.recent.map((call, i) => (
            <div key={i} className="px-4 py-2.5 text-xs">
              <div className="flex items-center gap-2">
                <Badge tone={call.kind === "llm_call" ? "violet" : "sky"}>{call.kind === "llm_call" ? "LLM" : "API"}</Badge>
                <span className="truncate font-medium text-slate-800">{call.who}</span>
                <span className="ml-auto shrink-0 text-slate-500">{call.cached ? "kept" : duration(call.seconds)}</span>
              </div>
              <div className="mt-0.5 flex items-center gap-2 text-slate-500">
                <span>{words(call.step)}</span>
                {call.status !== "ok" && <span className="truncate text-red-600">{call.error}</span>}
              </div>
            </div>
          ))}
        </Card>
      </div>
    </div>
  );
}

function StageRow({ stage, running }: { stage: StageProgress; running: boolean }) {
  const status = running ? "running" : stage.status;
  const icon = {
    done: <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500 text-[11px] text-white">✓</span>,
    running: <span className="pulse-ring h-5 w-5 rounded-full border-2 border-indigo-500 bg-white" />,
    failed: <span className="flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-[11px] text-white">!</span>,
    pending: <span className="h-5 w-5 rounded-full border-2 border-slate-200 bg-white" />,
  }[status];
  return (
    <div
      className={`flex items-center gap-3 rounded-lg px-3 py-2 ${running ? "bg-indigo-50" : status === "failed" ? "bg-red-50" : ""}`}
    >
      {icon}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className={`text-sm font-medium ${status === "pending" ? "text-slate-400" : "text-slate-900"}`}>
            {words(stage.name)}
          </span>
          {stage.runs > 1 && <Badge tone="indigo">×{stage.runs}</Badge>}
          {stage.failed_attempts > 0 && <Badge tone="amber">{stage.failed_attempts} failed</Badge>}
        </div>
        <div className="truncate text-xs text-slate-500">{stage.error ?? stage.summary}</div>
      </div>
      <div className="shrink-0 text-right text-xs text-slate-500">
        {stage.seconds > 0 && <div>{duration(stage.seconds)}</div>}
        {(stage.llm_calls > 0 || stage.api_calls > 0) && (
          <div>
            {stage.llm_calls} LLM · {stage.api_calls} API
          </div>
        )}
      </div>
    </div>
  );
}
