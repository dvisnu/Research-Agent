import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api";
import { Badge, Card, Empty, ErrorBox, LevelBadge, Spinner, StateBadge } from "../components/ui";
import { ago, percent } from "../format";
import { Link, navigate } from "../router";
import type { Run } from "../types";

export function RunsPage() {
  const { data, error, isLoading } = useQuery({ queryKey: ["runs"], queryFn: api.runs, refetchInterval: 5_000 });
  const [filter, setFilter] = useState("");

  const runs = (data ?? []).filter(
    (run) => !filter || run.question.toLowerCase().includes(filter.toLowerCase()) || run.run_id.includes(filter),
  );

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Runs</h1>
          <p className="mt-1 text-sm text-slate-500">
            Every question or idea the agent has worked on: its gaps, ideas and experiment plan.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter runs…"
            className="w-56 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm shadow-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 focus:outline-none"
          />
          <Link
            to="/new"
            className="rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white shadow-sm hover:bg-indigo-500"
          >
            New run
          </Link>
        </div>
      </div>

      {isLoading && <Spinner />}
      {error && <ErrorBox error={error} />}
      {data && !data.length && <Empty>No runs yet. Start one from “New run”.</Empty>}

      {!!runs.length && (
        <Card className="overflow-hidden">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium tracking-wide text-slate-500 uppercase">
              <tr>
                <th className="px-4 py-2.5">Question or idea</th>
                <th className="hidden px-4 py-2.5 lg:table-cell">Best idea</th>
                <th className="hidden px-4 py-2.5 md:table-cell">Plan coverage</th>
                <th className="px-4 py-2.5">Status</th>
                <th className="hidden px-4 py-2.5 sm:table-cell">Started</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {runs.map((run) => (
                <RunRow key={run.run_id} run={run} />
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}

function RunRow({ run }: { run: Run }) {
  const to = `/runs/${run.run_id}`;
  return (
    <tr onClick={() => navigate(to)} className="cursor-pointer align-top hover:bg-slate-50">
      <td className="max-w-xl px-4 py-3">
        <Link to={to} className="line-clamp-2 font-medium text-slate-900 hover:text-indigo-700">
          {run.question}
        </Link>
        <div className="mt-1 font-mono text-[11px] text-slate-400">{run.run_id}</div>
      </td>
      <td className="hidden max-w-xs px-4 py-3 lg:table-cell">
        {run.best_idea ? (
          <div className="space-y-1">
            <div className="line-clamp-2 text-slate-700">{run.best_idea.title}</div>
            <LevelBadge level={run.best_idea.level} />
          </div>
        ) : (
          <span className="text-slate-400">—</span>
        )}
      </td>
      <td className="hidden px-4 py-3 md:table-cell">
        {run.coverage ? (
          <span className="text-slate-700">
            {percent(run.coverage.coverage)}{" "}
            <span className="text-xs text-slate-500">
              ({run.coverage.supported}/{run.coverage.elements})
            </span>
          </span>
        ) : (
          <span className="text-slate-400">—</span>
        )}
      </td>
      <td className="px-4 py-3">
        <div className="flex flex-col items-start gap-1">
          <StateBadge state={run.state} />
          {run.diagnosis?.quota && (
            <Badge tone="amber" title={run.error ?? undefined}>
              quota: {run.diagnosis.keys.join(", ")}
            </Badge>
          )}
        </div>
      </td>
      <td className="hidden px-4 py-3 whitespace-nowrap text-slate-500 sm:table-cell">{ago(run.created_at)}</td>
    </tr>
  );
}
