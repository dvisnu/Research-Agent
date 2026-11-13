import { useQuery } from "@tanstack/react-query";
import { api } from "../../api";
import { Bar, Card, Empty, SectionTitle, Spinner, Stat } from "../../components/ui";
import { count, duration, percent, words } from "../../format";

export function CostTab({ runId, live }: { runId: string; live: boolean }) {
  const { data, error, isLoading } = useQuery({
    queryKey: ["cost", runId],
    queryFn: () => api.cost(runId),
    refetchInterval: live ? 10_000 : false,
  });
  if (isLoading) return <Spinner />;
  if (error) return <Empty>{error.message}</Empty>;
  if (!data) return null;

  const longest = Math.max(1, ...data.stages.map((s) => s.seconds));
  const calls = data.models.reduce((n, m) => n + m.calls, 0);
  const replies = data.reuse.filter((r) => r.kind !== "embedding");
  const reused = replies.reduce((n, r) => n + r.reused, 0);
  const asked = replies.reduce((n, r) => n + r.asked, 0);
  const requests = data.apis.reduce((n, a) => n + a.requests, 0);
  const kept = data.apis.reduce((n, a) => n + a.cached, 0);

  return (
    <div className="space-y-6">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Time in stages" value={duration(data.seconds)} hint={`${duration(data.elapsed)} from first stage to last`} />
        <Stat label="Model calls" value={count(calls)} hint={`${data.failed_attempts} failed stage attempts`} />
        <Stat
          label="LLM replies reused"
          value={reused + asked ? percent(reused / (reused + asked)) : "—"}
          hint={`${reused} kept from earlier, ${asked} asked`}
        />
        <Stat
          label="API responses kept"
          value={requests + kept ? percent(kept / (requests + kept)) : "—"}
          hint={`${kept} kept, ${requests} sent`}
        />
      </div>

      <Card className="p-5">
        <SectionTitle>Time per stage</SectionTitle>
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-slate-500">
            <tr>
              <th className="pb-2 font-medium">Stage</th>
              <th className="w-1/3 pb-2 font-medium" />
              <th className="pb-2 text-right font-medium">Time</th>
              <th className="pb-2 text-right font-medium">Runs</th>
              <th className="pb-2 text-right font-medium">LLM</th>
              <th className="pb-2 text-right font-medium">API</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {data.stages.map((s) => (
              <tr key={s.stage}>
                <td className="py-1.5 text-slate-800">{words(s.stage)}</td>
                <td className="py-1.5 pr-4">
                  <Bar value={s.seconds / longest} />
                </td>
                <td className="py-1.5 text-right text-slate-600">{duration(s.seconds)}</td>
                <td className="py-1.5 text-right text-slate-600">
                  {s.attempts}
                  {s.failed ? <span className="text-amber-700"> ({s.failed} failed)</span> : ""}
                </td>
                <td className="py-1.5 text-right text-slate-600">{s.llm_calls}</td>
                <td className="py-1.5 text-right text-slate-600">{s.api_requests}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="p-5">
          <SectionTitle>Models</SectionTitle>
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-slate-500">
              <tr>
                <th className="pb-2 font-medium">Model</th>
                <th className="pb-2 text-right font-medium">Calls</th>
                <th className="pb-2 text-right font-medium">Tokens in</th>
                <th className="pb-2 text-right font-medium">Tokens out</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.models.map((m) => (
                <tr key={m.model + m.role}>
                  <td className="py-1.5">
                    <div className="break-all text-slate-800">{m.model}</div>
                    <div className="text-xs text-slate-500">{m.role}</div>
                  </td>
                  <td className="py-1.5 text-right text-slate-600">
                    {m.calls}
                    {m.failed ? <span className="text-amber-700"> ({m.failed} failed)</span> : ""}
                  </td>
                  <td className="py-1.5 text-right text-slate-600">{count(m.input_tokens)}</td>
                  <td className="py-1.5 text-right text-slate-600">
                    {count(m.output_tokens)}
                    {m.reasoning_tokens ? (
                      <div className="text-xs text-slate-400">{count(m.reasoning_tokens)} thinking</div>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
        <Card className="p-5">
          <SectionTitle>APIs</SectionTitle>
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-slate-500">
              <tr>
                <th className="pb-2 font-medium">Service</th>
                <th className="pb-2 text-right font-medium">Sent</th>
                <th className="pb-2 text-right font-medium">Kept</th>
                <th className="pb-2 text-right font-medium">Failed</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.apis.map((a) => (
                <tr key={a.service}>
                  <td className="py-1.5 text-slate-800">{words(a.service)}</td>
                  <td className="py-1.5 text-right text-slate-600">{a.requests}</td>
                  <td className="py-1.5 text-right text-slate-600">{a.cached}</td>
                  <td className="py-1.5 text-right text-slate-600">{a.failed || ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3 className="mt-5 mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">Reused work</h3>
          <ul className="space-y-1.5 text-sm">
            {data.reuse.map((r) => (
              <li key={r.kind} className="flex items-center gap-3">
                <span className="w-32 shrink-0 text-slate-700">{words(r.kind)}</span>
                <Bar value={r.reused + r.asked ? r.reused / (r.reused + r.asked) : 0} tone="green" />
                <span className="w-20 shrink-0 text-right text-xs text-slate-500">
                  {r.reused}/{r.reused + r.asked}
                </span>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
}
