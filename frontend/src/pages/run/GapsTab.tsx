import { QuoteLink } from "../../components/Drawer";
import { Badge, Card, Empty, Spinner, type Tone } from "../../components/ui";
import { useReport } from "./useReport";

const KIND: Record<string, [Tone, string]> = {
  stated: ["indigo", "Stated by a paper"],
  structural: ["sky", "Structural (missing in the graph)"],
  conflict: ["rose", "Conflicting findings"],
};

export function GapsTab({ runId, live }: { runId: string; live: boolean }) {
  const { data, error, isLoading } = useReport(runId, live);
  if (isLoading) return <Spinner />;
  if (error) return <Empty>{error.message}</Empty>;
  if (!data?.research_gaps.length) return <Empty>No gaps yet: the gaps stage hasn't run.</Empty>;
  return (
    <div className="space-y-4">
      <p className="text-sm text-slate-500">
        Ranked by relevance to the question. Each gap rests on quoted sentences; click a quote to read it in its
        section.
      </p>
      {data.research_gaps.map((gap) => {
        const [tone, label] = KIND[gap.kind] ?? ["slate", gap.kind];
        return (
          <Card key={gap.rank} className="p-5">
            <div className="flex flex-wrap items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-slate-900 text-xs font-semibold text-white">
                {gap.rank}
              </span>
              <Badge tone={tone}>{label}</Badge>
              <span className="text-xs text-slate-500">
                relevance {gap.relevance.toFixed(3)} · evidence from {gap.papers} paper{gap.papers === 1 ? "" : "s"}
              </span>
            </div>
            <p className="mt-3 text-sm leading-relaxed font-medium text-slate-900">{gap.gap}</p>
            {gap.warrant && (
              <p className="mt-2 text-sm text-slate-600">
                <span className="font-medium text-slate-700">Warrant: </span>
                {gap.warrant}
              </p>
            )}
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              {gap.evidence.map((e) => (
                <QuoteLink key={e.sentence_id} sentenceId={e.sentence_id} quote={e.quote} paper={e.paper} />
              ))}
            </div>
          </Card>
        );
      })}
    </div>
  );
}
