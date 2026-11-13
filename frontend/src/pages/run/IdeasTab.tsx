import { useState } from "react";
import { QuoteLink } from "../../components/Drawer";
import { Badge, Card, Empty, LevelBadge, SimilarityBadge, Spinner } from "../../components/ui";
import { words } from "../../format";
import type { ClosestWork, Idea } from "../../types";
import { useReport } from "./useReport";

export function IdeasTab({ runId, live }: { runId: string; live: boolean }) {
  const { data, error, isLoading } = useReport(runId, live);
  if (isLoading) return <Spinner />;
  if (error) return <Empty>{error.message}</Empty>;
  if (!data?.ideas.length) return <Empty>No checked ideas yet.</Empty>;
  return (
    <div className="space-y-5">
      <p className="text-sm text-slate-500">
        The checked ideas, best first. Each was compared with the closest papers an independent search found; the level
        rates its novelty from 1 (already done) to 5.
      </p>
      {data.ideas.map((idea) => (
        <IdeaCard key={idea.idea_id} idea={idea} />
      ))}
    </div>
  );
}

function IdeaCard({ idea }: { idea: Idea }) {
  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-center gap-2">
        {idea.rank != null && (
          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-slate-900 text-xs font-semibold text-white">
            {idea.rank}
          </span>
        )}
        <LevelBadge level={idea.level} label={idea.label} />
        {idea.rating && <Badge tone="slate">{idea.rating}</Badge>}
        {idea.inconclusive && <Badge tone="amber">inconclusive</Badge>}
        <Badge tone="violet">{words(idea.origin)}</Badge>
        {idea.kind && <Badge>{words(idea.kind)}</Badge>}
      </div>
      <h3 className="mt-3 text-lg font-semibold text-slate-900">{idea.title}</h3>
      <dl className="mt-3 grid gap-3 text-sm md:grid-cols-3">
        {(
          [
            ["Purpose", idea.purpose],
            ["Mechanism", idea.mechanism],
            ["Evaluation", idea.evaluation],
          ] as const
        ).map(([label, text]) => (
          <div key={label}>
            <dt className="text-xs font-semibold tracking-wide text-slate-500 uppercase">{label}</dt>
            <dd className="mt-1 leading-relaxed text-slate-700">{text}</dd>
          </div>
        ))}
      </dl>
      {(idea.builds_on.length > 0 || idea.feasibility) && (
        <div className="mt-3 space-y-1 text-sm text-slate-600">
          {idea.builds_on.length > 0 && <p>Builds on: {idea.builds_on.join(", ")}</p>}
          {idea.feasibility && <p>Feasibility: {idea.feasibility}</p>}
        </div>
      )}
      {idea.reasons.length > 0 && (
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-700">
          {idea.reasons.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      )}

      {idea.gap && (
        <div className="mt-4 rounded-lg bg-slate-50 p-3">
          <div className="text-xs font-semibold tracking-wide text-slate-500 uppercase">
            The gap it answers{idea.gap_kind ? ` (${idea.gap_kind})` : ""}
          </div>
          <p className="mt-1 text-sm text-slate-800">{idea.gap}</p>
          {idea.addressed_by.length > 0 && (
            <p className="mt-1 text-xs text-amber-700">Possibly addressed by: {idea.addressed_by.join("; ")}</p>
          )}
          <div className="mt-2 space-y-2">
            {idea.gap_evidence.map((e) => (
              <QuoteLink key={e.sentence_id} sentenceId={e.sentence_id} quote={e.quote} paper={e.paper} />
            ))}
          </div>
        </div>
      )}

      {idea.closest.length > 0 && <Closest closest={idea.closest} />}

      {idea.history.length > 0 && (
        <details className="mt-4 text-sm">
          <summary className="cursor-pointer text-slate-600">Earlier versions ({idea.history.length})</summary>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-slate-600">
            {idea.history.map((h, i) => (
              <li key={i}>{h}</li>
            ))}
          </ul>
        </details>
      )}
    </Card>
  );
}

function Closest({ closest }: { closest: ClosestWork[] }) {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <div className="mt-4">
      <div className="mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">
        Closest papers ({closest.length})
      </div>
      <div className="divide-y divide-slate-100 rounded-lg border border-slate-200">
        {closest.map((work) => (
          <div key={work.position}>
            <button
              onClick={() => setOpen(open === work.position ? null : work.position)}
              className="flex w-full flex-wrap items-center gap-2 px-3 py-2 text-left hover:bg-slate-50"
            >
              <span className="w-5 text-xs text-slate-400">{work.position}</span>
              <span className="min-w-0 flex-1 text-sm text-slate-800">
                {work.title}
                {work.year && <span className="text-slate-500"> ({work.year})</span>}
              </span>
              <SimilarityBadge what="purpose" value={work.purpose} />
              <SimilarityBadge what="mechanism" value={work.mechanism} />
              <SimilarityBadge what="evaluation" value={work.evaluation} />
              <Badge>{work.read}</Badge>
              <span className="text-xs text-slate-400">{open === work.position ? "▲" : "▼"}</span>
            </button>
            {open === work.position && (
              <div className="space-y-2 bg-slate-50/60 px-3 pt-1 pb-3 text-sm text-slate-700">
                {work.paper_mechanism && (
                  <p>
                    <span className="font-medium">What the paper does: </span>
                    {work.paper_mechanism}
                  </p>
                )}
                {work.difference && (
                  <p>
                    <span className="font-medium">Difference: </span>
                    {work.difference}
                  </p>
                )}
                {work.solves_gap && (
                  <p>
                    <span className="font-medium">Already solves the gap? </span>
                    {work.solves_gap}
                  </p>
                )}
                <div className="space-y-2">
                  {work.evidence.map((e) => (
                    <QuoteLink key={e.sentence_id} sentenceId={e.sentence_id} quote={e.quote} />
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
