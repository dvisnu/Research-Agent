import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../../api";
import { Badge, Card, Empty, Spinner, type Tone } from "../../components/ui";
import { words } from "../../format";
import type { Paper } from "../../types";

const LEVEL: Record<string, [Tone, string]> = {
  full: ["indigo", "full text"],
  abstract: ["sky", "abstract"],
  excluded: ["slate", "not read"],
};
const RELEVANCE: Record<string, Tone> = { core: "green", related: "slate", off_topic: "red" };

export function PapersTab({ runId }: { runId: string }) {
  const { data, error, isLoading } = useQuery({ queryKey: ["papers", runId], queryFn: () => api.papers(runId) });
  const [level, setLevel] = useState<"read" | "all" | "full" | "abstract" | "excluded">("read");
  const [search, setSearch] = useState("");
  const [open, setOpen] = useState<string | null>(null);

  if (isLoading) return <Spinner />;
  if (error) return <Empty>{error.message}</Empty>;
  if (!data?.length) return <Empty>No papers yet: the search hasn't run.</Empty>;

  const counts = {
    full: data.filter((p) => p.read_level === "full").length,
    abstract: data.filter((p) => p.read_level === "abstract").length,
  };
  const shown = data.filter(
    (p) =>
      (level === "all" ||
        (level === "read" ? p.read_level === "full" || p.read_level === "abstract" : p.read_level === level)) &&
      (!search || p.title.toLowerCase().includes(search.toLowerCase())),
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        {(
          [
            ["read", `Read (${counts.full + counts.abstract})`],
            ["full", `Full text (${counts.full})`],
            ["abstract", `Abstract (${counts.abstract})`],
            ["excluded", "Not read"],
            ["all", `All candidates (${data.length})`],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setLevel(key)}
            className={`rounded-full px-3 py-1 ${level === key ? "bg-slate-900 text-white" : "bg-white text-slate-600 ring-1 ring-slate-300 hover:bg-slate-50"}`}
          >
            {label}
          </button>
        ))}
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter by title…"
          className="ml-auto w-64 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm shadow-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 focus:outline-none"
        />
      </div>
      <Card className="overflow-hidden">
        <table className="w-full text-sm">
          <thead className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium tracking-wide text-slate-500 uppercase">
            <tr>
              <th className="w-14 px-4 py-2.5">Rank</th>
              <th className="px-4 py-2.5">Paper</th>
              <th className="hidden px-4 py-2.5 md:table-cell">Cited by</th>
              <th className="px-4 py-2.5">Read</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {shown.slice(0, 300).map((paper) => (
              <PaperRow
                key={paper.paper_id}
                paper={paper}
                open={open === paper.paper_id}
                toggle={() => setOpen(open === paper.paper_id ? null : paper.paper_id)}
              />
            ))}
          </tbody>
        </table>
        {shown.length > 300 && (
          <div className="px-4 py-2 text-xs text-slate-500">Showing 300 of {shown.length}; filter to see the rest.</div>
        )}
      </Card>
    </div>
  );
}

function PaperRow({ paper, open, toggle }: { paper: Paper; open: boolean; toggle: () => void }) {
  const [tone, label] = LEVEL[paper.read_level ?? ""] ?? ["slate", "—"];
  return (
    <>
      <tr onClick={toggle} className="cursor-pointer align-top hover:bg-slate-50">
        <td className="px-4 py-2.5 text-slate-500">{paper.rank ?? "—"}</td>
        <td className="px-4 py-2.5">
          <div className="font-medium text-slate-900">
            {paper.title}
            {paper.year && <span className="font-normal text-slate-500"> ({paper.year})</span>}
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-slate-500">
            {paper.authors.length > 0 && (
              <span className="truncate">
                {paper.authors.slice(0, 3).join(", ")}
                {paper.authors.length > 3 ? " et al." : ""}
              </span>
            )}
            {paper.venue && <span>{paper.venue}</span>}
            {Object.entries(paper.links).map(([name, url]) => (
              <a
                key={name}
                href={url}
                target="_blank"
                rel="noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="text-indigo-600 hover:underline"
              >
                {name} ↗
              </a>
            ))}
          </div>
        </td>
        <td className="hidden px-4 py-2.5 text-slate-600 md:table-cell">{paper.citations?.toLocaleString() ?? "—"}</td>
        <td className="px-4 py-2.5">
          <div className="flex flex-col items-start gap-1">
            <Badge tone={tone}>{label}</Badge>
            {paper.relevance && <Badge tone={RELEVANCE[paper.relevance]}>{words(paper.relevance)}</Badge>}
          </div>
        </td>
      </tr>
      {open && (
        <tr className="bg-slate-50/70">
          <td />
          <td colSpan={3} className="px-4 pb-3 text-xs text-slate-600">
            <p>{paper.reason ?? "Not ranked yet."}</p>
            <p className="mt-1 text-slate-500">
              Found by: {paper.channels.map(words).join(", ") || "—"}
              {paper.text_source ? ` · text: ${words(paper.text_source)}` : ""}
            </p>
          </td>
        </tr>
      )}
    </>
  );
}
