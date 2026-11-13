import { Bar, Card, Empty, ErrorBox, Prose, SectionTitle, Spinner, Stat } from "../../components/ui";
import { count, percent } from "../../format";
import { Link } from "../../router";
import type { Report } from "../../types";
import { useReport } from "./useReport";

export function OverviewTab({ runId, live }: { runId: string; live: boolean }) {
  const { data, error, isLoading } = useReport(runId, live);
  if (isLoading) return <Spinner />;
  if (error) return <Empty>{error.message}</Empty>;
  if (!data) return null;
  const ner = data.novel_evidence_report;
  const cov = data.experiment_plan.provenance_coverage;
  const kg = data.knowledge_graph;
  const best = data.ideas[0];

  return (
    <div className="space-y-6">
      {!data.final && (
        <div className="rounded-lg bg-sky-50 px-4 py-2 text-sm text-sky-900">
          The run hasn't written its report yet; this is what it has found so far.
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Papers read" value={count(kg?.papers_in_corpus)} hint={kg ? `${count(kg.papers_extracted)} extracted` : undefined} />
        <Stat
          label="Quote check"
          value={kg?.quote_check?.pass_rate != null ? percent(kg.quote_check.pass_rate) : "—"}
          hint="extracted items whose quotes are found in their sentences"
        />
        <Stat label="Gaps" value={data.research_gaps.length} hint="evidence-backed, ranked" />
        <Stat
          label="Plan coverage"
          value={"coverage" in cov ? percent(cov.coverage) : "—"}
          hint={"coverage" in cov ? `${cov.supported} of ${cov.elements} elements supported` : "no plan yet"}
        />
      </div>

      <Card className="p-5">
        <SectionTitle aside={best && <Link to={`/runs/${runId}/ideas`} className="text-indigo-600 hover:underline">All ideas →</Link>}>
          Novel Evidence Report
        </SectionTitle>
        {!ner.proposed_idea && <p className="text-sm text-slate-500">No idea yet.</p>}
        {ner.proposed_idea && (
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="space-y-4">
              <Field title="Proposed idea">
                <Prose text={ner.proposed_idea} />
              </Field>
              <Field title="Novelty assessment">
                <Prose text={ner.novelty_assessment} />
              </Field>
              <Field title="Research gap">
                <Prose text={ner.research_gap} />
              </Field>
            </div>
            <div className="space-y-4">
              <Field title="Closest existing work">
                <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700">
                  {ner.closest_existing_work.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </Field>
              <Field title="Difference from existing work">
                <Prose text={ner.difference_from_existing_work} />
              </Field>
              <Field title="Evidence">
                <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700">
                  {ner.evidence.map((e, i) => (
                    <li key={i}>{e}</li>
                  ))}
                </ul>
              </Field>
            </div>
          </div>
        )}
      </Card>

      {"coverage" in cov && (
        <Card className="p-5">
          <SectionTitle aside={<Link to={`/runs/${runId}/plan`} className="text-indigo-600 hover:underline">Open the plan →</Link>}>
            Experiment plan: {data.experiment_plan.idea_title}
          </SectionTitle>
          <PlanFields report={data} />
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="p-5">
          <SectionTitle>Closest related work</SectionTitle>
          {!data.related_work.length && <p className="text-sm text-slate-500">None yet.</p>}
          <ol className="space-y-3">
            {data.related_work.map((paper, i) => (
              <li key={i} className="text-sm">
                <div className="font-medium text-slate-900">
                  {paper.title}
                  {paper.year && <span className="font-normal text-slate-500"> ({paper.year})</span>}
                </div>
                {paper.purpose && <p className="mt-0.5 line-clamp-2 text-slate-600">{paper.purpose}</p>}
                <p className="mt-0.5 text-xs text-slate-500">
                  similarity {paper.similarity.toFixed(2)}
                  {paper.shares.length ? ` · shares ${paper.shares.join(", ")}` : ""}
                </p>
              </li>
            ))}
          </ol>
        </Card>
        <Card className="p-5">
          <SectionTitle>Research landscape</SectionTitle>
          {data.landscape ? <Landscape grid={data.landscape} /> : <p className="text-sm text-slate-500">No grid yet.</p>}
        </Card>
      </div>
      {error && <ErrorBox error={error} />}
    </div>
  );
}

function Field({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="mb-1 text-xs font-semibold tracking-wide text-slate-500 uppercase">{title}</h3>
      {children}
    </div>
  );
}

function PlanFields({ report }: { report: Report }) {
  const cov = report.experiment_plan.provenance_coverage;
  if (!("fields" in cov) || !cov.fields) return null;
  return (
    <div className="grid gap-x-8 gap-y-2 sm:grid-cols-2">
      {Object.entries(cov.fields).map(([field, f]) => (
        <div key={field} className="text-sm">
          <div className="flex justify-between text-slate-700">
            <span className="capitalize">{field.replaceAll("_", " ")}</span>
            <span className="text-xs text-slate-500">
              {f.supported}/{f.elements} supported
            </span>
          </div>
          <Bar value={f.elements ? f.supported / f.elements : 0} tone="green" />
        </div>
      ))}
    </div>
  );
}

function Landscape({ grid }: { grid: NonNullable<Report["landscape"]> }) {
  return (
    <div className="overflow-x-auto">
      <table className="text-xs">
        <thead>
          <tr>
            <th className="sticky left-0 bg-white" />
            {grid.datasets.map((d) => (
              <th key={d} className="h-28 min-w-8 align-bottom">
                <div className="w-6 origin-bottom-left translate-x-4 -rotate-45 whitespace-nowrap text-slate-600">{d}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {grid.methods.map((m, i) => (
            <tr key={m}>
              <th className="sticky left-0 max-w-40 truncate bg-white pr-2 text-left font-normal text-slate-700" title={m}>
                {m}
              </th>
              {grid.cells[i].map((cell, j) => {
                const [style, meaning] = cell.startsWith("R")
                  ? ["bg-indigo-100 text-indigo-800", `${cell.slice(1)} result(s)`]
                  : cell.startsWith("p")
                    ? ["bg-amber-50 text-amber-800", `${cell.slice(1)} paper(s) use both, no result for the pair`]
                    : ["bg-slate-50 text-slate-300", "nothing"];
                return (
                  <td
                    key={j}
                    title={`${m} × ${grid.datasets[j]}: ${meaning}`}
                    className={`h-7 min-w-8 border border-white text-center ${style}`}
                  >
                    {cell === "-" ? "·" : cell}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-slate-500">
        Most-studied methods × datasets. <span className="text-indigo-700">R<i>n</i></span>: n results;{" "}
        <span className="text-amber-700">p<i>n</i></span>: n papers use both without a result for the pair (a possible
        structural gap); ·: nothing.
      </p>
    </div>
  );
}
