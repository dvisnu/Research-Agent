import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";
import { api } from "../api";
import { Badge, Button, Card, ErrorBox } from "../components/ui";
import { ago } from "../format";
import { Link, navigate } from "../router";

const EXAMPLES = [
  "How can mixture-of-experts language models balance load across experts without hurting model quality?",
  "Speed up LLM inference with speculative decoding that needs no separate draft model.",
];

export function NewRunPage() {
  const client = useQueryClient();
  const { data: status } = useQuery({ queryKey: ["status"], queryFn: api.status });
  const [question, setQuestion] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [refresh, setRefresh] = useState(false);

  const start = useMutation({
    mutationFn: () => {
      const form = new FormData();
      form.append("question", question);
      form.append("refresh", String(refresh));
      files.forEach((file) => form.append("uploads", file));
      return api.start(form);
    },
    onSuccess: ({ run_id }) => {
      client.invalidateQueries({ queryKey: ["runs"] });
      client.invalidateQueries({ queryKey: ["status"] });
      navigate(`/runs/${run_id}/progress`);
    },
  });

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (question.trim()) start.mutate();
  };

  const busy = status?.active;
  const elsewhere = status?.running.filter((r) => r.run_id !== status.active) ?? [];
  const setup = status?.setup;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <form onSubmit={onSubmit} className="space-y-5">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">New run</h1>
          <p className="mt-1 text-sm text-slate-500">
            Ask a research question, or describe an idea to develop. The agent reviews the literature, finds
            evidence-backed gaps, proposes ideas and checks their novelty, then writes an experiment plan whose every
            element cites its sources.
          </p>
        </div>

        <Card className="space-y-4 p-5">
          <label className="block">
            <span className="text-sm font-medium text-slate-800">Research question or idea</span>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={5}
              maxLength={4000}
              placeholder="e.g. How can retrieval-augmented generation handle knowledge conflicts between retrieved documents?"
              className="mt-1.5 block w-full rounded-lg border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 focus:outline-none"
            />
          </label>
          <div className="flex flex-wrap gap-2 text-xs">
            <span className="text-slate-500">Try:</span>
            {EXAMPLES.map((example) => (
              <button
                type="button"
                key={example}
                onClick={() => setQuestion(example)}
                className="max-w-md truncate rounded-full bg-slate-100 px-2.5 py-0.5 text-slate-600 hover:bg-indigo-50 hover:text-indigo-700"
              >
                {example}
              </button>
            ))}
          </div>

          <label className="block">
            <span className="text-sm font-medium text-slate-800">Papers to read in full (optional)</span>
            <span className="block text-xs text-slate-500">PDFs you provide are always read in full and marked as yours.</span>
            <input
              type="file"
              accept="application/pdf,.pdf"
              multiple
              onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
              className="mt-1.5 block w-full text-sm text-slate-600 file:mr-3 file:rounded-lg file:border-0 file:bg-slate-100 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-slate-700 hover:file:bg-slate-200"
            />
          </label>

          <label className="flex items-start gap-2 text-sm">
            <input
              type="checkbox"
              checked={refresh}
              onChange={(e) => setRefresh(e.target.checked)}
              className="mt-0.5 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
            />
            <span>
              <span className="font-medium text-slate-800">Search afresh</span>
              <span className="block text-xs text-slate-500">
                By default, search responses from the last 30 days are reused, so asking again finds the same papers.
              </span>
            </span>
          </label>
        </Card>

        {busy && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
            A run started here is still going (<Link to={`/runs/${busy}/progress`} className="underline">{busy}</Link>);
            one run at a time.
          </div>
        )}
        {!busy && !!elsewhere.length && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
            Another run is going outside this window (
            {elsewhere.map((r) => (
              <Link key={r.run_id} to={`/runs/${r.run_id}/progress`} className="underline">
                {r.run_id}
              </Link>
            ))}
            , last active {ago(elsewhere[0].last_activity)}). Two runs at once share the free-tier quotas and the
            machine's memory.
          </div>
        )}
        {start.error && <ErrorBox error={start.error} />}

        <div className="flex items-center gap-3">
          <Button type="submit" kind="primary" disabled={!question.trim() || start.isPending || !!busy}>
            {start.isPending ? "Starting…" : "Start run"}
          </Button>
          <span className="text-xs text-slate-500">
            A new topic takes about 1–2.5 hours on the free tiers; you can leave this page and come back.
          </span>
        </div>
      </form>

      <aside className="space-y-4">
        <Card className="p-4">
          <h2 className="text-sm font-semibold text-slate-900">Setup</h2>
          {setup?.error ? (
            <div className="mt-2">
              <ErrorBox error={`Invalid .env: ${setup.error}`} />
            </div>
          ) : (
            <dl className="mt-3 space-y-2 text-xs">
              {Object.entries(setup?.models ?? {}).map(([role, spec]) => (
                <div key={role}>
                  <dt className="font-medium text-slate-500 capitalize">{role}</dt>
                  <dd className="break-words text-slate-800">{spec ? spec.split(",").join(", then ") : "not set"}</dd>
                </div>
              ))}
              {setup?.corpus && (
                <div>
                  <dt className="font-medium text-slate-500">Corpus</dt>
                  <dd className="text-slate-800">
                    {setup.corpus.full_read} papers in full + {setup.corpus.abstract_only} by abstract
                  </dd>
                </div>
              )}
            </dl>
          )}
        </Card>
        <Card className="p-4">
          <h2 className="text-sm font-semibold text-slate-900">API keys in .env</h2>
          <ul className="mt-3 space-y-1.5 text-xs">
            {Object.entries(setup?.keys ?? {}).map(([key, set]) => (
              <li key={key} className="flex items-center justify-between gap-2">
                <code className="text-slate-700">{key}</code>
                {set ? <Badge tone="green">set</Badge> : <Badge tone="amber">missing</Badge>}
              </li>
            ))}
          </ul>
          <p className="mt-3 text-xs text-slate-500">
            When a quota runs out the run stops and says which key to switch. Edit .env, then press Resume: the run
            picks up the new key without restarting the UI.
          </p>
        </Card>
      </aside>
    </div>
  );
}
