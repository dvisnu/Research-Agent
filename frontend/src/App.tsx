import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { NewRunPage } from "./pages/NewRunPage";
import { RunPage } from "./pages/RunPage";
import { RunsPage } from "./pages/RunsPage";
import { Link, useRoute } from "./router";

export function App() {
  const route = useRoute();
  const { data: status } = useQuery({ queryKey: ["status"], queryFn: api.status, refetchInterval: 5_000 });
  const going = status?.active ?? status?.running[0]?.run_id ?? null;

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center gap-6 px-4 sm:px-6">
          <Link to="/" className="flex items-center gap-2 font-semibold text-slate-900">
            <svg viewBox="0 0 32 32" className="h-7 w-7" aria-hidden>
              <circle cx="16" cy="16" r="14" fill="#4f46e5" />
              <path
                d="M10 21l4-10 4 7 3-4"
                stroke="white"
                strokeWidth="2.5"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            Research Agent
          </Link>
          <nav className="flex items-center gap-1 text-sm">
            <Link
              to="/"
              className={`rounded-md px-3 py-1.5 ${route.page === "runs" ? "bg-slate-100 font-medium text-slate-900" : "text-slate-600 hover:text-slate-900"}`}
            >
              Runs
            </Link>
            <Link
              to="/new"
              className={`rounded-md px-3 py-1.5 ${route.page === "new" ? "bg-slate-100 font-medium text-slate-900" : "text-slate-600 hover:text-slate-900"}`}
            >
              New run
            </Link>
          </nav>
          <div className="ml-auto">
            {going && (
              <Link
                to={`/runs/${going}/progress`}
                className="flex items-center gap-2 rounded-full bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700 ring-1 ring-indigo-200 hover:bg-indigo-100"
              >
                <span className="h-2 w-2 animate-pulse rounded-full bg-indigo-500" />
                A run is going
              </Link>
            )}
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        {route.page === "runs" && <RunsPage />}
        {route.page === "new" && <NewRunPage />}
        {route.page === "run" && <RunPage runId={route.runId} tab={route.tab} />}
      </main>
    </div>
  );
}
