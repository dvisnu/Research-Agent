import type {
  Cost,
  ElementChain,
  GraphData,
  GraphItem,
  Paper,
  Progress,
  Report,
  Run,
  Sentence,
  Status,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  // The server refuses changes without this header, so other sites can't start runs.
  if (init.method && init.method !== "GET") headers.set("X-Research-Agent", "1");
  const response = await fetch(path, { ...init, headers });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      // not JSON: keep the status line
    }
    throw new ApiError(response.status, message);
  }
  return response.json() as Promise<T>;
}

const run = (id: string) => `/api/runs/${encodeURIComponent(id)}`;

export const api = {
  status: () => request<Status>("/api/status"),
  runs: () => request<Run[]>("/api/runs"),
  run: (id: string) => request<Run>(run(id)),
  progress: (id: string) => request<Progress>(`${run(id)}/progress`),
  report: (id: string) => request<Report>(`${run(id)}/report`),
  papers: (id: string) => request<Paper[]>(`${run(id)}/papers`),
  cost: (id: string) => request<Cost>(`${run(id)}/cost`),
  graph: (id: string) => request<GraphData>(`${run(id)}/graph`),
  graphItem: (kind: "node" | "edge", id: string) => request<GraphItem>(`/api/graph/${kind}/${encodeURIComponent(id)}`),
  element: (id: string, elementId: string) =>
    request<ElementChain>(`${run(id)}/plan/elements/${encodeURIComponent(elementId)}`),
  sentence: (id: string) => request<Sentence>(`/api/sentences/${encodeURIComponent(id)}`),
  start: (form: FormData) => request<{ run_id: string }>("/api/runs", { method: "POST", body: form }),
  resume: (id: string) => request<{ run_id: string }>(`${run(id)}/resume`, { method: "POST" }),
  stop: (id: string) => request<{ run_id: string }>(`${run(id)}/stop`, { method: "POST" }),
  fileUrl: (id: string, name: string) => `${run(id)}/files/${name}`,
};
