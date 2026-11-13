import { useQuery } from "@tanstack/react-query";
import { api } from "../../api";

// The run's report: report.json once written, else what the run has found so far.
export function useReport(runId: string, live: boolean) {
  return useQuery({
    queryKey: ["report", runId],
    queryFn: () => api.report(runId),
    refetchInterval: live ? 30_000 : false,
  });
}
