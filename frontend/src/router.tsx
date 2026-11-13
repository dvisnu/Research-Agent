// A small history router: the app has three pages, so no router library.
import { type MouseEvent, type ReactNode, useSyncExternalStore } from "react";

export const TABS = ["progress", "overview", "gaps", "ideas", "plan", "graph", "papers", "cost"] as const;
export type Tab = (typeof TABS)[number];

export type Route = { page: "runs" } | { page: "new" } | { page: "run"; runId: string; tab: Tab | null };

function parse(path: string): Route {
  const parts = path.split("/").filter(Boolean);
  if (parts[0] === "new") return { page: "new" };
  if (parts[0] === "runs" && parts[1]) {
    const tab = TABS.find((t) => t === parts[2]) ?? null;
    return { page: "run", runId: decodeURIComponent(parts[1]), tab };
  }
  return { page: "runs" };
}

const listeners = new Set<() => void>();
window.addEventListener("popstate", () => listeners.forEach((l) => l()));

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useRoute(): Route {
  const path = useSyncExternalStore(subscribe, () => window.location.pathname);
  return parse(path);
}

export function navigate(to: string, { replace = false } = {}) {
  if (to === window.location.pathname) return;
  if (replace) window.history.replaceState(null, "", to);
  else window.history.pushState(null, "", to);
  listeners.forEach((l) => l());
  if (!replace) window.scrollTo(0, 0);
}

export function Link({ to, className, children }: { to: string; className?: string; children: ReactNode }) {
  const onClick = (event: MouseEvent<HTMLAnchorElement>) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    navigate(to);
  };
  return (
    <a href={to} onClick={onClick} className={className}>
      {children}
    </a>
  );
}
