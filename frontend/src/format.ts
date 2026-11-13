export function duration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 60) return `${Math.round(seconds)} s`;
  const minutes = seconds / 60;
  if (minutes < 60) return `${minutes < 10 ? minutes.toFixed(1) : Math.round(minutes)} min`;
  const hours = Math.floor(minutes / 60);
  return `${hours} h ${Math.round(minutes - hours * 60)} min`;
}

export function since(iso: string | null | undefined, now = Date.now()): number {
  return iso ? (now - new Date(iso).getTime()) / 1000 : 0;
}

export function ago(iso: string | null | undefined): string {
  if (!iso) return "—";
  const seconds = since(iso);
  if (seconds < 60) return "just now";
  return `${duration(seconds)} ago`;
}

export function when(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function count(n: number | null | undefined): string {
  return n == null ? "—" : n.toLocaleString();
}

export function percent(x: number | null | undefined): string {
  return x == null ? "—" : `${Math.round(x * 100)}%`;
}

export function words(name: string): string {
  return name.replaceAll("_", " ");
}
