import type { ReactNode } from "react";
import type { RunState } from "../types";
import { words } from "../format";

export type Tone = "slate" | "green" | "amber" | "red" | "indigo" | "sky" | "violet" | "rose";

const TONES: Record<Tone, string> = {
  slate: "bg-slate-100 text-slate-700 ring-slate-200",
  green: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  amber: "bg-amber-50 text-amber-800 ring-amber-200",
  red: "bg-red-50 text-red-700 ring-red-200",
  indigo: "bg-indigo-50 text-indigo-700 ring-indigo-200",
  sky: "bg-sky-50 text-sky-700 ring-sky-200",
  violet: "bg-violet-50 text-violet-700 ring-violet-200",
  rose: "bg-rose-50 text-rose-700 ring-rose-200",
};

export function Badge({ tone = "slate", children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span
      title={title}
      className={`inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium whitespace-nowrap ring-1 ring-inset ${TONES[tone]}`}
    >
      {children}
    </span>
  );
}

const STATE: Record<RunState, [Tone, string]> = {
  running: ["indigo", "Running"],
  completed: ["green", "Completed"],
  failed: ["red", "Stopped"],
  interrupted: ["amber", "Interrupted"],
  stale: ["amber", "No activity"],
};

export function StateBadge({ state }: { state: RunState }) {
  const [tone, label] = STATE[state];
  return (
    <Badge tone={tone}>
      {state === "running" && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-indigo-500" />}
      {label}
    </Badge>
  );
}

const VERDICT: Record<string, [Tone, string]> = {
  supported: ["green", "Supported"],
  not_supported: ["amber", "Not supported"],
  contradicted: ["red", "Contradicted"],
  no_source: ["slate", "No source"],
  unverified: ["slate", "Unverified"],
};

export function VerdictBadge({ verdict }: { verdict: string | null }) {
  const [tone, label] = VERDICT[verdict ?? "unverified"] ?? ["slate", words(verdict ?? "unverified")];
  return <Badge tone={tone}>{label}</Badge>;
}

export function LevelBadge({ level, label }: { level: number | null; label?: string | null }) {
  if (level == null) return <Badge>not rated</Badge>;
  const tone: Tone = level >= 4 ? "green" : level === 3 ? "sky" : level === 2 ? "amber" : "red";
  return (
    <Badge tone={tone} title={label ?? undefined}>
      Level {level}/5{label ? ` · ${label}` : ""}
    </Badge>
  );
}

const SIMILARITY: Record<string, Tone> = { same: "red", similar: "amber", different: "green", unknown: "slate" };

export function SimilarityBadge({ what, value }: { what: string; value: string | null }) {
  if (!value) return null;
  return (
    <Badge tone={SIMILARITY[value] ?? "slate"}>
      {what}: {value}
    </Badge>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-xl border border-slate-200 bg-white shadow-sm ${className}`}>{children}</div>;
}

export function SectionTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mb-3 flex items-baseline justify-between gap-3">
      <h2 className="text-base font-semibold text-slate-900">{children}</h2>
      {aside && <div className="text-sm text-slate-500">{aside}</div>}
    </div>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-8 text-sm text-slate-500">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-indigo-600" />
      {label}…
    </div>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">{message}</div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-slate-300 px-6 py-10 text-center text-sm text-slate-500">
      {children}
    </div>
  );
}

export function Button({
  children,
  onClick,
  kind = "secondary",
  disabled,
  type = "button",
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  kind?: "primary" | "secondary" | "danger";
  disabled?: boolean;
  type?: "button" | "submit";
  title?: string;
}) {
  const styles = {
    primary: "bg-indigo-600 text-white hover:bg-indigo-500 shadow-sm",
    secondary: "bg-white text-slate-700 ring-1 ring-inset ring-slate-300 hover:bg-slate-50",
    danger: "bg-white text-red-700 ring-1 ring-inset ring-red-300 hover:bg-red-50",
  }[kind];
  return (
    <button
      type={type}
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${styles}`}
    >
      {children}
    </button>
  );
}

export function Bar({ value, tone = "indigo" }: { value: number; tone?: "indigo" | "green" | "amber" }) {
  const color = { indigo: "bg-indigo-500", green: "bg-emerald-500", amber: "bg-amber-500" }[tone];
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
      <div className={`h-full rounded-full ${color}`} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-4 py-3 shadow-sm">
      <div className="text-xs font-medium tracking-wide text-slate-500 uppercase">{label}</div>
      <div className="mt-1 text-xl font-semibold text-slate-900">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

// A little Markdown: paragraphs, "- " bullets and **bold**, enough for the report's texts.
export function Prose({ text }: { text: string | null | undefined }) {
  if (!text) return null;
  const blocks = text.trim().split(/\n\s*\n/);
  return (
    <div className="space-y-2 text-sm leading-relaxed text-slate-700">
      {blocks.map((block, i) => {
        const lines = block.split("\n");
        if (lines.every((line) => /^\s*[-*] /.test(line))) {
          return (
            <ul key={i} className="list-disc space-y-1 pl-5">
              {lines.map((line, j) => (
                <li key={j}>{inline(line.replace(/^\s*[-*] /, ""))}</li>
              ))}
            </ul>
          );
        }
        return <p key={i}>{inline(lines.join(" "))}</p>;
      })}
    </div>
  );
}

function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={i} className="font-semibold text-slate-900">
        {part.slice(2, -2)}
      </strong>
    ) : (
      part
    ),
  );
}
