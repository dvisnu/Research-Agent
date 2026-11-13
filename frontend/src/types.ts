// The shapes the server sends (src/research_agent/web/views.py and report.py).

export type RunState = "running" | "completed" | "failed" | "interrupted" | "stale";

export interface Diagnosis {
  quota: boolean;
  role: string | null;
  providers: string[];
  keys: string[];
}

export interface Run {
  run_id: string;
  question: string;
  status: "running" | "completed" | "failed";
  state: RunState;
  error: string | null;
  diagnosis: Diagnosis | null;
  created_at: string;
  updated_at: string;
  last_activity: string | null;
  best_idea: { title: string; level: number | null } | null;
  coverage: Coverage | null;
  started_here: boolean;
  console_tail?: string[];
  files?: string[];
}

export interface Status {
  active: string | null;
  running: Run[];
  stages: { name: string; phase: number; summary: string }[];
  setup: {
    models?: Record<string, string | null>;
    keys?: Record<string, boolean>;
    corpus?: { full_read: number; abstract_only: number };
    error: string | null;
  };
}

export interface StageProgress {
  name: string;
  phase: number;
  summary: string;
  status: "done" | "running" | "failed" | "pending";
  runs: number;
  failed_attempts: number;
  seconds: number;
  llm_calls: number;
  api_calls: number;
  error: string | null;
}

export interface Call {
  kind: "llm_call" | "api_call";
  step: string;
  who: string;
  status: string;
  error: string | null;
  cached: boolean;
  seconds: number;
  ended_at: string;
}

export interface Progress {
  run: Run;
  current: { stage: string; since: string | null; calls: number } | null;
  last_finished: string | null;
  stages: StageProgress[];
  recent: Call[];
}

export interface Coverage {
  elements: number;
  supported: number;
  coverage: number;
  removed?: number;
  flagged?: number;
  labels?: Record<string, number>;
  fields?: Record<string, { elements: number; supported: number }>;
}

export interface Quote {
  sentence_id: string;
  quote: string;
  paper?: string | null;
}

export interface PlanElement {
  element_id: string;
  label: string;
  text: string;
  name: string | null;
  basis: string | null;
  sources: number[];
  decision: string | null;
  alternatives: string[];
  derivation: string | null;
  verdict: string | null;
  verdict_reason: string | null;
  status: "kept" | "flagged" | "removed";
}

export interface PlanSource {
  number: number;
  paper: string;
  sentence_id: string;
  quote: string;
  section: string | null;
}

export const PLAN_FIELDS = [
  ["hypothesis", "Hypothesis"],
  ["research_question", "Research question"],
  ["method", "Method"],
  ["datasets", "Datasets"],
  ["baselines", "Baselines"],
  ["metrics", "Metrics"],
  ["conditions", "Experimental conditions"],
  ["ablations", "Ablation studies"],
  ["expected_results", "Expected results"],
  ["risks_and_limitations", "Risks / limitations"],
] as const;

export type PlanField = (typeof PLAN_FIELDS)[number][0];

export type ExperimentPlan = {
  idea_id: string | null;
  idea_title: string | null;
  hypothesis: PlanElement | null;
  research_question: PlanElement | null;
  provenance_coverage: Coverage | Record<string, never>;
  sources: PlanSource[];
  removed: PlanElement[];
  open_points: string[];
  revisions: string[];
  note: string | null;
} & { [K in Exclude<PlanField, "hypothesis" | "research_question">]: PlanElement[] };

export interface ClosestWork {
  position: number;
  title: string;
  year: number | null;
  read: string;
  purpose: string | null;
  mechanism: string | null;
  evaluation: string | null;
  paper_mechanism: string | null;
  solves_gap: string | null;
  difference: string | null;
  evidence: Quote[];
}

export interface Idea {
  rank: number | null;
  idea_id: string;
  origin: string;
  title: string;
  purpose: string;
  mechanism: string;
  evaluation: string;
  kind: string | null;
  builds_on: string[];
  feasibility: string | null;
  level: number | null;
  label: string | null;
  rating: string | null;
  inconclusive: boolean;
  reasons: string[];
  coverage: Record<string, unknown>;
  gap: string | null;
  gap_kind: string | null;
  gap_evidence: Quote[];
  addressed_by: string[];
  closest: ClosestWork[];
  history: string[];
}

export interface Gap {
  rank: number;
  kind: string;
  gap: string;
  warrant: string | null;
  relevance: number;
  papers: number;
  evidence: (Quote & { paper: string })[];
}

export interface Report {
  final: boolean;
  run_id: string;
  question: string;
  generated_at: string;
  novel_evidence_report: {
    proposed_idea: string | null;
    novelty_assessment: string | null;
    evidence: string[];
    research_gap: string | null;
    closest_existing_work: string[];
    difference_from_existing_work: string | null;
  };
  experiment_plan: ExperimentPlan;
  related_work: { title: string; year: number | null; similarity: number; shares: string[]; purpose: string | null }[];
  landscape: { methods: string[]; datasets: string[]; cells: string[][] } | null;
  research_gaps: Gap[];
  ideas: Idea[];
  knowledge_graph: {
    papers_in_corpus: number;
    papers_extracted: number;
    quote_check: Record<string, number>;
    nodes: Record<string, number>;
    edges: Record<string, number>;
    names_merged: number;
  } | null;
}

export interface Paper {
  paper_id: string;
  title: string;
  year: number | null;
  venue: string | null;
  authors: string[];
  citations: number | null;
  rank: number | null;
  read_level: "full" | "abstract" | "excluded" | null;
  relevance: "core" | "related" | "off_topic" | null;
  reason: string | null;
  text_source: string | null;
  channels: string[];
  links: Record<string, string>;
}

export interface Sentence {
  sentence_id: string;
  text: string;
  section: string | null;
  retired: boolean;
  context: { sentence_id: string; text: string }[];
  paper: { paper_id: string; title: string; year: number | null; links: Record<string, string> };
}

export interface Fact {
  fact_id: string;
  kind: string;
  text: string;
  origin: { from: string | null; query: string | null };
  papers: { paper: string; why: string[] }[];
  evidence: { sentence_id: string; quote: string; kept: boolean; sentence: Sentence | null }[];
}

export interface ElementChain {
  element_id: string;
  label: string;
  text: string;
  name: string | null;
  status: string;
  verdict: string | null;
  verification: { reason?: string; model?: string; [k: string]: unknown } | null;
  decision: { reason?: string; alternatives?: string[]; answers?: { point: number; problem: string }[] };
  derivation: { round?: number; change?: string; model?: string; prompt?: string; checks?: string[] };
  basis: string | null;
  facts: Fact[];
}

export interface GraphData {
  nodes: { id: string; type: string; name: string; evidence: number }[];
  edges: { id: string; type: string; source: string; target: string }[];
}

export interface GraphItem {
  kind: "node" | "edge";
  node_type?: string;
  edge_type?: string;
  name?: string;
  attrs: Record<string, unknown>;
  aliases?: string[];
  links?: { edge_id: string; type: string; direction: "in" | "out"; node_id: string; node_type: string; name: string }[];
  source?: { node_id: string; node_type: string; name: string } | null;
  target?: { node_id: string; node_type: string; name: string } | null;
  evidence: { sentence_id: string; quote: string; sentence: Sentence | null }[];
}

export interface Cost {
  seconds: number;
  elapsed: number;
  failed_attempts: number;
  stages: { stage: string; attempts: number; failed: number; seconds: number; llm_calls: number; api_requests: number }[];
  models: {
    model: string;
    role: string;
    calls: number;
    failed: number;
    input_tokens: number;
    output_tokens: number;
    reasoning_tokens: number;
  }[];
  apis: { service: string; requests: number; cached: number; failed: number }[];
  reuse: { kind: string; reused: number; asked: number }[];
}
