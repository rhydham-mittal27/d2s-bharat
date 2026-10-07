// Typed client for the D2S FastAPI backend (backend/src/d2s/api/main.py).

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export const AREAS = [
  "coding_skills",
  "maths_stats_skills",
  "ai_and_ml_skills",
  "big_data_skills",
  "dashboard_and_storytelling_skills",
] as const;
export type Area = (typeof AREAS)[number];

export const AREA_LABEL: Record<string, string> = {
  coding_skills: "Coding",
  maths_stats_skills: "Maths & stats",
  ai_and_ml_skills: "AI & ML",
  big_data_skills: "Big data",
  dashboard_and_storytelling_skills: "Dashboards & storytelling",
};

// ---- plan --------------------------------------------------------------------------------------
export type Scheme = "balanced" | "market-led" | "outcome-led";
export type Bound = "low" | "point" | "high";

export interface PlanRequest {
  budget: number;
  trainer_hours: number;
  scheme: Scheme;
  bound: Bound;
}

export interface CourseDecision {
  course_id: string;
  seats: number;
  cost: number;
  trainer_hours: number;
}

export interface SkillOutcome {
  skill_id: string;
  learners_short: number;
  learners_closed: number;
  closure_pct: number;
  weighted_value: number;
}

export interface Plan {
  status: string;
  objective: number;
  courses: CourseDecision[];
  skills: SkillOutcome[];
  total_cost: number;
  total_trainer_hours: number;
  total_seats: number;
  messages: string[];
}

export interface WhyNot {
  course_id: string;
  name: string;
  feasible: boolean;
  objective_delta: number | null;
  objective_delta_pct: number | null;
  reason: string;
}

export interface PlanResult {
  request: PlanRequest;
  plan: Plan;
  course_names: Record<string, string>;
  weights: Record<string, number>;
  gaps: Record<string, number>;
  binding: { verdict: string; budget_slack: number; hours_slack: number | null };
  why_not: WhyNot[];
}

export interface EvidenceArea {
  area: string;
  label: string;
  coverage_pct: number;
  market_demand_share: number;
  market_pay_or: number;
  junior_hike_or: number;
  learners_short: number;
  learners_closed_by_course: number;
}

export interface CourseTrail {
  course_id: string;
  name: string;
  seats: number;
  cost: number;
  trainer_hours: number;
  areas: EvidenceArea[];
  impact_lost_if_removed_pct: number;
  constraint_context: string;
}

export interface Brief {
  title: string;
  sections: { heading: string; paragraphs: string[] }[];
  numbers_checked: number;
  unverified_numbers: string[];
  faithful: boolean;
}

// ---- confidence / baselines / what-if -------------------------------------------------------------
export interface CourseConfidence {
  course_id: string;
  name: string;
  chosen_in: number;
  scenarios: number;
  level: "high" | "medium" | "low";
  seats_range: [number, number];
}

export interface Confidence {
  level: "high" | "medium" | "low";
  score: number;
  same_plan_share: number;
  scenarios: string[];
  courses: CourseConfidence[];
  explanation: string;
}

export interface BaselineRow {
  id: string;
  name: string;
  rule: string;
  impact: number;
  impact_pct_of_optimiser: number;
  cost: number;
  trainer_hours: number;
  gaps_fully_closed: number;
  courses: string[];
}

export interface BaselineComparison {
  optimiser: { impact: number; cost: number; trainer_hours: number; gaps_fully_closed: number; courses: string[] };
  baselines: BaselineRow[];
  best_naive: string;
  uplift_vs_best_pct: number;
  uplift_vs_average_pct: number;
  summary: string;
  sweep: { budget: number; rules: Record<string, number> }[] | null;
  worst_case_pct: Record<string, number> | null;
}

export interface Scenario {
  budget_pct?: number;
  trainer_hours_delta?: number;
  remove_courses?: string[];
  require_courses?: string[];
  cost_pct?: number;
  demand_pct?: Record<string, number>;
  cohort_pct?: number;
}

export interface Preset {
  id: string;
  title: string;
  scenario: Scenario;
}

export interface PlanSummary {
  feasible: boolean;
  courses: Record<string, number>;
  cost: number;
  trainer_hours: number;
  seats: number;
  impact: number;
  closure_pct: Record<string, number>;
  learners_closed: Record<string, number>;
}

export interface WhatIfResult {
  scenario: Scenario;
  title: string;
  today: PlanSummary;
  after: PlanSummary;
  today_still_fits: boolean;
  today_impact_after: number | null;
  added: string[];
  dropped: string[];
  seat_changes: Record<string, number>;
  impact_change_pct: number | null;
  replanning_gain_pct: number | null;
  narrative: string[];
  conflicts: string[];
  course_names: Record<string, string>;
}

export interface FullResponse {
  result: PlanResult;
  evidence: CourseTrail[];
  brief: Brief;
  brief_markdown: string;
  timings_ms: Record<string, number>;
  run_id: string | null;
  baselines: BaselineComparison | null;
}

export interface Alternative {
  label: string;
  plan: Plan;
  impact_pct_of_best: number;
  cost_delta: number;
  hours_delta: number;
  added: string[];
  dropped: string[];
  tradeoff: string;
  course_names: Record<string, string>;
}

// ---- cohort ------------------------------------------------------------------------------------
export interface AreaGap {
  area: string;
  label: string;
  target_score: number;
  learners_short: number;
  share_short: number;
  mean_shortfall: number;
  linked_roles: { role: string; share_of_postings: number }[];
  linked_courses: { id: string; name: string; coverage_pct: number }[];
}

export interface CohortProfile {
  n_learners: number;
  n_rejected: number;
  issues: { row: number; problem: string }[];
  mean_probability: number;
  bands: Record<string, number>;
  areas: AreaGap[];
  cohort_id: string | null;
}

// ---- market ------------------------------------------------------------------------------------
export interface Overview {
  postings_clean: number;
  data_role_postings: number;
  distinct_canonical_skills: number;
  openings_ds_jobs: number;
  companies_ds_jobs: number;
}

export interface AreaEvidence {
  area: string;
  demand_share: number;
  demand_ci_low: number;
  demand_ci_high: number;
  odds_ratio: number;
  or_ci_low: number;
  or_ci_high: number;
  q_fdr: number;
  rq2_odds_ratio: number;
  rq2_or_ci_low: number;
  rq2_or_ci_high: number;
}

export interface DemandRow {
  skill: string;
  posts: number;
  share: number;
  share_ci_low: number;
  share_ci_high: number;
}

export interface SkillHit {
  skill: string;
  similarity: number;
  posts_all: number;
  posts_data_roles: number;
  share_data_roles: number;
  median_salary_mid_lakh: number | null;
  top_role_family: string | null;
  pay_odds_ratio: number | null;
  pay_ci: [number, number] | null;
}

// ---- auth ---------------------------------------------------------------------------------------
export interface User {
  id: string;
  email: string;
  name: string;
}

export interface TokenOut {
  access_token: string;
  token_type: string;
  expires_at: string;
  user: User;
}

// The auth provider registers the current token and what to do when the server rejects it.
let authToken: string | null = null;
let onUnauthorized: (() => void) | null = null;
export function setAuth(token: string | null, unauthorized: (() => void) | null) {
  authToken = token;
  onUnauthorized = unauthorized;
}

// ---- transport ---------------------------------------------------------------------------------
export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
  }
}

/** FastAPI sends a string, or for validation errors a list of {loc, msg}. Turn either into a sentence. */
function readableDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d: { loc?: (string | number)[]; msg?: string }) => {
        const field = d.loc?.filter((x) => x !== "body").join(".");
        const msg = (d.msg ?? "invalid value").replace(/^Value error, /, "");
        return field ? `${field}: ${msg}` : msg;
      })
      .join("; ");
  }
  return JSON.stringify(detail);
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  const headers = new Headers(init?.headers);
  if (authToken) headers.set("Authorization", `Bearer ${authToken}`);
  try {
    res = await fetch(`${API_URL}${path}`, { ...init, headers });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(`Cannot reach the backend at ${API_URL}. Is the API server running?`);
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = readableDetail(body.detail);
    } catch {}
    if (res.status === 401 && authToken && !path.startsWith("/api/auth/")) onUnauthorized?.();
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return call<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

export interface PlanBody {
  plan: PlanRequest;
  gaps?: Record<string, number>;
}

// ---- agents -----------------------------------------------------------------------------------
export interface TokenUsage {
  model_calls: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface AgentStep {
  tokens?: number;
  node: string;
  detail: string;
  elapsed_ms?: number;
  ms?: number;
  tool?: string;
  ok?: boolean;
  path?: string;
}

export interface CopilotAnswer {
  text: string;
  sections: { title: string; lines: string[]; tool: string; ok: boolean }[];
  assumptions: string[];
  notes: string[];
  path: "fast" | "llm" | "none";
  ms: number;
  tokens?: TokenUsage;
}

export interface StressReport {
  summary: string[];
  top_risks: string[];
  mitigations: string[];
  breakpoints: Record<string, number>;
  today: { cost: number; hours: number; budget: number; trainer_hours: number };
  ms: number;
  tokens?: TokenUsage;
}

export interface IntakeTarget {
  target: string;
  label: string;
  source: string | null;
  method: string;
  scale: string | null;
  sample: string[];
}

export interface IntakeProposal {
  filename: string;
  rows: number;
  columns: string[];
  targets: IntakeTarget[];
  warnings: string[];
  scale_options: string[];
  trace: AgentStep[];
}

export interface AgentStatus {
  model: { available: boolean; model: string; loaded?: boolean; reason: string | null };
  examples: string[];
}

/** POST that answers with server-sent events: calls onStep for each step, resolves with the result. */
export async function streamAgent<T>(path: string, body: unknown, onStep: (s: AgentStep) => void,
                                     signal?: AbortSignal): Promise<T> {
  const headers = new Headers({ "Content-Type": "application/json", Accept: "text/event-stream" });
  if (authToken) headers.set("Authorization", `Bearer ${authToken}`);
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { method: "POST", headers, body: JSON.stringify(body), signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(`Cannot reach the backend at ${API_URL}. Is the API server running?`);
  }
  if (!res.ok || !res.body) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      detail = readableDetail((await res.json()).detail);
    } catch {}
    if (res.status === 401 && authToken) onUnauthorized?.();
    throw new ApiError(detail, res.status);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut: number;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const event = /^event: (.*)$/m.exec(block)?.[1];
      const data = /^data: (.*)$/m.exec(block)?.[1];
      if (!event || data === undefined) continue;
      const parsed = JSON.parse(data);
      if (event === "step") onStep(parsed as AgentStep);
      else if (event === "result") return parsed as T;
      else if (event === "error") throw new ApiError(parsed.detail ?? "the agent failed");
    }
  }
  throw new ApiError("the agent stopped without a result");
}

// ---- explainable AI (offline) --------------------------------------------------------------------
export interface XFactor {
  name: string;
  value: number | string | null;
  contribution: number | null;
  unit: string;
  source: string;
  note: string;
}

export interface Explanation<D = Record<string, unknown>> {
  subject: string;
  question: string;
  summary: string;
  factors: XFactor[];
  counterfactuals: string[];
  evidence: string[];
  confidence: number | null;
  method: string;
  details: D;
}

export interface LearnerScores {
  coding_skills: number;
  maths_stats_skills: number;
  ai_and_ml_skills: number;
  big_data_skills: number;
  dashboard_and_storytelling_skills: number;
}

export type WhyDetails = {
  probability: number;
  probability_low: number;
  probability_high: number;
  baseline_probability: number;
  sum_of_contributions: number;
  logodds_gap: number;
  warnings: string[];
};

export type WhatIfDetails = {
  probability_now: number;
  probability_after: number;
  target: number;
  reached: boolean;
  total_points: number;
  curves: Record<string, { score: number; probability: number }[]>;
};

export type TextMatch = { skill: string; matched: string; span: [number, number] | null; clause_is_boilerplate: boolean };

export const api = {
  register: (name: string, email: string, password: string) =>
    post<TokenOut>("/api/auth/register", { name, email, password }),
  login: (email: string, password: string) => post<TokenOut>("/api/auth/login", { email, password }),
  me: () => call<User>("/api/auth/me"),
  planFull: (b: PlanBody, signal?: AbortSignal) => post<FullResponse>("/api/plan/full", b, signal),
  alternatives: (b: PlanBody, signal?: AbortSignal) =>
    post<Alternative[]>("/api/plan/alternatives?k=3", b, signal),
  confidence: (b: PlanBody, signal?: AbortSignal) => post<Confidence>("/api/plan/confidence", b, signal),
  baselines: (b: PlanBody) => post<BaselineComparison>("/api/plan/baselines", b),
  whatIfPresets: () => call<Preset[]>("/api/plan/what-if/presets"),
  whatIf: (b: PlanBody, scenario: Scenario) => post<WhatIfResult>("/api/plan/what-if", { ...b, scenario }),
  catalogue: () => call<{ id: string; name: string }[]>("/api/plan/catalogue"),
  xaiGlobal: () => call<Explanation<{ cv_auc: number; n_train: number }>>("/api/xai/model/global"),
  xaiWhy: (scores: LearnerScores, signal?: AbortSignal) =>
    post<Explanation<WhyDetails>>("/api/xai/learner/why", { scores }, signal),
  xaiWhatIf: (scores: LearnerScores, target: number, signal?: AbortSignal) =>
    post<Explanation<WhatIfDetails>>("/api/xai/learner/what-if", { scores, target_probability: target }, signal),
  xaiConstraints: (b: PlanBody) => post<Explanation<{ tight: string[] }>>("/api/xai/plan/constraints", b),
  xaiCourse: (b: PlanBody, courseId: string, resource: string) =>
    post<Explanation<{ found?: boolean; minimal_increase?: number; new_plan_courses?: Record<string, number> }>>(
      `/api/xai/plan/course/${encodeURIComponent(courseId)}/what-if?resource=${resource}`, b),
  xaiTag: (tag: string) =>
    call<Explanation<{ step: string; skill?: string; similarity?: number; threshold?: number }>>(
      `/api/xai/skill/tag?tag=${encodeURIComponent(tag)}`),
  xaiText: (text: string) => post<Explanation<{ matches: TextMatch[] }>>("/api/xai/skill/text", { text }),
  agentStatus: () => call<AgentStatus>("/api/agents/status"),
  intakeStart: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return call<{ thread_id: string; proposal: IntakeProposal; ms: number; tokens: TokenUsage }>("/api/agents/intake", {
      method: "POST",
      body: fd,
    });
  },
  intakeDecide: (threadId: string, decision: { approve: boolean; mapping?: Record<string, string | null>; scales?: Record<string, string> }) =>
    post<{ profile?: CohortProfile; cancelled?: boolean; mapping: Record<string, string>; scales: Record<string, string>; unreadable_rows: number; ms: number }>(
      `/api/agents/intake/${threadId}`,
      decision,
    ),
  cohortSample: () => call<CohortProfile>("/api/cohort/sample"),
  cohortUpload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return call<CohortProfile>("/api/cohort/profile", { method: "POST", body: fd });
  },
  overview: () => call<Overview>("/api/overview"),
  triangulation: () => call<AreaEvidence[]>("/api/market/triangulation"),
  demand: (top = 15) => call<DemandRow[]>(`/api/market/demand?scope=data&top=${top}`),
  searchSkills: (q: string) => call<SkillHit[]>(`/api/market/skills/search?k=6&q=${encodeURIComponent(q)}`),
};

// ---- formatting --------------------------------------------------------------------------------
export const inr = (n: number) => `₹${Math.round(n).toLocaleString("en-IN")}`;
export const lakh = (n: number) => `₹${(n / 100_000).toFixed(2)} L`;
export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;

/** "0 tokens (no model call)" / "193 tokens (163 in + 30 out)" */
export function tokenLabel(t?: TokenUsage): string {
  if (!t) return "";
  if (!t.model_calls) return "0 tokens (no model call)";
  return `${t.total_tokens} tokens (${t.input_tokens} in + ${t.output_tokens} out${t.model_calls > 1 ? `, ${t.model_calls} calls` : ""})`;
}
