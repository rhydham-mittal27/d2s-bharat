"use client";

import Link from "next/link";
import { useState } from "react";
import {
  AREA_LABEL,
  api,
  inr,
  lakh,
  type BaselineComparison,
  type Bound,
  type Confidence,
  type FullResponse,
  type Scheme,
} from "@/lib/api";
import { useStore } from "@/lib/store";
import { usePlan } from "@/lib/use-plan";
import { useConfidence } from "@/components/plan-extras";

const PRIORITIES: { id: Scheme; label: string; hint: string }[] = [
  { id: "balanced", label: "Balanced", hint: "Market demand, pay and outcomes count equally" },
  { id: "market-led", label: "Market-led", hint: "Weights the skills employers ask for most" },
  { id: "outcome-led", label: "Outcome-led", hint: "Weights the skills linked to higher salary hikes" },
];
const EVIDENCE: { id: Bound; label: string }[] = [
  { id: "low", label: "Cautious" },
  { id: "point", label: "Expected" },
  { id: "high", label: "Optimistic" },
];
const RULE_COLS = [
  { id: "cheapest_seat", label: "Cheapest" },
  { id: "biggest_gap", label: "Biggest gap" },
  { id: "most_demanded", label: "In-demand" },
];
const LETTERS = ["A", "B", "C", "D", "E"];
const card = "rounded-lg border border-line bg-surface";

export default function PlanPage() {
  const { plan, setPlan, body, cohort } = useStore();
  const [retry, setRetry] = useState(0);
  const { full, alts, loading, error } = usePlan(body, { retry });
  const conf = useConfidence(body);
  const [pick, setPick] = useState(0);
  const [adv, setAdv] = useState(false);

  const res = full?.result;
  const status =
    error && !res ? { dot: "bg-bad", text: "Backend unreachable" }
      : !res ? { dot: "bg-warn", text: "Solving…" }
        : loading ? { dot: "bg-warn", text: "Re-optimising…" }
          : { dot: "bg-accent", text: `Solved in ${Math.round(full!.timings_ms.total)} ms · proven optimal` };

  return (
    <div>
      <div className="mb-5 flex flex-col gap-1.5">
        <h1 className="text-xl font-semibold tracking-tight">What should we teach next?</h1>
        <p className="max-w-[720px] text-sm leading-relaxed text-muted">
          Set your budget and trainer capacity. The optimiser picks the courses and seats that close the most
          demand-weighted skill gap for{" "}
          <Link href="/cohort" className="text-accent hover:text-ink hover:underline">{cohort.name}</Link>.
        </p>
      </div>

      <div className="grid items-start gap-5 lg:grid-cols-[280px_minmax(0,1fr)]">
        {/* ---- controls ---- */}
        <aside className={`${card} flex flex-col gap-5 p-4 lg:sticky lg:top-4`}>
          <Slider label="Training budget" value={plan.budget} min={50_000} max={1_000_000} step={10_000}
                  format={lakh} onChange={(v) => setPlan({ ...plan, budget: v })} />
          <Slider label="Trainer-hours available" value={plan.trainer_hours} min={40} max={400} step={10}
                  format={(v) => `${v} h`} onChange={(v) => setPlan({ ...plan, trainer_hours: v })} />

          <div className="flex flex-col gap-3.5 border-t border-line pt-3">
            <button type="button" onClick={() => setAdv(!adv)} aria-expanded={adv}
                    className="flex w-full items-center justify-between text-[13px] font-medium">
              <span>Advanced</span>
              <span className="text-xs font-normal text-muted">{adv ? "Hide" : "Show"}</span>
            </button>
            {adv && (
              <>
                <div className="flex flex-col gap-2">
                  <div className="text-xs text-muted">Priority</div>
                  <div role="radiogroup" aria-label="Priority" className="flex flex-col gap-1.5">
                    {PRIORITIES.map((o) => {
                      const on = plan.scheme === o.id;
                      return (
                        <button key={o.id} type="button" role="radio" aria-checked={on}
                                onClick={() => setPlan({ ...plan, scheme: o.id })}
                                className={`flex items-start gap-2.5 rounded-lg border px-2.5 py-2 text-left ${on ? "border-accent bg-accent-soft" : "border-line"}`}>
                          <RadioDot on={on} />
                          <span className="flex flex-col gap-0.5">
                            <span className="text-[13px] font-medium">{o.label}</span>
                            <span className="text-[11px] leading-snug text-muted">{o.hint}</span>
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
                <div className="flex flex-col gap-2">
                  <div className="text-xs text-muted">Evidence reading</div>
                  <div role="radiogroup" aria-label="Evidence reading" className="flex gap-0.5 rounded-lg bg-track p-0.5">
                    {EVIDENCE.map((o) => {
                      const on = plan.bound === o.id;
                      return (
                        <button key={o.id} type="button" role="radio" aria-checked={on}
                                onClick={() => setPlan({ ...plan, bound: o.id })}
                                className={`min-w-0 flex-1 rounded-md border px-1 py-1.5 text-xs ${on ? "border-line bg-surface font-semibold text-ink" : "border-transparent text-muted"}`}>
                          {o.label}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </>
            )}
          </div>

          <div role="status" aria-live="polite" className="flex items-center gap-2 border-t border-line pt-3 text-xs text-muted">
            <span className={`h-2 w-2 rounded-full ${status.dot}`} />
            <span>{status.text}</span>
          </div>
        </aside>

        {/* ---- results ---- */}
        <section className="flex min-w-0 flex-col gap-4">
          {error && !res && (
            <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-bad bg-bad-soft p-4">
              <div className="flex min-w-0 flex-[1_1_260px] flex-col gap-1">
                <div className="text-sm font-semibold text-bad">Cannot reach the backend</div>
                <div className="text-[13px] leading-relaxed">
                  The optimiser service did not respond, so no plan was computed. Check that it is running, then retry.
                </div>
                {!error.startsWith("Cannot reach") && <div className="text-xs text-muted">{error}</div>}
              </div>
              <button type="button" onClick={() => setRetry((r) => r + 1)}
                      className="rounded-lg border border-bad bg-surface px-3.5 py-1.5 text-[13px] font-medium text-bad">
                Retry
              </button>
            </div>
          )}

          {!res && !error && (
            <div className={`${card} flex flex-col items-center gap-1.5 px-6 py-14 text-center`}>
              <div className="text-sm font-medium">Solving…</div>
              <div className="text-xs text-muted">
                Finding the proven-optimal plan for {lakh(plan.budget)} and {plan.trainer_hours} h
              </div>
            </div>
          )}

          {res && full && (
            <Results full={full} alts={alts ?? []} conf={conf.data} confLoading={conf.loading || loading}
                     pick={pick} setPick={setPick} busy={loading} />
          )}
        </section>
      </div>
    </div>
  );
}

function Results(props: {
  full: FullResponse;
  alts: NonNullable<ReturnType<typeof usePlan>["alts"]>;
  conf: Confidence | null;
  confLoading: boolean;
  pick: number;
  setPick: (i: number) => void;
  busy: boolean;
}) {
  const { full, alts, conf, confLoading, busy } = props;
  const { body } = useStore();
  const res = full.result;
  const A = res.plan;
  const options = alts;
  const sel = Math.min(props.pick, Math.max(options.length - 1, 0));
  const shown = options[sel]?.plan ?? A;
  const names = { ...res.course_names, ...(options[sel]?.course_names ?? {}) };
  const confById = Object.fromEntries((conf?.courses ?? []).map((c) => [c.course_id, c]));

  const nothingLimits = res.binding.verdict.startsWith("all valuable gaps");
  const fullyClosed = A.skills.filter((s) => s.closure_pct >= 99.5).length;
  const confTone = conf?.level === "high" ? "bg-accent-soft text-accent" : conf?.level === "medium" ? "bg-warn-soft text-warn" : "bg-bad-soft text-bad";

  return (
    <div className={`flex flex-col gap-4 transition-opacity duration-200 ${busy ? "opacity-60" : ""}`}>
      {/* verdict */}
      <div className={`${card} overflow-hidden`}>
        <div className={`flex items-start gap-2.5 border-b border-line px-4 py-3 text-[13px] leading-snug ${nothingLimits ? "bg-accent-soft text-accent" : "bg-warn-soft text-warn"}`}>
          <span className={`mt-[5px] h-2 w-2 flex-none rounded-[2px] ${nothingLimits ? "bg-accent" : "bg-warn"}`} />
          <span>
            <strong className="font-semibold">{nothingLimits ? "Nothing limits this plan:" : "What limits this plan:"}</strong>{" "}
            {res.binding.verdict}.
          </span>
        </div>
        <div className="flex flex-col gap-[18px] p-4">
          {confLoading || !conf ? (
            <div className="flex min-h-[25px] items-center gap-2 text-xs text-muted">
              <span className="h-2 w-2 rounded-full border-[1.5px] border-muted" />
              Checking how robust this plan is…
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
              <span className={`whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-semibold ${confTone}`}>
                {conf.level[0].toUpperCase() + conf.level.slice(1)} confidence · {Math.round(conf.score * 100)}%
              </span>
              <span className="flex-[1_1_260px] text-xs leading-snug text-muted">{conf.explanation}</span>
            </div>
          )}
          <div className="grid grid-cols-[repeat(auto-fit,minmax(130px,1fr))] gap-y-3.5">
            <Figure label="Spend" value={lakh(A.total_cost)} sub={`of ${lakh(res.request.budget)}`} />
            <Figure label="Trainer-hours" value={`${A.total_trainer_hours} h`} sub={`of ${res.request.trainer_hours} h`} />
            <Figure label="Seats" value={String(A.total_seats)} sub={`in ${A.courses.length} courses`} />
            <Figure label="Gaps fully closed" value={String(fullyClosed)} sub={`of ${A.skills.length} skills`} />
          </div>
        </div>
      </div>

      {/* plan A / B / C */}
      {options.length > 0 && (
        <div role="tablist" aria-label="Alternative plans" className="grid grid-cols-[repeat(auto-fit,minmax(210px,1fr))] gap-3">
          {options.map((a, i) => {
            const on = i === sel;
            return (
              <button key={a.label} type="button" role="tab" aria-selected={on} onClick={() => props.setPick(i)}
                      className={`flex flex-col gap-2.5 rounded-lg border p-3.5 text-left transition-colors ${on ? "border-accent bg-accent-soft shadow-[inset_0_0_0_1px_var(--accent)]" : "border-line bg-surface hover:border-muted"}`}>
                <span className="flex items-center justify-between gap-2">
                  <span className="flex items-center gap-2 text-[13px] font-semibold">
                    <RadioDot on={on} />Plan {LETTERS[i]}
                  </span>
                  {i === 0 && <span className="text-[11px] font-medium text-accent">Recommended</span>}
                </span>
                <span className="flex items-baseline gap-1.5">
                  <span className="num text-2xl font-semibold leading-tight tracking-tight">{a.impact_pct_of_best.toFixed(1)}%</span>
                  <span className="text-xs text-muted">impact</span>
                </span>
                <span className="num text-xs font-medium">
                  {lakh(a.plan.total_cost)} · {a.plan.total_trainer_hours} h · {a.plan.total_seats} seats
                </span>
                <span className="text-xs leading-snug text-muted">
                  {i === 0 ? "Highest demand-weighted impact under every constraint (proven optimal)." : a.tradeoff}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {/* courses + gaps */}
      <div className="grid grid-cols-[repeat(auto-fit,minmax(300px,1fr))] items-start gap-4">
        <div className={`${card} overflow-hidden`}>
          <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-line px-4 py-3.5">
            <span className="text-sm font-semibold">Courses to run</span>
            <span className="text-xs text-muted">
              Plan {LETTERS[sel]} · {shown.courses.length} courses, {shown.total_seats} seats
            </span>
          </div>
          <Row className="border-b border-line py-2 text-[11px] text-muted">
            <span>Course</span><span className="text-right">Seats</span><span className="text-right">Cost</span><span className="text-right">Hours</span>
          </Row>
          {shown.courses.map((c) => {
            const cc = confById[c.course_id];
            const tone = cc?.level === "high" ? "bg-accent-soft text-accent" : cc?.level === "medium" ? "bg-warn-soft text-warn" : "bg-bad-soft text-bad";
            return (
              <Row key={c.course_id} className="border-b border-track py-2.5 text-[13px]">
                <span className="flex min-w-0 flex-wrap items-center gap-x-1.5 gap-y-1 leading-snug">
                  <span>{names[c.course_id] ?? c.course_id}</span>
                  {cc && (
                    <span title={`${cc.level} confidence: chosen under ${cc.chosen_in} of ${cc.scenarios} evidence readings`}
                          className={`rounded-full px-1.5 py-px text-[11px] font-semibold ${tone}`}>
                      {cc.chosen_in}/{cc.scenarios}
                    </span>
                  )}
                </span>
                <span className="num text-right">{c.seats}</span>
                <span className="num text-right">{inr(c.cost)}</span>
                <span className="num text-right">{c.trainer_hours}</span>
              </Row>
            );
          })}
          <Row className="py-2.5 text-[13px] font-semibold">
            <span>Total</span>
            <span className="num text-right">{shown.total_seats}</span>
            <span className="num text-right">{inr(shown.total_cost)}</span>
            <span className="num text-right">{shown.total_trainer_hours}</span>
          </Row>
        </div>

        <div className={`${card} flex flex-col gap-3.5 px-4 py-3.5`}>
          <div className="flex flex-col gap-0.5">
            <span className="text-sm font-semibold">Skill gaps closed</span>
            <span className="text-xs leading-snug text-muted">Share of learners below target that this plan brings up to target</span>
          </div>
          {[...shown.skills].sort((a, b) => b.closure_pct - a.closure_pct).map((s) => (
            <div key={s.skill_id} className="flex flex-col gap-1.5">
              <div className="flex justify-between gap-2 text-[13px]">
                <span>{AREA_LABEL[s.skill_id] ?? s.skill_id}</span>
                <span className="num whitespace-nowrap text-xs text-muted">
                  {Math.round(s.learners_closed)} of {s.learners_short} · {Math.round(s.closure_pct)}%
                </span>
              </div>
              <Track pct={s.closure_pct} color={s.closure_pct >= 50 ? "bg-accent" : "bg-warn"} />
            </div>
          ))}
        </div>
      </div>

      {full.baselines && <Rules cmp={full.baselines} budget={res.request.budget} bodyKey={JSON.stringify(body)} />}

      {sel === 0 && res.why_not.length > 0 && (
        <div className={`${card} overflow-hidden`}>
          <div className="border-b border-line px-4 py-3.5 text-sm font-semibold">Why these courses were left out</div>
          {res.why_not.map((w) => (
            <div key={w.course_id} className="flex flex-wrap justify-between gap-x-4 gap-y-1 border-b border-track px-4 py-2.5 text-[13px] last:border-0">
              <span className="font-medium">{w.name}</span>
              <span className="text-muted">
                {!w.feasible
                  ? "Does not fit the budget and trainer-hours (with its prerequisites)"
                  : w.objective_delta_pct != null && Math.abs(w.objective_delta_pct) >= 0.05
                    ? `Including it lowers impact by ${Math.abs(w.objective_delta_pct).toFixed(1)}%`
                    : "Including it adds no extra impact"}
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="flex justify-end">
        <Link href="/brief"
              className="inline-flex min-h-11 w-full items-center justify-center gap-1.5 rounded-lg bg-accent px-[18px] py-2.5 text-sm font-medium text-on-accent sm:w-auto">
          Open decision brief →
        </Link>
      </div>
    </div>
  );
}

function Rules({ cmp, budget, bodyKey }: { cmp: BaselineComparison; budget: number; bodyKey: string }) {
  const { body } = useStore();
  const [open, setOpen] = useState(false);
  const [computed, setComputed] = useState<{ key: string; data: BaselineComparison } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sweep = computed?.key === bodyKey ? computed.data : null;
  const visible = open && sweep !== null;

  async function toggle() {
    if (visible) return setOpen(false);
    setOpen(true);
    if (sweep) return;
    setBusy(true);
    setError(null);
    try {
      setComputed({ key: bodyKey, data: await api.baselines(body) });
    } catch (e) {
      setError((e as Error).message);
      setOpen(false);
    } finally {
      setBusy(false);
    }
  }

  const rows = [
    { id: "opt", name: "D2S optimiser", pct: 100, sub: `${lakh(cmp.optimiser.cost)} · ${cmp.optimiser.trainer_hours} h`, strong: true },
    ...cmp.baselines.slice().sort((a, b) => b.impact_pct_of_optimiser - a.impact_pct_of_optimiser)
      .map((b) => ({ id: b.id, name: b.name, pct: b.impact_pct_of_optimiser, sub: b.rule, strong: false })),
  ];
  const summary = cmp.uplift_vs_average_pct < 0.05
    ? "At these settings, the simple rules happen to match the optimiser."
    : `At these settings, the optimiser closes ${cmp.uplift_vs_average_pct.toFixed(1)}% more demand-weighted skill gap than a simple rule on average.`;
  const red = (v: number) => (v < 80 ? "text-bad" : "");

  return (
    <div className={`${card} flex flex-col gap-3.5 px-4 py-3.5`}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-sm font-semibold">Compared with simple rules</span>
        <span className="text-xs text-muted">Demand-weighted impact, same budget and hours</span>
      </div>
      {rows.map((r) => (
        <div key={r.id} className="flex flex-col gap-1.5">
          <div className="flex justify-between gap-2 text-[13px]">
            <span className={r.strong ? "font-semibold" : ""}>{r.name}</span>
            <span className="num font-semibold">{r.pct === 100 ? "100%" : `${r.pct.toFixed(1)}%`}</span>
          </div>
          <Track pct={r.pct} color={r.strong ? "bg-accent" : "bg-grey"} />
          <span className="text-xs text-muted">{r.sub}</span>
        </div>
      ))}
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2.5 border-t border-line pt-3">
        <span className="flex-[1_1_300px] text-[13px] leading-relaxed">{summary}</span>
        <button type="button" onClick={toggle} disabled={busy}
                className="flex-none whitespace-nowrap rounded-lg border border-line bg-surface px-3 py-1.5 text-[13px] font-medium disabled:opacity-60">
          {busy ? "Comparing…" : visible ? "Hide budget comparison" : "Compare across budgets"}
        </button>
      </div>
      {error && <p className="text-xs text-bad">{error}</p>}
      {visible && sweep?.sweep && sweep.worst_case_pct && (
        <div className="overflow-hidden rounded-lg border border-line text-xs">
          <SweepRow className="border-b border-line bg-bg py-2 text-muted">
            <span>Budget</span>
            {RULE_COLS.map((c) => <span key={c.id} className="text-right">{c.label}</span>)}
            <span className="text-right">Optimiser</span>
          </SweepRow>
          {sweep.sweep.map((r) => {
            const cur = r.budget === budget;
            return (
              <SweepRow key={r.budget} className={`num border-b border-track py-[7px] ${cur ? "bg-accent-soft font-semibold" : ""}`}>
                <span>{lakh(r.budget)}</span>
                {RULE_COLS.map((c) => <span key={c.id} className={`text-right ${red(r.rules[c.id])}`}>{r.rules[c.id].toFixed(1)}%</span>)}
                <span className="text-right text-accent">100%</span>
              </SweepRow>
            );
          })}
          <SweepRow className="num border-t border-line py-2 font-semibold">
            <span>Worst case</span>
            {RULE_COLS.map((c) => (
              <span key={c.id} className={`text-right ${red(sweep.worst_case_pct![c.id])}`}>{sweep.worst_case_pct![c.id].toFixed(1)}%</span>
            ))}
            <span className="text-right text-accent">100%</span>
          </SweepRow>
        </div>
      )}
    </div>
  );
}

// ---- small pieces -------------------------------------------------------------------------------------
function Slider(props: { label: string; value: number; min: number; max: number; step: number;
                         format: (v: number) => string; onChange: (v: number) => void }) {
  const id = props.label.replace(/\W+/g, "-").toLowerCase();
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-[13px] font-medium">{props.label}</label>
        <span className="num text-lg font-semibold">{props.format(props.value)}</span>
      </div>
      <input id={id} type="range" min={props.min} max={props.max} step={props.step} value={props.value}
             onChange={(e) => props.onChange(Number(e.target.value))} className="m-0 w-full cursor-pointer" />
      <div className="num flex justify-between text-[11px] text-muted">
        <span>{props.format(props.min)}</span><span>{props.format(props.max)}</span>
      </div>
    </div>
  );
}

function Figure({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="flex flex-col gap-1 border-l border-line px-3.5">
      <span className="text-xs text-muted">{label}</span>
      <span className="flex flex-wrap items-baseline gap-1.5">
        <span className="num text-2xl font-semibold leading-tight tracking-tight">{value}</span>
        <span className="text-xs text-muted">{sub}</span>
      </span>
    </div>
  );
}

function RadioDot({ on }: { on: boolean }) {
  return (
    <span className={`mt-0.5 flex h-3.5 w-3.5 flex-none items-center justify-center rounded-full border-[1.5px] ${on ? "border-accent" : "border-muted"}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${on ? "bg-accent" : "bg-transparent"}`} />
    </span>
  );
}

function Track({ pct, color }: { pct: number; color: string }) {
  return (
    <div className="h-2 overflow-hidden rounded bg-track">
      <div className={`h-2 rounded ${color} transition-[width] duration-300`} style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
    </div>
  );
}

function Row({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`grid grid-cols-[minmax(0,1fr)_44px_74px_44px] items-center gap-2.5 px-4 ${className}`}>{children}</div>;
}

function SweepRow({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`grid grid-cols-[minmax(52px,0.8fr)_repeat(4,minmax(0,1fr))] gap-2 px-3 ${className}`}>{children}</div>;
}
