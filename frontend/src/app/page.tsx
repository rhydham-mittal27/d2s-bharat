"use client";

import Link from "next/link";
import { useState } from "react";
import { AREA_LABEL, inr, lakh, type Alternative, type Bound, type PlanResult, type Scheme } from "@/lib/api";
import { useStore } from "@/lib/store";
import { usePlan } from "@/lib/use-plan";
import { Bar, Card, ErrorBox, H2, Muted, Stat } from "@/components/ui";
import {
  BaselinePanel,
  ConfidenceSummary,
  CourseConfidenceTag,
  useConfidence,
} from "@/components/plan-extras";

const SCHEMES: { id: Scheme; label: string; hint: string }[] = [
  { id: "balanced", label: "Balanced", hint: "market demand, pay and learner outcomes equally" },
  { id: "market-led", label: "Market-led", hint: "weight what employers ask for most" },
  { id: "outcome-led", label: "Outcome-led", hint: "weight what predicts higher salary hikes" },
];
const BOUNDS: { id: Bound; label: string }[] = [
  { id: "low", label: "Cautious" },
  { id: "point", label: "Expected" },
  { id: "high", label: "Optimistic" },
];

export default function PlanPage() {
  const { plan, setPlan, body, cohort } = useStore();
  const { full, alts, loading, error } = usePlan(body);
  const conf = useConfidence(body);
  const confById = Object.fromEntries((conf.data?.courses ?? []).map((c) => [c.course_id, c]));
  const [pick, setPick] = useState(0);

  const res = full?.result;
  const options = alts ?? [];
  const selected: Alternative | undefined = options[Math.min(pick, options.length - 1)];
  const shown = selected?.plan ?? res?.plan;
  const names = { ...(res?.course_names ?? {}), ...(selected?.course_names ?? {}) };

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">What should we teach next?</h1>
        <Muted>
          Set your budget and trainer capacity. The optimiser picks the courses and seats that close the most
          demand-weighted skill gap for <Link className="underline" href="/cohort">{cohort.name}</Link>.
        </Muted>
      </div>

      <div className="grid gap-5 lg:grid-cols-[280px_1fr]">
        {/* ---- controls ---- */}
        <Card className="h-fit space-y-5 lg:sticky lg:top-4">
          <Slider
            label="Training budget"
            value={plan.budget}
            min={50_000}
            max={1_000_000}
            step={10_000}
            format={lakh}
            onChange={(v) => setPlan({ ...plan, budget: v })}
          />
          <Slider
            label="Trainer-hours available"
            value={plan.trainer_hours}
            min={40}
            max={400}
            step={10}
            format={(v) => `${v} h`}
            onChange={(v) => setPlan({ ...plan, trainer_hours: v })}
          />
          <details className="text-sm">
            <summary className="cursor-pointer text-muted hover:text-ink">Advanced</summary>
            <fieldset className="mt-3 space-y-1.5">
              <legend className="mb-1 text-xs text-muted">Priority</legend>
              {SCHEMES.map((s) => (
                <label key={s.id} className="flex items-start gap-2">
                  <input
                    type="radio"
                    name="scheme"
                    className="mt-1 accent-[var(--accent)]"
                    checked={plan.scheme === s.id}
                    onChange={() => setPlan({ ...plan, scheme: s.id })}
                  />
                  <span>
                    {s.label}
                    <span className="block text-xs text-muted">{s.hint}</span>
                  </span>
                </label>
              ))}
            </fieldset>
            <fieldset className="mt-3">
              <legend className="mb-1 text-xs text-muted">Evidence reading</legend>
              <div className="flex rounded-md border border-line p-0.5">
                {BOUNDS.map((b) => (
                  <button
                    key={b.id}
                    type="button"
                    onClick={() => setPlan({ ...plan, bound: b.id })}
                    className={`flex-1 rounded px-2 py-1 text-xs ${plan.bound === b.id ? "bg-accent-soft text-accent font-medium" : "text-muted"}`}
                  >
                    {b.label}
                  </button>
                ))}
              </div>
            </fieldset>
          </details>
          <p className="text-xs text-muted" aria-live="polite">
            {loading ? "Re-optimising…" : res ? `Solved in ${Math.round(full!.timings_ms.total)} ms · proven optimal` : ""}
          </p>
        </Card>

        {/* ---- results ---- */}
        <div className={`space-y-5 transition-opacity ${loading && res ? "opacity-60" : ""}`}>
          {error && <ErrorBox message={error} />}
          {!res && !error && <Card><Muted>Solving…</Muted></Card>}
          {res && shown && (
            <>
              <Verdict res={res} confidence={<ConfidenceSummary conf={conf.data} loading={conf.loading} />} />

              {options.length > 0 && (
                <div className="grid gap-3 sm:grid-cols-3" role="tablist" aria-label="Alternative plans">
                  {options.map((a, i) => (
                    <button
                      key={a.label}
                      role="tab"
                      aria-selected={i === pick}
                      onClick={() => setPick(i)}
                      className={`rounded-lg border p-3 text-left transition-colors ${i === pick ? "border-accent bg-accent-soft" : "border-line bg-surface hover:border-muted"}`}
                    >
                      <div className="flex items-baseline justify-between">
                        <span className="font-semibold">{a.label}</span>
                        <span className="num text-sm">{a.impact_pct_of_best.toFixed(1)}% impact</span>
                      </div>
                      <div className="num mt-1 text-sm">
                        {lakh(a.plan.total_cost)} · {a.plan.total_trainer_hours} h · {a.plan.total_seats} seats
                      </div>
                      <p className="mt-2 text-xs text-muted">{a.tradeoff}</p>
                    </button>
                  ))}
                </div>
              )}

              <div className="grid gap-5 md:grid-cols-2">
                <Card>
                  <H2 hint={`${shown.courses.length} courses, ${shown.total_seats} seats`}>Courses to run</H2>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-xs text-muted">
                        <th className="pb-1 font-normal">Course</th>
                        <th className="pb-1 pl-3 text-right font-normal">Seats</th>
                        <th className="pb-1 pl-3 text-right font-normal">Cost</th>
                        <th className="pb-1 pl-3 text-right font-normal">Hours</th>
                      </tr>
                    </thead>
                    <tbody className="num">
                      {shown.courses.map((c) => (
                        <tr key={c.course_id} className="border-t border-line">
                          <td className="py-1.5 pr-2">
                            {names[c.course_id] ?? c.course_id}{" "}
                            {pick === 0 && <CourseConfidenceTag c={confById[c.course_id]} />}
                          </td>
                          <td className="py-1.5 pl-3 text-right">{c.seats}</td>
                          <td className="py-1.5 pl-3 text-right">{inr(c.cost)}</td>
                          <td className="py-1.5 pl-3 text-right">{c.trainer_hours}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Card>

                <Card>
                  <H2 hint="Share of learners below target that this plan brings up to target">Skill gaps closed</H2>
                  <ul className="space-y-3">
                    {[...shown.skills]
                      .sort((a, b) => b.closure_pct - a.closure_pct)
                      .map((s) => (
                        <li key={s.skill_id}>
                          <div className="mb-1 flex justify-between text-sm">
                            <span>{AREA_LABEL[s.skill_id] ?? s.skill_id}</span>
                            <span className="num text-muted">
                              {Math.round(s.learners_closed)} of {s.learners_short} · {Math.round(s.closure_pct)}%
                            </span>
                          </div>
                          <Bar value={s.closure_pct / 100} tone={s.closure_pct < 50 ? "warn" : "accent"} />
                        </li>
                      ))}
                  </ul>
                </Card>
              </div>

              {full?.baselines && <BaselinePanel cmp={full.baselines} body={body} />}

              {pick === 0 && res.why_not.length > 0 && (
                <Card>
                  <H2 hint="Each was tested by forcing it into the plan and re-solving">Why these courses were left out</H2>
                  <ul className="divide-y divide-line text-sm">
                    {res.why_not.map((w) => (
                      <li key={w.course_id} className="flex flex-wrap items-baseline justify-between gap-2 py-2">
                        <span>{w.name}</span>
                        <span className="text-xs text-muted">
                          {!w.feasible
                            ? "does not fit the budget and trainer-hours (with its prerequisites)"
                            : w.objective_delta_pct != null
                              ? `including it lowers impact by ${Math.abs(w.objective_delta_pct).toFixed(1)}%`
                              : w.reason}
                        </span>
                      </li>
                    ))}
                  </ul>
                </Card>
              )}

              <div className="flex justify-end">
                <Link href="/brief" className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white dark:text-black">
                  Open decision brief →
                </Link>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Verdict({ res, confidence }: { res: PlanResult; confidence: React.ReactNode }) {
  const p = res.plan;
  const req = res.request;
  return (
    <Card className="space-y-4">
      <div className="rounded-md bg-warn-soft px-3 py-2 text-sm">
        <span className="font-medium text-warn">What limits this plan: </span>
        {res.binding.verdict}.
      </div>
      {confidence}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Spend" value={lakh(p.total_cost)} sub={`of ${lakh(req.budget)}`} />
        <Stat label="Trainer-hours" value={`${p.total_trainer_hours} h`} sub={`of ${req.trainer_hours} h`} />
        <Stat label="Seats" value={p.total_seats} />
        <Stat
          label="Gaps fully closed"
          value={`${p.skills.filter((s) => s.closure_pct >= 99.5).length} of ${p.skills.length}`}
        />
      </div>
    </Card>
  );
}

function Slider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
}) {
  const id = props.label.replace(/\W+/g, "-").toLowerCase();
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <label htmlFor={id} className="text-sm font-medium">{props.label}</label>
        <span className="num text-sm">{props.format(props.value)}</span>
      </div>
      <input
        id={id}
        type="range"
        className="w-full"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
      <div className="num flex justify-between text-[11px] text-muted">
        <span>{props.format(props.min)}</span>
        <span>{props.format(props.max)}</span>
      </div>
    </div>
  );
}
