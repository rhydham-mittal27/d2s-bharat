"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { AREA_LABEL, AREAS, api, inr, lakh, type Preset, type Scenario, type WhatIfResult } from "@/lib/api";
import { useStore } from "@/lib/store";
import { Card, ErrorBox, H2, Muted } from "@/components/ui";
import { StressTest } from "@/components/stress-test";

const EMPTY: Required<Scenario> = {
  budget_pct: 0,
  trainer_hours_delta: 0,
  cost_pct: 0,
  cohort_pct: 0,
  demand_pct: {},
  remove_courses: [],
  require_courses: [],
};

export default function WhatIfPage() {
  const { body, plan, cohort } = useStore();
  const [presets, setPresets] = useState<Preset[]>([]);
  const [courses, setCourses] = useState<{ id: string; name: string }[]>([]);
  const [sc, setSc] = useState<Required<Scenario>>(EMPTY);
  const [active, setActive] = useState<string | null>(null);
  const [result, setResult] = useState<WhatIfResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.whatIfPresets().then(setPresets).catch((e) => setError(e.message));
    api.catalogue().then((c) => setCourses(c.map((x) => ({ id: x.id, name: x.name })))).catch(() => {});
  }, []);

  async function run(s: Required<Scenario>, presetId: string | null = null) {
    setSc(s);
    setActive(presetId);
    setBusy(true);
    setError(null);
    try {
      setResult(await api.whatIf(body, s));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const set = (patch: Partial<Scenario>) => {
    setSc({ ...sc, ...patch } as Required<Scenario>);
    setActive(null);
  };
  const toggle = (list: "remove_courses" | "require_courses", id: string) => {
    const other = list === "remove_courses" ? "require_courses" : "remove_courses";
    const on = sc[list].includes(id);
    set({ [list]: on ? sc[list].filter((x) => x !== id) : [...sc[list], id], [other]: sc[other].filter((x) => x !== id) });
  };

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">What happens if…</h1>
        <Muted>
          Change the conditions and see how the best plan responds. Starting point: {cohort.name}, {lakh(plan.budget)},{" "}
          {plan.trainer_hours} trainer-hours (<Link href="/" className="underline">change on the Plan page</Link>).
        </Muted>
      </div>

      <StressTest />

      <div className="flex flex-wrap gap-2">
        {presets.map((p) => (
          <button
            key={p.id}
            type="button"
            disabled={busy}
            onClick={() => run({ ...EMPTY, ...p.scenario } as Required<Scenario>, p.id)}
            className={`rounded-full border px-3 py-1.5 text-sm transition-colors disabled:opacity-60 ${active === p.id ? "border-accent bg-accent-soft text-accent" : "border-line bg-surface hover:border-muted"}`}
          >
            {p.title}
          </button>
        ))}
      </div>

      <div className="grid gap-5 lg:grid-cols-[300px_1fr]">
        <Card className="h-fit space-y-4">
          <H2 hint="Combine any changes, then run">Build your own</H2>
          <Pct label="Training budget" value={sc.budget_pct} min={-80} max={200} step={5} onChange={(v) => set({ budget_pct: v })} />
          <Num label="Trainer-hours" value={sc.trainer_hours_delta} min={-150} max={300} step={10} unit="h"
               onChange={(v) => set({ trainer_hours_delta: v })} />
          <Pct label="Course prices" value={sc.cost_pct} min={-50} max={100} step={5} onChange={(v) => set({ cost_pct: v })} />
          <Pct label="Learners short (next batch)" value={sc.cohort_pct} min={-50} max={200} step={10}
               onChange={(v) => set({ cohort_pct: v })} />
          <details className="text-sm">
            <summary className="cursor-pointer text-muted hover:text-ink">Market demand per skill area</summary>
            <div className="mt-3 space-y-3">
              {AREAS.map((a) => (
                <Pct key={a} label={AREA_LABEL[a]} value={sc.demand_pct[a] ?? 0} min={-80} max={200} step={10}
                     onChange={(v) => set({ demand_pct: { ...sc.demand_pct, [a]: v } })} />
              ))}
            </div>
          </details>
          <details className="text-sm">
            <summary className="cursor-pointer text-muted hover:text-ink">Courses: drop or require</summary>
            <ul className="mt-3 space-y-1.5">
              {courses.map((c) => (
                <li key={c.id} className="flex items-center justify-between gap-2 text-xs">
                  <span className="min-w-0 truncate" title={c.name}>{c.name}</span>
                  <span className="flex shrink-0 gap-1">
                    <Toggle on={sc.remove_courses.includes(c.id)} onClick={() => toggle("remove_courses", c.id)} tone="bad">drop</Toggle>
                    <Toggle on={sc.require_courses.includes(c.id)} onClick={() => toggle("require_courses", c.id)} tone="accent">require</Toggle>
                  </span>
                </li>
              ))}
            </ul>
          </details>
          <div className="flex gap-2">
            <button type="button" disabled={busy} onClick={() => run(sc)}
                    className="flex-1 rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:text-black">
              {busy ? "Simulating…" : "Run simulation"}
            </button>
            <button type="button" onClick={() => { setSc(EMPTY); setActive(null); setResult(null); }}
                    className="rounded-md border border-line px-3 py-2 text-sm">Reset</button>
          </div>
        </Card>

        <div className={`space-y-5 transition-opacity ${busy && result ? "opacity-60" : ""}`}>
          {error && <ErrorBox message={error} />}
          {!result && !error && (
            <Card><Muted>Pick a scenario above, or build your own and press Run simulation.</Muted></Card>
          )}
          {result && <ResultView r={result} />}
        </div>
      </div>
    </div>
  );
}

function ResultView({ r }: { r: WhatIfResult }) {
  const name = (id: string) => r.course_names[id] ?? id;
  const ids = [...new Set([...Object.keys(r.today.courses), ...Object.keys(r.after.courses)])];
  return (
    <>
      <Card className="space-y-3">
        <h2 className="text-lg font-semibold">{r.title}</h2>
        <ul className="space-y-1.5 text-sm">
          {r.narrative.map((n, i) => (
            <li key={i} className="flex gap-2"><span className="text-accent">•</span><span>{n}</span></li>
          ))}
        </ul>
        {r.conflicts.length > 0 && (
          <div className="rounded-md bg-bad/10 px-3 py-2 text-sm text-bad">
            Conflicting requirements: {r.conflicts.join("; ")}
          </div>
        )}
      </Card>

      {r.after.feasible && (
        <>
          <Card>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Compare label="Spend" a={lakh(r.today.cost)} b={lakh(r.after.cost)} />
              <Compare label="Trainer-hours" a={`${r.today.trainer_hours} h`} b={`${r.after.trainer_hours} h`} />
              <Compare label="Seats" a={String(r.today.seats)} b={String(r.after.seats)} />
              <div>
                <div className="text-xs text-muted">{r.impact_change_pct != null ? "Impact change" : "Gain from re-planning"}</div>
                <div className="num text-lg font-semibold">
                  {r.impact_change_pct != null
                    ? `${r.impact_change_pct > 0 ? "+" : ""}${r.impact_change_pct}%`
                    : r.replanning_gain_pct != null ? `+${r.replanning_gain_pct}%` : "re-plan needed"}
                </div>
                <div className="text-xs text-muted">
                  {r.impact_change_pct != null ? "demand-weighted gap closed" : "vs keeping today's plan"}
                </div>
              </div>
            </div>
          </Card>

          <div className="grid gap-5 md:grid-cols-2">
            <Card>
              <H2>Courses: today → after</H2>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-muted">
                    <th className="pb-1 font-normal">Course</th>
                    <th className="pb-1 pl-3 text-right font-normal">Today</th>
                    <th className="pb-1 pl-3 text-right font-normal">After</th>
                  </tr>
                </thead>
                <tbody className="num">
                  {ids.map((id) => {
                    const t = r.today.courses[id];
                    const a = r.after.courses[id];
                    const tone = !t ? "text-accent font-medium" : !a ? "text-bad line-through" : "";
                    return (
                      <tr key={id} className="border-t border-line">
                        <td className={`py-1.5 pr-2 ${tone}`}>{name(id)}</td>
                        <td className="py-1.5 pl-3 text-right text-muted">{t ?? "–"}</td>
                        <td className={`py-1.5 pl-3 text-right ${a !== t ? "font-medium" : ""}`}>{a ?? "–"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="mt-2 text-[11px] text-muted">
                Seats per course. <span className="text-accent">Green</span>: added, <span className="text-bad">struck</span>: dropped.
                Today&apos;s cost {inr(r.today.cost)}.
              </p>
            </Card>

            <Card>
              <H2 hint="Share of each gap closed: today (grey) and after (colour)">Skill gaps</H2>
              <ul className="space-y-3">
                {Object.keys(r.after.closure_pct).map((a) => {
                  const t = r.today.closure_pct[a] ?? 0;
                  const v = r.after.closure_pct[a];
                  return (
                    <li key={a}>
                      <div className="mb-1 flex justify-between text-sm">
                        <span>{AREA_LABEL[a] ?? a}</span>
                        <span className="num text-muted">{Math.round(t)}% → <span className="text-ink">{Math.round(v)}%</span></span>
                      </div>
                      <div className="relative h-2 w-full overflow-hidden rounded-full bg-track">
                        <div className="absolute inset-y-0 left-0 rounded-full bg-muted/40" style={{ width: `${t}%` }} />
                        <div className={`absolute inset-y-0 left-0 rounded-full ${v >= t ? "bg-accent" : "bg-warn"} transition-[width] duration-300`}
                             style={{ width: `${v}%`, opacity: 0.85 }} />
                      </div>
                    </li>
                  );
                })}
              </ul>
            </Card>
          </div>
        </>
      )}
    </>
  );
}

function Compare({ label, a, b }: { label: string; a: string; b: string }) {
  return (
    <div>
      <div className="text-xs text-muted">{label}</div>
      <div className="num text-lg font-semibold">{b}</div>
      <div className="num text-xs text-muted">{a === b ? "unchanged" : `was ${a}`}</div>
    </div>
  );
}

function Pct(props: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return <Num {...props} unit="%" />;
}

function Num(props: { label: string; value: number; min: number; max: number; step: number; unit: string; onChange: (v: number) => void }) {
  const id = `wi-${props.label.replace(/\W+/g, "-").toLowerCase()}`;
  const shown = `${props.value > 0 ? "+" : ""}${props.value}${props.unit === "%" ? "%" : ` ${props.unit}`}`;
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-sm">
        <label htmlFor={id}>{props.label}</label>
        <span className={`num ${props.value === 0 ? "text-muted" : "font-medium"}`}>{props.value === 0 ? "no change" : shown}</span>
      </div>
      <input id={id} type="range" className="w-full" min={props.min} max={props.max} step={props.step} value={props.value}
             onChange={(e) => props.onChange(Number(e.target.value))} />
    </div>
  );
}

function Toggle({ on, onClick, tone, children }: { on: boolean; onClick: () => void; tone: "bad" | "accent"; children: string }) {
  const onCls = tone === "bad" ? "border-bad bg-bad/10 text-bad" : "border-accent bg-accent-soft text-accent";
  return (
    <button type="button" aria-pressed={on} onClick={onClick}
            className={`rounded border px-1.5 py-0.5 ${on ? onCls : "border-line text-muted hover:text-ink"}`}>
      {children}
    </button>
  );
}
