"use client";

import { useEffect, useState } from "react";
import {
  api,
  lakh,
  type BaselineComparison,
  type Confidence,
  type CourseConfidence,
  type PlanBody,
} from "@/lib/api";
import { Bar, Card, ErrorBox, H2 } from "./ui";

// ---- confidence ----------------------------------------------------------------------------------
const confCache = new Map<string, Confidence>();

/** Confidence for the current settings: 9 re-optimisations, so fetched separately and debounced. */
export function useConfidence(body: PlanBody): { data: Confidence | null; loading: boolean } {
  const key = JSON.stringify(body);
  const [state, setState] = useState<{ key: string; data: Confidence | null }>({
    key,
    data: confCache.get(key) ?? null,
  });

  useEffect(() => {
    if (confCache.has(key)) return;
    const ctrl = new AbortController();
    const t = setTimeout(() => {
      api
        .confidence(JSON.parse(key) as PlanBody, ctrl.signal)
        .then((c) => {
          confCache.set(key, c);
          setState({ key, data: c });
        })
        .catch(() => {}); // confidence is supplementary; the plan itself reports errors
    }, 600);
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  }, [key]);

  const data = confCache.get(key) ?? (state.key === key ? state.data : null);
  return { data, loading: !data };
}

const TONE = {
  high: "bg-accent-soft text-accent",
  medium: "bg-warn-soft text-warn",
  low: "bg-bad/10 text-bad",
} as const;

export function ConfidenceBadge({ level, label }: { level: "high" | "medium" | "low"; label?: string }) {
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ${TONE[level]}`}>
      {label ?? `${level[0].toUpperCase()}${level.slice(1)} confidence`}
    </span>
  );
}

export function CourseConfidenceTag({ c }: { c: CourseConfidence | undefined }) {
  if (!c) return null;
  return (
    <span title={`Chosen in ${c.chosen_in} of ${c.scenarios} ways of reading the evidence`}>
      <ConfidenceBadge level={c.level} label={`${c.chosen_in}/${c.scenarios}`} />
    </span>
  );
}

export function ConfidenceSummary({ conf, loading }: { conf: Confidence | null; loading: boolean }) {
  if (loading || !conf) return <p className="text-xs text-muted">Checking how robust this plan is…</p>;
  return (
    <div className="flex flex-wrap items-start gap-2 text-sm">
      <ConfidenceBadge level={conf.level} label={`${conf.level[0].toUpperCase()}${conf.level.slice(1)} confidence · ${Math.round(conf.score * 100)}%`} />
      <span className="text-xs text-muted">{conf.explanation}</span>
    </div>
  );
}

// ---- naive baselines ------------------------------------------------------------------------------
export function BaselinePanel({ cmp, body }: { cmp: BaselineComparison; body: PlanBody }) {
  // the sweep belongs to the settings it was computed for; once they change it is simply not shown
  const bodyKey = JSON.stringify(body);
  const [computed, setComputed] = useState<{ key: string; data: BaselineComparison } | null>(null);
  const sweep = computed?.key === bodyKey ? computed.data : null;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadSweep() {
    setBusy(true);
    setError(null);
    try {
      setComputed({ key: bodyKey, data: await api.baselines(body) });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const rows = [
    { id: "opt", name: "D2S optimiser", pct: 100, note: `${lakh(cmp.optimiser.cost)} · ${cmp.optimiser.trainer_hours} h`, strong: true },
    ...cmp.baselines
      .slice()
      .sort((a, b) => b.impact_pct_of_optimiser - a.impact_pct_of_optimiser)
      .map((b) => ({ id: b.id, name: b.name, pct: b.impact_pct_of_optimiser, note: b.rule, strong: false })),
  ];
  return (
    <Card>
      <H2 hint="Same budget, trainer-hours and catalogue. Bars: demand-weighted skill gap closed, as % of the optimiser.">
        Compared with simple rules
      </H2>
      <ul className="space-y-3">
        {rows.map((r) => (
          <li key={r.id}>
            <div className="mb-1 flex flex-wrap justify-between gap-2 text-sm">
              <span className={r.strong ? "font-medium" : ""}>{r.name}</span>
              <span className="num text-muted">{r.pct.toFixed(1)}%</span>
            </div>
            <Bar value={r.pct / 100} tone={r.strong ? "accent" : "muted"} />
            <p className="mt-0.5 text-[11px] text-muted">{r.note}</p>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-sm">{cmp.summary}</p>

      {!sweep && (
        <button type="button" onClick={loadSweep} disabled={busy}
                className="mt-3 rounded-md border border-line px-3 py-1.5 text-xs disabled:opacity-50">
          {busy ? "Comparing…" : "Compare across budgets"}
        </button>
      )}
      {error && <div className="mt-3"><ErrorBox message={error} /></div>}
      {sweep?.sweep && (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[420px] text-xs">
            <thead>
              <tr className="text-left text-muted">
                <th className="pb-1 font-normal">Budget</th>
                {sweep.baselines.map((b) => (
                  <th key={b.id} className="pb-1 pl-3 text-right font-normal">{b.name}</th>
                ))}
                <th className="pb-1 pl-3 text-right font-normal">Optimiser</th>
              </tr>
            </thead>
            <tbody className="num">
              {sweep.sweep.map((r) => (
                <tr key={r.budget} className="border-t border-line">
                  <td className="py-1">{lakh(r.budget)}</td>
                  {sweep.baselines.map((b) => {
                    const v = r.rules[b.id];
                    return (
                      <td key={b.id} className={`py-1 pl-3 text-right ${v >= 99.95 ? "text-accent" : v < 80 ? "text-bad" : ""}`}>
                        {v.toFixed(1)}%
                      </td>
                    );
                  })}
                  <td className="py-1 pl-3 text-right font-medium">100%</td>
                </tr>
              ))}
              {sweep.worst_case_pct && (
                <tr className="border-t border-line font-medium">
                  <td className="py-1">Worst case</td>
                  {sweep.baselines.map((b) => (
                    <td key={b.id} className="py-1 pl-3 text-right">{sweep.worst_case_pct![b.id].toFixed(1)}%</td>
                  ))}
                  <td className="py-1 pl-3 text-right">100%</td>
                </tr>
              )}
            </tbody>
          </table>
          <p className="mt-2 text-[11px] text-muted">
            A rule can match the optimiser at one budget and fall far short at another; you cannot know in advance
            which rule to trust. The optimiser is best at every budget.
          </p>
        </div>
      )}
    </Card>
  );
}
