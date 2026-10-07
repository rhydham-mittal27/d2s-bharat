"use client";

import Link from "next/link";
import { inr, lakh, pct } from "@/lib/api";
import { useStore } from "@/lib/store";
import { usePlan } from "@/lib/use-plan";
import { Card, ErrorBox, H2, Muted } from "@/components/ui";

export default function BriefPage() {
  const { body, plan, cohort } = useStore();
  const { full, loading, error } = usePlan(body, { alternatives: false, delayMs: 0 });

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Decision brief</h1>
          <Muted>
            {cohort.name} · budget {lakh(plan.budget)} · {plan.trainer_hours} trainer-hours ·{" "}
            <Link href="/" className="underline">change</Link>
          </Muted>
        </div>
        <button
          type="button"
          onClick={() => window.print()}
          disabled={!full}
          className="no-print rounded-md border border-line px-4 py-2 text-sm disabled:opacity-50"
        >
          Print / save as PDF
        </button>
      </div>

      {error && <ErrorBox message={error} />}
      {loading && !full && <Card><Muted>Preparing the brief…</Muted></Card>}

      {full && (
        <>
          <Card className="space-y-5">
            <h2 className="text-lg font-semibold">{full.brief.title}</h2>
            {full.brief.sections.map((s) => (
              <div key={s.heading}>
                <h3 className="mb-1.5 text-sm font-semibold text-accent">{s.heading}</h3>
                <div className="space-y-1 text-sm leading-relaxed">
                  {s.paragraphs.map((p, i) =>
                    p.startsWith("- ") ? (
                      <p key={i} className="pl-4 -indent-3">• {p.slice(2)}</p>
                    ) : (
                      <p key={i}>{p}</p>
                    ),
                  )}
                </div>
              </div>
            ))}
            <p className={`rounded-md px-3 py-2 text-xs ${full.brief.faithful ? "bg-accent-soft text-accent" : "bg-warn-soft text-warn"}`}>
              {full.brief.faithful
                ? `All ${full.brief.numbers_checked} numbers in this brief were checked against the optimiser's output.`
                : `Unverified numbers: ${full.brief.unverified_numbers.join(", ")}`}
              {" "}Written from templates; no language model is involved.
            </p>
          </Card>

          <Card>
            <H2 hint="For each chosen course: the skill areas it serves and the evidence behind them">Evidence trail</H2>
            <div className="space-y-4">
              {full.evidence.map((c) => (
                <div key={c.course_id} className="border-t border-line pt-3 first:border-0 first:pt-0">
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <span className="font-medium">{c.name}</span>
                    <span className="num text-xs text-muted">
                      {c.seats} seats · {inr(c.cost)} · removing it loses {c.impact_lost_if_removed_pct}% of impact
                    </span>
                  </div>
                  <table className="mt-2 w-full text-xs">
                    <thead>
                      <tr className="text-left text-muted">
                        <th className="font-normal">Skill area</th>
                        <th className="pl-3 text-right font-normal">In data-role postings</th>
                        <th className="pl-3 text-right font-normal">Pay odds ratio</th>
                        <th className="pl-3 text-right font-normal">Junior hike odds ratio</th>
                        <th className="pl-3 text-right font-normal">Learners helped</th>
                      </tr>
                    </thead>
                    <tbody className="num">
                      {c.areas.map((a) => (
                        <tr key={a.area}>
                          <td className="py-0.5">{a.label}</td>
                          <td className="pl-3 text-right">{pct(a.market_demand_share, 1)}</td>
                          <td className="pl-3 text-right">{a.market_pay_or.toFixed(2)}</td>
                          <td className="pl-3 text-right">{a.junior_hike_or.toFixed(2)}</td>
                          <td className="pl-3 text-right">
                            {Math.round(a.learners_closed_by_course)} of {a.learners_short}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
            <p className="mt-4 text-xs text-muted">
              Odds ratios above 1 mean the skill goes with higher pay (market) or a higher salary hike (juniors).
              They are associations in the SAS sample data, not causal effects.
            </p>
          </Card>
        </>
      )}
    </div>
  );
}
