"use client";

import { useEffect, useState } from "react";
import {
  AREA_LABEL,
  AREAS,
  api,
  pct,
  type Explanation,
  type LearnerScores,
  type WhatIfDetails,
  type WhyDetails,
} from "@/lib/api";
import { Card, ErrorBox, H2, Muted } from "./ui";

const EXAMPLE: LearnerScores = {
  coding_skills: 3.5,
  maths_stats_skills: 3.4,
  ai_and_ml_skills: 4.0,
  big_data_skills: 3.8,
  dashboard_and_storytelling_skills: 3.3,
};

/** Why does this learner get this predicted chance, and what is the cheapest way to raise it? */
export function ExplainLearner() {
  const [scores, setScores] = useState<LearnerScores>(EXAMPLE);
  const [target, setTarget] = useState(0.7);
  const [why, setWhy] = useState<Explanation<WhyDetails> | null>(null);
  const [path, setPath] = useState<Explanation<WhatIfDetails> | null>(null);
  const [global, setGlobal] = useState<Explanation<{ cv_auc: number; n_train: number }> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.xaiGlobal().then(setGlobal).catch((e) => setError(e.message));
  }, []);

  const key = JSON.stringify([scores, target]);
  useEffect(() => {
    const ctrl = new AbortController();
    const t = setTimeout(() => {
      Promise.all([api.xaiWhy(scores, ctrl.signal), api.xaiWhatIf(scores, target, ctrl.signal)])
        .then(([w, p]) => {
          setWhy(w);
          setPath(p);
          setError(null);
        })
        .catch((e) => e.name !== "AbortError" && setError(e.message));
    }, 250);
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  const d = why?.details;
  return (
    <div className="space-y-5">
      <Muted>
        A junior learner&apos;s chance of a high salary hike, from the model trained on the SAS junior cohort
        {global ? ` (AUC ${global.details.cv_auc.toFixed(2)}, ${global.details.n_train} learners)` : ""}. Move the
        scores: every explanation below is recomputed exactly, offline.
      </Muted>
      {error && <ErrorBox message={error} />}

      <div className="grid gap-5 lg:grid-cols-[280px_1fr]">
        <Card className="h-fit space-y-4">
          <H2>Self-ratings (1–5)</H2>
          {AREAS.map((a) => (
            <div key={a}>
              <div className="mb-1 flex justify-between text-sm">
                <label htmlFor={`sc-${a}`}>{AREA_LABEL[a]}</label>
                <span className="num font-medium">{scores[a as keyof LearnerScores].toFixed(1)}</span>
              </div>
              <input id={`sc-${a}`} type="range" min={1} max={5} step={0.1} className="w-full"
                     value={scores[a as keyof LearnerScores]}
                     onChange={(e) => setScores({ ...scores, [a]: Number(e.target.value) })} />
            </div>
          ))}
          <div className="border-t border-line pt-3">
            <div className="mb-1 flex justify-between text-sm">
              <label htmlFor="target">Target chance</label>
              <span className="num font-medium">{pct(target)}</span>
            </div>
            <input id="target" type="range" min={0.5} max={0.95} step={0.05} className="w-full" value={target}
                   onChange={(e) => setTarget(Number(e.target.value))} />
          </div>
          <button type="button" onClick={() => setScores(EXAMPLE)} className="text-xs text-muted underline">
            Reset to the example learner
          </button>
        </Card>

        <div className="space-y-5">
          {why && d && (
            <Card className="space-y-4">
              <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
                <span className="num text-3xl font-semibold">{pct(d.probability)}</span>
                <span className="text-sm text-muted">
                  chance of a high hike (likely range {pct(d.probability_low)}–{pct(d.probability_high)}) ·
                  average junior {pct(d.baseline_probability)}
                </span>
              </div>
              <p className="text-sm">{why.summary}</p>
              <H2 hint="Shapley values: how far each skill moves this learner from the average junior (log-odds). They add up exactly to the difference.">
                What pushes the prediction up or down
              </H2>
              <Diverging factors={why.factors.map((f) => ({ name: f.name, value: f.contribution ?? 0, note: `score ${f.value} · ${f.note}` }))} />
              <p className="text-[11px] text-muted">
                Sum of contributions {d.sum_of_contributions.toFixed(3)} = log-odds gap to the average junior{" "}
                {d.logodds_gap.toFixed(3)} (exact, no sampling). {why.method}.
              </p>
            </Card>
          )}

          {path && (
            <Card className="space-y-3">
              <H2 hint="The fewest total scale points that reach the target (provably the cheapest for this model)">
                Cheapest way to reach {pct(path.details.target)}
              </H2>
              <p className="text-sm">{path.summary}</p>
              {path.factors.length > 0 && (
                <ul className="grid gap-2 sm:grid-cols-3">
                  {path.factors.map((f) => (
                    <li key={f.name} className="rounded-md border border-line px-3 py-2 text-sm">
                      <div className="font-medium">{f.name}</div>
                      <div className="num text-muted">{String(f.value)} (+{f.contribution})</div>
                    </li>
                  ))}
                </ul>
              )}
              <Curves curves={path.details.curves} scores={scores} target={path.details.target} />
            </Card>
          )}

          {global && (
            <Card className="space-y-3">
              <H2 hint="Across all juniors: how much one standard deviation more of each skill multiplies the odds of a high hike, with 90% bootstrap intervals">
                What matters most overall
              </H2>
              <p className="text-sm">{global.summary}</p>
              <ul className="space-y-2">
                {global.factors.map((f) => {
                  const or = f.contribution ?? 1;
                  return (
                    <li key={f.name} className="grid grid-cols-[150px_1fr_60px] items-center gap-2 text-sm" title={f.note}>
                      <span>{f.name}</span>
                      <div className="h-2 overflow-hidden rounded-full bg-track">
                        <div className="h-full rounded-full bg-accent" style={{ width: `${Math.min(100, ((or - 1) / 2.5) * 100)}%` }} />
                      </div>
                      <span className="num text-right text-xs text-muted">×{or.toFixed(2)}</span>
                    </li>
                  );
                })}
              </ul>
              <p className="text-[11px] text-muted">{global.method}. Associations in the SAS sample, not causes; not for decisions about individuals.</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

/** Horizontal bars around zero: negative to the left (holds back), positive to the right (helps). */
function Diverging({ factors }: { factors: { name: string; value: number; note: string }[] }) {
  const max = Math.max(0.5, ...factors.map((f) => Math.abs(f.value)));
  return (
    <ul className="space-y-2">
      {factors.map((f) => {
        const w = (Math.abs(f.value) / max) * 50;
        const neg = f.value < 0;
        return (
          <li key={f.name} className="grid grid-cols-[150px_1fr_60px] items-center gap-2 text-sm" title={f.note}>
            <span>{f.name}</span>
            <div className="relative h-3 rounded bg-track">
              <div className="absolute inset-y-0 left-1/2 w-px bg-muted/60" />
              <div className={`absolute inset-y-0 rounded ${neg ? "bg-warn" : "bg-accent"}`}
                   style={neg ? { right: "50%", width: `${w}%` } : { left: "50%", width: `${w}%` }} />
            </div>
            <span className={`num text-right text-xs ${neg ? "text-warn" : "text-accent"}`}>
              {f.value > 0 ? "+" : ""}{f.value.toFixed(2)}
            </span>
          </li>
        );
      })}
      <li className="grid grid-cols-[150px_1fr_60px] text-[11px] text-muted">
        <span />
        <span className="flex justify-between"><span>← holds back</span><span>helps →</span></span>
        <span />
      </li>
    </ul>
  );
}

/** One small chart per skill: predicted chance as that skill moves from 1 to 5 (others fixed). */
function Curves({ curves, scores, target }: { curves: WhatIfDetails["curves"]; scores: LearnerScores; target: number }) {
  const W = 160, H = 70, P = 4;
  const x = (s: number) => P + ((s - 1) / 4) * (W - 2 * P);
  const y = (p: number) => H - P - p * (H - 2 * P);
  return (
    <div>
      <p className="mb-2 text-xs text-muted">Predicted chance as one skill changes, the others held at their current score. Dot: now. Dashed line: target.</p>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {AREAS.map((a) => {
          const pts = curves[a] ?? [];
          const now = scores[a as keyof LearnerScores];
          const nearest = pts.reduce((b, p) => (Math.abs(p.score - now) < Math.abs(b.score - now) ? p : b), pts[0]);
          return (
            <figure key={a} className="rounded-md border border-line p-2">
              <figcaption className="mb-1 text-[11px] text-muted">{AREA_LABEL[a]}</figcaption>
              <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img"
                   aria-label={`Chance against ${AREA_LABEL[a]} score`}>
                <line x1={P} x2={W - P} y1={y(target)} y2={y(target)} stroke="var(--muted)" strokeDasharray="3 3" strokeWidth="1" />
                <polyline fill="none" stroke="var(--accent)" strokeWidth="2"
                          points={pts.map((p) => `${x(p.score)},${y(p.probability)}`).join(" ")} />
                {nearest && <circle cx={x(now)} cy={y(nearest.probability)} r="3.5" fill="var(--ink)" />}
              </svg>
              <div className="num flex justify-between text-[10px] text-muted"><span>1</span><span>5</span></div>
            </figure>
          );
        })}
      </div>
    </div>
  );
}
