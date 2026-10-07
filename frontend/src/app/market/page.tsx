"use client";

import { useEffect, useState, type FormEvent } from "react";
import {
  AREA_LABEL,
  api,
  pct,
  type AreaEvidence,
  type DemandRow,
  type Overview,
  type SkillHit,
} from "@/lib/api";
import { Bar, Card, ErrorBox, H2, Muted, Stat } from "@/components/ui";

export default function MarketPage() {
  const [ov, setOv] = useState<Overview | null>(null);
  const [areas, setAreas] = useState<AreaEvidence[] | null>(null);
  const [demand, setDemand] = useState<DemandRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.overview(), api.triangulation(), api.demand(15)])
      .then(([o, t, d]) => {
        setOv(o);
        setAreas(t);
        setDemand(d);
      })
      .catch((e) => setError(e.message));
  }, []);

  const maxShare = demand ? Math.max(...demand.map((d) => d.share_ci_high)) : 1;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Market evidence</h1>
        <Muted>
          Where the plan&apos;s priorities come from: SAS job postings (demand and pay) and the SAS junior cohort
          (salary hikes). Every figure has a 95% interval.
        </Muted>
      </div>
      {error && <ErrorBox message={error} />}

      {ov && (
        <Card className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Job postings analysed" value={ov.postings_clean.toLocaleString("en-IN")} />
          <Stat label="Data-role postings" value={ov.data_role_postings.toLocaleString("en-IN")} />
          <Stat label="Distinct skills mapped" value={ov.distinct_canonical_skills.toLocaleString("en-IN")} />
          <Stat
            label="Data-science openings"
            value={ov.openings_ds_jobs.toLocaleString("en-IN")}
            sub={`${ov.companies_ds_jobs} companies`}
          />
        </Card>
      )}

      {areas && (
        <Card>
          <H2 hint="The three signals the optimiser weighs for each skill area">Evidence per skill area</H2>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-sm">
              <thead>
                <tr className="text-left text-xs text-muted">
                  <th className="pb-2 font-normal">Skill area</th>
                  <th className="pb-2 text-right font-normal">Share of data-role postings</th>
                  <th className="pb-2 text-right font-normal">Pay odds ratio (market)</th>
                  <th className="pb-2 text-right font-normal">Hike odds ratio (juniors)</th>
                </tr>
              </thead>
              <tbody className="num">
                {[...areas]
                  .sort((a, b) => b.demand_share - a.demand_share)
                  .map((a) => (
                    <tr key={a.area} className="border-t border-line">
                      <td className="py-2">{AREA_LABEL[a.area] ?? a.area}</td>
                      <td className="py-2 text-right">
                        {pct(a.demand_share, 1)}{" "}
                        <span className="text-xs text-muted">
                          [{pct(a.demand_ci_low, 1)}–{pct(a.demand_ci_high, 1)}]
                        </span>
                      </td>
                      <td className="py-2 text-right">
                        {a.odds_ratio.toFixed(2)}{" "}
                        <span className="text-xs text-muted">
                          [{a.or_ci_low.toFixed(2)}–{a.or_ci_high.toFixed(2)}]
                        </span>
                      </td>
                      <td className="py-2 text-right">
                        {a.rq2_odds_ratio.toFixed(2)}{" "}
                        <span className="text-xs text-muted">
                          [{a.rq2_or_ci_low.toFixed(2)}–{a.rq2_or_ci_high.toFixed(2)}]
                        </span>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-muted">
            Odds ratio above 1: postings or learners with this skill are more likely to be in the high-pay /
            high-hike group. Associations, not causes.
          </p>
        </Card>
      )}

      <div className="grid gap-5 md:grid-cols-2">
        {demand && (
          <Card>
            <H2 hint="Share of data-role postings that ask for each skill">Most demanded skills</H2>
            <ul className="space-y-2">
              {demand.map((d) => (
                <li key={d.skill} className="grid grid-cols-[130px_1fr_48px] items-center gap-2 text-sm">
                  <span className="truncate" title={d.skill}>{d.skill}</span>
                  <Bar value={d.share / maxShare} />
                  <span className="num text-right text-xs text-muted">{pct(d.share, 1)}</span>
                </li>
              ))}
            </ul>
          </Card>
        )}
        <SkillSearch />
      </div>
    </div>
  );
}

function SkillSearch() {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<SkillHit[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!q.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setHits(await api.searchSkills(q.trim()));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <H2 hint="Type any skill; meaning-based search finds the closest skills in the postings">Look up a skill</H2>
      <form onSubmit={submit} className="flex gap-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="e.g. pytorch, power bi, hadoop"
          aria-label="Skill"
          className="min-w-0 flex-1 rounded-md border border-line bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
        />
        <button type="submit" disabled={busy} className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:text-black">
          {busy ? "…" : "Search"}
        </button>
      </form>
      {error && <div className="mt-3"><ErrorBox message={error} /></div>}
      {hits && (
        <ul className="mt-3 divide-y divide-line text-sm">
          {hits.length === 0 && <li className="py-2 text-muted">No match.</li>}
          {hits.map((h) => (
            <li key={h.skill} className="py-2">
              <div className="flex justify-between gap-2">
                <span className="font-medium">{h.skill}</span>
                <span className="num text-xs text-muted">match {Math.round(h.similarity * 100)}%</span>
              </div>
              <div className="num text-xs text-muted">
                {h.posts_data_roles} data-role postings ({pct(h.share_data_roles, 2)})
                {h.median_salary_mid_lakh != null && ` · median pay ₹${h.median_salary_mid_lakh} L`}
                {h.top_role_family && ` · mostly ${h.top_role_family}`}
                {h.pay_odds_ratio != null && ` · pay odds ratio ${h.pay_odds_ratio.toFixed(2)}`}
              </div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
