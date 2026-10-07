"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { api, pct, type CohortProfile } from "@/lib/api";
import { useStore } from "@/lib/store";
import { Bar, Card, ErrorBox, H2, Muted, Stat } from "@/components/ui";
import { IntakeAgent } from "@/components/intake-agent";

const TEMPLATE =
  "id,coding_skills,maths_stats_skills,ai_and_ml_skills,big_data_skills,dashboard_and_storytelling_skills\n" +
  "s01,4,3,4,2,3\ns02,5,4,3,3,4\n";

export default function CohortPage() {
  const { cohort, applyProfile } = useStore();
  const [profile, setProfile] = useState<CohortProfile | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // show the sample until the user uploads something
    api.cohortSample().then(setProfile).catch((e) => setError(e.message));
  }, []);

  async function upload(file: File) {
    setBusy(true);
    setError(null);
    try {
      const p = await api.cohortUpload(file);
      setProfile(p);
      setFileName(file.name);
      if (p.n_learners > 0) applyProfile(p, file.name, false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  }

  async function useSample() {
    setBusy(true);
    setError(null);
    try {
      const p = await api.cohortSample();
      setProfile(p);
      setFileName(null);
      applyProfile(p, "SAS sample cohort", true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const templateHref = `data:text/csv;charset=utf-8,${encodeURIComponent(TEMPLATE)}`;
  const showing = fileName ?? "SAS sample cohort";

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Your learners</h1>
        <Muted>
          Upload one row per learner with five self-ratings from 1 to 5. We work out how many learners fall short of the
          target in each skill area; those gaps drive the plan.
        </Muted>
      </div>

      <Card className="flex flex-wrap items-center gap-3">
        <input
          ref={input}
          type="file"
          accept=".csv,.xlsx,.xls"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
        />
        <button
          type="button"
          disabled={busy}
          onClick={() => input.current?.click()}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:text-black"
        >
          {busy ? "Reading…" : "Upload CSV or Excel"}
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={useSample}
          className="rounded-md border border-line px-4 py-2 text-sm disabled:opacity-50"
        >
          Use the SAS sample (139 learners)
        </button>
        <a href={templateHref} download="cohort_template.csv" className="text-sm text-muted underline">
          Download template
        </a>
        <span className="ml-auto text-xs text-muted">
          Planning for: <span className="text-ink">{cohort.name}</span>
        </span>
      </Card>

      <IntakeAgent
        onDone={(p, name) => {
          setProfile(p);
          setFileName(name);
          if (p.n_learners > 0) applyProfile(p, name, false);
        }}
      />

      {error && <ErrorBox message={error} />}

      {profile && (
        <>
          <Card>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Stat label="Showing" value={<span className="text-base">{showing}</span>} />
              <Stat label="Learners" value={profile.n_learners} />
              <Stat label="Rows rejected" value={profile.n_rejected} />
              <Stat
                label="Avg. chance of a high salary hike"
                value={pct(profile.mean_probability)}
                sub="from the junior hike model"
              />
            </div>
            {profile.issues.length > 0 && (
              <details className="mt-3 text-sm">
                <summary className="cursor-pointer text-warn">
                  {profile.n_rejected} {profile.n_rejected === 1 ? "row was" : "rows were"} rejected. See why
                </summary>
                <ul className="mt-2 space-y-0.5 text-xs text-muted">
                  {profile.issues.map((i) => (
                    <li key={`${i.row}-${i.problem}`}>Row {i.row}: {i.problem}</li>
                  ))}
                </ul>
              </details>
            )}
          </Card>

          {profile.n_learners === 0 ? (
            <ErrorBox message="No valid learner rows in this file, so the active cohort was not changed." />
          ) : (
            <Card>
              <H2 hint="Learners below the target score in each area. The target is the score typical of juniors with a high hike.">
                Skill gaps
              </H2>
              <ul className="divide-y divide-line">
                {[...profile.areas]
                  .sort((a, b) => b.share_short - a.share_short)
                  .map((a) => (
                    <li key={a.area} className="grid gap-2 py-3 sm:grid-cols-[180px_1fr_1fr] sm:items-center">
                      <div>
                        <div className="text-sm font-medium">{a.label}</div>
                        <div className="text-xs text-muted">target {a.target_score.toFixed(1)} / 5</div>
                      </div>
                      <div>
                        <div className="num mb-1 text-sm">
                          {a.learners_short} short <span className="text-muted">({pct(a.share_short)})</span>
                        </div>
                        <Bar value={a.share_short} tone={a.share_short > 0.5 ? "warn" : "accent"} />
                      </div>
                      <div className="text-xs text-muted">
                        {a.linked_courses.length > 0
                          ? `Courses: ${a.linked_courses.map((c) => c.name).join(", ")}`
                          : "No catalogue course covers this area"}
                      </div>
                    </li>
                  ))}
              </ul>
            </Card>
          )}

          <div className="flex justify-end">
            <Link href="/" className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white dark:text-black">
              Plan for this cohort →
            </Link>
          </div>
        </>
      )}
    </div>
  );
}
