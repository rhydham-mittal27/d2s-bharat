"use client";

import { useEffect, useState } from "react";
import { api, type Explanation } from "@/lib/api";
import { useStore } from "@/lib/store";
import { Card, ErrorBox, H2, Muted } from "./ui";

const RESOURCES = [
  { id: "budget", label: "More budget" },
  { id: "trainer_hours", label: "More trainer-hours" },
  { id: "weight", label: "More priority on its skill area" },
];

/** Which limits really bind the plan, and what it would take for a left-out course to be chosen. */
export function ExplainPlan() {
  const { body } = useStore();
  const [report, setReport] = useState<Explanation<{ tight: string[] }> | null>(null);
  const [courses, setCourses] = useState<{ id: string; name: string }[]>([]);
  const [course, setCourse] = useState("dl_nlp");
  const [resource, setResource] = useState("budget");
  const [answer, setAnswer] = useState<Explanation<{ found?: boolean; new_plan_courses?: Record<string, number> }> | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const key = JSON.stringify(body);

  useEffect(() => {
    api.catalogue().then((c) => setCourses(c.map((x) => ({ id: x.id, name: x.name })))).catch(() => {});
  }, []);
  useEffect(() => {
    api.xaiConstraints(JSON.parse(key)).then(setReport).catch((e) => setError(e.message));
  }, [key]);

  async function ask() {
    setBusy(true);
    setError(null);
    try {
      setAnswer(await api.xaiCourse(body, course, resource));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const names = Object.fromEntries(courses.map((c) => [c.id, c.name]));
  return (
    <div className="space-y-5">
      <Muted>For your current plan settings. Every statement is checked by re-solving the plan exactly, not estimated.</Muted>
      {error && <ErrorBox message={error} />}
      {report && (
        <Card className="space-y-3">
          <H2 hint="A limit is binding if relaxing it by 25% and re-solving improves the plan. Small slack alone does not tell: courses come in whole blocks.">
            What really limits the plan
          </H2>
          <p className="text-sm">{report.summary}</p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[520px] text-sm">
              <thead>
                <tr className="text-left text-xs text-muted">
                  <th className="pb-1 font-normal">Limit</th>
                  <th className="pb-1 pl-3 font-normal">Used / available</th>
                  <th className="pb-1 pl-3 font-normal">Verdict</th>
                </tr>
              </thead>
              <tbody>
                {report.factors.map((f) => {
                  const binding = /^binding|^tight/.test(f.note);
                  return (
                    <tr key={f.name} className="border-t border-line">
                      <td className="py-1.5">{f.name}</td>
                      <td className="num py-1.5 pl-3">{f.value}</td>
                      <td className={`py-1.5 pl-3 text-xs ${binding ? "font-medium text-warn" : "text-muted"}`}>{f.note}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-[11px] text-muted">Method: {report.method}.</p>
        </Card>
      )}

      <Card className="space-y-3">
        <H2 hint="The smallest increase that makes the optimiser choose the course by itself (searched by doubling and bisection over exact re-solves)">
          What would it take to include a course?
        </H2>
        <div className="flex flex-wrap items-end gap-2">
          <label className="text-sm">
            <span className="mb-1 block text-xs text-muted">Course</span>
            <select value={course} onChange={(e) => setCourse(e.target.value)}
                    className="rounded border border-line bg-bg px-2 py-1.5 text-sm">
              {courses.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-xs text-muted">If we had…</span>
            <select value={resource} onChange={(e) => setResource(e.target.value)}
                    className="rounded border border-line bg-bg px-2 py-1.5 text-sm">
              {RESOURCES.map((r) => <option key={r.id} value={r.id}>{r.label}</option>)}
            </select>
          </label>
          <button type="button" onClick={ask} disabled={busy}
                  className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:text-black">
            {busy ? "Searching…" : "Find out"}
          </button>
        </div>
        {answer && (
          <div className="rounded-md border border-line px-3 py-2 text-sm">
            <p>{answer.summary}</p>
            {answer.details.found && answer.details.new_plan_courses && (
              <p className="mt-1 text-xs text-muted">
                The plan would then be:{" "}
                {Object.entries(answer.details.new_plan_courses).map(([id, n]) => `${names[id] ?? id} (${n})`).join(", ")}
              </p>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
