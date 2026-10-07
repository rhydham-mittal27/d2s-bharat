"use client";

import { useState } from "react";
import { streamAgent, tokenLabel, type AgentStep, type StressReport } from "@/lib/api";
import { useStore } from "@/lib/store";
import { AgentTrace } from "./agent-trace";
import { Card, ErrorBox, H2 } from "./ui";

/** Agent 2: finds where today's plan breaks and what to do about it. */
export function StressTest() {
  const { body } = useStore();
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [report, setReport] = useState<StressReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    setReport(null);
    setSteps([]);
    try {
      setReport(await streamAgent<StressReport>("/api/agents/stress-test", body, (s) => setSteps((x) => [...x, s])));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <H2 hint="An agent tries about 20 shocks, then the combinations most likely to hurt, and reports where the plan breaks.">
          Stress-test this plan
        </H2>
        <button type="button" onClick={run} disabled={busy}
                className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:text-black">
          {busy ? "Testing…" : report ? "Run again" : "Run stress test"}
        </button>
      </div>
      {busy && (
        <p className="text-sm text-muted" aria-live="polite">{steps.length ? steps[steps.length - 1].detail : "Starting…"}</p>
      )}
      {error && <ErrorBox message={error} />}
      {report && (
        <div className="grid gap-4 md:grid-cols-3">
          <div>
            <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted">Findings</h3>
            <ul className="space-y-1.5 text-sm">{report.summary.map((s) => <li key={s}>{s}</li>)}</ul>
          </div>
          <div>
            <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted">Biggest risks</h3>
            <ol className="list-decimal space-y-1.5 pl-4 text-sm">{report.top_risks.map((s) => <li key={s}>{s}</li>)}</ol>
          </div>
          <div>
            <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted">What to do</h3>
            <ul className="space-y-1.5 text-sm">
              {report.mitigations.map((s) => (
                <li key={s} className="flex gap-2"><span className="text-accent">→</span><span>{s}</span></li>
              ))}
            </ul>
          </div>
        </div>
      )}
      {report && <p className="text-[11px] text-muted">Finished in {(report.ms / 1000).toFixed(1)} s · {tokenLabel(report.tokens)} · every figure is an exact re-optimisation.</p>}
      {steps.length > 0 && <AgentTrace steps={steps} open={busy} />}
    </Card>
  );
}
