"use client";

import { useRef, useState } from "react";
import { api, tokenLabel, type CohortProfile, type IntakeProposal, type TokenUsage } from "@/lib/api";
import { AgentTrace } from "./agent-trace";
import { Card, ErrorBox, H2 } from "./ui";

const SCALE_LABEL: Record<string, string> = {
  "1-5": "1–5 (as is)",
  "0-5": "0–5",
  "1-10": "1–10",
  "0-10": "0–10",
  "0-100": "0–100 / %",
  words: "words (poor…excellent)",
};

/** Agent 3: maps a messy cohort file to the expected format; nothing is used until the user approves. */
export function IntakeAgent({ onDone }: { onDone: (p: CohortProfile, name: string) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [pending, setPending] = useState<{ id: string; p: IntakeProposal; tokens: TokenUsage } | null>(null);
  const [mapping, setMapping] = useState<Record<string, string | null>>({});
  const [scales, setScales] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);

  async function upload(file: File) {
    setBusy(true);
    setError(null);
    setDone(null);
    try {
      const r = await api.intakeStart(file);
      setPending({ id: r.thread_id, p: r.proposal, tokens: r.tokens });
      setMapping(Object.fromEntries(r.proposal.targets.map((t) => [t.target, t.source])));
      setScales(Object.fromEntries(r.proposal.targets.filter((t) => t.scale).map((t) => [t.target, t.scale!])));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  }

  async function decide(approve: boolean) {
    if (!pending) return;
    setBusy(true);
    setError(null);
    try {
      const r = await api.intakeDecide(pending.id, { approve, mapping, scales });
      if (r.profile) {
        onDone(r.profile, pending.p.filename);
        setDone(`Imported ${r.profile.n_learners} learners from ${pending.p.filename}` +
                (r.unreadable_rows ? ` (${r.unreadable_rows} rows with unreadable scores were skipped)` : "") + ".");
      }
      setPending(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const p = pending?.p;
  return (
    <Card className="space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <H2 hint="Any column names, scores out of 10 or 100, or words like 'good'. The agent proposes a mapping; you approve it.">
          Messy file? Let the intake agent map it
        </H2>
        <input ref={input} type="file" accept=".csv,.xlsx,.xls" className="hidden"
               onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
        <button type="button" disabled={busy} onClick={() => input.current?.click()}
                className="rounded-md border border-line px-4 py-2 text-sm disabled:opacity-50">
          {busy && !pending ? "Reading…" : "Upload any file"}
        </button>
      </div>
      {error && <ErrorBox message={error} />}
      {done && <p className="rounded-md bg-accent-soft px-3 py-2 text-sm text-accent">{done}</p>}

      {p && (
        <div className="space-y-3">
          <p className="text-sm">
            <span className="font-medium">{p.filename}</span> · {p.rows} rows. Check the proposed mapping:
            <span className="ml-2 text-xs text-muted">({tokenLabel(pending?.tokens)})</span>
          </p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-sm">
              <thead>
                <tr className="text-left text-xs text-muted">
                  <th className="pb-1 font-normal">We need</th>
                  <th className="pb-1 font-normal">Your column</th>
                  <th className="pb-1 font-normal">Scale</th>
                  <th className="pb-1 font-normal">Found by</th>
                  <th className="pb-1 font-normal">Sample</th>
                </tr>
              </thead>
              <tbody>
                {p.targets.map((t) => (
                  <tr key={t.target} className="border-t border-line align-top">
                    <td className="py-1.5 pr-2">{t.label}</td>
                    <td className="py-1.5 pr-2">
                      <select value={mapping[t.target] ?? ""} aria-label={`Column for ${t.label}`}
                              onChange={(e) => setMapping({ ...mapping, [t.target]: e.target.value || null })}
                              className="w-full rounded border border-line bg-bg px-1.5 py-1 text-xs">
                        <option value="">{t.target === "id" ? "(number the rows)" : "— none —"}</option>
                        {p.columns.map((c) => <option key={c} value={c}>{c}</option>)}
                      </select>
                    </td>
                    <td className="py-1.5 pr-2">
                      {t.target !== "id" && (
                        <select value={scales[t.target] ?? "1-5"} aria-label={`Scale for ${t.label}`}
                                onChange={(e) => setScales({ ...scales, [t.target]: e.target.value })}
                                className="rounded border border-line bg-bg px-1.5 py-1 text-xs">
                          {p.scale_options.map((s) => <option key={s} value={s}>{SCALE_LABEL[s] ?? s}</option>)}
                        </select>
                      )}
                    </td>
                    <td className={`py-1.5 pr-2 text-xs ${t.method.startsWith("model") ? "text-warn" : "text-muted"}`}>
                      {t.method}
                    </td>
                    <td className="py-1.5 text-xs text-muted">{t.sample.join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {p.warnings.length > 0 && (
            <ul className="space-y-0.5 text-xs text-warn">{p.warnings.map((w) => <li key={w}>• {w}</li>)}</ul>
          )}
          <div className="flex gap-2">
            <button type="button" disabled={busy} onClick={() => decide(true)}
                    className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:text-black">
              {busy ? "Importing…" : "Approve and import"}
            </button>
            <button type="button" disabled={busy} onClick={() => decide(false)}
                    className="rounded-md border border-line px-4 py-2 text-sm">Cancel</button>
          </div>
          <AgentTrace steps={p.trace} />
        </div>
      )}
    </Card>
  );
}
