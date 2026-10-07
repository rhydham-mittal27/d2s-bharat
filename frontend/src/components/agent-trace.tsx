"use client";

import type { AgentStep } from "@/lib/api";

const NODE_LABEL: Record<string, string> = {
  route: "Understand",
  choose: "Local model",
  act: "Run tool",
  answer: "Write answer",
  baseline: "Today's plan",
  probes: "Single shocks",
  explore: "Choose combinations",
  follow_ups: "Combined shocks",
  report: "Report",
};

/** The agent's steps, as they happened: what it did, and how long each step took. */
export function AgentTrace({ steps, open = false }: { steps: AgentStep[]; open?: boolean }) {
  return (
    <details className="text-xs" open={open}>
      <summary className="cursor-pointer text-muted hover:text-ink">How I got this ({steps.length} steps)</summary>
      <ol className="mt-2 space-y-1 border-l border-line pl-3">
        {steps.map((s, i) => (
          <li key={i} className="flex flex-wrap gap-x-2">
            <span className={`font-medium ${s.ok === false ? "text-warn" : "text-accent"}`}>
              {NODE_LABEL[s.node] ?? s.node}
            </span>
            <span className="text-muted">{s.detail}</span>
            {s.ms != null && <span className="num text-muted">({Math.round(s.ms)} ms)</span>}
            {s.tokens != null && <span className="num text-muted">· {s.tokens} tokens</span>}
          </li>
        ))}
      </ol>
    </details>
  );
}
