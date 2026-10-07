"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { api, streamAgent, tokenLabel, type AgentStatus, type AgentStep, type CopilotAnswer } from "@/lib/api";
import { useStore } from "@/lib/store";
import { Card, ErrorBox, Muted } from "@/components/ui";
import { AgentTrace } from "@/components/agent-trace";

interface Turn {
  id: number;
  question: string;
  steps: AgentStep[];
  answer?: CopilotAnswer;
  error?: string;
}

const FALLBACK_EXAMPLES = [
  "What happens if our budget is cut by 30%?",
  "A trainer is leaving next term. What changes?",
  "How sure are you about this plan?",
  "Why didn't you pick the deep learning course?",
  "Is this better than just funding the biggest gap?",
  "Budget drops 20% and AI demand jumps. What should we do?",
];

export default function CopilotPage() {
  const { body, cohort } = useStore();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.agentStatus().then(setStatus).catch(() => {});
  }, []);
  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }), [turns]);

  async function ask(question: string) {
    const text = question.trim();
    if (!text || busy) return;
    const id = Date.now();
    setTurns((t) => [...t, { id, question: text, steps: [] }]);
    setQ("");
    setBusy(true);
    const update = (fn: (t: Turn) => Turn) => setTurns((all) => all.map((t) => (t.id === id ? fn(t) : t)));
    try {
      const answer = await streamAgent<CopilotAnswer>("/api/agents/copilot", { ...body, question: text }, (s) =>
        update((t) => ({ ...t, steps: [...t.steps, s] })),
      );
      update((t) => ({ ...t, answer }));
    } catch (e) {
      update((t) => ({ ...t, error: (e as Error).message }));
    } finally {
      setBusy(false);
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    ask(q);
  }

  const examples = status?.examples?.length ? status.examples : FALLBACK_EXAMPLES;
  const model = status?.model;
  return (
    <div className="mx-auto max-w-3xl space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Planning Copilot</h1>
          <Muted>Ask about your plan in plain English. Every number comes from the optimiser, not the model.</Muted>
        </div>
        {model && (
          <span className="text-xs text-muted" title={model.reason ?? undefined}>
            Local model {model.model}:{" "}
            <span className={model.available ? "text-accent" : "text-warn"}>
              {model.available ? (model.loaded ? "ready" : "available") : "off (fast answers only)"}
            </span>
          </span>
        )}
      </div>

      {turns.length === 0 && (
        <Card>
          <p className="mb-3 text-sm text-muted">
            Planning for <span className="text-ink">{cohort.name}</span>. Try one of these:
          </p>
          <div className="flex flex-wrap gap-2">
            {examples.map((e) => (
              <button key={e} type="button" onClick={() => ask(e)}
                      className="rounded-full border border-line bg-bg px-3 py-1.5 text-left text-sm hover:border-muted">
                {e}
              </button>
            ))}
          </div>
        </Card>
      )}

      <div className="space-y-4">
        {turns.map((t) => (
          <div key={t.id} className="space-y-2">
            <div className="ml-auto w-fit max-w-[85%] rounded-lg bg-accent px-3 py-2 text-sm text-white dark:text-black">
              {t.question}
            </div>
            <Card className="space-y-3">
              {!t.answer && !t.error && (
                <p className="text-sm text-muted" aria-live="polite">
                  {t.steps.length ? t.steps[t.steps.length - 1].detail : "Thinking…"}
                </p>
              )}
              {t.error && <ErrorBox message={t.error} />}
              {t.answer && <AnswerView a={t.answer} />}
              {t.steps.length > 0 && <AgentTrace steps={t.steps} open={!t.answer && !t.error} />}
            </Card>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form onSubmit={submit} className="sticky bottom-3 flex gap-2 rounded-lg border border-line bg-surface p-2 shadow-sm">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="e.g. What if prices rise 15% and a trainer leaves?"
          aria-label="Question"
          maxLength={500}
          className="min-w-0 flex-1 rounded-md bg-bg px-3 py-2 text-sm outline-none"
        />
        <button type="submit" disabled={busy || !q.trim()}
                className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:text-black">
          {busy ? "Working…" : "Ask"}
        </button>
      </form>
    </div>
  );
}

function AnswerView({ a }: { a: CopilotAnswer }) {
  const badge = a.path === "fast" ? "Fast path" : a.path === "llm" ? "Local model" : "Help";
  return (
    <div className="space-y-3">
      {a.sections.map((s) => (
        <div key={s.title}>
          <h3 className={`text-sm font-semibold ${s.ok ? "" : "text-warn"}`}>{s.title}</h3>
          <ul className="mt-1 space-y-1 text-sm">
            {s.lines.map((l, i) => (
              <li key={i} className="flex gap-2"><span className="text-accent">•</span><span>{l}</span></li>
            ))}
          </ul>
        </div>
      ))}
      {a.assumptions.length > 0 && (
        <p className="rounded-md bg-warn-soft px-3 py-2 text-xs text-warn">Assumed: {a.assumptions.join(" ")}</p>
      )}
      {a.notes.map((n) => (
        <p key={n} className="text-sm text-muted">{n}</p>
      ))}
      <p className="text-[11px] text-muted">
        {badge} · {(a.ms / 1000).toFixed(1)} s · {tokenLabel(a.tokens)}
        {a.sections.length ? " · numbers from the optimiser and simulator" : ""}
      </p>
    </div>
  );
}
