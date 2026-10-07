"use client";

import { useState, type FormEvent } from "react";
import { api, type Explanation, type TextMatch } from "@/lib/api";
import { Bar, Card, ErrorBox, H2, Muted } from "./ui";

const STEP_LABEL: Record<string, string> = {
  exact: "Exact name",
  contains: "Contains a known skill",
  semantic: "Closest meaning (accepted)",
  review: "Closest meaning (needs review)",
  unmapped: "No confident match",
};
const SAMPLE_TEXT =
  "We are hiring a data analyst with strong SQL and Python. Experience with Tableau dashboards and " +
  "machine learning is a plus. Knowledge of Hadoop or Spark preferred.";

/** Why a job-posting tag maps to a standard skill, and which skill names a text contains. */
export function ExplainSkills() {
  const [tag, setTag] = useState("Core Java");
  const [tagX, setTagX] = useState<Explanation<{ step: string; similarity?: number; threshold?: number }> | null>(null);
  const [text, setText] = useState(SAMPLE_TEXT);
  const [textX, setTextX] = useState<Explanation<{ matches: TextMatch[] }> | null>(null);
  const [busy, setBusy] = useState<"tag" | "text" | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run(kind: "tag" | "text", e?: FormEvent) {
    e?.preventDefault();
    setBusy(kind);
    setError(null);
    try {
      if (kind === "tag") setTagX(await api.xaiTag(tag.trim()));
      else setTextX(await api.xaiText(text));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="space-y-5">
      <Muted>
        How messy job-posting language becomes standard ESCO/O*NET skills. The first lookup loads the 16,000-skill
        vocabulary and can take about 15 seconds; after that it is instant.
      </Muted>
      {error && <ErrorBox message={error} />}

      <Card className="space-y-3">
        <H2 hint="Matching steps, in order: exact name → a known skill name inside the tag → closest meaning (Sentence Transformers)">
          Why does a tag map to a skill?
        </H2>
        <form onSubmit={(e) => run("tag", e)} className="flex gap-2">
          <input value={tag} onChange={(e) => setTag(e.target.value)} aria-label="Job tag" maxLength={100}
                 className="min-w-0 flex-1 rounded-md border border-line bg-bg px-3 py-2 text-sm outline-none focus:border-accent" />
          <button type="submit" disabled={busy !== null || !tag.trim()}
                  className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:text-black">
            {busy === "tag" ? "Explaining…" : "Explain"}
          </button>
        </form>
        <div className="flex flex-wrap gap-1.5 text-xs">
          {["Core Java", "pytorch developer", "Analytical", "HTML", "data viz"].map((t) => (
            <button key={t} type="button" onClick={() => setTag(t)}
                    className="rounded-full border border-line px-2 py-0.5 text-muted hover:text-ink">{t}</button>
          ))}
        </div>
        {tagX && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-accent-soft px-2 py-0.5 text-xs font-medium text-accent">
                {STEP_LABEL[tagX.details.step] ?? tagX.details.step}
              </span>
              <p className="text-sm">{tagX.summary}</p>
            </div>
            <ul className="space-y-2">
              {tagX.factors.map((f, i) => (
                <li key={`${f.name}-${i}`} className="grid grid-cols-[1fr_120px_48px] items-center gap-2 text-sm">
                  <span className={i === 0 && !f.unit ? "font-medium" : ""}>
                    {f.name}
                    {f.note && <span className="ml-1 text-[11px] text-muted">({f.note})</span>}
                  </span>
                  <Bar value={typeof f.value === "number" ? f.value : 0} tone={i === 0 && !f.unit ? "accent" : "muted"} />
                  <span className="num text-right text-xs text-muted">{typeof f.value === "number" ? f.value.toFixed(2) : ""}</span>
                </li>
              ))}
            </ul>
            {tagX.details.threshold != null && (
              <p className="text-[11px] text-muted">
                Meaning-based matches are accepted at similarity ≥ {tagX.details.threshold}; this one scored{" "}
                {tagX.details.similarity?.toFixed(3)}.
              </p>
            )}
            <p className="text-[11px] text-muted">Method: {tagX.method}.</p>
          </div>
        )}
      </Card>

      <Card className="space-y-3">
        <H2 hint="Known skill names found in a piece of text, highlighted where they occur">Which skills does a text mention?</H2>
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} maxLength={5000} aria-label="Text"
                  className="w-full rounded-md border border-line bg-bg px-3 py-2 text-sm outline-none focus:border-accent" />
        <button type="button" onClick={() => run("text")} disabled={busy !== null || !text.trim()}
                className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:text-black">
          {busy === "text" ? "Reading…" : "Find skills"}
        </button>
        {textX && (
          <div className="space-y-2">
            <p className="text-sm">{textX.summary}</p>
            <Highlighted text={text} matches={textX.details.matches} />
          </div>
        )}
      </Card>
    </div>
  );
}

function Highlighted({ text, matches }: { text: string; matches: TextMatch[] }) {
  const spans = matches.filter((m) => m.span).sort((a, b) => a.span![0] - b.span![0]);
  const parts: React.ReactNode[] = [];
  let at = 0;
  for (const m of spans) {
    const [a, b] = m.span!;
    if (a < at) continue;
    parts.push(text.slice(at, a));
    parts.push(
      <mark key={a} title={m.skill} className="rounded bg-accent-soft px-0.5 text-accent">{text.slice(a, b)}</mark>,
    );
    at = b;
  }
  parts.push(text.slice(at));
  return <p className="rounded-md border border-line bg-bg px-3 py-2 text-sm leading-relaxed">{parts}</p>;
}
