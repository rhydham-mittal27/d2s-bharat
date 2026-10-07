"use client";

import { useState } from "react";
import { ExplainLearner } from "@/components/explain-learner";
import { ExplainPlan } from "@/components/explain-plan";
import { ExplainSkills } from "@/components/explain-skills";
import { Muted } from "@/components/ui";

const TABS = [
  { id: "learner", label: "Learner prediction", body: <ExplainLearner /> },
  { id: "plan", label: "Plan limits", body: <ExplainPlan /> },
  { id: "skills", label: "Skill matching", body: <ExplainSkills /> },
] as const;

export default function ExplainPage() {
  const [tab, setTab] = useState<(typeof TABS)[number]["id"]>("learner");
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Explain</h1>
        <Muted>
          Why the system says what it says: exact explanations computed offline from the models themselves, no
          language model and no approximation.
        </Muted>
      </div>
      <div role="tablist" aria-label="What to explain" className="flex flex-wrap gap-1 rounded-lg border border-line bg-surface p-1">
        {TABS.map((t) => (
          <button key={t.id} role="tab" type="button" aria-selected={tab === t.id} onClick={() => setTab(t.id)}
                  className={`rounded-md px-3 py-1.5 text-sm ${tab === t.id ? "bg-accent-soft font-medium text-accent" : "text-muted hover:text-ink"}`}>
            {t.label}
          </button>
        ))}
      </div>
      <div role="tabpanel">{TABS.find((t) => t.id === tab)!.body}</div>
    </div>
  );
}
