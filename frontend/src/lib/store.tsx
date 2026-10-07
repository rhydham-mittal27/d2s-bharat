"use client";

// Shared state across pages: the active cohort (its gaps feed every plan) and the plan settings.
// Kept in sessionStorage so a page reload keeps the user's choices.

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import type { CohortProfile, PlanBody, PlanRequest } from "./api";

export interface ActiveCohort {
  name: string; // "SAS sample" or the uploaded file name
  sample: boolean;
  n_learners: number;
  gaps: Record<string, number>;
}

const DEFAULT_PLAN: PlanRequest = { budget: 400_000, trainer_hours: 200, scheme: "balanced", bound: "point" };
const SAMPLE: ActiveCohort = { name: "SAS sample cohort", sample: true, n_learners: 139, gaps: {} };

interface Store {
  cohort: ActiveCohort;
  setCohort: (c: ActiveCohort) => void;
  applyProfile: (p: CohortProfile, name: string, sample: boolean) => void;
  plan: PlanRequest;
  setPlan: (p: PlanRequest) => void;
  body: PlanBody;
}

const Ctx = createContext<Store | null>(null);

function load<T>(key: string, fallback: T): T {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function save(key: string, value: unknown) {
  try {
    sessionStorage.setItem(key, JSON.stringify(value));
  } catch {}
}

export function StoreProvider({ children }: { children: ReactNode }) {
  const [cohort, setCohortState] = useState<ActiveCohort>(SAMPLE);
  const [plan, setPlanState] = useState<PlanRequest>(DEFAULT_PLAN);

  useEffect(() => {
    // restore after mount (sessionStorage is not available during server rendering)
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setCohortState(load("d2s.cohort", SAMPLE));
    setPlanState(load("d2s.plan", DEFAULT_PLAN));
  }, []);

  const setCohort = (c: ActiveCohort) => {
    setCohortState(c);
    save("d2s.cohort", c);
  };
  const setPlan = (p: PlanRequest) => {
    setPlanState(p);
    save("d2s.plan", p);
  };
  const applyProfile = (p: CohortProfile, name: string, sample: boolean) =>
    setCohort({
      name,
      sample,
      n_learners: p.n_learners,
      gaps: Object.fromEntries(p.areas.map((a) => [a.area, a.learners_short])),
    });

  const body: PlanBody = cohort.sample ? { plan } : { plan, gaps: cohort.gaps };
  return (
    <Ctx.Provider value={{ cohort, setCohort, applyProfile, plan, setPlan, body }}>{children}</Ctx.Provider>
  );
}

export function useStore(): Store {
  const s = useContext(Ctx);
  if (!s) throw new Error("useStore outside StoreProvider");
  return s;
}
