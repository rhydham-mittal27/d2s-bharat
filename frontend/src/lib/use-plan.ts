"use client";

// Fetches the full plan (+ evidence + brief) and Plan A/B/C for the current settings.
// Debounced so dragging a slider sends one request; stale requests are aborted; results are cached
// by request so moving between pages does not recompute.

import { useEffect, useState } from "react";
import { api, type Alternative, type FullResponse, type PlanBody } from "./api";

interface PlanState {
  full: FullResponse | null;
  alts: Alternative[] | null;
  loading: boolean;
  error: string | null;
}

const cache = new Map<string, { full: FullResponse; alts: Alternative[] | null }>();

/** Results can hold a user's cohort gaps: forget them on sign-out. */
export function clearPlanCache() {
  cache.clear();
}

export function usePlan(body: PlanBody, opts: { alternatives?: boolean; delayMs?: number; retry?: number } = {}): PlanState {
  const { alternatives = true, delayMs = 400, retry = 0 } = opts; // bump `retry` to try again after an error
  const key = JSON.stringify(body);
  const hit = cache.get(key);
  const [state, setState] = useState<PlanState>({
    full: hit?.full ?? null,
    alts: hit?.alts ?? null,
    loading: !hit,
    error: null,
  });

  useEffect(() => {
    const cached = cache.get(key);
    if (cached && (cached.alts || !alternatives)) {
      setState({ full: cached.full, alts: cached.alts, loading: false, error: null });
      return;
    }
    setState((s) => ({ ...s, loading: true, error: null }));
    const ctrl = new AbortController();
    const parsed = JSON.parse(key) as PlanBody;
    const timer = setTimeout(async () => {
      try {
        const [full, alts] = await Promise.all([
          api.planFull(parsed, ctrl.signal),
          alternatives ? api.alternatives(parsed, ctrl.signal) : Promise.resolve(null),
        ]);
        cache.set(key, { full, alts });
        setState({ full, alts, loading: false, error: null });
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        setState((s) => ({ ...s, loading: false, error: (e as Error).message }));
      }
    }, delayMs);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [key, alternatives, delayMs, retry]);

  return state;
}
