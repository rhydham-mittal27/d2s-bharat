"use client";

// Session = the JWT from /api/auth/login|register plus the user it names.
// Stored in localStorage so it survives reloads; cleared on sign-out, on expiry, and when the API
// answers 401. Pages outside PUBLIC_PATHS redirect to /login while signed out.

import { usePathname, useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, setAuth, type TokenOut, type User } from "./api";
import { clearPlanCache } from "./use-plan";

const KEY = "d2s.session";
const SESSION_KEYS = ["d2s.cohort", "d2s.plan"];
export const PUBLIC_PATHS = ["/login", "/register"];

interface Session {
  token: string;
  expiresAt: string;
  user: User;
}

interface AuthState {
  ready: boolean; // session restored from storage
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  logout: (reason?: string) => void;
}

const Ctx = createContext<AuthState | null>(null);

function readSession(): Session | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const s = JSON.parse(raw) as Session;
    return new Date(s.expiresAt).getTime() > Date.now() ? s : null;
  } catch {
    return null;
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);

  const logout = useCallback(
    (reason?: string) => {
      try {
        localStorage.removeItem(KEY);
        SESSION_KEYS.forEach((k) => sessionStorage.removeItem(k));
      } catch {}
      clearPlanCache();
      setAuth(null, null);
      // a full page load, not a client navigation: it drops every bit of in-memory state and cannot
      // race with AuthGate's own redirect
      window.location.assign(reason ? `/login?reason=${encodeURIComponent(reason)}` : "/login");
    },
    [],
  );

  useEffect(() => {
    const s = readSession();
    setAuth(s?.token ?? null, s ? () => logout("Your session ended. Please sign in again.") : null);
    // one-time restore from localStorage (not available during server rendering)
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSession(s);
    setReady(true);
  }, [logout]);

  // sign out exactly when the token expires
  useEffect(() => {
    if (!session) return;
    const ms = new Date(session.expiresAt).getTime() - Date.now();
    const t = setTimeout(() => logout("Your session expired. Please sign in again."), Math.max(0, ms));
    return () => clearTimeout(t);
  }, [session, logout]);

  const start = (t: TokenOut) => {
    const s: Session = { token: t.access_token, expiresAt: t.expires_at, user: t.user };
    try {
      localStorage.setItem(KEY, JSON.stringify(s));
    } catch {}
    setAuth(s.token, () => logout("Your session ended. Please sign in again."));
    setSession(s);
  };

  const value: AuthState = {
    ready,
    user: session?.user ?? null,
    login: async (email, password) => start(await api.login(email, password)),
    register: async (name, email, password) => start(await api.register(name, email, password)),
    logout,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAuth(): AuthState {
  const a = useContext(Ctx);
  if (!a) throw new Error("useAuth outside AuthProvider");
  return a;
}

/** Where to go after sign-in: only same-site paths, never an absolute URL (open-redirect guard). */
function safeNext(search: string): string {
  const next = new URLSearchParams(search).get("next") ?? "/";
  return next.startsWith("/") && !next.startsWith("//") && !PUBLIC_PATHS.includes(next) ? next : "/";
}

/** Renders protected pages only for a signed-in user; otherwise sends them to /login?next=… */
export function AuthGate({ children }: { children: ReactNode }) {
  const { ready, user } = useAuth();
  const path = usePathname();
  const router = useRouter();
  const isPublic = PUBLIC_PATHS.includes(path);

  useEffect(() => {
    if (!ready) return;
    if (!user && !isPublic) router.replace(`/login?next=${encodeURIComponent(path)}`);
    if (user && isPublic) router.replace(safeNext(window.location.search));
  }, [ready, user, isPublic, path, router]);

  if (isPublic) return <>{children}</>;
  if (!ready || !user) return <p className="text-sm text-muted">Checking your session…</p>;
  return <>{children}</>;
}
