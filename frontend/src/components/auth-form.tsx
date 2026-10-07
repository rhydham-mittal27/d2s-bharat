"use client";

import Link from "next/link";
import { useEffect, useState, type FormEvent } from "react";
import { useAuth } from "@/lib/auth";
import { ErrorBox } from "./ui";

type Mode = "login" | "register";

export function AuthForm({ mode }: { mode: Mode }) {
  const { login, register } = useAuth();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    // e.g. "Your session expired" after an automatic sign-out
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setNotice(new URLSearchParams(window.location.search).get("reason"));
  }, []);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "login") await login(email, password);
      else await register(name, email, password);
      // AuthGate moves a signed-in user on to ?next= (or the Plan page)
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  const isLogin = mode === "login";
  const field = "w-full rounded-md border border-line bg-bg px-3 py-2 text-sm outline-none focus:border-accent";
  return (
    <div className="mx-auto mt-10 w-full max-w-sm">
      <h1 className="text-xl font-semibold tracking-tight">{isLogin ? "Sign in" : "Create your account"}</h1>
      <p className="mt-1 text-sm text-muted">
        {isLogin
          ? "Plan training for your institution with D2S Bharat."
          : "Your cohorts and saved plans are private to your account."}
      </p>

      <form onSubmit={submit} className="mt-6 space-y-4 rounded-lg border border-line bg-surface p-5" noValidate={false}>
        {notice && !error && (
          <div className="rounded-md bg-warn-soft px-3 py-2 text-sm text-warn">{notice}</div>
        )}
        {error && <ErrorBox message={error} />}
        {!isLogin && (
          <div>
            <label htmlFor="name" className="mb-1 block text-sm font-medium">Name</label>
            <input id="name" required maxLength={120} autoComplete="name" value={name}
                   onChange={(e) => setName(e.target.value)} className={field} />
          </div>
        )}
        <div>
          <label htmlFor="email" className="mb-1 block text-sm font-medium">Email</label>
          <input id="email" type="email" required autoComplete="email" value={email}
                 onChange={(e) => setEmail(e.target.value)} className={field} />
        </div>
        <div>
          <label htmlFor="password" className="mb-1 block text-sm font-medium">Password</label>
          <input id="password" type="password" required minLength={isLogin ? 1 : 8} maxLength={128}
                 autoComplete={isLogin ? "current-password" : "new-password"} value={password}
                 onChange={(e) => setPassword(e.target.value)} className={field} />
          {!isLogin && <p className="mt-1 text-xs text-muted">At least 8 characters.</p>}
        </div>
        <button type="submit" disabled={busy}
                className="w-full rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:text-black">
          {busy ? (isLogin ? "Signing in…" : "Creating account…") : isLogin ? "Sign in" : "Create account"}
        </button>
      </form>

      <p className="mt-4 text-center text-sm text-muted">
        {isLogin ? (
          <>No account yet? <Link href="/register" className="text-ink underline">Create one</Link></>
        ) : (
          <>Already registered? <Link href="/login" className="text-ink underline">Sign in</Link></>
        )}
      </p>
    </div>
  );
}
