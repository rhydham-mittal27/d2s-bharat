"use client";

import Link from "next/link";
import { useEffect, useId, useState, type FormEvent, type ReactNode } from "react";
import { ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";

type Mode = "login" | "register";
type Field = "name" | "email" | "password";
type Banner = { text: string; tone: "bad" | "warn"; signInLink?: boolean };

/** Set to "split" to show the value panel beside the form on wide screens (design option). */
const LAYOUT: "centered" | "split" = "centered";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;
const WEAK_PASSWORD = "Choose a stronger password: no spaces at the start or end, and more varied characters.";

/** The same three rules the server enforces (d2s/api/auth.py). */
function passwordChecks(pw: string) {
  return [
    { label: "8 to 128 characters", ok: pw.length >= 8 && pw.length <= 128 },
    { label: "At least 4 different characters", ok: new Set(pw).size >= 4 },
    { label: "No spaces at the start or end", ok: pw.length > 0 && pw === pw.trim() },
  ];
}

export function AuthForm({ mode }: { mode: Mode }) {
  const { login, register } = useAuth();
  const reg = mode === "register";
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [busy, setBusy] = useState(false);
  const [tried, setTried] = useState(false);
  const [errors, setErrors] = useState<Partial<Record<Field, string>>>({});
  const [banner, setBanner] = useState<Banner | null>(null);
  const [query, setQuery] = useState("");
  // unique per form: Next.js keeps the previous page mounted (hidden) for instant back-navigation,
  // so the sign-in and register forms can both be in the DOM at once
  const uid = useId();
  const id = { name: `${uid}name`, email: `${uid}email`, pw: `${uid}pw`, rules: `${uid}rules` };

  useEffect(() => {
    // "?reason=Your session expired…" after an automatic sign-out; keep ?next= when switching pages
    const q = new URLSearchParams(window.location.search);
    const reason = q.get("reason");
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (reason) setBanner({ text: reason, tone: "warn" });
    const next = q.get("next");
    setQuery(next ? `?next=${encodeURIComponent(next)}` : "");
  }, []);

  function validate(): Partial<Record<Field, string>> {
    const e: Partial<Record<Field, string>> = {};
    if (reg) {
      if (!name.trim()) e.name = "Enter your name.";
      else if (name.length > 120) e.name = "Keep your name under 120 characters.";
    }
    if (!email.trim()) e.email = "Enter your email address.";
    else if (!EMAIL_RE.test(email.trim())) e.email = "Enter a valid email address.";
    if (!password) e.password = "Enter a password.";
    else if (reg && passwordChecks(password).some((c) => !c.ok)) e.password = WEAK_PASSWORD;
    return e;
  }

  /** Server answers -> the design's messages (field errors where we know the field). */
  function showServerError(err: unknown) {
    const status = err instanceof ApiError ? err.status : undefined;
    const msg = (err as Error).message ?? "";
    if (status === 409) {
      setErrors({ email: "An account with this email already exists." });
      setBanner({ text: "An account with this email already exists.", tone: "bad", signInLink: true });
    } else if (status === 401) {
      setBanner({ text: "Email or password is incorrect.", tone: "bad" });
    } else if (status === 422 && /^(name|email|password):/.test(msg)) {
      const fieldErrors: Partial<Record<Field, string>> = {};
      for (const part of msg.split("; ")) {
        const [f, ...rest] = part.split(": ");
        const text = rest.join(": ");
        if (f === "password") fieldErrors.password = /stronger/.test(text) ? WEAK_PASSWORD : capitalise(text);
        else if (f === "email") fieldErrors.email = "Enter a valid email address.";
        else if (f === "name") fieldErrors.name = capitalise(text);
      }
      setErrors(fieldErrors);
    } else {
      setBanner({ text: msg || "Something went wrong. Try again.", tone: "bad" });
    }
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (busy) return;
    setTried(true);
    const found = validate();
    setErrors(found);
    setBanner(null);
    if (Object.keys(found).length) return;
    setBusy(true);
    try {
      if (reg) await register(name.trim(), email.trim(), password);
      else await login(email.trim(), password);
      // AuthGate moves a signed-in user on to ?next= (or the Plan page)
    } catch (err) {
      showServerError(err);
      setBusy(false);
    }
  }

  function edit(field: Field, value: string) {
    if (field === "name") setName(value);
    if (field === "email") setEmail(value);
    if (field === "password") setPassword(value);
    setErrors(({ [field]: _drop, ...rest }) => rest); // eslint-disable-line @typescript-eslint/no-unused-vars
    if (banner?.tone === "bad") setBanner(null);
  }

  const input = (field: Field) =>
    `h-10 w-full rounded-md border bg-bg px-3 text-sm outline-none transition-shadow focus:border-accent focus:shadow-[0_0_0_3px_var(--accent-soft)] ${errors[field] ? "border-bad" : "border-line"}`;

  const form = (
    <div className="flex w-full max-w-[384px] flex-col gap-5 justify-self-center">
      <div className="flex flex-col gap-1.5">
        <h1 className="text-xl font-semibold tracking-tight">{reg ? "Create your account" : "Sign in"}</h1>
        <p className="text-sm leading-relaxed text-muted">
          {reg ? "Your cohorts and saved plans are private to your account." : "Plan training for your institution with D2S Bharat."}
        </p>
      </div>

      <form onSubmit={submit} noValidate className="flex flex-col gap-4 rounded-lg border border-line bg-surface p-5">
        {banner && (
          <div role="alert" className={`flex flex-col gap-1 rounded-md px-3 py-2.5 text-[13px] leading-snug ${banner.tone === "bad" ? "bg-bad-soft text-bad" : "bg-warn-soft text-warn"}`}>
            <span>{banner.text}</span>
            {banner.signInLink && (
              <Link href={`/login${query}`} className="font-semibold underline">Sign in instead</Link>
            )}
          </div>
        )}

        {reg && (
          <FieldBox id={id.name} label="Name" error={errors.name}>
            <input id={id.name} name="name" autoComplete="name" maxLength={120} value={name} placeholder="Ananya Iyer"
                   aria-invalid={!!errors.name} aria-describedby={errors.name ? `${id.name}-err` : undefined}
                   onChange={(e) => edit("name", e.target.value)} className={input("name")} />
          </FieldBox>
        )}

        <FieldBox id={id.email} label="Email" error={errors.email}>
          <input id={id.email} name="email" type="email" inputMode="email" autoComplete="email" autoCapitalize="none"
                 spellCheck={false} value={email} placeholder="ananya@college.edu"
                 aria-invalid={!!errors.email} aria-describedby={errors.email ? `${id.email}-err` : undefined}
                 onChange={(e) => edit("email", e.target.value)} className={input("email")} />
        </FieldBox>

        <div className="flex flex-col gap-1.5">
          <label htmlFor={id.pw} className="text-sm font-medium">Password</label>
          <div className="relative">
            <input id={id.pw} name="password" type={showPw ? "text" : "password"} maxLength={128}
                   autoComplete={reg ? "new-password" : "current-password"} value={password}
                   aria-invalid={!!errors.password}
                   aria-describedby={[errors.password ? `${id.pw}-err` : "", reg ? id.rules : ""].join(" ").trim() || undefined}
                   onChange={(e) => edit("password", e.target.value)} className={`${input("password")} pr-16`} />
            <button type="button" onClick={() => setShowPw(!showPw)} aria-label={showPw ? "Hide password" : "Show password"}
                    className="absolute right-1 top-1 h-8 rounded px-2.5 text-xs font-medium text-muted hover:text-ink">
              {showPw ? "Hide" : "Show"}
            </button>
          </div>
          {errors.password && <span id={`${id.pw}-err`} className="text-xs leading-snug text-bad">{errors.password}</span>}
          {reg && (
            <div id={id.rules} data-testid="pw-rules" className="flex flex-col gap-1.5 pt-0.5">
              {passwordChecks(password).map((c) => {
                const fail = !c.ok && tried && password.length > 0;
                return (
                  <div key={c.label} className={`flex items-center gap-2 text-xs ${c.ok ? "text-ink" : fail ? "text-bad" : "text-muted"}`}>
                    <span aria-hidden
                          className={`flex h-3.5 w-3.5 flex-none items-center justify-center rounded-full border-[1.5px] text-[9px] font-bold leading-none text-on-accent ${c.ok ? "border-accent bg-accent" : fail ? "border-bad" : "border-line"}`}>
                      {c.ok ? "✓" : ""}
                    </span>
                    <span>{c.label}</span>
                    <span className="sr-only">{c.ok ? "(met)" : "(not met)"}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <button type="submit" disabled={busy}
                className={`mt-1 h-11 rounded-lg bg-accent text-sm font-medium text-on-accent ${busy ? "cursor-default opacity-70" : ""}`}>
          {busy ? (reg ? "Creating account…" : "Signing in…") : reg ? "Create account" : "Sign in"}
        </button>
      </form>

      <div className="flex justify-center gap-1.5 text-[13px] text-muted">
        <span>{reg ? "Already registered?" : "No account yet?"}</span>
        <Link href={`${reg ? "/login" : "/register"}${query}`} className="font-medium text-accent hover:text-ink hover:underline">
          {reg ? "Sign in" : "Create one"}
        </Link>
      </div>
    </div>
  );

  if (LAYOUT === "centered") return <div className="mx-auto w-full max-w-[432px] pt-6 sm:pt-10">{form}</div>;
  return (
    <div className="mx-auto grid w-full max-w-[960px] items-center gap-12 pt-6 sm:pt-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,384px)]">
      <ValuePanel />
      {form}
    </div>
  );
}

function FieldBox({ id, label, error, children }: { id: string; label: string; error?: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} className="text-sm font-medium">{label}</label>
      {children}
      {error && <span id={`${id}-err`} className="text-xs text-bad">{error}</span>}
    </div>
  );
}

function ValuePanel() {
  const points = [
    ["Plans in under a second", "Move a slider and the plan recomputes."],
    ["Proven optimal", "Every plan is checked against simple rules of thumb."],
    ["Runs offline", "Your cohort data stays on your institution's server."],
  ];
  return (
    <div className="hidden max-w-[400px] flex-col gap-7 lg:flex">
      <div className="flex flex-col gap-2">
        <span className="text-xs font-medium text-accent">For deans, training heads and programme managers</span>
        <span className="text-[26px] font-semibold leading-tight tracking-tight">Decide what to teach next, with a plan you can defend.</span>
      </div>
      <div className="flex flex-col border-t border-line">
        {points.map(([t, d]) => (
          <div key={t} className="flex flex-col gap-0.5 border-b border-line py-3.5">
            <span className="text-sm font-medium">{t}</span>
            <span className="text-[13px] leading-snug text-muted">{d}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function capitalise(s: string): string {
  const t = s.trim();
  return t ? t[0].toUpperCase() + t.slice(1) + (/[.!?]$/.test(t) ? "" : ".") : t;
}
