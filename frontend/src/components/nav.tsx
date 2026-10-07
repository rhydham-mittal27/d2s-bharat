"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { useStore } from "@/lib/store";

const LINKS = [
  { href: "/", label: "Plan" },
  { href: "/cohort", label: "Cohort" },
  { href: "/copilot", label: "Copilot" },
  { href: "/what-if", label: "What if" },
  { href: "/brief", label: "Brief" },
  { href: "/explain", label: "Explain" },
  { href: "/market", label: "Market evidence" },
] as const;

export function Nav() {
  const path = usePathname();
  const { cohort } = useStore();
  const { user, logout } = useAuth();
  return (
    <header className="border-b border-line bg-surface">
      <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
        <Link href="/" className="font-semibold tracking-tight">
          D2S <span className="text-accent">Bharat</span>
        </Link>
        {user && <nav className="flex flex-wrap gap-1 text-sm">
          {LINKS.map((l) => {
            const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded-md px-3 py-1.5 ${active ? "bg-accent-soft text-accent font-medium" : "text-muted hover:text-ink"}`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>}
        {user && (
          <div className="ml-auto flex items-center gap-4 text-xs text-muted">
            <Link href="/cohort" className="hover:text-ink" title="Change cohort">
              Cohort: <span className="text-ink">{cohort.name}</span> · {cohort.n_learners} learners
            </Link>
            <span className="hidden sm:inline" title={user.email}>{user.name}</span>
            <button type="button" onClick={() => logout()} className="rounded-md border border-line px-2 py-1 hover:text-ink">
              Sign out
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
