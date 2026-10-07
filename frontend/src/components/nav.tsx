"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";
import { useStore } from "@/lib/store";

const LINKS = [
  { href: "/", label: "Plan" },
  { href: "/cohort", label: "Cohort" },
  { href: "/copilot", label: "Copilot" },
  { href: "/what-if", label: "What-if" },
  { href: "/brief", label: "Brief" },
  { href: "/explain", label: "Explain" },
  { href: "/market", label: "Market" },
] as const;

export function Nav() {
  const path = usePathname();
  const { cohort } = useStore();
  const { user, logout } = useAuth();
  return (
    <header className="border-b border-line bg-surface">
      <div className="mx-auto flex min-h-[52px] w-full max-w-[1150px] flex-wrap items-center gap-x-5 gap-y-2 px-4 py-2.5 sm:px-6">
        <Link href="/" className="flex items-center gap-2 text-sm font-semibold tracking-tight text-ink">
          <span className="h-2.5 w-2.5 rounded-[2px] bg-accent" aria-hidden />
          D2S Bharat
        </Link>
        {user && (
          <nav className="order-3 flex w-full flex-wrap gap-0.5 md:order-none md:w-auto md:flex-1">
            {LINKS.map((l) => {
              const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
              return (
                <Link key={l.href} href={l.href} aria-current={active ? "page" : undefined}
                      className={`rounded-md px-2.5 py-1.5 text-[13px] ${active ? "bg-track font-medium text-ink" : "text-muted hover:text-ink"}`}>
                  {l.label}
                </Link>
              );
            })}
          </nav>
        )}
        <div className="ml-auto flex items-center gap-3 text-xs text-muted">
          {user && (
            <>
              <Link href="/cohort" title="Change cohort"
                    className="hidden whitespace-nowrap rounded-full border border-line px-2 py-0.5 text-muted hover:text-ink md:inline">
                {cohort.name}
              </Link>
              <span className="hidden whitespace-nowrap text-ink md:inline" title={user.email}>{user.name}</span>
              <button type="button" onClick={() => logout()} className="text-muted hover:text-ink">Sign out</button>
            </>
          )}
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

/** Light/dark switch; remembers the choice (applied before paint by the script in layout.tsx). */
function ThemeToggle() {
  const [dark, setDark] = useState<boolean | null>(null);

  useEffect(() => {
    const set = document.documentElement.dataset.theme;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setDark(set ? set === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches);
  }, []);

  function toggle() {
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("d2s.theme", next);
    } catch {}
    setDark(!dark);
  }

  if (dark === null) return <span className="inline-block w-[62px]" aria-hidden />;
  return (
    <button type="button" onClick={toggle} aria-label={`Switch to ${dark ? "light" : "dark"} mode`}
            className="flex items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-xs text-muted hover:text-ink">
      <span className={`h-2.5 w-2.5 rounded-full border-[1.5px] border-current ${dark ? "" : "bg-current"}`} />
      {dark ? "Light" : "Dark"}
    </button>
  );
}
