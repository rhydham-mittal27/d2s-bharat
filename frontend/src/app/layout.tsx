import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/nav";
import { AuthGate, AuthProvider } from "@/lib/auth";
import { StoreProvider } from "@/lib/store";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "D2S Bharat",
  description: "Turn labour-market demand into the training plan an institution can actually run.",
};

const THEME_BOOT =
  "try{var t=localStorage.getItem('d2s.theme');if(t==='light'||t==='dark')document.documentElement.dataset.theme=t}catch(e){}";

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" suppressHydrationWarning className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <head>
        {/* apply the saved light/dark choice before first paint (no flash) */}
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT }} />
      </head>
      <body className="min-h-full flex flex-col font-sans">
        <AuthProvider>
          <StoreProvider>
            <Nav />
            <main className="mx-auto w-full max-w-[1150px] flex-1 px-4 pb-12 pt-6 sm:px-6">
              <AuthGate>{children}</AuthGate>
            </main>
          </StoreProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
