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

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col font-sans">
        <AuthProvider>
          <StoreProvider>
            <Nav />
            <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6">
              <AuthGate>{children}</AuthGate>
            </main>
          </StoreProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
