import Providers from '@/components/Providers';
import Header from '@/components/layout/Header';
import Sidebar from '@/components/layout/Sidebar';
import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  display: 'swap', // Improve font loading performance
  preload: true,
});

export const metadata: Metadata = {
  title: 'RestaurantAI - Smart Restaurant Management',
  description: 'AI-powered restaurant management platform with intelligent recommendations, demand forecasting, and customer analytics',
  keywords: ['restaurant', 'POS', 'AI', 'management', 'analytics'],
  authors: [{ name: 'RestaurantAI' }],
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#3b82f6',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {process.env.NEXT_PUBLIC_API_URL && (
          <>
            <link rel="preconnect" href={new URL(process.env.NEXT_PUBLIC_API_URL).origin} />
            <link rel="dns-prefetch" href={new URL(process.env.NEXT_PUBLIC_API_URL).origin} />
          </>
        )}
      </head>
      <body className={inter.className}>
        <Providers>
          <div className="flex h-screen bg-slate-50 dark:bg-slate-900">
            <Sidebar />
            <div className="flex-1 flex flex-col overflow-hidden">
              <Header />
              <main className="flex-1 overflow-y-auto p-6">
                {children}
              </main>
            </div>
          </div>
        </Providers>
      </body>
    </html>
  );
}
