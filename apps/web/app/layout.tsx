import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { ThemeProvider } from '@/components/theme-provider';
import { Toaster } from '@/components/ui/toaster';
import { LayoutClient } from './layout-client';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'YUGMĀSTRA - Autonomous Cyber Defense',
  description: 'Autonomous Adversary-Defender Co-Evolution Platform',
  icons: {
    icon: '/favicon.svg',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          <LayoutClient>{children}</LayoutClient>
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
