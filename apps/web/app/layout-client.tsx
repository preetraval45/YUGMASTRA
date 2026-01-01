'use client';

import { usePathname } from 'next/navigation';
import { ThemeProvider } from '@/contexts/theme-context';
import { SocketProvider } from '@/contexts/socket-context';
import { ProfessionalNavbar } from '@/components/professional-navbar';

export function LayoutClient({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isAuthPage = pathname?.startsWith('/auth');

  if (isAuthPage) {
    return (
      <ThemeProvider>
        {children}
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider>
      <SocketProvider>
        <ProfessionalNavbar />
        {children}
      </SocketProvider>
    </ThemeProvider>
  );
}
