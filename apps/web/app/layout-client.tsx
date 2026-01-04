'use client';

import { usePathname } from 'next/navigation';
import { ThemeProvider } from '@/contexts/theme-context';
import { SocketProvider } from '@/contexts/socket-context';
import { TooltipProvider } from '@/components/ui/tooltip';
import { ProfessionalNavbar } from '@/components/professional-navbar';
import { MobileBottomNav } from '@/components/mobile-bottom-nav';
import { OnboardingTour } from '@/components/onboarding-tour';
import { Breadcrumb } from '@/components/ui/breadcrumb';
import { KeyboardShortcuts } from '@/components/keyboard-shortcuts';

export function LayoutClient({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isAuthPage = pathname?.startsWith('/auth');
  const isHomePage = pathname === '/';

  if (isAuthPage) {
    return (
      <ThemeProvider>
        <TooltipProvider>
          {children}
        </TooltipProvider>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider>
      <TooltipProvider>
        <SocketProvider>
          <ProfessionalNavbar />
          <KeyboardShortcuts />
          <OnboardingTour />
          {!isHomePage && (
            <div className="container-responsive pt-4 sm:pt-6">
              <Breadcrumb />
            </div>
          )}
          <div className="pb-20 lg:pb-0">
            {children}
          </div>
          <MobileBottomNav />
        </SocketProvider>
      </TooltipProvider>
    </ThemeProvider>
  );
}
