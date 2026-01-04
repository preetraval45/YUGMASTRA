'use client';

import { usePathname } from 'next/navigation';
import Link from 'next/link';
import {
  Home,
  Swords,
  Brain,
  Shield,
  Activity
} from 'lucide-react';

const navItems = [
  {
    name: 'Dashboard',
    href: '/dashboard',
    icon: Home,
  },
  {
    name: 'Battle',
    href: '/live-battle',
    icon: Swords,
  },
  {
    name: 'AI Tools',
    href: '/ai-assistant',
    icon: Brain,
  },
  {
    name: 'Security',
    href: '/threat-intelligence',
    icon: Shield,
  },
  {
    name: 'Research',
    href: '/evolution',
    icon: Activity,
  },
];

export function MobileBottomNav() {
  const pathname = usePathname();

  // Don't show on auth pages or home page
  if (pathname?.startsWith('/auth') || pathname === '/') {
    return null;
  }

  return (
    <nav className="lg:hidden fixed bottom-0 left-0 right-0 z-50 bg-background/98 backdrop-blur-xl border-t border-border/40 shadow-2xl dark:shadow-primary/10">
      <div className="flex items-center justify-around px-2 py-2 safe-area-inset-bottom">
        {navItems.map((item) => {
          const isActive = pathname === item.href || pathname?.startsWith(item.href + '/');

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex flex-col items-center justify-center gap-1 px-3 py-2 rounded-xl transition-all duration-200 min-w-[60px] min-h-[60px] ${
                isActive
                  ? 'bg-primary/15 dark:bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:text-foreground hover:bg-accent/50'
              }`}
            >
              <item.icon className={`h-6 w-6 ${isActive ? 'stroke-[2.5]' : 'stroke-2'}`} />
              <span className={`text-[10px] font-semibold ${isActive ? 'text-primary' : 'text-muted-foreground'}`}>
                {item.name}
              </span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
