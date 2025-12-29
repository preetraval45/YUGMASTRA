'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Home, TrendingUp, Swords, Shield, Network, Settings, LogOut, Zap, Lightbulb } from 'lucide-react';
import { cn } from '@/lib/utils';

const navItems = [
  { href: '/dashboard', icon: Home, label: 'Dashboard' },
  { href: '/live-battle', icon: Zap, label: 'Live Battle', highlight: true },
  { href: '/recommendations', icon: Lightbulb, label: 'Recommendations' },
  { href: '/evolution', icon: TrendingUp, label: 'Evolution' },
  { href: '/attacks', icon: Swords, label: 'Attacks' },
  { href: '/defenses', icon: Shield, label: 'Defenses' },
  { href: '/knowledge-graph', icon: Network, label: 'Knowledge Graph' },
  { href: '/settings', icon: Settings, label: 'Settings' },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="flex h-screen w-64 flex-col bg-card border-r">
      <div className="flex h-16 items-center border-b px-6">
        <h1 className="text-xl font-bold">YUGMĀSTRA</h1>
      </div>

      <nav className="flex-1 space-y-1 p-4">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          const isHighlight = 'highlight' in item && item.highlight;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : isHighlight
                  ? 'text-yellow-400 hover:bg-yellow-500/20 border border-yellow-500/30 animate-pulse'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              )}
            >
              <item.icon className="h-5 w-5" />
              {item.label}
              {isHighlight && !isActive && (
                <span className="ml-auto text-xs bg-yellow-500/20 text-yellow-400 px-2 py-0.5 rounded">
                  LIVE
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      <div className="border-t p-4">
        <button className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground">
          <LogOut className="h-5 w-5" />
          Logout
        </button>
      </div>
    </div>
  );
}
