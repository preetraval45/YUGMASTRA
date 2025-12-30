'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Home, TrendingUp, Swords, Shield, Network, Settings, LogOut, Zap, Lightbulb, Menu, X, User, Sparkles, Activity, Target, Brain, Server, BarChart3 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Logo } from '@/components/logo';

const navItems = [
  { href: '/dashboard', icon: Home, label: 'Dashboard' },
  { href: '/threat-intelligence', icon: Activity, label: 'Threat Intel', highlight: true },
  { href: '/attack-simulator', icon: Target, label: 'Attack Sim', highlight: true },
  { href: '/ai-assistant', icon: Sparkles, label: 'AI Assistant' },
  { href: '/live-battle', icon: Zap, label: 'Live Battle' },
  { href: '/cyber-range', icon: Server, label: 'Cyber Range', highlight: true },
  { href: '/model-training', icon: Brain, label: 'Model Training', highlight: true },
  { href: '/analytics', icon: BarChart3, label: 'ML Analytics', highlight: true },
  { href: '/recommendations', icon: Lightbulb, label: 'Recommendations' },
  { href: '/evolution', icon: TrendingUp, label: 'Evolution' },
  { href: '/attacks', icon: Swords, label: 'Attacks' },
  { href: '/defenses', icon: Shield, label: 'Defenses' },
  { href: '/knowledge-graph', icon: Network, label: 'Knowledge Graph' },
  { href: '/settings', icon: Settings, label: 'Settings' },
];

export function Sidebar() {
  const pathname = usePathname();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
    <>
      {/* Mobile Menu Button */}
      <button
        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 rounded-md bg-card border hover:bg-accent"
      >
        {isMobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </button>

      {/* Mobile Overlay */}
      {isMobileMenuOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 z-40"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={cn(
        "flex h-screen w-64 flex-col bg-card border-r fixed lg:static z-40 transition-transform duration-300",
        isMobileMenuOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
      )}>
      <div className="flex h-16 items-center border-b px-4">
        <Link href="/dashboard" className="flex items-center w-full">
          <Logo size="sm" showText={true} />
        </Link>
      </div>

      <nav className="flex-1 space-y-1 p-4">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          const isHighlight = 'highlight' in item && item.highlight;
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={() => setIsMobileMenuOpen(false)}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : isHighlight
                  ? 'text-amber-700 dark:text-yellow-400 hover:bg-amber-100 dark:hover:bg-yellow-500/20 border border-amber-300 dark:border-yellow-500/30 animate-pulse'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              )}
            >
              <item.icon className="h-5 w-5" />
              {item.label}
              {isHighlight && !isActive && (
                <span className="ml-auto text-xs bg-amber-100 dark:bg-yellow-500/20 text-amber-800 dark:text-yellow-400 px-2 py-0.5 rounded font-semibold">
                  LIVE
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Profile and Logout Section */}
      <div className="border-t p-4 space-y-2">
        {/* User Profile */}
        <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-accent/50">
          <div className="h-10 w-10 rounded-full bg-primary flex items-center justify-center">
            <User className="h-5 w-5 text-primary-foreground" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold truncate">Preet Raval</p>
            <p className="text-xs text-muted-foreground truncate">preetraval45@gmail.com</p>
          </div>
        </div>

        {/* Logout Button */}
        <button
          onClick={async () => {
            if (confirm('Are you sure you want to logout?')) {
              try {
                await fetch('/api/auth/logout', { method: 'POST' });
                localStorage.removeItem('yugmastra_user');
                window.location.href = '/auth/login';
              } catch (error) {
                console.error('Logout error:', error);
                localStorage.removeItem('yugmastra_user');
                window.location.href = '/auth/login';
              }
            }
          }}
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-red-600 dark:text-red-400 transition-colors hover:bg-red-50 dark:hover:bg-red-950"
        >
          <LogOut className="h-5 w-5" />
          Logout
        </button>
      </div>
      </div>
    </>
  );
}
