'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Logo } from './logo';
import { ThemeToggle } from './theme-toggle';
import {
  LayoutDashboard,
  Swords,
  Shield,
  Activity,
  Brain,
  Settings,
  ChevronDown,
  Bell,
  User,
  Bug,
  FileCode,
  Search,
  Code,
  GitBranch,
  Monitor,
  Microscope,
  Crosshair
} from 'lucide-react';
import { useState, useEffect } from 'react';

// Navigation configuration for YUGMASTRA platform

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  {
    name: 'Live Simulations',
    icon: Swords,
    hasLive: true,
    children: [
      { name: 'Attack Simulator', href: '/attack-simulator', hasLive: true },
      { name: 'Live Battle', href: '/live-battle', hasLive: true },
      { name: 'Cyber Range', href: '/cyber-range', hasLive: true },
    ]
  },
  {
    name: 'AI & Analytics',
    icon: Brain,
    hasLive: true,
    children: [
      { name: 'AI Assistant', href: '/ai-assistant', hasLive: true },
      { name: 'Threat Hunting', href: '/threat-hunting', hasLive: true, icon: Search },
      { name: 'Code Review', href: '/code-review', hasLive: true, icon: Code },
      { name: 'Incident Response', href: '/incident-response', hasLive: true, icon: Shield },
      { name: 'Zero-Day Discovery', href: '/zero-day', hasLive: true, icon: Bug },
      { name: 'SIEM Rules', href: '/siem-rules', hasLive: true, icon: FileCode },
      { name: 'Model Training', href: '/model-training', hasLive: true },
      { name: 'ML Analytics', href: '/analytics', hasLive: true },
    ]
  },
  {
    name: 'Security',
    icon: Shield,
    children: [
      { name: 'Threat Intel', href: '/threat-intelligence' },
      { name: 'Attacks', href: '/attacks' },
      { name: 'Defenses', href: '/defenses' },
    ]
  },
  {
    name: 'Research',
    icon: Activity,
    children: [
      { name: 'Evolution', href: '/evolution' },
      { name: 'Knowledge Graph', href: '/knowledge-graph' },
      { name: 'Recommendations', href: '/recommendations' },
    ]
  },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export function Navbar() {
  const pathname = usePathname();
  const [openDropdown, setOpenDropdown] = useState<string | null>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (!target.closest('.navbar-dropdown')) {
        setOpenDropdown(null);
      }
    };

    if (openDropdown) {
      document.addEventListener('click', handleClickOutside);
      return () => document.removeEventListener('click', handleClickOutside);
    }
  }, [openDropdown]);

  const isActive = (item: any) => {
    if (item.href) {
      return pathname === item.href || pathname?.startsWith(item.href + '/');
    }
    if (item.children) {
      return item.children.some((child: any) => pathname === child.href || pathname?.startsWith(child.href + '/'));
    }
    return false;
  };

  return (
    <nav className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between px-4">
        {/* Left: Logo Only */}
        <Link href="/" className="flex-shrink-0">
          <Logo size="md" />
        </Link>

        {/* Center: Navigation Tabs */}
        <div className="flex flex-1 items-center justify-center">
          <div className="flex gap-1">
            {navigation.map((item) => {
              const active = isActive(item);

              if (item.children) {
                return (
                  <div
                    key={item.name}
                    className="relative navbar-dropdown"
                  >
                    <button
                      onClick={() => setOpenDropdown(openDropdown === item.name ? null : item.name)}
                      className={`flex items-center gap-1.5 text-sm font-medium transition-all hover:text-primary px-3 py-2 rounded-md whitespace-nowrap ${
                        active
                          ? 'text-primary bg-primary/10'
                          : 'text-muted-foreground hover:bg-accent'
                      }`}
                    >
                      <item.icon className="h-4 w-4" />
                      {item.name}
                      {item.hasLive && (
                        <span className="ml-1 px-1 py-0.5 text-[9px] font-bold bg-red-500 text-white rounded animate-pulse">
                          LIVE
                        </span>
                      )}
                      <ChevronDown className={`h-3 w-3 transition-transform ${openDropdown === item.name ? 'rotate-180' : ''}`} />
                    </button>

                    {openDropdown === item.name && (
                      <div className="absolute top-full left-0 mt-1 min-w-[200px] bg-popover border rounded-md shadow-lg py-1 z-50">
                        {item.children.map((child) => (
                          <Link
                            key={child.href}
                            href={child.href}
                            onClick={() => setOpenDropdown(null)}
                            className={`flex items-center justify-between px-4 py-2 text-sm transition-colors hover:bg-accent ${
                              pathname === child.href || pathname?.startsWith(child.href + '/')
                                ? 'text-primary bg-accent'
                                : 'text-foreground'
                            }`}
                          >
                            <span>{child.name}</span>
                            {child.hasLive && (
                              <span className="px-1.5 py-0.5 text-[9px] font-bold bg-red-500 text-white rounded animate-pulse">
                                LIVE
                              </span>
                            )}
                          </Link>
                        ))}
                      </div>
                    )}
                  </div>
                );
              }

              return (
                <Link
                  key={item.name}
                  href={item.href || '#'}
                  className={`flex items-center gap-1.5 text-sm font-medium transition-all hover:text-primary px-3 py-2 rounded-md whitespace-nowrap ${
                    active
                      ? 'text-primary bg-primary/10'
                      : 'text-muted-foreground hover:bg-accent'
                  }`}
                >
                  <item.icon className="h-4 w-4" />
                  {item.name}
                  {item.hasLive && (
                    <span className="ml-1 px-1 py-0.5 text-[9px] font-bold bg-red-500 text-white rounded">
                      LIVE
                    </span>
                  )}
                </Link>
              );
            })}
          </div>
        </div>

        {/* Right: Profile, Notification and Theme Toggle */}
        <div className="flex items-center gap-3 flex-shrink-0">
          <div className="flex items-center gap-3 bg-card/50 backdrop-blur-lg rounded-lg px-4 py-2 border">
            <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center">
              <User className="h-4 w-4 text-primary-foreground" />
            </div>
            <div className="hidden md:block">
              <p className="text-sm font-semibold">Preet Raval</p>
              <p className="text-xs text-muted-foreground">System Owner</p>
            </div>
          </div>
          <button className="relative rounded-md p-2 text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors">
            <Bell className="h-5 w-5" />
            <span className="absolute top-1 right-1 h-2 w-2 bg-red-500 rounded-full"></span>
          </button>
          <ThemeToggle />
        </div>
      </div>
    </nav>
  );
}
