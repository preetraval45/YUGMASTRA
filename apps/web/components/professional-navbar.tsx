'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { ThemeToggle } from './theme-toggle';
import { CommandPalette } from './command-palette';
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
  Menu,
  LogOut,
  UserCircle
} from 'lucide-react';
import { useState, useEffect } from 'react';

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
      { name: 'Threat Hunting', href: '/threat-hunting', hasLive: true },
      { name: 'Code Review', href: '/code-review', hasLive: true },
      { name: 'Incident Response', href: '/incident-response', hasLive: true },
      { name: 'Zero-Day Discovery', href: '/zero-day', hasLive: true },
      { name: 'SIEM Rules', href: '/siem-rules', hasLive: true },
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

export function ProfessionalNavbar() {
  const pathname = usePathname();
  const router = useRouter();
  const [openDropdown, setOpenDropdown] = useState<string | null>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [notificationOpen, setNotificationOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (!target.closest('.navbar-dropdown') && !target.closest('.notification-dropdown') && !target.closest('.profile-dropdown')) {
        setOpenDropdown(null);
        setNotificationOpen(false);
        setProfileOpen(false);
      }
    };

    if (openDropdown || notificationOpen || profileOpen) {
      document.addEventListener('click', handleClickOutside);
      return () => document.removeEventListener('click', handleClickOutside);
    }
  }, [openDropdown, notificationOpen, profileOpen]);

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
    <nav className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/98 backdrop-blur-xl supports-[backdrop-filter]:bg-background/95 shadow-lg dark:shadow-primary/5">
      <div className="w-full px-3 sm:px-4 md:px-6 lg:px-8">
        <div className="flex h-16 sm:h-20 lg:h-24 items-center justify-between gap-2 sm:gap-4 lg:gap-8">
          {/* LEFT: Logo - Full SVG Logo */}
          <div className="flex-shrink-0">
            <Link href="/" className="block group">
              <img
                src="/logo-full-adaptive.svg"
                alt="YUGMĀSTRA"
                className="h-10 sm:h-12 md:h-14 lg:h-16 w-auto transition-all group-hover:scale-105"
              />
            </Link>
          </div>

          {/* CENTER: Navigation Tabs - Circular pills */}
          <div className="hidden lg:flex flex-1 items-center justify-center">
            <div className="flex items-center gap-2 bg-muted/40 dark:bg-muted/20 rounded-full px-3 py-2 shadow-inner border border-border/40 dark:border-border/30">
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
                        className={`flex items-center gap-2.5 px-4 py-2.5 rounded-full text-sm font-semibold transition-all duration-200 whitespace-nowrap ${
                          active
                            ? 'bg-primary text-primary-foreground shadow-md dark:shadow-primary/20'
                            : 'hover:bg-accent/90 dark:hover:bg-accent/70 text-muted-foreground hover:text-foreground'
                        }`}
                      >
                        <item.icon className="h-4 w-4" />
                        <span className="hidden xl:inline">{item.name}</span>
                        <ChevronDown className={`h-3.5 w-3.5 transition-transform duration-200 ${openDropdown === item.name ? 'rotate-180' : ''}`} />
                      </button>

                      {openDropdown === item.name && (
                        <div className="absolute top-full left-0 mt-2 min-w-[240px] bg-white dark:bg-card/95 backdrop-blur-xl border-2 border-gray-200 dark:border-border/40 rounded-2xl shadow-2xl dark:shadow-primary/10 py-2 z-50 animate-in slide-in-from-top-2 duration-200">
                          {item.children.map((child) => (
                            <Link
                              key={child.href}
                              href={child.href}
                              onClick={() => setOpenDropdown(null)}
                              className={`flex items-center justify-between px-4 py-3 mx-2 rounded-xl text-base font-medium transition-all duration-150 ${
                                pathname === child.href || pathname?.startsWith(child.href + '/')
                                  ? 'bg-primary/15 dark:bg-primary/10 text-primary dark:text-primary font-semibold'
                                  : 'hover:bg-gray-100 dark:hover:bg-accent/50 text-gray-900 dark:text-foreground/90 hover:text-gray-950 dark:hover:text-foreground'
                              }`}
                            >
                              <span>{child.name}</span>
                              {child.hasLive && (
                                <span className="px-2 py-0.5 text-[10px] font-bold bg-red-500 text-white rounded-full animate-pulse shadow-sm">
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
                    className={`flex items-center gap-2.5 px-4 py-2.5 rounded-full text-sm font-semibold transition-all duration-200 whitespace-nowrap ${
                      active
                        ? 'bg-primary text-primary-foreground shadow-md dark:shadow-primary/20'
                        : 'hover:bg-accent/90 dark:hover:bg-accent/70 text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <item.icon className="h-4 w-4" />
                    <span className="hidden xl:inline">{item.name}</span>
                  </Link>
                );
              })}
            </div>
          </div>

          {/* RIGHT: Search, Profile, Notifications, Theme - Bigger */}
          <div className="flex items-center gap-1 sm:gap-2 lg:gap-3 flex-shrink-0">
            {/* Command Palette Search */}
            <CommandPalette />

            {/* Notifications */}
            <div className="relative notification-dropdown">
              <button
                onClick={() => setNotificationOpen(!notificationOpen)}
                className="relative flex items-center justify-center w-9 h-9 sm:w-10 sm:h-10 lg:w-11 lg:h-11 rounded-full hover:bg-accent/90 dark:hover:bg-accent/70 text-muted-foreground hover:text-foreground transition-all"
              >
                <Bell className="h-4 w-4 sm:h-5 sm:w-5" />
                <span className="absolute top-1 right-1 sm:top-1.5 sm:right-1.5 h-1.5 w-1.5 sm:h-2 sm:w-2 bg-red-500 rounded-full animate-pulse shadow-sm"></span>
              </button>

              {notificationOpen && (
                <div className="absolute top-full right-0 mt-2 w-80 bg-white dark:bg-card/95 backdrop-blur-xl border-2 border-gray-200 dark:border-border/40 rounded-2xl shadow-2xl dark:shadow-primary/10 py-2 z-50 animate-in slide-in-from-top-2 duration-200">
                  <div className="px-4 py-2 border-b border-gray-200 dark:border-border/40">
                    <h3 className="font-semibold text-sm text-gray-900 dark:text-foreground">Notifications</h3>
                  </div>
                  <div className="max-h-96 overflow-y-auto">
                    <div className="px-2 py-2 space-y-1">
                      <div className="px-3 py-2.5 rounded-xl hover:bg-gray-100 dark:hover:bg-accent/30 cursor-pointer transition-all">
                        <div className="flex items-start gap-3">
                          <div className="h-2 w-2 bg-red-500 rounded-full mt-1.5 flex-shrink-0"></div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 dark:text-foreground leading-[1.4]">New attack detected</p>
                            <p className="text-xs text-gray-600 dark:text-muted-foreground leading-[1.4] mt-0.5">SQL injection attempt blocked</p>
                            <p className="text-[10px] text-gray-500 dark:text-muted-foreground/70 leading-[1.4] mt-1">2 minutes ago</p>
                          </div>
                        </div>
                      </div>
                      <div className="px-3 py-2.5 rounded-xl hover:bg-gray-100 dark:hover:bg-accent/30 cursor-pointer transition-all">
                        <div className="flex items-start gap-3">
                          <div className="h-2 w-2 bg-blue-500 rounded-full mt-1.5 flex-shrink-0"></div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 dark:text-foreground leading-[1.4]">Model training complete</p>
                            <p className="text-xs text-gray-600 dark:text-muted-foreground leading-[1.4] mt-0.5">Red Team AI v2.3 ready</p>
                            <p className="text-[10px] text-gray-500 dark:text-muted-foreground/70 leading-[1.4] mt-1">1 hour ago</p>
                          </div>
                        </div>
                      </div>
                      <div className="px-3 py-2.5 rounded-xl hover:bg-gray-100 dark:hover:bg-accent/30 cursor-pointer transition-all">
                        <div className="flex items-start gap-3">
                          <div className="h-2 w-2 bg-green-500 rounded-full mt-1.5 flex-shrink-0"></div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 dark:text-foreground leading-[1.4]">Defense updated</p>
                            <p className="text-xs text-gray-600 dark:text-muted-foreground leading-[1.4] mt-0.5">New firewall rules applied</p>
                            <p className="text-[10px] text-gray-500 dark:text-muted-foreground/70 leading-[1.4] mt-1">3 hours ago</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="px-4 py-2 border-t border-gray-200 dark:border-border/40">
                    <Link
                      href="/notifications"
                      onClick={() => setNotificationOpen(false)}
                      className="block w-full text-center text-xs font-semibold text-primary hover:text-primary/80 transition-colors py-1"
                    >
                      View all notifications
                    </Link>
                  </div>
                </div>
              )}
            </div>

            {/* Theme Toggle */}
            <div className="scale-95">
              <ThemeToggle />
            </div>

            {/* Profile */}
            <div className="relative profile-dropdown">
              <button
                onClick={() => setProfileOpen(!profileOpen)}
                className="flex items-center gap-2.5 px-3 py-2 rounded-full hover:bg-accent/90 dark:hover:bg-accent/70 transition-all group"
              >
                <div className="h-9 w-9 rounded-full bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-md group-hover:shadow-lg transition-shadow">
                  <User className="h-4 w-4 text-white" />
                </div>
                <span className="hidden md:inline text-sm font-semibold text-foreground leading-[1.4]">Preet</span>
                <ChevronDown className={`hidden md:inline h-3.5 w-3.5 transition-transform duration-200 ${profileOpen ? 'rotate-180' : ''}`} />
              </button>

              {profileOpen && (
                <div className="absolute top-full right-0 mt-2 w-56 bg-white dark:bg-card/95 backdrop-blur-xl border-2 border-gray-200 dark:border-border/40 rounded-2xl shadow-2xl dark:shadow-primary/10 py-2 z-50 animate-in slide-in-from-top-2 duration-200">
                  <Link
                    href="/profile"
                    onClick={() => setProfileOpen(false)}
                    className="flex items-center gap-3 px-4 py-3 mx-2 rounded-xl text-base font-medium transition-all duration-150 hover:bg-gray-100 dark:hover:bg-accent/50 text-gray-900 dark:text-foreground"
                  >
                    <UserCircle className="h-5 w-5" />
                    <span>Profile</span>
                  </Link>
                  <div className="mx-2 my-1 border-t border-gray-200 dark:border-border/40"></div>
                  <button
                    onClick={async () => {
                      setProfileOpen(false);
                      try {
                        // Call logout API
                        await fetch('/api/auth/logout', { method: 'POST' });
                        // Clear any local storage
                        localStorage.clear();
                        sessionStorage.clear();
                        // Redirect to auth page
                        router.push('/auth');
                      } catch (error) {
                        console.error('Logout error:', error);
                        // Redirect anyway
                        router.push('/auth');
                      }
                    }}
                    className="w-full flex items-center gap-3 px-4 py-3 mx-2 rounded-xl text-base font-medium transition-all duration-150 hover:bg-red-50 dark:hover:bg-red-950/20 text-red-600 dark:text-red-400"
                  >
                    <LogOut className="h-5 w-5" />
                    <span>Logout</span>
                  </button>
                </div>
              )}
            </div>

            {/* Mobile Menu */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden flex items-center justify-center w-11 h-11 rounded-full hover:bg-accent/90 dark:hover:bg-accent/70 text-foreground transition-all"
            >
              <Menu className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="lg:hidden border-t bg-background/95 backdrop-blur-xl px-4 py-4 animate-in slide-in-from-top-2 duration-200">
          <div className="space-y-2">
            {navigation.map((item) => (
              <div key={item.name}>
                {item.children ? (
                  <div>
                    <button
                      onClick={() => setOpenDropdown(openDropdown === item.name ? null : item.name)}
                      className="w-full flex items-center justify-between px-4 py-3 rounded-xl hover:bg-accent/50 transition-all"
                    >
                      <div className="flex items-center gap-3">
                        <item.icon className="h-5 w-5" />
                        <span className="font-medium">{item.name}</span>
                      </div>
                      <ChevronDown className={`h-4 w-4 transition-transform ${openDropdown === item.name ? 'rotate-180' : ''}`} />
                    </button>
                    {openDropdown === item.name && (
                      <div className="ml-4 mt-2 space-y-1">
                        {item.children.map((child) => (
                          <Link
                            key={child.href}
                            href={child.href}
                            onClick={() => setMobileMenuOpen(false)}
                            className="block px-4 py-2 rounded-lg hover:bg-accent/50 text-sm"
                          >
                            {child.name}
                          </Link>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <Link
                    href={item.href || '#'}
                    onClick={() => setMobileMenuOpen(false)}
                    className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-accent/50 transition-all"
                  >
                    <item.icon className="h-5 w-5" />
                    <span className="font-medium">{item.name}</span>
                  </Link>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </nav>
  );
}
