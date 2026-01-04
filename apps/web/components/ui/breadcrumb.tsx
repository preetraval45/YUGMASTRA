'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ChevronRight, Home } from 'lucide-react';
import { cn } from '@/lib/utils';

export function Breadcrumb() {
  const pathname = usePathname();

  if (!pathname || pathname === '/' || pathname === '/auth') return null;

  const paths = pathname.split('/').filter(Boolean);

  const getBreadcrumbName = (path: string) => {
    const names: Record<string, string> = {
      'dashboard': 'Dashboard',
      'ai-assistant': 'AI Assistant',
      'threat-hunting': 'Threat Hunting',
      'code-review': 'Code Review',
      'incident-response': 'Incident Response',
      'zero-day': 'Zero-Day Discovery',
      'siem-rules': 'SIEM Rules',
      'model-training': 'Model Training',
      'analytics': 'ML Analytics',
      'attack-simulator': 'Attack Simulator',
      'live-battle': 'Live Battle',
      'cyber-range': 'Cyber Range',
      'threat-intelligence': 'Threat Intelligence',
      'attacks': 'Attacks',
      'defenses': 'Defenses',
      'evolution': 'Evolution',
      'knowledge-graph': 'Knowledge Graph',
      'recommendations': 'Recommendations',
      'settings': 'Settings',
      'profile': 'Profile',
      'notifications': 'Notifications',
    };
    return names[path] || path.charAt(0).toUpperCase() + path.slice(1);
  };

  return (
    <nav className="flex items-center space-x-1 text-sm text-muted-foreground mb-4 sm:mb-6">
      <Link
        href="/dashboard"
        className="hover:text-foreground transition-colors flex items-center"
      >
        <Home className="h-4 w-4" />
      </Link>
      {paths.map((path, index) => {
        const href = '/' + paths.slice(0, index + 1).join('/');
        const isLast = index === paths.length - 1;
        return (
          <div key={path} className="flex items-center">
            <ChevronRight className="h-4 w-4 mx-1" />
            {isLast ? (
              <span className="text-foreground font-medium">
                {getBreadcrumbName(path)}
              </span>
            ) : (
              <Link
                href={href}
                className="hover:text-foreground transition-colors"
              >
                {getBreadcrumbName(path)}
              </Link>
            )}
          </div>
        );
      })}
    </nav>
  );
}
