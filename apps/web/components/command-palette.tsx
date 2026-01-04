'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from '@/components/ui/command';
import {
  Shield, Swords, Brain, Target, Zap, Database, Network, BarChart3,
  Search, Home, Activity, FileCode, AlertTriangle, Bug, HelpCircle,
  Settings, User, BookOpen
} from 'lucide-react';

interface CommandItem {
  title: string;
  description: string;
  href: string;
  icon: any;
  category: string;
  keywords: string[];
}

const commands: CommandItem[] = [
  // Main Pages
  { title: 'Dashboard', description: 'Command center and system overview', href: '/dashboard', icon: Home, category: 'Pages', keywords: ['home', 'main', 'overview'] },
  { title: 'Live Battle Arena', description: 'Watch AI agents compete in real-time', href: '/live-battle', icon: Target, category: 'Pages', keywords: ['battle', 'fight', 'arena', 'watch', 'live'] },
  { title: 'Evolution Timeline', description: 'View AI training progress', href: '/evolution', icon: Activity, category: 'Pages', keywords: ['evolution', 'progress', 'training', 'timeline'] },

  // AI Tools
  { title: 'Threat Hunting', description: 'Hunt for advanced threats with AI', href: '/threat-hunting', icon: Search, category: 'AI Tools', keywords: ['threat', 'hunt', 'search', 'detect', 'find'] },
  { title: 'Code Review', description: 'AI-powered vulnerability scanning', href: '/code-review', icon: FileCode, category: 'AI Tools', keywords: ['code', 'review', 'scan', 'vulnerability', 'owasp'] },
  { title: 'Incident Response', description: 'Automated incident handling', href: '/incident-response', icon: AlertTriangle, category: 'AI Tools', keywords: ['incident', 'response', 'nist', 'playbook', 'ransomware'] },
  { title: 'Zero-Day Discovery', description: 'Find unknown vulnerabilities', href: '/zero-day', icon: Bug, category: 'AI Tools', keywords: ['zero-day', 'vulnerability', 'cve', 'cvss', 'discovery'] },
  { title: 'SIEM Rules', description: 'Generate detection rules', href: '/siem-rules', icon: Shield, category: 'AI Tools', keywords: ['siem', 'rules', 'detection', 'sigma', 'splunk', 'elastic'] },

  // Platform Features
  { title: 'AI Assistant', description: 'Chat with security AI', href: '/ai-assistant', icon: Brain, category: 'Features', keywords: ['assistant', 'chat', 'ai', 'help', 'ask'] },
  { title: 'Attack Simulator', description: 'Simulate red vs blue battles', href: '/attack-simulator', icon: Swords, category: 'Features', keywords: ['attack', 'simulate', 'red team', 'blue team'] },
  { title: 'Cyber Range', description: 'Practice environment', href: '/cyber-range', icon: Zap, category: 'Features', keywords: ['range', 'practice', 'training', 'lab'] },
  { title: 'Model Training', description: 'Train AI defense models', href: '/model-training', icon: Brain, category: 'Features', keywords: ['model', 'training', 'ml', 'ai', 'machine learning'] },
  { title: 'Threat Intelligence', description: 'Latest threat feeds', href: '/threat-intelligence', icon: Database, category: 'Features', keywords: ['threat', 'intelligence', 'feed', 'ioc', 'ttp'] },
  { title: 'ML Analytics', description: 'Performance metrics', href: '/analytics', icon: BarChart3, category: 'Features', keywords: ['analytics', 'metrics', 'stats', 'performance'] },
  { title: 'Knowledge Graph', description: 'Attack pattern visualization', href: '/knowledge-graph', icon: Network, category: 'Features', keywords: ['knowledge', 'graph', 'mitre', 'attack', 'visualization'] },
];

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((open) => !open);
      }
    };

    document.addEventListener('keydown', down);
    return () => document.removeEventListener('keydown', down);
  }, []);

  const handleSelect = (href: string) => {
    setOpen(false);
    router.push(href);
  };

  return (
    <>
      {/* Search Trigger Button - Visible in header */}
      <button
        onClick={() => setOpen(true)}
        className="hidden sm:flex items-center gap-2 px-3 py-1.5 text-sm text-muted-foreground bg-muted/50 hover:bg-muted rounded-md border transition-colors"
      >
        <Search className="h-4 w-4" />
        <span>Search...</span>
        <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground opacity-100">
          <span className="text-xs">⌘</span>K
        </kbd>
      </button>

      {/* Mobile Search Button */}
      <button
        onClick={() => setOpen(true)}
        className="sm:hidden p-2 text-muted-foreground hover:text-foreground transition-colors"
        aria-label="Search"
      >
        <Search className="h-5 w-5" />
      </button>

      <CommandDialog open={open} onOpenChange={setOpen}>
        <CommandInput placeholder="Search tools, pages, and features..." />
        <CommandList>
          <CommandEmpty>No results found.</CommandEmpty>

          <CommandGroup heading="AI Tools">
            {commands
              .filter((cmd) => cmd.category === 'AI Tools')
              .map((cmd) => (
                <CommandItem
                  key={cmd.href}
                  onSelect={() => handleSelect(cmd.href)}
                  className="flex items-center gap-3 px-4 py-3"
                >
                  <cmd.icon className="h-4 w-4 text-primary" />
                  <div className="flex-1">
                    <p className="font-medium">{cmd.title}</p>
                    <p className="text-xs text-muted-foreground">{cmd.description}</p>
                  </div>
                </CommandItem>
              ))}
          </CommandGroup>

          <CommandSeparator />

          <CommandGroup heading="Features">
            {commands
              .filter((cmd) => cmd.category === 'Features')
              .map((cmd) => (
                <CommandItem
                  key={cmd.href}
                  onSelect={() => handleSelect(cmd.href)}
                  className="flex items-center gap-3 px-4 py-3"
                >
                  <cmd.icon className="h-4 w-4 text-primary" />
                  <div className="flex-1">
                    <p className="font-medium">{cmd.title}</p>
                    <p className="text-xs text-muted-foreground">{cmd.description}</p>
                  </div>
                </CommandItem>
              ))}
          </CommandGroup>

          <CommandSeparator />

          <CommandGroup heading="Pages">
            {commands
              .filter((cmd) => cmd.category === 'Pages')
              .map((cmd) => (
                <CommandItem
                  key={cmd.href}
                  onSelect={() => handleSelect(cmd.href)}
                  className="flex items-center gap-3 px-4 py-3"
                >
                  <cmd.icon className="h-4 w-4 text-primary" />
                  <div className="flex-1">
                    <p className="font-medium">{cmd.title}</p>
                    <p className="text-xs text-muted-foreground">{cmd.description}</p>
                  </div>
                </CommandItem>
              ))}
          </CommandGroup>
        </CommandList>
      </CommandDialog>
    </>
  );
}
