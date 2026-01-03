'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Command, Search, Home, Settings, HelpCircle } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';

const shortcuts = [
  { key: 'Ctrl+K', description: 'Open command palette', icon: Command },
  { key: 'Ctrl+/', description: 'Toggle this help', icon: HelpCircle },
  { key: 'Ctrl+H', description: 'Go to dashboard', icon: Home },
  { key: 'Ctrl+,', description: 'Open settings', icon: Settings },
  { key: 'Esc', description: 'Close dialog', icon: null },
];

const quickLinks = [
  { name: 'Dashboard', href: '/dashboard', icon: '🏠' },
  { name: 'AI Assistant', href: '/ai-assistant', icon: '🤖' },
  { name: 'Threat Hunting', href: '/threat-hunting', icon: '🔍' },
  { name: 'Code Review', href: '/code-review', icon: '💻' },
  { name: 'Incident Response', href: '/incident-response', icon: '🚨' },
  { name: 'Zero-Day Discovery', href: '/zero-day', icon: '🐛' },
  { name: 'SIEM Rules', href: '/siem-rules', icon: '📋' },
  { name: 'Live Battle', href: '/live-battle', icon: '⚔️' },
  { name: 'Settings', href: '/settings', icon: '⚙️' },
];

export function KeyboardShortcuts() {
  const [commandOpen, setCommandOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [search, setSearch] = useState('');
  const router = useRouter();

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setCommandOpen((open) => !open);
      }
      if (e.key === '/' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setHelpOpen((open) => !open);
      }
      if (e.key === 'h' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        router.push('/dashboard');
      }
      if (e.key === ',' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        router.push('/settings');
      }
    };

    document.addEventListener('keydown', down);
    return () => document.removeEventListener('keydown', down);
  }, [router]);

  const filteredLinks = quickLinks.filter((link) =>
    link.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <>
      {/* Command Palette */}
      <Dialog open={commandOpen} onOpenChange={setCommandOpen}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Search className="h-5 w-5" />
              Quick Navigation
            </DialogTitle>
            <DialogDescription>
              Search and navigate to any page quickly
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <Input
              placeholder="Type to search..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="text-base"
              autoFocus
            />
            <div className="max-h-[400px] overflow-y-auto space-y-1">
              {filteredLinks.map((link) => (
                <button
                  key={link.href}
                  onClick={() => {
                    router.push(link.href);
                    setCommandOpen(false);
                    setSearch('');
                  }}
                  className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-accent transition-colors text-left"
                >
                  <span className="text-2xl">{link.icon}</span>
                  <span className="font-medium">{link.name}</span>
                </button>
              ))}
              {filteredLinks.length === 0 && (
                <p className="text-center py-8 text-muted-foreground">
                  No results found
                </p>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Help Dialog */}
      <Dialog open={helpOpen} onOpenChange={setHelpOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <HelpCircle className="h-5 w-5" />
              Keyboard Shortcuts
            </DialogTitle>
            <DialogDescription>
              Speed up your workflow with these shortcuts
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            {shortcuts.map((shortcut) => (
              <div
                key={shortcut.key}
                className="flex items-center justify-between py-2 px-3 rounded-lg bg-muted/30"
              >
                <div className="flex items-center gap-2">
                  {shortcut.icon && <shortcut.icon className="h-4 w-4 text-muted-foreground" />}
                  <span className="text-sm">{shortcut.description}</span>
                </div>
                <kbd className="px-2 py-1 text-xs font-semibold text-foreground bg-background border border-border rounded">
                  {shortcut.key}
                </kbd>
              </div>
            ))}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
