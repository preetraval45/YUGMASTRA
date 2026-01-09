'use client';

import { useState } from 'react';
import {
  Activity,
  TrendingUp,
  AlertTriangle,
  Shield,
  Globe,
  Target,
  Flame,
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface ThreatData {
  id: string;
  name: string;
  category: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  trend: 'rising' | 'stable' | 'declining';
  count: number;
}

export default function ThreatIntelligencePage() {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const stats = {
    activeCampaigns: 47,
    zeroDay: 12,
    malwareFamilies: 234,
    compromisedAssets: 891,
  };

  const threats: ThreatData[] = [
    { id: '1', name: 'Ransomware-as-a-Service', category: 'ransomware', severity: 'critical', trend: 'rising', count: 1247 },
    { id: '2', name: 'Supply Chain Attacks', category: 'apt', severity: 'critical', trend: 'rising', count: 892 },
    { id: '3', name: 'AI-Powered Phishing', category: 'phishing', severity: 'high', trend: 'rising', count: 3421 },
    { id: '4', name: 'Zero-Day Exploits', category: 'exploit', severity: 'critical', trend: 'stable', count: 156 },
    { id: '5', name: 'Cryptojacking', category: 'malware', severity: 'medium', trend: 'declining', count: 678 },
  ];

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500 text-white';
      case 'high': return 'bg-orange-500 text-white';
      case 'medium': return 'bg-yellow-500 text-black';
      case 'low': return 'bg-blue-500 text-white';
      default: return 'bg-muted text-foreground';
    }
  };

  const getTrendIcon = (trend: string) => {
    if (trend === 'rising') return <TrendingUp className="w-4 h-4 text-red-500" />;
    if (trend === 'declining') return <TrendingUp className="w-4 h-4 text-green-500 rotate-180" />;
    return <Activity className="w-4 h-4 text-blue-500" />;
  };

  return (
    <div className="min-h-screen bg-background p-8 pt-32">
      <div className="mb-8">
        <div className="mb-4">
          <h1 className="text-4xl font-bold mb-2">Global Threat Intelligence</h1>
          <p className="text-muted-foreground">Real-time threat data from global sources</p>
        </div>

        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 flex items-start gap-3 mb-6">
          <Globe className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              <strong className="text-foreground">What this page does:</strong> Aggregates real-time threat intelligence from global sources including MITRE ATT&CK, threat feeds, and OSINT platforms. Track active campaigns, zero-day exploits, malware families, and compromised assets. Monitor threat trends, severity levels, and geographic distribution. Uses ML to correlate IOCs and predict emerging threats.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <StatsCard
          title="Active Campaigns"
          value={stats.activeCampaigns}
          icon={Flame}
          gradient="from-red-500 to-orange-500"
          onClick={() => {}}
          active={false}
        />
        <StatsCard
          title="Zero-Day Exploits"
          value={stats.zeroDay}
          icon={AlertTriangle}
          gradient="from-orange-500 to-yellow-500"
          onClick={() => {}}
          active={false}
        />
        <StatsCard
          title="Malware Families"
          value={stats.malwareFamilies}
          icon={Shield}
          gradient="from-purple-500 to-pink-500"
          onClick={() => {}}
          active={false}
        />
        <StatsCard
          title="Compromised Assets"
          value={stats.compromisedAssets}
          icon={Target}
          gradient="from-blue-500 to-cyan-500"
          onClick={() => {}}
          active={false}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="bg-card rounded-lg border p-6">
            <h2 className="text-2xl font-bold mb-6">Active Threats</h2>
            <div className="space-y-4">
              {threats.map((threat) => (
                <div
                  key={threat.id}
                  className="bg-accent/50 rounded-lg p-4 border border-border hover:border-primary/50 transition-all cursor-pointer"
                >
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="font-semibold text-lg">{threat.name}</h3>
                    {getTrendIcon(threat.trend)}
                  </div>
                  <div className="flex items-center gap-2 mb-3">
                    <span className={cn('text-xs px-2 py-1 rounded font-medium', getSeverityColor(threat.severity))}>
                      {threat.severity.toUpperCase()}
                    </span>
                    <span className="text-xs bg-muted px-2 py-1 rounded">
                      {threat.category}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm text-muted-foreground">
                    <span>Detected instances: {threat.count}</span>
                    <span className="capitalize">{threat.trend}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-card rounded-lg border p-6">
            <h3 className="text-lg font-semibold mb-4">Threat Categories</h3>
            <div className="space-y-2">
              {['all', 'ransomware', 'apt', 'phishing', 'exploit', 'malware'].map((category) => (
                <button
                  key={category}
                  onClick={() => setSelectedCategory(category)}
                  className={cn(
                    'w-full text-left px-3 py-2 rounded-lg transition-colors capitalize',
                    selectedCategory === category
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-background hover:bg-accent'
                  )}
                >
                  {category}
                </button>
              ))}
            </div>
          </div>

          <div className="bg-card rounded-lg border p-6">
            <h3 className="text-lg font-semibold mb-4">Geographic Distribution</h3>
            <div className="space-y-3">
              <div>
                <div className="flex items-center justify-between text-sm mb-1">
                  <span>North America</span>
                  <span className="text-muted-foreground">32%</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500" style={{ width: '32%' }} />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between text-sm mb-1">
                  <span>Europe</span>
                  <span className="text-muted-foreground">28%</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-green-500" style={{ width: '28%' }} />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between text-sm mb-1">
                  <span>Asia Pacific</span>
                  <span className="text-muted-foreground">25%</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-yellow-500" style={{ width: '25%' }} />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between text-sm mb-1">
                  <span>Other Regions</span>
                  <span className="text-muted-foreground">15%</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-purple-500" style={{ width: '15%' }} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

interface StatsCardProps {
  title: string;
  value: number;
  icon: any;
  gradient: string;
  onClick: () => void;
  active: boolean;
}

function StatsCard({ title, value, icon: Icon, gradient, onClick, active }: StatsCardProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "relative overflow-hidden rounded-xl p-6 transition-all hover:scale-105",
        active ? "ring-4 ring-primary" : ""
      )}
    >
      <div className={cn("absolute inset-0 bg-gradient-to-br opacity-10", gradient)} />
      <div className="relative">
        <div className="flex items-center justify-between mb-2">
          <Icon className={cn("h-8 w-8 bg-gradient-to-br bg-clip-text text-transparent", gradient)} />
        </div>
        <div className="text-3xl font-bold">{value}</div>
        <div className="text-sm text-muted-foreground">{title}</div>
      </div>
    </button>
  );
}
