'use client';

import { useState, useEffect } from 'react';
import {
  Activity,
  TrendingUp,
  AlertTriangle,
  Shield,
  Zap,
  Globe,
  Target,
  Crosshair,
  Flame,
  Lock
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface ThreatData {
  id: string;
  name: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  type: string;
  source: string;
  targets: number;
  confidence: number;
  timestamp: Date;
}

export default function ThreatIntelligencePage() {
  const [threats, setThreats] = useState<ThreatData[]>([]);
  const [selectedThreat, setSelectedThreat] = useState<ThreatData | null>(null);
  const [filter, setFilter] = useState<'all' | 'critical' | 'high' | 'medium' | 'low'>('all');
  const [realTimeMode, setRealTimeMode] = useState(true);

  // Simulate real-time threat data
  useEffect(() => {
    const generateMockThreats = (): ThreatData[] => {
      const threatTypes = ['APT', 'Ransomware', 'Phishing', 'DDoS', 'Zero-Day', 'Malware', 'Data Breach'];
      const sources = ['North Korea', 'Russia', 'China', 'Iran', 'Unknown', 'Cybercriminal Group'];
      const names = [
        'APT29 (Cozy Bear)',
        'Lazarus Group',
        'DarkSide Ransomware',
        'Emotet Botnet',
        'SolarWinds Supply Chain',
        'Log4Shell Exploitation',
        'ProxyShell Attack',
        'Kaseya VSA Breach'
      ];

      return Array.from({ length: 15 }, (_, i) => ({
        id: `threat-${i}`,
        name: names[Math.floor(Math.random() * names.length)],
        severity: ['critical', 'high', 'medium', 'low'][Math.floor(Math.random() * 4)] as any,
        type: threatTypes[Math.floor(Math.random() * threatTypes.length)],
        source: sources[Math.floor(Math.random() * sources.length)],
        targets: Math.floor(Math.random() * 10000),
        confidence: Math.floor(Math.random() * 30) + 70,
        timestamp: new Date(Date.now() - Math.random() * 86400000)
      }));
    };

    setThreats(generateMockThreats());

    if (realTimeMode) {
      const interval = setInterval(() => {
        setThreats(prev => {
          const newThreats = generateMockThreats();
          return [...newThreats.slice(0, 3), ...prev.slice(0, 12)];
        });
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [realTimeMode]);

  const filteredThreats = filter === 'all'
    ? threats
    : threats.filter(t => t.severity === filter);

  const stats = {
    total: threats.length,
    critical: threats.filter(t => t.severity === 'critical').length,
    high: threats.filter(t => t.severity === 'high').length,
    medium: threats.filter(t => t.severity === 'medium').length,
    low: threats.filter(t => t.severity === 'low').length,
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500 text-white';
      case 'high': return 'bg-orange-500 text-white';
      case 'medium': return 'bg-yellow-500 text-black';
      case 'low': return 'bg-blue-500 text-white';
      default: return 'bg-gray-500 text-white';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/5 p-6">
      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-red-500 via-orange-500 to-yellow-500 bg-clip-text text-transparent">
            Global Threat Intelligence
          </h1>
          <p className="text-muted-foreground mt-2">Real-time cyber threat monitoring and analysis</p>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={() => setRealTimeMode(!realTimeMode)}
            className={cn(
              "px-4 py-2 rounded-lg font-semibold transition-all",
              realTimeMode
                ? "bg-green-500 text-white animate-pulse"
                : "bg-gray-700 text-gray-300"
            )}
          >
            {realTimeMode ? '🔴 LIVE' : 'Paused'}
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
        <StatsCard
          title="Total Threats"
          value={stats.total}
          icon={Activity}
          gradient="from-blue-500 to-cyan-500"
          onClick={() => setFilter('all')}
          active={filter === 'all'}
        />
        <StatsCard
          title="Critical"
          value={stats.critical}
          icon={AlertTriangle}
          gradient="from-red-500 to-rose-500"
          onClick={() => setFilter('critical')}
          active={filter === 'critical'}
        />
        <StatsCard
          title="High"
          value={stats.high}
          icon={Flame}
          gradient="from-orange-500 to-amber-500"
          onClick={() => setFilter('high')}
          active={filter === 'high'}
        />
        <StatsCard
          title="Medium"
          value={stats.medium}
          icon={TrendingUp}
          gradient="from-yellow-500 to-orange-500"
          onClick={() => setFilter('medium')}
          active={filter === 'medium'}
        />
        <StatsCard
          title="Low"
          value={stats.low}
          icon={Shield}
          gradient="from-blue-500 to-indigo-500"
          onClick={() => setFilter('low')}
          active={filter === 'low'}
        />
      </div>

      {/* Threat Map */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Threat List */}
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-card border rounded-xl p-6">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <Target className="h-6 w-6 text-red-500" />
              Active Threats
            </h2>
            <div className="space-y-3 max-h-[600px] overflow-y-auto">
              {filteredThreats.map((threat) => (
                <div
                  key={threat.id}
                  onClick={() => setSelectedThreat(threat)}
                  className={cn(
                    "p-4 rounded-lg border-2 cursor-pointer transition-all hover:scale-[1.02]",
                    selectedThreat?.id === threat.id
                      ? "border-primary bg-primary/10"
                      : "border-border bg-card hover:border-primary/50"
                  )}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <span className={cn(
                          "px-3 py-1 rounded-full text-xs font-bold uppercase",
                          getSeverityColor(threat.severity)
                        )}>
                          {threat.severity}
                        </span>
                        <span className="text-xs bg-accent px-2 py-1 rounded">
                          {threat.type}
                        </span>
                      </div>
                      <h3 className="font-bold text-lg">{threat.name}</h3>
                      <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                        <span className="flex items-center gap-1">
                          <Globe className="h-4 w-4" />
                          {threat.source}
                        </span>
                        <span className="flex items-center gap-1">
                          <Crosshair className="h-4 w-4" />
                          {threat.targets.toLocaleString()} targets
                        </span>
                        <span className="flex items-center gap-1">
                          <Zap className="h-4 w-4" />
                          {threat.confidence}% confidence
                        </span>
                      </div>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {new Date(threat.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Threat Details */}
        <div className="space-y-4">
          {selectedThreat ? (
            <div className="bg-card border rounded-xl p-6">
              <h2 className="text-2xl font-bold mb-4">Threat Details</h2>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-muted-foreground">Name</label>
                  <p className="font-bold text-lg">{selectedThreat.name}</p>
                </div>
                <div>
                  <label className="text-sm text-muted-foreground">Severity</label>
                  <span className={cn(
                    "inline-block px-3 py-1 rounded-full text-sm font-bold mt-1",
                    getSeverityColor(selectedThreat.severity)
                  )}>
                    {selectedThreat.severity.toUpperCase()}
                  </span>
                </div>
                <div>
                  <label className="text-sm text-muted-foreground">Type</label>
                  <p className="font-semibold">{selectedThreat.type}</p>
                </div>
                <div>
                  <label className="text-sm text-muted-foreground">Source</label>
                  <p className="font-semibold">{selectedThreat.source}</p>
                </div>
                <div>
                  <label className="text-sm text-muted-foreground">Affected Targets</label>
                  <p className="font-semibold">{selectedThreat.targets.toLocaleString()}</p>
                </div>
                <div>
                  <label className="text-sm text-muted-foreground">Confidence</label>
                  <div className="mt-2">
                    <div className="h-2 bg-accent rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
                        style={{ width: `${selectedThreat.confidence}%` }}
                      />
                    </div>
                    <p className="text-sm font-semibold mt-1">{selectedThreat.confidence}%</p>
                  </div>
                </div>
                <div className="pt-4 border-t">
                  <button className="w-full bg-red-500 hover:bg-red-600 text-white font-bold py-3 rounded-lg transition-colors flex items-center justify-center gap-2">
                    <Lock className="h-5 w-5" />
                    Deploy Countermeasures
                  </button>
                  <button className="w-full mt-2 bg-primary hover:bg-primary/80 text-primary-foreground font-bold py-3 rounded-lg transition-colors">
                    Analyze with AI
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-card border rounded-xl p-6 text-center text-muted-foreground">
              <Target className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>Select a threat to view details</p>
            </div>
          )}
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
