'use client';

import { useState, useEffect } from 'react';
import { Shield, Swords, AlertCircle, CheckCircle2, Clock, TrendingUp, Filter } from 'lucide-react';

export default function AttacksPage() {
  const [animateStats, setAnimateStats] = useState(false);
  const [selectedFilter, setSelectedFilter] = useState<'all' | 'successful' | 'detected' | 'undetected'>('all');

  const [recentAttacks] = useState([
    { id: 1, type: 'SQL Injection', target: 'web_server', success: true, detected: false, impact: 0.85, time: '2 min ago', severity: 'critical' as const },
    { id: 2, type: 'Phishing', target: 'endpoint_1', success: true, detected: true, impact: 0.65, time: '5 min ago', severity: 'high' as const },
    { id: 3, type: 'Port Scan', target: 'all', success: true, detected: true, impact: 0.15, time: '8 min ago', severity: 'low' as const },
    { id: 4, type: 'Privilege Escalation', target: 'web_server', success: false, detected: true, impact: 0.0, time: '12 min ago', severity: 'high' as const },
    { id: 5, type: 'Lateral Movement', target: 'db_server', success: true, detected: false, impact: 0.95, time: '15 min ago', severity: 'critical' as const },
  ]);

  useEffect(() => {
    setTimeout(() => setAnimateStats(true), 100);
  }, []);

  const filteredAttacks = recentAttacks.filter(attack => {
    if (selectedFilter === 'all') return true;
    if (selectedFilter === 'successful') return attack.success;
    if (selectedFilter === 'detected') return attack.detected;
    if (selectedFilter === 'undetected') return !attack.detected;
    return true;
  });

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-500/20';
      case 'high': return 'text-orange-600 bg-orange-500/20';
      case 'medium': return 'text-yellow-600 bg-yellow-500/20';
      default: return 'text-blue-600 bg-blue-500/20';
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-8 pt-32">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Attack Analytics</h1>
          <p className="text-muted-foreground">Red Team performance and attack patterns</p>
        </div>

        {/* Filter Buttons */}
        <div className="flex gap-2 mb-6 flex-wrap">
          <button
            onClick={() => setSelectedFilter('all')}
            className={`px-4 py-2 rounded-lg transition-all ${
              selectedFilter === 'all'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-card border hover:bg-accent'
            }`}
          >
            <Filter className="w-4 h-4 inline mr-2" />
            All Attacks
          </button>
          <button
            onClick={() => setSelectedFilter('successful')}
            className={`px-4 py-2 rounded-lg transition-all ${
              selectedFilter === 'successful'
                ? 'bg-green-600 text-white shadow-lg'
                : 'bg-card border hover:bg-accent'
            }`}
          >
            Successful
          </button>
          <button
            onClick={() => setSelectedFilter('detected')}
            className={`px-4 py-2 rounded-lg transition-all ${
              selectedFilter === 'detected'
                ? 'bg-orange-600 text-white shadow-lg'
                : 'bg-card border hover:bg-accent'
            }`}
          >
            Detected
          </button>
          <button
            onClick={() => setSelectedFilter('undetected')}
            className={`px-4 py-2 rounded-lg transition-all ${
              selectedFilter === 'undetected'
                ? 'bg-red-600 text-white shadow-lg'
                : 'bg-card border hover:bg-accent'
            }`}
          >
            Undetected
          </button>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className={`bg-card rounded-lg p-6 border hover:border-red-500/50 hover:shadow-xl hover:shadow-red-500/10 transition-all duration-500 transform ${animateStats ? 'scale-100 opacity-100' : 'scale-95 opacity-0'}`}>
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-red-500/10 rounded-lg">
                <Swords className="w-6 h-6 text-red-500 animate-pulse" />
              </div>
              <span className="text-xs bg-red-500/20 text-red-500 px-2 py-1 rounded animate-pulse">Live</span>
            </div>
            <h3 className="text-2xl font-bold mb-1">1,523</h3>
            <p className="text-sm text-muted-foreground">Total Attacks</p>
            <div className="mt-2 flex items-center gap-1 text-xs text-green-500">
              <TrendingUp className="w-3 h-3" />
              +8% this hour
            </div>
          </div>

          <div className={`bg-card rounded-lg p-6 border hover:border-green-500/50 hover:shadow-xl hover:shadow-green-500/10 transition-all duration-500 delay-75 transform ${animateStats ? 'scale-100 opacity-100' : 'scale-95 opacity-0'}`}>
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-green-500/10 rounded-lg">
                <CheckCircle2 className="w-6 h-6 text-green-500" />
              </div>
              <span className="text-xs text-green-500 font-semibold">58.5%</span>
            </div>
            <h3 className="text-2xl font-bold mb-1">891</h3>
            <p className="text-sm text-muted-foreground">Successful</p>
            <div className="mt-2 flex items-center gap-1 text-xs text-red-500">
              <TrendingUp className="w-3 h-3" />
              +12% vs last week
            </div>
          </div>

          <div className={`bg-card rounded-lg p-6 border hover:border-orange-500/50 hover:shadow-xl hover:shadow-orange-500/10 transition-all duration-500 delay-150 transform ${animateStats ? 'scale-100 opacity-100' : 'scale-95 opacity-0'}`}>
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-orange-500/10 rounded-lg">
                <AlertCircle className="w-6 h-6 text-orange-500" />
              </div>
              <span className="text-xs text-orange-500 font-semibold">42%</span>
            </div>
            <h3 className="text-2xl font-bold mb-1">642</h3>
            <p className="text-sm text-muted-foreground">Detected</p>
            <div className="mt-2 flex items-center gap-1 text-xs text-blue-500">
              <TrendingUp className="w-3 h-3" />
              Detection improving
            </div>
          </div>

          <div className={`bg-card rounded-lg p-6 border hover:border-purple-500/50 hover:shadow-xl hover:shadow-purple-500/10 transition-all duration-500 delay-200 transform ${animateStats ? 'scale-100 opacity-100' : 'scale-95 opacity-0'}`}>
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-purple-500/10 rounded-lg">
                <Clock className="w-6 h-6 text-purple-500" />
              </div>
              <span className="text-xs text-purple-500 font-semibold">Avg</span>
            </div>
            <h3 className="text-2xl font-bold mb-1">45.3s</h3>
            <p className="text-sm text-muted-foreground">Time to Detect</p>
            <div className="mt-2 flex items-center gap-1 text-xs text-green-500">
              <TrendingUp className="w-3 h-3 rotate-180" />
              -5s improvement
            </div>
          </div>
        </div>

        {/* Attack Types Distribution */}
        <div className="bg-card rounded-lg p-6 border mb-6">
          <h2 className="text-xl font-bold mb-4">Attack Type Distribution</h2>
          <div className="space-y-3">
            {[
              { type: 'Web Exploit', count: 342, percentage: 22 },
              { type: 'Phishing', count: 298, percentage: 20 },
              { type: 'Lateral Movement', count: 267, percentage: 18 },
              { type: 'Privilege Escalation', count: 245, percentage: 16 },
              { type: 'Data Exfiltration', count: 189, percentage: 12 },
              { type: 'Port Scanning', count: 182, percentage: 12 },
            ].map((item, i) => (
              <div key={i}>
                <div className="flex justify-between text-sm mb-2">
                  <span>{item.type}</span>
                  <span className="font-semibold">{item.count}</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-red-600 to-red-400"
                    style={{ width: `${item.percentage * 4}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Attacks Table */}
        <div className="bg-card rounded-lg p-6 border">
          <h2 className="text-xl font-bold mb-4">Recent Attacks</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-3 px-4 text-sm font-semibold">Type</th>
                  <th className="text-left py-3 px-4 text-sm font-semibold">Target</th>
                  <th className="text-left py-3 px-4 text-sm font-semibold">Status</th>
                  <th className="text-left py-3 px-4 text-sm font-semibold">Detected</th>
                  <th className="text-left py-3 px-4 text-sm font-semibold">Impact</th>
                  <th className="text-left py-3 px-4 text-sm font-semibold">Time</th>
                </tr>
              </thead>
              <tbody>
                {filteredAttacks.map((attack, index) => (
                  <tr
                    key={attack.id}
                    className="border-b hover:bg-accent/50 transition-all duration-300 animate-in fade-in"
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <Swords className="w-4 h-4 text-red-500" />
                        <div className="flex flex-col">
                          <span className="font-medium">{attack.type}</span>
                          <span className={`text-xs px-2 py-0.5 rounded-full w-fit ${getSeverityColor(attack.severity)}`}>
                            {attack.severity}
                          </span>
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <code className="text-xs bg-muted px-2 py-1 rounded">{attack.target}</code>
                    </td>
                    <td className="py-3 px-4">
                      {attack.success ? (
                        <span className="inline-flex items-center gap-1 text-xs bg-green-500/20 text-green-500 px-2 py-1 rounded">
                          <CheckCircle2 className="w-3 h-3" />
                          Success
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-xs bg-red-500/20 text-red-500 px-2 py-1 rounded">
                          <AlertCircle className="w-3 h-3" />
                          Failed
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4">
                      {attack.detected ? (
                        <Shield className="w-4 h-4 text-orange-500" />
                      ) : (
                        <span className="text-muted-foreground text-xs">Undetected</span>
                      )}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden max-w-[100px]">
                          <div
                            className="h-full bg-gradient-to-r from-orange-600 to-red-600"
                            style={{ width: `${attack.impact * 100}%` }}
                          />
                        </div>
                        <span className="text-xs font-medium">{(attack.impact * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-sm text-muted-foreground">{attack.time}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
