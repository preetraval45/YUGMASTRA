'use client';

import { Shield, Swords, AlertCircle, CheckCircle2, Clock } from 'lucide-react';

export default function AttacksPage() {
  const recentAttacks = [
    { id: 1, type: 'SQL Injection', target: 'web_server', success: true, detected: false, impact: 0.85, time: '2 min ago' },
    { id: 2, type: 'Phishing', target: 'endpoint_1', success: true, detected: true, impact: 0.65, time: '5 min ago' },
    { id: 3, type: 'Port Scan', target: 'all', success: true, detected: true, impact: 0.15, time: '8 min ago' },
    { id: 4, type: 'Privilege Escalation', target: 'web_server', success: false, detected: true, impact: 0.0, time: '12 min ago' },
    { id: 5, type: 'Lateral Movement', target: 'db_server', success: true, detected: false, impact: 0.95, time: '15 min ago' },
  ];

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Attack Analytics</h1>
          <p className="text-muted-foreground">Red Team performance and attack patterns</p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-card rounded-lg p-6 border">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-red-500/10 rounded-lg">
                <Swords className="w-6 h-6 text-red-500" />
              </div>
              <span className="text-xs bg-red-500/20 text-red-500 px-2 py-1 rounded">Live</span>
            </div>
            <h3 className="text-2xl font-bold mb-1">1,523</h3>
            <p className="text-sm text-muted-foreground">Total Attacks</p>
          </div>

          <div className="bg-card rounded-lg p-6 border">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-green-500/10 rounded-lg">
                <CheckCircle2 className="w-6 h-6 text-green-500" />
              </div>
              <span className="text-xs text-green-500 font-semibold">58.5%</span>
            </div>
            <h3 className="text-2xl font-bold mb-1">891</h3>
            <p className="text-sm text-muted-foreground">Successful</p>
          </div>

          <div className="bg-card rounded-lg p-6 border">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-orange-500/10 rounded-lg">
                <AlertCircle className="w-6 h-6 text-orange-500" />
              </div>
              <span className="text-xs text-orange-500 font-semibold">42%</span>
            </div>
            <h3 className="text-2xl font-bold mb-1">642</h3>
            <p className="text-sm text-muted-foreground">Detected</p>
          </div>

          <div className="bg-card rounded-lg p-6 border">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-purple-500/10 rounded-lg">
                <Clock className="w-6 h-6 text-purple-500" />
              </div>
              <span className="text-xs text-purple-500 font-semibold">Avg</span>
            </div>
            <h3 className="text-2xl font-bold mb-1">45.3s</h3>
            <p className="text-sm text-muted-foreground">Time to Detect</p>
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
                {recentAttacks.map((attack) => (
                  <tr key={attack.id} className="border-b hover:bg-accent/50 transition-colors">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <Swords className="w-4 h-4 text-red-500" />
                        <span className="font-medium">{attack.type}</span>
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
