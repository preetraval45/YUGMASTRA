'use client';

import { useState, useEffect } from 'react';
import { Shield, CheckCircle2, XCircle, Clock, TrendingUp, Sparkles, Activity } from 'lucide-react';
import { sendNotification } from '@/hooks/use-notifications';

export default function DefensesPage() {
  const [stats, setStats] = useState({
    totalDetections: 642,
    truePositives: 588,
    falsePositives: 54,
    responseTime: 12.4
  });

  const [detectionRules, setDetectionRules] = useState([
    { id: 1, name: 'SQL Injection Pattern', confidence: 0.92, fp_rate: 0.03, detections: 145, active: true },
    { id: 2, name: 'Lateral Movement Detector', confidence: 0.87, fp_rate: 0.08, detections: 89, active: true },
    { id: 3, name: 'Anomalous Traffic Pattern', confidence: 0.79, fp_rate: 0.12, detections: 234, active: true },
    { id: 4, name: 'Privilege Escalation Alert', confidence: 0.94, fp_rate: 0.02, detections: 67, active: true },
    { id: 5, name: 'Data Exfiltration Monitor', confidence: 0.88, fp_rate: 0.06, detections: 123, active: true },
  ]);

  const [recentActivity, setRecentActivity] = useState<string[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  // Simulate live updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Random detection event
      if (Math.random() > 0.7) {
        setStats(prev => ({
          ...prev,
          totalDetections: prev.totalDetections + 1,
          truePositives: prev.truePositives + (Math.random() > 0.15 ? 1 : 0),
          falsePositives: prev.falsePositives + (Math.random() > 0.85 ? 1 : 0)
        }));

        // Update random rule
        const ruleIndex = Math.floor(Math.random() * detectionRules.length);
        setDetectionRules(prev => prev.map((rule, idx) =>
          idx === ruleIndex
            ? { ...rule, detections: rule.detections + 1 }
            : rule
        ));

        const activities = [
          'New threat pattern detected and blocked',
          'Defense rule auto-tuned for better accuracy',
          'Anomaly detection threshold adjusted',
          'Zero-day exploit pattern learned',
          'Attack vector successfully neutralized'
        ];

        setRecentActivity(prev => [
          activities[Math.floor(Math.random() * activities.length)],
          ...prev.slice(0, 4)
        ]);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const handleGenerateRule = () => {
    setIsGenerating(true);

    setTimeout(() => {
      const newRule = {
        id: detectionRules.length + 1,
        name: `Advanced Pattern ${detectionRules.length + 1}`,
        confidence: 0.75 + Math.random() * 0.2,
        fp_rate: Math.random() * 0.1,
        detections: 0,
        active: true
      };

      setDetectionRules(prev => [newRule, ...prev]);
      setIsGenerating(false);
      sendNotification('defense', 'New Rule Generated', `Created ${newRule.name} with ${(newRule.confidence * 100).toFixed(0)}% confidence`, 'low');
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Defense Analytics</h1>
          <p className="text-muted-foreground">Blue Team performance and detection capabilities</p>
        </div>

        {/* Recent Activity Feed */}
        {recentActivity.length > 0 && (
          <div className="mb-6 bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-blue-500 animate-pulse" />
              <h3 className="font-semibold text-blue-500">Live Defense Activity</h3>
            </div>
            <div className="space-y-1">
              {recentActivity.map((activity, idx) => (
                <p key={idx} className="text-sm text-muted-foreground animate-in fade-in slide-in-from-top">
                  • {activity}
                </p>
              ))}
            </div>
          </div>
        )}

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-card rounded-lg p-6 border hover:border-blue-500/50 hover:shadow-lg transition-all">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-blue-500/10 rounded-lg">
                <Shield className="w-6 h-6 text-blue-500" />
              </div>
              <span className="text-xs bg-blue-500/20 text-blue-500 px-2 py-1 rounded animate-pulse">Live</span>
            </div>
            <h3 className="text-2xl font-bold mb-1 transition-all">{stats.totalDetections}</h3>
            <p className="text-sm text-muted-foreground">Total Detections</p>
          </div>

          <div className="bg-card rounded-lg p-6 border hover:border-green-500/50 hover:shadow-lg transition-all">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-green-500/10 rounded-lg">
                <CheckCircle2 className="w-6 h-6 text-green-500" />
              </div>
              <span className="text-xs text-green-500 font-semibold">
                {((stats.truePositives / stats.totalDetections) * 100).toFixed(1)}%
              </span>
            </div>
            <h3 className="text-2xl font-bold mb-1 transition-all">{stats.truePositives}</h3>
            <p className="text-sm text-muted-foreground">True Positives</p>
          </div>

          <div className="bg-card rounded-lg p-6 border hover:border-orange-500/50 hover:shadow-lg transition-all">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-orange-500/10 rounded-lg">
                <XCircle className="w-6 h-6 text-orange-500" />
              </div>
              <span className="text-xs text-orange-500 font-semibold">
                {((stats.falsePositives / stats.totalDetections) * 100).toFixed(1)}%
              </span>
            </div>
            <h3 className="text-2xl font-bold mb-1 transition-all">{stats.falsePositives}</h3>
            <p className="text-sm text-muted-foreground">False Positives</p>
          </div>

          <div className="bg-card rounded-lg p-6 border hover:border-purple-500/50 hover:shadow-lg transition-all">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-purple-500/10 rounded-lg">
                <Clock className="w-6 h-6 text-purple-500" />
              </div>
              <span className="text-xs text-purple-500 font-semibold">Avg</span>
            </div>
            <h3 className="text-2xl font-bold mb-1">{stats.responseTime.toFixed(1)}s</h3>
            <p className="text-sm text-muted-foreground">Response Time</p>
          </div>
        </div>

        {/* Detection Rate Trend */}
        <div className="bg-card rounded-lg p-6 border mb-6">
          <h2 className="text-xl font-bold mb-4">Detection Rate Trend</h2>
          <div className="h-64 flex items-end gap-2">
            {Array.from({ length: 30 }, (_, i) => {
              const rate = 0.6 + Math.sin(i / 3) * 0.15 + (Math.random() - 0.5) * 0.05;
              return (
                <div key={i} className="flex-1 flex flex-col justify-end">
                  <div
                    className="bg-gradient-to-t from-blue-600 to-blue-400 rounded-t hover:from-blue-500 hover:to-blue-300 transition-colors cursor-pointer"
                    style={{ height: `${rate * 100}%` }}
                    title={`Day ${i + 1}: ${(rate * 100).toFixed(1)}%`}
                  />
                </div>
              );
            })}
          </div>
          <div className="flex justify-between mt-4 text-sm text-muted-foreground">
            <span>30 days ago</span>
            <span>Today</span>
          </div>
        </div>

        {/* AI-Generated Detection Rules */}
        <div className="bg-card rounded-lg p-6 border mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold">AI-Generated Detection Rules</h2>
            <button
              onClick={handleGenerateRule}
              disabled={isGenerating}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm hover:bg-primary/90 disabled:bg-primary/50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
            >
              <Sparkles className={`w-4 h-4 ${isGenerating ? 'animate-spin' : ''}`} />
              {isGenerating ? 'Generating...' : 'Generate New Rule'}
            </button>
          </div>
          <div className="space-y-4">
            {detectionRules.map((rule) => (
              <div key={rule.id} className="p-4 bg-accent/50 rounded-lg border hover:bg-accent transition-all animate-in fade-in slide-in-from-left">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="font-semibold mb-1">{rule.name}</h3>
                    <p className="text-xs text-muted-foreground">Generated by Blue Team AI</p>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded ${rule.active ? 'bg-blue-500/20 text-blue-500' : 'bg-muted/50 text-muted-foreground'}`}>
                    {rule.active ? 'Active' : 'Inactive'}
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Confidence</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-green-600 to-green-400"
                          style={{ width: `${rule.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-semibold">{(rule.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">False Positive Rate</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-orange-600 to-orange-400"
                          style={{ width: `${rule.fp_rate * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-semibold">{(rule.fp_rate * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Total Detections</p>
                    <p className="text-sm font-semibold">{rule.detections}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-card rounded-lg p-6 border">
            <h2 className="text-xl font-bold mb-4">Adaptive Thresholds</h2>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Anomaly Detection Threshold</span>
                  <span className="font-semibold">0.73</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500" style={{ width: '73%' }} />
                </div>
                <p className="text-xs text-muted-foreground mt-1">↑ Auto-adjusted +0.05</p>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Alert Correlation Sensitivity</span>
                  <span className="font-semibold">0.68</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-green-500" style={{ width: '68%' }} />
                </div>
                <p className="text-xs text-muted-foreground mt-1">→ Stable</p>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Response Urgency Level</span>
                  <span className="font-semibold">0.81</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-orange-500" style={{ width: '81%' }} />
                </div>
                <p className="text-xs text-muted-foreground mt-1">↓ Auto-adjusted -0.03</p>
              </div>
            </div>
          </div>

          <div className="bg-card rounded-lg p-6 border">
            <h2 className="text-xl font-bold mb-4">Learning Progress</h2>
            <div className="space-y-4">
              <div className="p-4 bg-accent/50 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-4 h-4 text-green-500" />
                  <span className="font-semibold">Improvement Rate</span>
                </div>
                <p className="text-2xl font-bold mb-1">+18.2%</p>
                <p className="text-xs text-muted-foreground">Detection accuracy over 30 days</p>
              </div>
              <div className="p-4 bg-accent/50 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Shield className="w-4 h-4 text-blue-500" />
                  <span className="font-semibold">Strategies Learned</span>
                </div>
                <p className="text-2xl font-bold mb-1">127</p>
                <p className="text-xs text-muted-foreground">Unique defense patterns discovered</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
