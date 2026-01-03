'use client';

import { useState, useEffect } from 'react';
import { BarChart3, Brain, TrendingUp, Target, AlertTriangle, Shield, Zap, PieChart, LineChart, Activity } from 'lucide-react';

interface PredictionData {
  attackType: string;
  probability: number;
  confidence: number;
  timeframe: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

interface PatternData {
  pattern: string;
  frequency: number;
  trend: 'increasing' | 'stable' | 'decreasing';
  impact: number;
}

export default function AnalyticsPage() {
  const [predictions] = useState<PredictionData[]>([
    { attackType: 'SQL Injection', probability: 0.87, confidence: 0.92, timeframe: 'Next 24h', severity: 'high' },
    { attackType: 'Brute Force', probability: 0.76, confidence: 0.88, timeframe: 'Next 12h', severity: 'critical' },
    { attackType: 'XSS Attack', probability: 0.65, confidence: 0.81, timeframe: 'Next 48h', severity: 'medium' },
    { attackType: 'DDoS', probability: 0.54, confidence: 0.75, timeframe: 'Next 72h', severity: 'high' },
    { attackType: 'Zero-Day Exploit', probability: 0.23, confidence: 0.45, timeframe: 'Next Week', severity: 'critical' },
  ]);

  const [patterns] = useState<PatternData[]>([
    { pattern: 'Authentication attempts spike at 2-4 AM', frequency: 847, trend: 'increasing', impact: 8.5 },
    { pattern: 'Port scanning from Eastern Europe IPs', frequency: 623, trend: 'stable', impact: 6.2 },
    { pattern: 'API rate limit exceeded patterns', frequency: 512, trend: 'increasing', impact: 7.8 },
    { pattern: 'Suspicious file upload attempts', frequency: 289, trend: 'decreasing', impact: 5.4 },
    { pattern: 'Database query anomalies', frequency: 156, trend: 'stable', impact: 9.1 },
  ]);

  const [attackVectorDistribution] = useState([
    { name: 'Web Application', value: 35, color: 'bg-red-500' },
    { name: 'Network', value: 25, color: 'bg-orange-500' },
    { name: 'API', value: 20, color: 'bg-yellow-500' },
    { name: 'Database', value: 12, color: 'bg-blue-500' },
    { name: 'Other', value: 8, color: 'bg-purple-500' },
  ]);

  const [defenseEffectiveness] = useState([
    { defense: 'WAF', blocked: 1247, effectiveness: 94.2 },
    { defense: 'IDS/IPS', blocked: 983, effectiveness: 87.6 },
    { defense: 'Rate Limiting', blocked: 756, effectiveness: 92.1 },
    { defense: 'Input Validation', blocked: 634, effectiveness: 88.9 },
    { defense: 'Authentication', blocked: 521, effectiveness: 79.3 },
  ]);

  const [timeSeriesData] = useState([
    { time: '00:00', attacks: 45, blocks: 42 },
    { time: '04:00', attacks: 78, blocks: 71 },
    { time: '08:00', attacks: 52, blocks: 49 },
    { time: '12:00', attacks: 89, blocks: 84 },
    { time: '16:00', attacks: 67, blocks: 63 },
    { time: '20:00', attacks: 94, blocks: 88 },
  ]);

  const [mlMetrics] = useState({
    modelAccuracy: 0.924,
    precision: 0.891,
    recall: 0.867,
    f1Score: 0.879,
    falsePositiveRate: 0.08,
    falseNegativeRate: 0.12,
  });

  const [anomalyScore, setAnomalyScore] = useState(7.3);

  useEffect(() => {
    // Simulate real-time anomaly score updates
    const interval = setInterval(() => {
      setAnomalyScore(prev => {
        const change = (Math.random() - 0.5) * 2;
        return Math.max(0, Math.min(10, prev + change));
      });
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'low': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'increasing': return <TrendingUp className="w-4 h-4 text-red-500" />;
      case 'decreasing': return <TrendingUp className="w-4 h-4 text-green-500 rotate-180" />;
      case 'stable': return <Activity className="w-4 h-4 text-blue-500" />;
      default: return null;
    }
  };

  return (
    <div className="min-h-screen bg-background p-8 pt-32">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
              <Brain className="w-8 h-8 text-purple-500" />
              ML-Powered Analytics
            </h1>
            <p className="text-muted-foreground">
              AI-driven threat prediction and pattern analysis - Preet Raval's Security Intelligence
            </p>
          </div>
          <div className="bg-card rounded-lg px-6 py-3 border">
            <p className="text-sm text-muted-foreground">Anomaly Score</p>
            <p className={`text-3xl font-bold font-mono ${
              anomalyScore > 7 ? 'text-red-500' :
              anomalyScore > 4 ? 'text-yellow-500' :
              'text-green-500'
            }`}>
              {anomalyScore.toFixed(1)} / 10
            </p>
          </div>
        </div>
      </div>

      {/* ML Model Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-6">
        <div className="bg-card rounded-lg p-4 border border-green-500/30">
          <div className="flex items-center gap-2 mb-1">
            <Target className="w-4 h-4 text-green-500" />
            <h3 className="text-xs text-muted-foreground">Accuracy</h3>
          </div>
          <p className="text-2xl font-bold text-green-500">{(mlMetrics.modelAccuracy * 100).toFixed(1)}%</p>
        </div>

        <div className="bg-card rounded-lg p-4 border border-blue-500/30">
          <div className="flex items-center gap-2 mb-1">
            <Zap className="w-4 h-4 text-blue-500" />
            <h3 className="text-xs text-muted-foreground">Precision</h3>
          </div>
          <p className="text-2xl font-bold text-blue-500">{(mlMetrics.precision * 100).toFixed(1)}%</p>
        </div>

        <div className="bg-card rounded-lg p-4 border border-purple-500/30">
          <div className="flex items-center gap-2 mb-1">
            <Activity className="w-4 h-4 text-purple-500" />
            <h3 className="text-xs text-muted-foreground">Recall</h3>
          </div>
          <p className="text-2xl font-bold text-purple-500">{(mlMetrics.recall * 100).toFixed(1)}%</p>
        </div>

        <div className="bg-card rounded-lg p-4 border border-yellow-500/30">
          <div className="flex items-center gap-2 mb-1">
            <TrendingUp className="w-4 h-4 text-yellow-500" />
            <h3 className="text-xs text-muted-foreground">F1 Score</h3>
          </div>
          <p className="text-2xl font-bold text-yellow-500">{(mlMetrics.f1Score * 100).toFixed(1)}%</p>
        </div>

        <div className="bg-card rounded-lg p-4 border border-red-500/30">
          <div className="flex items-center gap-2 mb-1">
            <AlertTriangle className="w-4 h-4 text-red-500" />
            <h3 className="text-xs text-muted-foreground">FP Rate</h3>
          </div>
          <p className="text-2xl font-bold text-red-500">{(mlMetrics.falsePositiveRate * 100).toFixed(1)}%</p>
        </div>

        <div className="bg-card rounded-lg p-4 border border-orange-500/30">
          <div className="flex items-center gap-2 mb-1">
            <AlertTriangle className="w-4 h-4 text-orange-500" />
            <h3 className="text-xs text-muted-foreground">FN Rate</h3>
          </div>
          <p className="text-2xl font-bold text-orange-500">{(mlMetrics.falseNegativeRate * 100).toFixed(1)}%</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Threat Predictions */}
        <div className="bg-card rounded-lg p-6 border">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Brain className="w-6 h-6 text-purple-500" />
            AI Threat Predictions
          </h2>
          <div className="space-y-3">
            {predictions.map((pred, i) => (
              <div key={i} className="p-4 bg-accent/50 rounded-lg border">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="font-semibold mb-1">{pred.attackType}</h3>
                    <p className="text-xs text-muted-foreground">{pred.timeframe}</p>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded border ${getSeverityColor(pred.severity)}`}>
                    {pred.severity.toUpperCase()}
                  </span>
                </div>
                <div className="space-y-2">
                  <div>
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className="text-muted-foreground">Probability</span>
                      <span className="font-semibold">{(pred.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-red-600 to-red-400"
                        style={{ width: `${pred.probability * 100}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className="text-muted-foreground">Confidence</span>
                      <span className="font-semibold">{(pred.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-blue-600 to-blue-400"
                        style={{ width: `${pred.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Attack Patterns */}
        <div className="bg-card rounded-lg p-6 border">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <LineChart className="w-6 h-6 text-blue-500" />
            Detected Patterns
          </h2>
          <div className="space-y-3">
            {patterns.map((pattern, i) => (
              <div key={i} className="p-4 bg-accent/50 rounded-lg border">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <h3 className="font-semibold text-sm mb-1">{pattern.pattern}</h3>
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <span>Occurrences: <span className="font-semibold">{pattern.frequency}</span></span>
                      <div className="flex items-center gap-1">
                        {getTrendIcon(pattern.trend)}
                        <span className="capitalize">{pattern.trend}</span>
                      </div>
                    </div>
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between text-sm mb-1">
                    <span className="text-muted-foreground">Impact Score</span>
                    <span className={`font-semibold ${
                      pattern.impact > 7 ? 'text-red-500' :
                      pattern.impact > 5 ? 'text-yellow-500' :
                      'text-green-500'
                    }`}>
                      {pattern.impact.toFixed(1)} / 10
                    </span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div
                      className={`h-full ${
                        pattern.impact > 7 ? 'bg-gradient-to-r from-red-600 to-red-400' :
                        pattern.impact > 5 ? 'bg-gradient-to-r from-yellow-600 to-yellow-400' :
                        'bg-gradient-to-r from-green-600 to-green-400'
                      }`}
                      style={{ width: `${(pattern.impact / 10) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Attack Vector Distribution & Defense Effectiveness */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Attack Vector Distribution */}
        <div className="bg-card rounded-lg p-6 border">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <PieChart className="w-6 h-6 text-orange-500" />
            Attack Vector Distribution
          </h2>
          <div className="space-y-4">
            {attackVectorDistribution.map((vector, i) => (
              <div key={i}>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span>{vector.name}</span>
                  <span className="font-semibold">{vector.value}%</span>
                </div>
                <div className="h-3 bg-muted rounded-full overflow-hidden">
                  <div
                    className={`h-full ${vector.color}`}
                    style={{ width: `${vector.value}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Defense Effectiveness */}
        <div className="bg-card rounded-lg p-6 border">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Shield className="w-6 h-6 text-green-500" />
            Defense Effectiveness
          </h2>
          <div className="space-y-4">
            {defenseEffectiveness.map((defense, i) => (
              <div key={i}>
                <div className="flex items-center justify-between text-sm mb-2">
                  <div>
                    <span className="font-semibold">{defense.defense}</span>
                    <span className="text-muted-foreground ml-2">({defense.blocked} blocked)</span>
                  </div>
                  <span className="font-semibold text-green-500">{defense.effectiveness.toFixed(1)}%</span>
                </div>
                <div className="h-3 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-green-600 to-green-400"
                    style={{ width: `${defense.effectiveness}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Time Series Analysis */}
      <div className="bg-card rounded-lg p-6 border">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <BarChart3 className="w-6 h-6 text-purple-500" />
          Attack & Defense Time Series
        </h2>
        <div className="h-64 bg-accent/20 rounded-lg p-4 relative">
          {/* Y-axis labels */}
          <div className="absolute inset-0 flex flex-col justify-between p-4 pointer-events-none">
            {[100, 80, 60, 40, 20, 0].map((val) => (
              <div key={val} className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-8">{val}</span>
                <div className="flex-1 border-t border-gray-700/50" />
              </div>
            ))}
          </div>

          {/* Chart bars */}
          <div className="absolute inset-0 p-4 pl-12 pr-4 flex items-end gap-2">
            {timeSeriesData.map((data, i) => (
              <div key={i} className="flex-1 flex flex-col items-center gap-1">
                <div className="w-full flex gap-1 items-end" style={{ height: '200px' }}>
                  <div
                    className="flex-1 bg-gradient-to-t from-red-600 to-red-400 rounded-t hover:from-red-500 hover:to-red-300 transition-all"
                    style={{ height: `${(data.attacks / 100) * 100}%` }}
                    title={`Attacks: ${data.attacks}`}
                  />
                  <div
                    className="flex-1 bg-gradient-to-t from-green-600 to-green-400 rounded-t hover:from-green-500 hover:to-green-300 transition-all"
                    style={{ height: `${(data.blocks / 100) * 100}%` }}
                    title={`Blocks: ${data.blocks}`}
                  />
                </div>
                <span className="text-xs text-muted-foreground mt-2">{data.time}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="flex items-center justify-center gap-6 mt-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-gradient-to-r from-red-600 to-red-400 rounded" />
            <span className="text-muted-foreground">Attacks</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-gradient-to-r from-green-600 to-green-400 rounded" />
            <span className="text-muted-foreground">Blocks</span>
          </div>
        </div>
      </div>
    </div>
  );
}
