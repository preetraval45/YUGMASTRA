'use client';

import { useState, useEffect } from 'react';
import { TrendingUp, Users, Target, BarChart3, Zap, Trophy, Activity, Flame } from 'lucide-react';

export default function EvolutionPage() {
  const [timeRange, setTimeRange] = useState('24h');
  const [generation, setGeneration] = useState(1247);
  const [nashDistance, setNashDistance] = useState(0.23);
  const [redWinRate, setRedWinRate] = useState(52.3);
  const [blueDetection, setBlueDetection] = useState(48.9);
  const [winRateHistory, setWinRateHistory] = useState<number[]>([]);

  // Simulate evolution progress
  useEffect(() => {
    // Initialize history
    const initialHistory = Array.from({ length: 50 }, (_, i) =>
      50 + Math.sin(i / 5) * 10 + (Math.random() - 0.5) * 5
    );
    setWinRateHistory(initialHistory);

    const interval = setInterval(() => {
      // Update generation
      setGeneration(prev => prev + 1);

      // Gradually decrease Nash distance (converging)
      setNashDistance(prev => Math.max(0.05, prev - 0.001 + Math.random() * 0.002));

      // Oscillate win rates around 50% (equilibrium)
      setRedWinRate(prev => {
        const target = 50;
        const change = (target - prev) * 0.1 + (Math.random() - 0.5) * 2;
        return Math.max(45, Math.min(55, prev + change));
      });

      setBlueDetection(prev => {
        const target = 100 - redWinRate;
        const change = (target - prev) * 0.1 + (Math.random() - 0.5) * 2;
        return Math.max(45, Math.min(55, prev + change));
      });

      // Update history
      setWinRateHistory(prev => {
        const newHistory = [...prev.slice(1), redWinRate];
        return newHistory;
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [redWinRate]);

  return (
    <div className="min-h-screen bg-background p-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-4xl font-bold  mb-2 flex items-center gap-3">
              <Activity className="w-8 h-8 text-purple-500 animate-pulse" />
              Co-Evolution Arena
            </h1>
            <p className="text-muted-foreground">
              Watch adversarial AI agents evolve through competition - Preet Raval's System
            </p>
          </div>
          <div className="bg-card rounded-lg px-6 py-3 border border-purple-500/30">
            <p className="text-sm text-muted-foreground">Generation</p>
            <p className="text-3xl font-bold  font-mono">#{generation}</p>
            <p className="text-xs text-purple-500 animate-pulse">Evolving...</p>
          </div>
        </div>

        {/* Time Range Selector */}
        <div className="flex gap-2">
          {['1h', '24h', '7d', '30d', 'All'].map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                timeRange === range
                  ? 'bg-purple-600  shadow-lg shadow-purple-500/50'
                  : 'bg-card text-muted-foreground hover:bg-white/20'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Key Metrics with Animations */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-card rounded-lg p-6 border border-blue-500/30 hover:border-blue-500/50 transition-all">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-blue-500/20 rounded-lg animate-pulse">
              <TrendingUp className="w-6 h-6 text-primary" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Nash Distance</p>
              <h3 className="text-3xl font-bold  font-mono">{nashDistance.toFixed(3)}</h3>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-green-500 flex items-center gap-1">
              <TrendingUp className="w-3 h-3" />
              Converging
            </span>
            <span className="text-muted-foreground">to equilibrium</span>
          </div>
          <div className="mt-3 h-1 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all duration-1000"
              style={{ width: `${(1 - nashDistance) * 100}%` }}
            />
          </div>
        </div>

        <div className="bg-card rounded-lg p-6 border border-purple-500/30 hover:border-purple-500/50 transition-all">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-purple-500/20 rounded-lg">
              <Users className="w-6 h-6 text-purple-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Strategy Diversity</p>
              <h3 className="text-3xl font-bold ">High</h3>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-purple-500">127 unique</span>
            <span className="text-muted-foreground">strategies</span>
          </div>
          <div className="mt-3 grid grid-cols-8 gap-1">
            {Array.from({ length: 24 }).map((_, i) => (
              <div
                key={i}
                className="h-6 rounded"
                style={{
                  backgroundColor: `hsl(${i * 15}, 70%, ${50 + Math.random() * 20}%)`,
                  animation: `pulse ${1 + Math.random()}s infinite`
                }}
              />
            ))}
          </div>
        </div>

        <div className="bg-card rounded-lg p-6 border border-red-500/30 hover:border-red-500/50 transition-all">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-red-500/20 rounded-lg">
              <Flame className="w-6 h-6 text-red-500 animate-pulse" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Red Win Rate</p>
              <h3 className="text-3xl font-bold  font-mono">{redWinRate.toFixed(1)}%</h3>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className={`${redWinRate > 50 ? 'text-red-500' : 'text-primary'}`}>
              {redWinRate > 50 ? '↑ Attacking' : '↓ Defending'}
            </span>
          </div>
          <div className="mt-3 h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-red-600 to-red-400 transition-all duration-1000"
              style={{ width: `${redWinRate}%` }}
            />
          </div>
        </div>

        <div className="bg-card rounded-lg p-6 border border-green-500/30 hover:border-green-500/50 transition-all">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-green-500/20 rounded-lg">
              <Target className="w-6 h-6 text-green-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Blue Detection</p>
              <h3 className="text-3xl font-bold  font-mono">{blueDetection.toFixed(1)}%</h3>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className={`${blueDetection > 50 ? 'text-green-500' : 'text-red-500'}`}>
              {blueDetection > 50 ? '↑ Learning' : '↓ Adapting'}
            </span>
          </div>
          <div className="mt-3 h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-green-600 to-green-400 transition-all duration-1000"
              style={{ width: `${blueDetection}%` }}
            />
          </div>
        </div>
      </div>

      {/* Main Evolution Chart */}
      <div className="bg-card rounded-lg p-6 border border mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold  flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-purple-500" />
            Win Rate Evolution
          </h2>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-gradient-to-r from-red-600 to-red-400 rounded" />
              <span className="text-sm ">Red Team</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-gradient-to-r from-blue-600 to-blue-400 rounded" />
              <span className="text-sm ">Blue Team</span>
            </div>
          </div>
        </div>
        <div className="relative h-96 bg-accent/20 rounded-lg p-4">
          {/* Grid lines */}
          <div className="absolute inset-0 flex flex-col justify-between p-4 pointer-events-none">
            {[0, 25, 50, 75, 100].map((val) => (
              <div key={val} className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-8">{val}%</span>
                <div className="flex-1 border-t border-gray-700/50" />
              </div>
            ))}
          </div>

          {/* Chart */}
          <div className="absolute inset-0 p-4 pl-12 flex items-end gap-1">
            {winRateHistory.map((rate, i) => {
              const redRate = rate / 100;
              const blueRate = (100 - rate) / 100;
              return (
                <div key={i} className="flex-1 flex flex-col justify-end gap-0.5 group relative">
                  <div
                    className="bg-gradient-to-t from-red-600 to-red-400 rounded-t transition-all duration-500 hover:from-red-500 hover:to-red-300"
                    style={{ height: `${redRate * 100}%` }}
                  />
                  <div
                    className="bg-gradient-to-t from-blue-600 to-blue-400 transition-all duration-500 hover:from-blue-500 hover:to-blue-300"
                    style={{ height: `${blueRate * 100}%` }}
                  />
                  {/* Tooltip on hover */}
                  <div className="absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 hidden group-hover:block bg-popover  text-xs px-2 py-1 rounded whitespace-nowrap">
                    Gen {generation - (50 - i)}<br/>
                    R: {rate.toFixed(1)}%<br/>
                    B: {(100 - rate).toFixed(1)}%
                  </div>
                </div>
              );
            })}
          </div>

          {/* Equilibrium line */}
          <div className="absolute top-1/2 left-12 right-4 border-t-2 border-yellow-500/50 border-dashed pointer-events-none">
            <span className="absolute right-0 top-0 transform -translate-y-1/2 bg-yellow-500/20 text-yellow-300 text-xs px-2 py-1 rounded">
              Nash Equilibrium (50%)
            </span>
          </div>
        </div>
      </div>

      {/* Population Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="bg-card rounded-lg p-6 border border-red-500/30">
          <h2 className="text-2xl font-bold  mb-4 flex items-center gap-2">
            <Flame className="w-6 h-6 text-red-500" />
            Red Team Population (Attackers)
          </h2>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-muted-foreground">Average Attack Success</span>
                <span className="font-semibold  font-mono">65.2%</span>
              </div>
              <div className="h-3 bg-muted rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-red-600 to-red-400 rounded-full transition-all duration-1000" style={{ width: '65.2%' }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-muted-foreground">Best Agent Performance</span>
                <span className="font-semibold  font-mono">82.7%</span>
              </div>
              <div className="h-3 bg-muted rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-red-700 to-red-500 rounded-full transition-all duration-1000" style={{ width: '82.7%' }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-muted-foreground">Strategy Diversity</span>
                <span className="font-semibold  font-mono">70.3%</span>
              </div>
              <div className="h-3 bg-muted rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-orange-600 to-orange-400 rounded-full transition-all duration-1000" style={{ width: '70.3%' }} />
              </div>
            </div>
          </div>

          {/* Top strategies */}
          <div className="mt-6 pt-6 border-t border">
            <h3 className="text-sm font-semibold  mb-3">Evolved Attack Strategies:</h3>
            <div className="space-y-2">
              {[
                { name: 'Multi-stage SQLi Chain', usage: 23 },
                { name: 'Polymorphic XSS Variants', usage: 19 },
                { name: 'Adaptive Timing Attacks', usage: 16 },
                { name: 'Hybrid Exploit Combos', usage: 14 },
              ].map((strategy, i) => (
                <div key={i} className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">{strategy.name}</span>
                  <span className="bg-red-500/20 text-red-300 px-2 py-1 rounded">
                    {strategy.usage}% usage
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="bg-card rounded-lg p-6 border border-blue-500/30">
          <h2 className="text-2xl font-bold  mb-4 flex items-center gap-2">
            <Target className="w-6 h-6 text-primary" />
            Blue Team Population (Defenders)
          </h2>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-muted-foreground">Average Detection Rate</span>
                <span className="font-semibold  font-mono">58.4%</span>
              </div>
              <div className="h-3 bg-muted rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full transition-all duration-1000" style={{ width: '58.4%' }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-muted-foreground">Best Agent Performance</span>
                <span className="font-semibold  font-mono">75.9%</span>
              </div>
              <div className="h-3 bg-muted rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-blue-700 to-blue-500 rounded-full transition-all duration-1000" style={{ width: '75.9%' }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-muted-foreground">Defense Diversity</span>
                <span className="font-semibold  font-mono">65.8%</span>
              </div>
              <div className="h-3 bg-muted rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-cyan-600 to-cyan-400 rounded-full transition-all duration-1000" style={{ width: '65.8%' }} />
              </div>
            </div>
          </div>

          {/* Top strategies */}
          <div className="mt-6 pt-6 border-t border">
            <h3 className="text-sm font-semibold  mb-3">Learned Defense Patterns:</h3>
            <div className="space-y-2">
              {[
                { name: 'Behavioral Anomaly Detection', usage: 28 },
                { name: 'Pattern-based Signatures', usage: 22 },
                { name: 'Threshold Adaptation', usage: 18 },
                { name: 'Deception Honeypots', usage: 11 },
              ].map((strategy, i) => (
                <div key={i} className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">{strategy.name}</span>
                  <span className="bg-blue-500/20 text-blue-300 px-2 py-1 rounded">
                    {strategy.usage}% usage
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Convergence Analysis */}
      <div className="bg-card rounded-lg p-6 border border-purple-500/30">
        <h2 className="text-2xl font-bold  mb-4 flex items-center gap-2">
          <Trophy className="w-6 h-6 text-yellow-500" />
          Convergence Analysis
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-4 bg-accent/50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-5 h-5 text-purple-500" />
              <h3 className="font-semibold ">Equilibrium Status</h3>
            </div>
            <p className="text-2xl font-bold text-purple-500 mb-1">Approaching</p>
            <p className="text-sm text-muted-foreground">Distance: {nashDistance.toFixed(3)} (target: &lt; 0.10)</p>
            <div className="mt-3 flex items-center gap-2">
              <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-purple-600 to-purple-400 rounded-full"
                  style={{ width: `${(1 - nashDistance / 0.5) * 100}%` }}
                />
              </div>
              <span className="text-xs text-muted-foreground">{((1 - nashDistance / 0.5) * 100).toFixed(0)}%</span>
            </div>
          </div>

          <div className="p-4 bg-accent/50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-5 h-5 text-green-500" />
              <h3 className="font-semibold ">Learning Rate</h3>
            </div>
            <p className="text-2xl font-bold text-green-500 mb-1">Active</p>
            <p className="text-sm text-muted-foreground">Both teams improving steadily</p>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <div className="bg-red-500/20 text-red-300 px-2 py-1 rounded text-center">
                Red: +2.3% /gen
              </div>
              <div className="bg-blue-500/20 text-blue-300 px-2 py-1 rounded text-center">
                Blue: +1.8% /gen
              </div>
            </div>
          </div>

          <div className="p-4 bg-accent/50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Trophy className="w-5 h-5 text-yellow-500" />
              <h3 className="font-semibold ">Predicted Winner</h3>
            </div>
            <p className="text-2xl font-bold text-yellow-500 mb-1">Balanced</p>
            <p className="text-sm text-muted-foreground">Converging to 50/50 split</p>
            <div className="mt-3 text-xs text-center bg-yellow-500/20 text-yellow-300 px-2 py-2 rounded">
              🏆 Perfect Nash Equilibrium Expected
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
