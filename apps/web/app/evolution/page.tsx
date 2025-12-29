'use client';

import { useState } from 'react';
import { TrendingUp, Users, Target, BarChart3 } from 'lucide-react';

export default function EvolutionPage() {
  const [timeRange, setTimeRange] = useState('24h');

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Co-Evolution Metrics</h1>
          <p className="text-muted-foreground">Track adversarial training progress and Nash equilibrium</p>
        </div>

        {/* Time Range Selector */}
        <div className="flex gap-2 mb-6">
          {['1h', '24h', '7d', '30d', 'All'].map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-lg transition-colors ${
                timeRange === range
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-card hover:bg-accent'
              }`}
            >
              {range}
            </button>
          ))}
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-card rounded-lg p-6 border">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-blue-500/10 rounded-lg">
                <TrendingUp className="w-6 h-6 text-blue-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Nash Distance</p>
                <h3 className="text-2xl font-bold">0.23</h3>
              </div>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-green-500">↓ 12%</span>
              <span className="text-muted-foreground">vs last period</span>
            </div>
          </div>

          <div className="bg-card rounded-lg p-6 border">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-purple-500/10 rounded-lg">
                <Users className="w-6 h-6 text-purple-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Strategy Diversity</p>
                <h3 className="text-2xl font-bold">0.78</h3>
              </div>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-green-500">↑ 5%</span>
              <span className="text-muted-foreground">vs last period</span>
            </div>
          </div>

          <div className="bg-card rounded-lg p-6 border">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-red-500/10 rounded-lg">
                <Target className="w-6 h-6 text-red-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Red Win Rate</p>
                <h3 className="text-2xl font-bold">52.3%</h3>
              </div>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-red-500">↑ 2.1%</span>
              <span className="text-muted-foreground">vs last period</span>
            </div>
          </div>

          <div className="bg-card rounded-lg p-6 border">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-green-500/10 rounded-lg">
                <BarChart3 className="w-6 h-6 text-green-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Blue Detection</p>
                <h3 className="text-2xl font-bold">48.9%</h3>
              </div>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-green-500">↑ 3.4%</span>
              <span className="text-muted-foreground">vs last period</span>
            </div>
          </div>
        </div>

        {/* Main Chart */}
        <div className="bg-card rounded-lg p-6 border mb-6">
          <h2 className="text-xl font-bold mb-4">Win Rate Evolution</h2>
          <div className="h-96 flex items-end gap-2">
            {Array.from({ length: 50 }, (_, i) => {
              const redRate = 0.5 + Math.sin(i / 5) * 0.1 + (Math.random() - 0.5) * 0.05;
              const blueRate = 1 - redRate;
              return (
                <div key={i} className="flex-1 flex flex-col justify-end gap-1">
                  <div
                    className="bg-gradient-to-t from-red-600 to-red-400 rounded-t"
                    style={{ height: `${redRate * 100}%` }}
                  />
                  <div
                    className="bg-gradient-to-t from-blue-600 to-blue-400"
                    style={{ height: `${blueRate * 100}%` }}
                  />
                </div>
              );
            })}
          </div>
          <div className="flex justify-center gap-6 mt-4">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-red-500 rounded" />
              <span className="text-sm">Red Team</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-blue-500 rounded" />
              <span className="text-sm">Blue Team</span>
            </div>
          </div>
        </div>

        {/* Population Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-card rounded-lg p-6 border">
            <h2 className="text-xl font-bold mb-4">Red Team Population</h2>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Avg Fitness</span>
                  <span className="font-semibold">0.65</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-red-500" style={{ width: '65%' }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Best Fitness</span>
                  <span className="font-semibold">0.82</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-red-600" style={{ width: '82%' }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Diversity</span>
                  <span className="font-semibold">0.70</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-red-400" style={{ width: '70%' }} />
                </div>
              </div>
            </div>
          </div>

          <div className="bg-card rounded-lg p-6 border">
            <h2 className="text-xl font-bold mb-4">Blue Team Population</h2>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Avg Fitness</span>
                  <span className="font-semibold">0.58</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500" style={{ width: '58%' }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Best Fitness</span>
                  <span className="font-semibold">0.75</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-blue-600" style={{ width: '75%' }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Diversity</span>
                  <span className="font-semibold">0.65</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-blue-400" style={{ width: '65%' }} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
