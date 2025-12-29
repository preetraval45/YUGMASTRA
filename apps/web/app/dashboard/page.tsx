'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Activity, Shield, Swords, TrendingUp, AlertTriangle, CheckCircle, Play, Eye, RefreshCw } from 'lucide-react';
import { sendNotification } from '@/hooks/use-notifications';

interface DashboardMetrics {
  totalEpisodes: number;
  redWins: number;
  blueWins: number;
  currentPhase: string;
  redWinRate: number;
  blueDetectionRate: number;
  activeAttacks: number;
  blockedAttacks: number;
}

export default function Dashboard() {
  const router = useRouter();
  const [metrics, setMetrics] = useState<DashboardMetrics>({
    totalEpisodes: 523,
    redWins: 271,
    blueWins: 252,
    currentPhase: 'exploration',
    redWinRate: 0.52,
    blueDetectionRate: 0.48,
    activeAttacks: 12,
    blockedAttacks: 34,
  });

  const [realtimeUpdates, setRealtimeUpdates] = useState<string[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isResetting, setIsResetting] = useState(false);

  const handleStartTraining = () => {
    setIsTraining(true);
    sendNotification('system', 'Training Started', 'New co-evolution training session initiated', 'low');

    setTimeout(() => {
      setMetrics(prev => ({
        ...prev,
        totalEpisodes: prev.totalEpisodes + 1
      }));
      setIsTraining(false);
      sendNotification('system', 'Training Complete', 'Episode finished successfully', 'low');
    }, 3000);
  };

  const handleViewEvolution = () => {
    router.push('/evolution');
  };

  const handleResetCyberRange = () => {
    setIsResetting(true);
    sendNotification('system', 'Resetting Cyber Range', 'All systems are being reset to initial state', 'medium');

    setTimeout(() => {
      setMetrics({
        totalEpisodes: 0,
        redWins: 0,
        blueWins: 0,
        currentPhase: 'initialization',
        redWinRate: 0,
        blueDetectionRate: 0,
        activeAttacks: 0,
        blockedAttacks: 0,
      });
      setRealtimeUpdates([]);
      setIsResetting(false);
      sendNotification('system', 'Reset Complete', 'Cyber Range has been reset successfully', 'low');
    }, 2000);
  };

  useEffect(() => {
    // Simulate real-time updates
    const interval = setInterval(() => {
      const updates = [
        'Red agent discovered new attack path',
        'Blue agent updated detection rule',
        'Nash equilibrium distance decreased to 0.23',
        'New vulnerability chain detected',
        'Defense strategy adapted successfully',
      ];
      const randomUpdate = updates[Math.floor(Math.random() * updates.length)];
      setRealtimeUpdates(prev => [randomUpdate, ...prev.slice(0, 9)]);
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 p-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">Dashboard</h1>
            <p className="text-blue-200">Real-time co-evolution metrics and system status</p>
          </div>
          <div className="bg-white/10 backdrop-blur-lg rounded-lg px-6 py-3 border border-blue-500/30">
            <p className="text-sm text-gray-300">Defending system owned by</p>
            <p className="text-lg font-semibold text-white">Preet Raval</p>
            <p className="text-xs text-blue-300">preetraval45@gmail.com</p>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {/* Total Episodes */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20 hover:bg-white/15 hover:border-blue-400/50 transition-all cursor-pointer transform hover:scale-105 hover:shadow-xl hover:shadow-blue-500/20">
          <div className="flex items-center justify-between mb-4">
            <div className="p-3 bg-blue-500/20 rounded-lg">
              <Activity className="w-6 h-6 text-blue-400" />
            </div>
            <span className="text-xs text-green-400 font-semibold">+12 today</span>
          </div>
          <h3 className="text-2xl font-bold text-white">{metrics.totalEpisodes}</h3>
          <p className="text-sm text-gray-300">Total Episodes</p>
        </div>

        {/* Red Team Wins */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20 hover:bg-white/15 hover:border-red-400/50 transition-all cursor-pointer transform hover:scale-105 hover:shadow-xl hover:shadow-red-500/20">
          <div className="flex items-center justify-between mb-4">
            <div className="p-3 bg-red-500/20 rounded-lg">
              <Swords className="w-6 h-6 text-red-400" />
            </div>
            <span className="text-xs text-red-400 font-semibold">
              {(metrics.redWinRate * 100).toFixed(1)}%
            </span>
          </div>
          <h3 className="text-2xl font-bold text-white">{metrics.redWins}</h3>
          <p className="text-sm text-gray-300">Red Team Wins</p>
        </div>

        {/* Blue Team Wins */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20 hover:bg-white/15 hover:border-blue-400/50 transition-all cursor-pointer transform hover:scale-105 hover:shadow-xl hover:shadow-blue-500/20">
          <div className="flex items-center justify-between mb-4">
            <div className="p-3 bg-blue-500/20 rounded-lg">
              <Shield className="w-6 h-6 text-blue-400" />
            </div>
            <span className="text-xs text-blue-400 font-semibold">
              {(metrics.blueDetectionRate * 100).toFixed(1)}%
            </span>
          </div>
          <h3 className="text-2xl font-bold text-white">{metrics.blueWins}</h3>
          <p className="text-sm text-gray-300">Blue Team Wins</p>
        </div>

        {/* Evolution Phase */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20 hover:bg-white/15 hover:border-purple-400/50 transition-all cursor-pointer transform hover:scale-105 hover:shadow-xl hover:shadow-purple-500/20">
          <div className="flex items-center justify-between mb-4">
            <div className="p-3 bg-purple-500/20 rounded-lg">
              <TrendingUp className="w-6 h-6 text-purple-400" />
            </div>
            <span className="text-xs text-purple-400 font-semibold uppercase">
              {metrics.currentPhase}
            </span>
          </div>
          <h3 className="text-2xl font-bold text-white">Active</h3>
          <p className="text-sm text-gray-300">Evolution Phase</p>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Real-time Activity */}
        <div className="lg:col-span-2 bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
          <h2 className="text-xl font-bold text-white mb-4">Real-time Activity</h2>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {realtimeUpdates.map((update, index) => (
              <div
                key={index}
                className="flex items-start gap-3 p-3 bg-white/5 rounded-lg border border-white/10 hover:bg-white/10 transition-colors"
              >
                <div className="p-2 bg-blue-500/20 rounded">
                  <Activity className="w-4 h-4 text-blue-400" />
                </div>
                <div className="flex-1">
                  <p className="text-sm text-white">{update}</p>
                  <p className="text-xs text-gray-400 mt-1">Just now</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* System Status */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
          <h2 className="text-xl font-bold text-white mb-4">System Status</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <span className="text-sm text-white">Cyber Range</span>
              </div>
              <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded">
                Online
              </span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <span className="text-sm text-white">Red Team AI</span>
              </div>
              <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded">
                Training
              </span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <span className="text-sm text-white">Blue Team AI</span>
              </div>
              <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded">
                Training
              </span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-yellow-400" />
                <span className="text-sm text-white">Knowledge Graph</span>
              </div>
              <span className="text-xs bg-yellow-500/20 text-yellow-400 px-2 py-1 rounded">
                Indexing
              </span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <span className="text-sm text-white">API Gateway</span>
              </div>
              <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded">
                Healthy
              </span>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="mt-6 pt-6 border-t border-white/10">
            <h3 className="text-sm font-semibold text-white mb-3">Quick Actions</h3>
            <div className="space-y-2">
              <button
                onClick={handleStartTraining}
                disabled={isTraining}
                className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed text-white text-sm rounded-lg transition-all flex items-center justify-center gap-2"
              >
                <Play className="w-4 h-4" />
                {isTraining ? 'Training...' : 'Start New Training'}
              </button>
              <button
                onClick={handleViewEvolution}
                className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded-lg transition-all flex items-center justify-center gap-2"
              >
                <Eye className="w-4 h-4" />
                View Evolution
              </button>
              <button
                onClick={handleResetCyberRange}
                disabled={isResetting}
                className="w-full px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-red-400 disabled:cursor-not-allowed text-white text-sm rounded-lg transition-all flex items-center justify-center gap-2"
              >
                <RefreshCw className={`w-4 h-4 ${isResetting ? 'animate-spin' : ''}`} />
                {isResetting ? 'Resetting...' : 'Reset Cyber Range'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Charts Preview */}
      <div className="mt-6 bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
        <h2 className="text-xl font-bold text-white mb-4">Performance Overview</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-white/5 rounded-lg">
            <p className="text-sm text-gray-300 mb-2">Win Rate Trend</p>
            <div className="h-20 flex items-end gap-1">
              {[0.45, 0.48, 0.52, 0.50, 0.53, 0.52, 0.54].map((val, i) => (
                <div
                  key={i}
                  className="flex-1 bg-gradient-to-t from-blue-600 to-blue-400 rounded-t"
                  style={{ height: `${val * 100}%` }}
                />
              ))}
            </div>
          </div>
          <div className="p-4 bg-white/5 rounded-lg">
            <p className="text-sm text-gray-300 mb-2">Detection Rate</p>
            <div className="h-20 flex items-end gap-1">
              {[0.42, 0.45, 0.47, 0.48, 0.46, 0.49, 0.48].map((val, i) => (
                <div
                  key={i}
                  className="flex-1 bg-gradient-to-t from-green-600 to-green-400 rounded-t"
                  style={{ height: `${val * 100}%` }}
                />
              ))}
            </div>
          </div>
          <div className="p-4 bg-white/5 rounded-lg">
            <p className="text-sm text-gray-300 mb-2">Equilibrium Distance</p>
            <div className="h-20 flex items-end gap-1">
              {[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25].map((val, i) => (
                <div
                  key={i}
                  className="flex-1 bg-gradient-to-t from-purple-600 to-purple-400 rounded-t"
                  style={{ height: `${val * 100}%` }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
