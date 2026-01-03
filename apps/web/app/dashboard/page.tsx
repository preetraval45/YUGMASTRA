'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Activity,
  Shield,
  Swords,
  TrendingUp,
  CheckCircle,
  Zap,
  Target,
  Brain,
  Database,
  Network,
  ArrowRight,
  Sparkles,
  BarChart3
} from 'lucide-react';

interface DashboardMetrics {
  totalEpisodes: number;
  redWins: number;
  blueWins: number;
  activeThreats: number;
  blockedAttacks: number;
  systemHealth: number;
}

export default function Dashboard() {
  const router = useRouter();
  const [metrics] = useState<DashboardMetrics>({
    totalEpisodes: 523,
    redWins: 271,
    blueWins: 252,
    activeThreats: 3,
    blockedAttacks: 847,
    systemHealth: 94,
  });

  const quickActions = [
    {
      title: 'Live Attack Simulation',
      description: 'Real-time red vs blue team battle',
      icon: Swords,
      href: '/attack-simulator',
      color: 'from-red-500 to-orange-500',
      badge: 'LIVE',
    },
    {
      title: 'AI Assistant',
      description: 'Chat with security AI',
      icon: Brain,
      href: '/ai-assistant',
      color: 'from-purple-500 to-pink-500',
      badge: 'LIVE',
    },
    {
      title: 'Live Battle Arena',
      description: 'Multi-agent combat zone',
      icon: Target,
      href: '/live-battle',
      color: 'from-blue-500 to-cyan-500',
      badge: 'LIVE',
    },
    {
      title: 'Cyber Range',
      description: 'Practice environment',
      icon: Zap,
      href: '/cyber-range',
      color: 'from-green-500 to-emerald-500',
      badge: 'LIVE',
    },
    {
      title: 'Model Training',
      description: 'Train AI defense models',
      icon: TrendingUp,
      href: '/model-training',
      color: 'from-yellow-500 to-orange-500',
      badge: 'LIVE',
    },
    {
      title: 'Threat Intelligence',
      description: 'Latest threat feeds',
      icon: Database,
      href: '/threat-intelligence',
      color: 'from-indigo-500 to-purple-500',
    },
    {
      title: 'ML Analytics',
      description: 'Performance metrics',
      icon: BarChart3,
      href: '/analytics',
      color: 'from-pink-500 to-rose-500',
      badge: 'LIVE',
    },
    {
      title: 'Knowledge Graph',
      description: 'Attack pattern visualization',
      icon: Network,
      href: '/knowledge-graph',
      color: 'from-teal-500 to-cyan-500',
    },
  ];

  const systemStatus = [
    { name: 'Red Team AI', status: 'active', icon: Swords, color: 'text-red-500' },
    { name: 'Blue Team AI', status: 'active', icon: Shield, color: 'text-blue-500' },
    { name: 'Cyber Range', status: 'online', icon: Target, color: 'text-green-500' },
    { name: 'Knowledge Graph', status: 'syncing', icon: Network, color: 'text-yellow-500' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/20">
      <div className="container-responsive py-4 sm:py-6 md:py-8 pt-20 sm:pt-24 md:pt-28 lg:pt-32">
        {/* Header */}
        <div className="mb-6 sm:mb-8">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-2">
            <div>
              <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold bg-gradient-to-r from-primary to-purple-500 bg-clip-text text-transparent">
                Command Center
              </h1>
              <p className="text-muted-foreground mt-1 sm:mt-2 text-sm sm:text-base">
                Real-time autonomous cyber defense operations
              </p>
            </div>
            <div className="flex items-center gap-2 sm:gap-3 bg-card/50 backdrop-blur-lg rounded-lg px-3 sm:px-4 md:px-6 py-2 sm:py-3 border shadow-lg">
              <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
              <div>
                <p className="text-xs text-muted-foreground">System Owner</p>
                <p className="text-xs sm:text-sm font-semibold">Preet Raval</p>
              </div>
            </div>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 md:gap-6 mb-6 sm:mb-8">
          <div className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 backdrop-blur-lg rounded-lg sm:rounded-xl p-4 sm:p-5 md:p-6 border border-blue-500/20 shadow-lg hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
            <div className="flex items-center justify-between mb-3 sm:mb-4">
              <div className="p-2 sm:p-2.5 md:p-3 bg-blue-500/20 rounded-lg">
                <Activity className="w-4 h-4 sm:w-5 sm:h-5 md:w-6 md:h-6 text-blue-500" />
              </div>
              <span className="text-xs text-blue-400 font-semibold">+12 today</span>
            </div>
            <h3 className="text-2xl sm:text-3xl font-bold text-blue-500">{metrics.totalEpisodes}</h3>
            <p className="text-xs sm:text-sm text-muted-foreground mt-1">Total Episodes</p>
          </div>

          <div className="bg-gradient-to-br from-red-500/10 to-orange-500/10 backdrop-blur-lg rounded-lg sm:rounded-xl p-4 sm:p-5 md:p-6 border border-red-500/20 shadow-lg hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
            <div className="flex items-center justify-between mb-3 sm:mb-4">
              <div className="p-2 sm:p-2.5 md:p-3 bg-red-500/20 rounded-lg">
                <Swords className="w-4 h-4 sm:w-5 sm:h-5 md:w-6 md:h-6 text-red-500" />
              </div>
              <span className="text-xs text-red-400 font-semibold">
                {((metrics.redWins / metrics.totalEpisodes) * 100).toFixed(1)}%
              </span>
            </div>
            <h3 className="text-2xl sm:text-3xl font-bold text-red-500">{metrics.redWins}</h3>
            <p className="text-sm text-muted-foreground mt-1">Red Team Wins</p>
          </div>

          <div className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 backdrop-blur-lg rounded-xl p-6 border border-green-500/20 shadow-lg hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-green-500/20 rounded-lg">
                <Shield className="w-6 h-6 text-green-500" />
              </div>
              <span className="text-xs text-green-400 font-semibold">
                {((metrics.blueWins / metrics.totalEpisodes) * 100).toFixed(1)}%
              </span>
            </div>
            <h3 className="text-3xl font-bold text-green-500">{metrics.blueWins}</h3>
            <p className="text-sm text-muted-foreground mt-1">Blue Team Wins</p>
          </div>

          <div className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 backdrop-blur-lg rounded-xl p-6 border border-purple-500/20 shadow-lg hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-purple-500/20 rounded-lg">
                <CheckCircle className="w-6 h-6 text-purple-500" />
              </div>
              <span className="text-xs text-purple-400 font-semibold">{metrics.systemHealth}%</span>
            </div>
            <h3 className="text-3xl font-bold text-purple-500">{metrics.blockedAttacks}</h3>
            <p className="text-sm text-muted-foreground mt-1">Blocked Attacks</p>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="w-5 h-5 text-primary" />
            <h2 className="text-2xl font-bold">Quick Actions</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {quickActions.map((action) => (
              <button
                key={action.href}
                onClick={() => router.push(action.href)}
                className="group relative bg-card/50 backdrop-blur-lg rounded-xl p-6 border hover:border-primary/50 shadow-lg hover:shadow-xl transition-all cursor-pointer overflow-hidden text-left"
              >
                <div className={`absolute inset-0 bg-gradient-to-br ${action.color} opacity-0 group-hover:opacity-10 transition-opacity`}></div>
                <div className="relative z-10">
                  <div className="flex items-start justify-between mb-3">
                    <div className={`p-3 bg-gradient-to-br ${action.color} rounded-lg`}>
                      <action.icon className="w-5 h-5 text-white" />
                    </div>
                    {action.badge && (
                      <span className="px-2 py-1 text-[10px] font-bold bg-red-500 text-white rounded animate-pulse">
                        {action.badge}
                      </span>
                    )}
                  </div>
                  <h3 className="font-semibold mb-1 group-hover:text-primary transition-colors">
                    {action.title}
                  </h3>
                  <p className="text-xs text-muted-foreground mb-3">
                    {action.description}
                  </p>
                  <div className="flex items-center text-xs text-primary opacity-0 group-hover:opacity-100 transition-opacity">
                    <span>Launch</span>
                    <ArrowRight className="w-3 h-3 ml-1 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* System Status */}
        <div className="bg-card/50 backdrop-blur-lg rounded-xl p-6 border shadow-lg">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-primary" />
            System Status
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {systemStatus.map((system) => (
              <div
                key={system.name}
                className="flex items-center justify-between p-4 bg-accent/50 rounded-lg border hover:border-primary/50 transition-all"
              >
                <div className="flex items-center gap-3">
                  <system.icon className={`w-5 h-5 ${system.color}`} />
                  <div>
                    <p className="text-sm font-medium">{system.name}</p>
                    <p className="text-xs text-muted-foreground capitalize">{system.status}</p>
                  </div>
                </div>
                <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
