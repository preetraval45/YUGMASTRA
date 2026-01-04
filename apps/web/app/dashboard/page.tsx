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
  BarChart3,
  HelpCircle,
  Info
} from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

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
      helpText: 'Watch autonomous AI agents attack and defend in real-time. Red Team AI attempts various attack strategies while Blue Team AI learns to detect and block them.',
      icon: Swords,
      href: '/attack-simulator',
      color: 'from-red-500 to-orange-500',
      badge: 'LIVE',
    },
    {
      title: 'AI Assistant',
      description: 'Chat with security AI',
      helpText: 'Ask questions about cybersecurity, get vulnerability analysis, and receive expert recommendations from our AI-powered security assistant.',
      icon: Brain,
      href: '/ai-assistant',
      color: 'from-purple-500 to-pink-500',
      badge: 'LIVE',
    },
    {
      title: 'Live Battle Arena',
      description: 'Multi-agent combat zone',
      helpText: 'Advanced multi-agent reinforcement learning environment where multiple red and blue team agents compete simultaneously to develop sophisticated attack and defense strategies.',
      icon: Target,
      href: '/live-battle',
      color: 'from-blue-500 to-cyan-500',
      badge: 'LIVE',
    },
    {
      title: 'Cyber Range',
      description: 'Practice environment',
      helpText: 'Isolated training environment for testing security tools, practicing incident response, and experimenting with attack techniques safely.',
      icon: Zap,
      href: '/cyber-range',
      color: 'from-green-500 to-emerald-500',
      badge: 'LIVE',
    },
    {
      title: 'Model Training',
      description: 'Train AI defense models',
      helpText: 'Configure and train custom machine learning models for threat detection, behavior analysis, and automated response using your security data.',
      icon: TrendingUp,
      href: '/model-training',
      color: 'from-yellow-500 to-orange-500',
      badge: 'LIVE',
    },
    {
      title: 'Threat Intelligence',
      description: 'Latest threat feeds',
      helpText: 'Real-time threat intelligence feeds aggregating data from global sources, providing IOCs, TTPs, and emerging threat trends.',
      icon: Database,
      href: '/threat-intelligence',
      color: 'from-indigo-500 to-purple-500',
    },
    {
      title: 'ML Analytics',
      description: 'Performance metrics',
      helpText: 'Detailed analytics on AI model performance, training progress, win rates, and strategy effectiveness over time.',
      icon: BarChart3,
      href: '/analytics',
      color: 'from-pink-500 to-rose-500',
      badge: 'LIVE',
    },
    {
      title: 'Knowledge Graph',
      description: 'Attack pattern visualization',
      helpText: 'Interactive visualization of attack patterns, relationships between TTPs, and how threats evolve through the MITRE ATT&CK framework.',
      icon: Network,
      href: '/knowledge-graph',
      color: 'from-teal-500 to-cyan-500',
    },
  ];

  const systemStatus = [
    { name: 'Red Team AI', status: 'active', icon: Swords, color: 'text-red-500', description: 'Autonomous attack agent exploring exploitation strategies' },
    { name: 'Blue Team AI', status: 'active', icon: Shield, color: 'text-blue-500', description: 'Adaptive defense system learning detection patterns' },
    { name: 'Cyber Range', status: 'online', icon: Target, color: 'text-green-500', description: 'Isolated training and testing environment' },
    { name: 'Knowledge Graph', status: 'syncing', icon: Network, color: 'text-yellow-500', description: 'Attack pattern relationship database' },
  ];

  return (
    <div className="min-h-screen bg-background p-8 pt-32">
      <div className="container-responsive">
        {/* Header with Help */}
        <div className="mb-6 sm:mb-8">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-2">
            <div>
              <div className="flex items-center gap-3">
                <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold bg-gradient-to-r from-primary to-purple-500 bg-clip-text text-transparent">
                  Command Center
                </h1>
                <Dialog>
                  <DialogTrigger asChild>
                    <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                      <HelpCircle className="h-5 w-5 text-primary" />
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
                    <DialogHeader>
                      <DialogTitle className="text-2xl">Dashboard Guide</DialogTitle>
                      <DialogDescription className="text-base">
                        Understanding Your Command Center
                      </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-6 pt-4">
                      <div>
                        <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                          <Info className="h-5 w-5 text-blue-500" />
                          What You See Here
                        </h3>
                        <p className="text-muted-foreground">
                          Your Command Center provides real-time visibility into YUGMASTRA's autonomous adversary-defender co-evolution system.
                          Monitor AI training progress, system health, and access all security tools from this central hub.
                        </p>
                      </div>

                      <div>
                        <h3 className="text-lg font-semibold mb-3">Key Metrics Explained</h3>
                        <div className="space-y-3">
                          <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-3">
                            <p className="font-semibold text-blue-400 mb-1">Total Episodes</p>
                            <p className="text-sm text-muted-foreground">
                              Number of complete attack-defense simulation rounds. Each episode represents a full cycle where Red Team AI attempts attacks and Blue Team AI defends.
                              Higher numbers indicate more training data and model refinement.
                            </p>
                          </div>

                          <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                            <p className="font-semibold text-red-400 mb-1">Red Team Wins</p>
                            <p className="text-sm text-muted-foreground">
                              Episodes where Red Team AI successfully compromised defenses. This measures attack effectiveness and helps identify defense gaps.
                              A balanced win rate (~50%) indicates healthy co-evolution.
                            </p>
                          </div>

                          <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-3">
                            <p className="font-semibold text-green-400 mb-1">Blue Team Wins</p>
                            <p className="text-sm text-muted-foreground">
                              Episodes where Blue Team AI successfully blocked all attacks. This measures defense effectiveness and detection capability.
                              Improving win rate over time shows the defense AI is learning.
                            </p>
                          </div>

                          <div className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-3">
                            <p className="font-semibold text-purple-400 mb-1">Blocked Attacks</p>
                            <p className="text-sm text-muted-foreground">
                              Total number of individual attack attempts that were detected and prevented. This includes malware execution blocks, network intrusion prevention, and anomaly detection triggers.
                            </p>
                          </div>
                        </div>
                      </div>

                      <div>
                        <h3 className="text-lg font-semibold mb-2">Getting Started</h3>
                        <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                          <li>Click <strong>Live Attack Simulation</strong> to watch AI agents battle in real-time</li>
                          <li>Use <strong>AI Assistant</strong> to ask cybersecurity questions and get expert guidance</li>
                          <li>Explore <strong>Cyber Range</strong> to practice security techniques safely</li>
                          <li>Check <strong>ML Analytics</strong> to see how AI models are improving over time</li>
                          <li>View <strong>Knowledge Graph</strong> to visualize attack patterns and relationships</li>
                        </ol>
                      </div>

                      <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4">
                        <h4 className="font-semibold text-yellow-400 mb-2 flex items-center gap-2">
                          <Sparkles className="h-4 w-4" />
                          Pro Tip
                        </h4>
                        <p className="text-sm text-muted-foreground">
                          The "LIVE" badges indicate real-time features that update automatically. These tools show active AI agent behavior as it happens.
                          Try opening multiple tools simultaneously to see how different agents interact!
                        </p>
                      </div>
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
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

          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 flex items-start gap-3">
            <Activity className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                <strong className="text-foreground">What this page does:</strong> Your central command hub for YUGMASTRA's autonomous adversary-defender co-evolution system. Monitor real-time AI training metrics, track Red vs Blue team battle outcomes, launch security tools, and view system health. Access live attack simulations, AI assistants, cyber ranges, and threat intelligence feeds from this unified dashboard.
              </p>
            </div>
          </div>
        </div>

        {/* Metrics Grid with Tooltips */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 md:gap-6 mb-6 sm:mb-8">
          <Dialog>
            <DialogTrigger asChild>
              <div className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 backdrop-blur-lg rounded-lg sm:rounded-xl p-4 sm:p-5 md:p-6 border border-blue-500/20 shadow-lg hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
                <div className="flex items-center justify-between mb-3 sm:mb-4">
                  <div className="p-2 sm:p-2.5 md:p-3 bg-blue-500/20 rounded-lg">
                    <Activity className="w-4 h-4 sm:w-5 sm:h-5 md:w-6 md:h-6 text-blue-500" />
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-xs text-blue-400 font-semibold">+12 today</span>
                    <HelpCircle className="h-3 w-3 text-blue-400" />
                  </div>
                </div>
                <h3 className="text-2xl sm:text-3xl font-bold text-blue-500">{metrics.totalEpisodes}</h3>
                <p className="text-xs sm:text-sm text-muted-foreground mt-1">Total Episodes</p>
              </div>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Total Training Episodes</DialogTitle>
                <DialogDescription>Complete attack-defense simulation rounds</DialogDescription>
              </DialogHeader>
              <div className="space-y-4 pt-4">
                <p className="text-foreground">
                  <strong>Current: {metrics.totalEpisodes} episodes</strong> (+12 completed today)
                </p>
                <p className="text-muted-foreground">
                  Each episode represents one full cycle of adversarial training where:
                </p>
                <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                  <li>Red Team AI attempts various attack strategies on your system</li>
                  <li>Blue Team AI detects, analyzes, and responds to threats</li>
                  <li>Both agents learn from the outcomes to improve their tactics</li>
                  <li>Attack patterns and defense strategies are logged for analysis</li>
                </ul>
                <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-3">
                  <p className="text-sm text-muted-foreground">
                    <strong className="text-blue-400">Why it matters:</strong> More episodes mean more diverse attack scenarios tested,
                    better-trained AI models, and more robust defenses. Your system has experienced {metrics.totalEpisodes} unique security situations.
                  </p>
                </div>
              </div>
            </DialogContent>
          </Dialog>

          <Dialog>
            <DialogTrigger asChild>
              <div className="bg-gradient-to-br from-red-500/10 to-orange-500/10 backdrop-blur-lg rounded-lg sm:rounded-xl p-4 sm:p-5 md:p-6 border border-red-500/20 shadow-lg hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
                <div className="flex items-center justify-between mb-3 sm:mb-4">
                  <div className="p-2 sm:p-2.5 md:p-3 bg-red-500/20 rounded-lg">
                    <Swords className="w-4 h-4 sm:w-5 sm:h-5 md:w-6 md:h-6 text-red-500" />
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-xs text-red-400 font-semibold">
                      {((metrics.redWins / metrics.totalEpisodes) * 100).toFixed(1)}%
                    </span>
                    <HelpCircle className="h-3 w-3 text-red-400" />
                  </div>
                </div>
                <h3 className="text-2xl sm:text-3xl font-bold text-red-500">{metrics.redWins}</h3>
                <p className="text-sm text-muted-foreground mt-1">Red Team Wins</p>
              </div>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Red Team Attack Success</DialogTitle>
                <DialogDescription>Offensive AI effectiveness measurement</DialogDescription>
              </DialogHeader>
              <div className="space-y-4 pt-4">
                <p className="text-foreground">
                  <strong>Win Rate: {((metrics.redWins / metrics.totalEpisodes) * 100).toFixed(1)}%</strong> ({metrics.redWins} successful breaches out of {metrics.totalEpisodes} attempts)
                </p>
                <p className="text-muted-foreground">
                  Red Team wins occur when the offensive AI successfully:
                </p>
                <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                  <li>Bypasses security controls and gains unauthorized access</li>
                  <li>Executes malicious payloads without detection</li>
                  <li>Achieves attack objectives (data exfiltration, privilege escalation, etc.)</li>
                  <li>Evades Blue Team detection for the duration of the episode</li>
                </ul>
                <div className="bg-orange-500/10 border border-orange-500/20 rounded-lg p-3">
                  <p className="text-sm text-muted-foreground">
                    <strong className="text-orange-400">Ideal balance:</strong> A win rate around 50% indicates healthy co-evolution.
                    Too high means defenses need improvement; too low might mean attacks aren't challenging enough to drive innovation.
                  </p>
                </div>
              </div>
            </DialogContent>
          </Dialog>

          <Dialog>
            <DialogTrigger asChild>
              <div className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 backdrop-blur-lg rounded-xl p-6 border border-green-500/20 shadow-lg hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
                <div className="flex items-center justify-between mb-4">
                  <div className="p-3 bg-green-500/20 rounded-lg">
                    <Shield className="w-6 h-6 text-green-500" />
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-xs text-green-400 font-semibold">
                      {((metrics.blueWins / metrics.totalEpisodes) * 100).toFixed(1)}%
                    </span>
                    <HelpCircle className="h-3 w-3 text-green-400" />
                  </div>
                </div>
                <h3 className="text-3xl font-bold text-green-500">{metrics.blueWins}</h3>
                <p className="text-sm text-muted-foreground mt-1">Blue Team Wins</p>
              </div>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Blue Team Defense Success</DialogTitle>
                <DialogDescription>Defensive AI effectiveness measurement</DialogDescription>
              </DialogHeader>
              <div className="space-y-4 pt-4">
                <p className="text-foreground">
                  <strong>Win Rate: {((metrics.blueWins / metrics.totalEpisodes) * 100).toFixed(1)}%</strong> ({metrics.blueWins} successful defenses out of {metrics.totalEpisodes} attempts)
                </p>
                <p className="text-muted-foreground">
                  Blue Team wins occur when the defensive AI successfully:
                </p>
                <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                  <li>Detects all attack attempts before they achieve objectives</li>
                  <li>Blocks malicious activities through automated response</li>
                  <li>Maintains system integrity throughout the episode</li>
                  <li>Prevents data exfiltration and privilege escalation</li>
                </ul>
                <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-3">
                  <p className="text-sm text-muted-foreground">
                    <strong className="text-green-400">Continuous improvement:</strong> Blue Team AI learns from both wins and losses.
                    Each episode refines detection patterns, response strategies, and threat intelligence for better future performance.
                  </p>
                </div>
              </div>
            </DialogContent>
          </Dialog>

          <Dialog>
            <DialogTrigger asChild>
              <div className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 backdrop-blur-lg rounded-xl p-6 border border-purple-500/20 shadow-lg hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
                <div className="flex items-center justify-between mb-4">
                  <div className="p-3 bg-purple-500/20 rounded-lg">
                    <CheckCircle className="w-6 h-6 text-purple-500" />
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-xs text-purple-400 font-semibold">{metrics.systemHealth}%</span>
                    <HelpCircle className="h-3 w-3 text-purple-400" />
                  </div>
                </div>
                <h3 className="text-3xl font-bold text-purple-500">{metrics.blockedAttacks}</h3>
                <p className="text-sm text-muted-foreground mt-1">Blocked Attacks</p>
              </div>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Blocked Attack Attempts</DialogTitle>
                <DialogDescription>Total prevented malicious activities</DialogDescription>
              </DialogHeader>
              <div className="space-y-4 pt-4">
                <p className="text-foreground">
                  <strong>{metrics.blockedAttacks} attacks blocked</strong> with {metrics.systemHealth}% system health
                </p>
                <p className="text-muted-foreground">
                  This metric tracks individual attack attempts that were successfully prevented:
                </p>
                <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                  <li><strong>Malware execution blocks:</strong> Prevented malicious code from running</li>
                  <li><strong>Network intrusion prevention:</strong> Stopped unauthorized network access attempts</li>
                  <li><strong>Anomaly detection triggers:</strong> Identified and blocked suspicious behavior patterns</li>
                  <li><strong>Privilege escalation prevention:</strong> Stopped attempts to gain higher access levels</li>
                  <li><strong>Data exfiltration blocks:</strong> Prevented unauthorized data transfers</li>
                </ul>
                <div className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-3">
                  <p className="text-sm text-muted-foreground">
                    <strong className="text-purple-400">System health:</strong> Overall defense effectiveness score based on blocked attacks,
                    response time, false positive rate, and resource utilization. {metrics.systemHealth}% indicates excellent security posture.
                  </p>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        {/* Quick Actions */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="w-5 h-5 text-primary" />
            <h2 className="text-2xl font-bold">Quick Actions</h2>
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="ghost" size="sm" className="h-6 w-6 p-0 ml-1">
                  <HelpCircle className="h-4 w-4 text-muted-foreground" />
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Quick Actions Guide</DialogTitle>
                  <DialogDescription>Launch security tools and training environments</DialogDescription>
                </DialogHeader>
                <div className="space-y-3 pt-4">
                  <p className="text-muted-foreground">
                    Quick Actions provide one-click access to YUGMASTRA's core features. Tools marked with a "LIVE" badge update in real-time.
                  </p>
                  <div className="space-y-2">
                    {quickActions.slice(0, 4).map((action) => (
                      <div key={action.href} className="bg-card/50 border rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <action.icon className="h-4 w-4 text-primary" />
                          <p className="font-semibold">{action.title}</p>
                          {action.badge && <Badge variant="destructive" className="text-[10px]">{action.badge}</Badge>}
                        </div>
                        <p className="text-sm text-muted-foreground">{action.helpText}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {quickActions.map((action) => (
              <Dialog key={action.href}>
                <div className="group relative bg-card/50 backdrop-blur-lg rounded-xl p-6 border hover:border-primary/50 shadow-lg hover:shadow-xl transition-all overflow-hidden">
                  <div className={`absolute inset-0 bg-gradient-to-br ${action.color} opacity-0 group-hover:opacity-10 transition-opacity`}></div>
                  <div className="relative z-10">
                    <div className="flex items-start justify-between mb-3">
                      <div className={`p-3 bg-gradient-to-br ${action.color} rounded-lg`}>
                        <action.icon className="w-5 h-5 text-white" />
                      </div>
                      <div className="flex items-center gap-1">
                        {action.badge && (
                          <span className="px-2 py-1 text-[10px] font-bold bg-red-500 text-white rounded animate-pulse">
                            {action.badge}
                          </span>
                        )}
                        <DialogTrigger asChild>
                          <button className="p-1 hover:bg-primary/10 rounded transition-colors">
                            <Info className="h-3 w-3 text-muted-foreground" />
                          </button>
                        </DialogTrigger>
                      </div>
                    </div>
                    <h3 className="font-semibold mb-1 group-hover:text-primary transition-colors">
                      {action.title}
                    </h3>
                    <p className="text-xs text-muted-foreground mb-3">
                      {action.description}
                    </p>
                    <button
                      onClick={() => router.push(action.href)}
                      className="flex items-center text-xs text-primary opacity-0 group-hover:opacity-100 transition-opacity w-full"
                    >
                      <span>Launch</span>
                      <ArrowRight className="w-3 h-3 ml-1 group-hover:translate-x-1 transition-transform" />
                    </button>
                  </div>
                </div>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                      <action.icon className="h-5 w-5 text-primary" />
                      {action.title}
                    </DialogTitle>
                    <DialogDescription>{action.description}</DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 pt-4">
                    <p className="text-foreground">{action.helpText}</p>
                    <Button
                      onClick={() => router.push(action.href)}
                      className={`w-full bg-gradient-to-r ${action.color}`}
                    >
                      Launch {action.title}
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
            ))}
          </div>
        </div>

        {/* System Status */}
        <div className="bg-card/50 backdrop-blur-lg rounded-xl p-6 border shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary" />
              System Status
            </h2>
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                  <HelpCircle className="h-4 w-4 text-muted-foreground" />
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>System Component Status</DialogTitle>
                  <DialogDescription>Health monitoring for critical AI systems</DialogDescription>
                </DialogHeader>
                <div className="space-y-3 pt-4">
                  {systemStatus.map((system) => (
                    <div key={system.name} className="bg-card/50 border rounded-lg p-3">
                      <div className="flex items-center gap-2 mb-1">
                        <system.icon className={`h-4 w-4 ${system.color}`} />
                        <p className="font-semibold">{system.name}</p>
                        <Badge variant="outline" className="text-[10px]">{system.status}</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">{system.description}</p>
                    </div>
                  ))}
                </div>
              </DialogContent>
            </Dialog>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {systemStatus.map((system) => (
              <div
                key={system.name}
                className="flex items-center justify-between p-4 bg-accent/50 rounded-lg border hover:border-primary/50 transition-all group cursor-help"
                title={system.description}
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
