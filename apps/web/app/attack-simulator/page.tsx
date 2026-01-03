'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Input } from '@/components/ui/input';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  AreaChart, Area, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import {
  Shield, Swords, Activity, Zap, Terminal, AlertTriangle,
  CheckCircle2, XCircle, TrendingUp, TrendingDown, Clock,
  Users, Network, Target, Eye, Ban, Play, Pause, RotateCcw
} from 'lucide-react';

const COLORS = {
  red: '#ef4444',
  blue: '#3b82f6',
  green: '#10b981',
  yellow: '#f59e0b',
  purple: '#8b5cf6',
  pink: '#ec4899',
};

export default function LiveSimulationPage() {
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [attackData, setAttackData] = useState<any[]>([]);
  const [defenseData, setDefenseData] = useState<any[]>([]);
  const [networkHealth, setNetworkHealth] = useState(100);
  const [score, setScore] = useState(0);
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [currentCommand, setCurrentCommand] = useState('');
  const terminalRef = useRef<HTMLDivElement>(null);

  const [stats, setStats] = useState({
    totalAttacks: 0,
    blockedAttacks: 0,
    successfulAttacks: 0,
    detectedThreats: 0,
    activeAgents: 8,
    compromisedAssets: 0,
  });

  const [agents, setAgents] = useState([
    { id: 1, name: 'APT29', type: 'Nation-State', status: 'active', attacks: 12, success: 8 },
    { id: 2, name: 'ScriptKid42', type: 'Script Kiddie', status: 'active', attacks: 5, success: 1 },
    { id: 3, name: 'RansomCrew', type: 'Ransomware', status: 'active', attacks: 8, success: 4 },
    { id: 4, name: 'InsiderX', type: 'Insider Threat', status: 'detected', attacks: 3, success: 2 },
    { id: 5, name: 'BlueDefender1', type: 'SOC Analyst', status: 'active', blocks: 15 },
    { id: 6, name: 'HunterBot', type: 'Threat Hunter', status: 'active', blocks: 8 },
    { id: 7, name: 'ResponseTeam', type: 'Incident Response', status: 'active', blocks: 12 },
    { id: 8, name: 'WhiteHat01', type: 'Ethical Hacker', status: 'active', blocks: 6 },
  ]);

  const [liveEvents, setLiveEvents] = useState<any[]>([]);
  const [realWorldIncidents, setRealWorldIncidents] = useState([
    {
      id: 1,
      title: 'SolarWinds Supply Chain Attack',
      severity: 'CRITICAL',
      date: '2024-01-15',
      status: 'Resolved',
      attacker: 'APT29 (Cozy Bear)',
      target: 'SolarWinds Orion Platform',
      technique: 'Supply Chain Compromise',
      impact: '18,000+ organizations affected',
      detection: 'Anomalous network traffic detected by FireEye',
      mitigation: 'Emergency patches deployed, affected systems isolated',
      resolution: 'Complete system rebuild and security overhaul',
      lessons: 'Implement zero-trust architecture, enhanced supply chain security'
    },
    {
      id: 2,
      title: 'Colonial Pipeline Ransomware',
      severity: 'HIGH',
      date: '2024-02-20',
      status: 'Resolved',
      attacker: 'DarkSide Ransomware Group',
      target: 'Colonial Pipeline OT Systems',
      technique: 'VPN Credential Compromise',
      impact: 'Major fuel supply disruption on East Coast',
      detection: 'Encryption activity detected on file servers',
      mitigation: 'Systems taken offline, ransom paid (later recovered)',
      resolution: 'Systems restored from backups, security enhanced',
      lessons: 'MFA implementation, network segmentation, backup strategy'
    },
  ]);

  useEffect(() => {
    if (isSimulationRunning) {
      const interval = setInterval(() => {
        setTimeElapsed(prev => prev + 1);
        simulateLiveAttackDefense();
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [isSimulationRunning]);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [commandHistory]);

  const simulateLiveAttackDefense = () => {
    const attackSuccess = Math.random() > 0.5;
    const timestamp = new Date().toLocaleTimeString();

    const attackTypes = ['SQL Injection', 'Phishing', 'DDoS', 'Malware', 'Credential Theft', 'Lateral Movement', 'Data Exfiltration'];
    const attackType = attackTypes[Math.floor(Math.random() * attackTypes.length)];

    const defenseActions = ['Blocked by Firewall', 'Detected by IDS', 'Quarantined', 'Threat Neutralized', 'Alert Triggered'];
    const defenseAction = defenseActions[Math.floor(Math.random() * defenseActions.length)];

    setStats(prev => ({
      ...prev,
      totalAttacks: prev.totalAttacks + 1,
      blockedAttacks: attackSuccess ? prev.blockedAttacks : prev.blockedAttacks + 1,
      successfulAttacks: attackSuccess ? prev.successfulAttacks + 1 : prev.successfulAttacks,
      detectedThreats: prev.detectedThreats + (Math.random() > 0.3 ? 1 : 0),
    }));

    setAttackData(prev => [...prev.slice(-19), {
      time: timestamp,
      attacks: stats.totalAttacks + 1,
      blocked: stats.blockedAttacks + (attackSuccess ? 0 : 1),
      successful: stats.successfulAttacks + (attackSuccess ? 1 : 0),
    }]);

    const newEvent = {
      id: Date.now(),
      time: timestamp,
      type: attackSuccess ? 'attack-success' : 'defense-success',
      attacker: agents[Math.floor(Math.random() * 4)].name,
      defender: agents[4 + Math.floor(Math.random() * 4)].name,
      action: attackSuccess ? `${attackType} - Attack Succeeded` : `${attackType} - ${defenseAction}`,
      severity: attackSuccess ? 'high' : 'low',
    };

    setLiveEvents(prev => [newEvent, ...prev].slice(0, 50));

    if (attackSuccess) {
      setNetworkHealth(prev => Math.max(0, prev - Math.random() * 5));
      setScore(prev => Math.max(0, prev - 5));
    } else {
      setScore(prev => prev + 10);
      setNetworkHealth(prev => Math.min(100, prev + 1));
    }
  };

  const startSimulation = () => {
    setIsSimulationRunning(true);
    addCommandOutput('Simulation started - Live attack/defense battle initiated');
    addCommandOutput('Multiple AI agents deployed: Nation-State APTs, Script Kiddies, Ransomware Operators');
    addCommandOutput('Defense team activated: SOC Analysts, Threat Hunters, Incident Responders');
  };

  const pauseSimulation = () => {
    setIsSimulationRunning(false);
    addCommandOutput('Simulation paused');
  };

  const resetSimulation = () => {
    setIsSimulationRunning(false);
    setTimeElapsed(0);
    setAttackData([]);
    setNetworkHealth(100);
    setScore(0);
    setStats({
      totalAttacks: 0,
      blockedAttacks: 0,
      successfulAttacks: 0,
      detectedThreats: 0,
      activeAgents: 8,
      compromisedAssets: 0,
    });
    setLiveEvents([]);
    setCommandHistory([]);
    addCommandOutput('Simulation reset - All systems restored');
  };

  const executeCommand = (cmd: string) => {
    const command = cmd.trim().toLowerCase();
    addCommandOutput(`> ${cmd}`);

    if (command === 'help') {
      addCommandOutput('Available commands:');
      addCommandOutput('  block [ip] - Block IP address');
      addCommandOutput('  isolate [asset] - Isolate compromised asset');
      addCommandOutput('  patch [system] - Deploy emergency patch');
      addCommandOutput('  hunt - Start threat hunting');
      addCommandOutput('  status - Show system status');
      addCommandOutput('  agents - List all active agents');
    } else if (command.startsWith('block')) {
      const ip = command.split(' ')[1] || '192.168.1.100';
      addCommandOutput(`Blocking IP: ${ip}`);
      addCommandOutput('Firewall rule added successfully');
      setScore(prev => prev + 15);
    } else if (command.startsWith('isolate')) {
      addCommandOutput('Isolating compromised asset from network');
      addCommandOutput('Quarantine successful - Threat contained');
      setScore(prev => prev + 20);
    } else if (command.startsWith('patch')) {
      addCommandOutput('Deploying emergency security patch');
      addCommandOutput('Patch applied successfully');
      setNetworkHealth(prev => Math.min(100, prev + 10));
      setScore(prev => prev + 25);
    } else if (command === 'hunt') {
      addCommandOutput('Initiating threat hunting protocol');
      addCommandOutput('Scanning for IOCs and anomalous behavior');
      addCommandOutput('3 suspicious activities detected');
      setScore(prev => prev + 30);
    } else if (command === 'status') {
      addCommandOutput(`Network Health: ${networkHealth.toFixed(1)}%`);
      addCommandOutput(`Active Threats: ${stats.successfulAttacks}`);
      addCommandOutput(`Blocked Attacks: ${stats.blockedAttacks}`);
      addCommandOutput(`Current Score: ${score}`);
    } else if (command === 'agents') {
      addCommandOutput('Active Agents:');
      agents.forEach(agent => {
        addCommandOutput(`  ${agent.name} (${agent.type}) - ${agent.status}`);
      });
    } else {
      addCommandOutput(`Unknown command: ${cmd}`);
    }

    setCurrentCommand('');
  };

  const addCommandOutput = (text: string) => {
    setCommandHistory(prev => [...prev, text]);
  };

  const agentTypeColors: any = {
    'Nation-State': 'bg-red-500',
    'Script Kiddie': 'bg-yellow-500',
    'Ransomware': 'bg-orange-500',
    'Insider Threat': 'bg-purple-500',
    'SOC Analyst': 'bg-blue-500',
    'Threat Hunter': 'bg-cyan-500',
    'Incident Response': 'bg-green-500',
    'Ethical Hacker': 'bg-emerald-500',
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/20 p-6 pt-32">
      <div className="max-w-[1800px] mx-auto space-y-6">

        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 bg-clip-text text-transparent mb-2">
              Live Cyber Warfare Simulation
            </h1>
            <p className="text-muted-foreground text-lg">Real-time Attack & Defense Battle - Multiple AI Agents Engaged</p>
          </div>
          <div className="flex gap-3">
            {!isSimulationRunning ? (
              <Button onClick={startSimulation} size="lg" className="bg-green-600 hover:bg-green-700">
                <Play className="mr-2 h-5 w-5" /> Start Battle
              </Button>
            ) : (
              <Button onClick={pauseSimulation} size="lg" className="bg-yellow-600 hover:bg-yellow-700">
                <Pause className="mr-2 h-5 w-5" /> Pause
              </Button>
            )}
            <Button onClick={resetSimulation} size="lg" variant="outline">
              <RotateCcw className="mr-2 h-5 w-5" /> Reset
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="bg-gradient-to-br from-red-900/50 to-red-800/30 border-red-500/50">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Swords className="h-4 w-4" /> Total Attacks
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-4xl font-bold text-red-400">{stats.totalAttacks}</div>
              <div className="flex items-center gap-2 mt-2 text-xs text-red-300">
                <TrendingUp className="h-3 w-3" />
                <span>{stats.successfulAttacks} succeeded</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-green-900/50 to-green-800/30 border-green-500/50">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Shield className="h-4 w-4" /> Blocked
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-4xl font-bold text-green-400">{stats.blockedAttacks}</div>
              <div className="flex items-center gap-2 mt-2 text-xs text-green-300">
                <CheckCircle2 className="h-3 w-3" />
                <span>{((stats.blockedAttacks / Math.max(stats.totalAttacks, 1)) * 100).toFixed(0)}% blocked</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 border-blue-500/50">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Activity className="h-4 w-4" /> Network Health
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-4xl font-bold text-blue-400">{networkHealth.toFixed(0)}%</div>
              <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                <div
                  className="bg-gradient-to-r from-blue-500 to-cyan-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${networkHealth}%` }}
                />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-purple-900/50 to-purple-800/30 border-purple-500/50">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Target className="h-4 w-4" /> Score
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-4xl font-bold text-purple-400">{score}</div>
              <div className="flex items-center gap-2 mt-2 text-xs text-purple-300">
                <Zap className="h-3 w-3" />
                <span>Combat effectiveness</span>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          <Card className="lg:col-span-2 bg-card/50 backdrop-blur-lg border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-foreground">
                <Activity className="h-5 w-5 text-blue-500" />
                Real-Time Attack vs Defense Battle
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={attackData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" />
                  <YAxis stroke="hsl(var(--muted-foreground))" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Area type="monotone" dataKey="attacks" stackId="1" stroke={COLORS.red} fill={COLORS.red} fillOpacity={0.6} name="Total Attacks" />
                  <Area type="monotone" dataKey="blocked" stackId="2" stroke={COLORS.green} fill={COLORS.green} fillOpacity={0.6} name="Blocked" />
                  <Area type="monotone" dataKey="successful" stackId="3" stroke={COLORS.yellow} fill={COLORS.yellow} fillOpacity={0.6} name="Breached" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card className="bg-card/50 backdrop-blur-lg border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-foreground">
                <Users className="h-5 w-5 text-purple-500" />
                Active Agents ({agents.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[280px]">
                <div className="space-y-2">
                  {agents.map(agent => (
                    <div key={agent.id} className="p-3 rounded-lg bg-accent/50 border hover:border-primary/50 transition-colors">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-sm text-foreground">{agent.name}</span>
                        <Badge className={`${agentTypeColors[agent.type]} text-xs`}>
                          {agent.status}
                        </Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">{agent.type}</div>
                      {agent.attacks !== undefined && (
                        <div className="text-xs text-red-400 mt-1">
                          {agent.attacks} attacks, {agent.success} successful
                        </div>
                      )}
                      {agent.blocks !== undefined && (
                        <div className="text-xs text-green-400 mt-1">
                          {agent.blocks} threats blocked
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          <Card className="bg-card/50 backdrop-blur-lg border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-foreground">
                <Terminal className="h-5 w-5 text-green-500" />
                Command Terminal - User Intervention
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div
                ref={terminalRef}
                className="bg-black dark:bg-gray-950 rounded-lg p-4 font-mono text-sm h-[300px] overflow-y-auto mb-3 border border-green-500/30"
              >
                {commandHistory.map((line, idx) => (
                  <div key={idx} className={line.startsWith('>') ? 'text-green-400' : 'text-gray-300'}>
                    {line}
                  </div>
                ))}
                <div className="flex items-center gap-2 mt-2">
                  <span className="text-green-400">$</span>
                  <span className="animate-pulse">_</span>
                </div>
              </div>
              <div className="flex gap-2">
                <Input
                  value={currentCommand}
                  onChange={(e) => setCurrentCommand(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && executeCommand(currentCommand)}
                  placeholder="Type 'help' for commands"
                  className="bg-background border font-mono text-green-500"
                />
                <Button onClick={() => executeCommand(currentCommand)} className="bg-green-600 hover:bg-green-700">
                  Execute
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-card/50 backdrop-blur-lg border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-foreground">
                <Zap className="h-5 w-5 text-yellow-500" />
                Live Event Stream
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[350px]">
                <div className="space-y-2">
                  {liveEvents.map(event => (
                    <div
                      key={event.id}
                      className={`p-3 rounded-lg border ${
                        event.type === 'attack-success'
                          ? 'bg-red-900/20 border-red-500/50'
                          : 'bg-green-900/20 border-green-500/50'
                      }`}
                    >
                      <div className="flex items-start justify-between mb-1">
                        <div className="flex items-center gap-2">
                          {event.type === 'attack-success' ? (
                            <XCircle className="h-4 w-4 text-red-400" />
                          ) : (
                            <CheckCircle2 className="h-4 w-4 text-green-400" />
                          )}
                          <span className="text-xs text-gray-400">{event.time}</span>
                        </div>
                        <Badge variant={event.severity === 'high' ? 'destructive' : 'default'} className="text-xs">
                          {event.severity}
                        </Badge>
                      </div>
                      <div className="text-sm font-medium mb-1">{event.action}</div>
                      <div className="text-xs text-gray-400">
                        Attacker: <span className="text-red-400">{event.attacker}</span> vs
                        Defender: <span className="text-blue-400"> {event.defender}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        <Card className="bg-card/50 backdrop-blur-lg border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-foreground">
              <AlertTriangle className="h-5 w-5 text-orange-500" />
              Real-World Cyber Incidents - Learn from Actual Attacks
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {realWorldIncidents.map(incident => (
                <div key={incident.id} className="p-4 rounded-lg bg-accent/50 border">
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="font-bold text-lg text-foreground">{incident.title}</h3>
                    <Badge className={
                      incident.severity === 'CRITICAL' ? 'bg-red-500' :
                      incident.severity === 'HIGH' ? 'bg-orange-500' : 'bg-yellow-500'
                    }>
                      {incident.severity}
                    </Badge>
                  </div>

                  <div className="space-y-2 text-sm">
                    <div><span className="text-muted-foreground">Date:</span> <span className="text-foreground">{incident.date}</span></div>
                    <div><span className="text-muted-foreground">Status:</span> <Badge variant="outline" className="ml-2">{incident.status}</Badge></div>
                    <div><span className="text-muted-foreground">Attacker:</span> <span className="text-red-500">{incident.attacker}</span></div>
                    <div><span className="text-muted-foreground">Target:</span> <span className="text-foreground">{incident.target}</span></div>
                    <div><span className="text-muted-foreground">Technique:</span> <span className="text-purple-500">{incident.technique}</span></div>

                    <div className="pt-2 border-t mt-3">
                      <div className="font-semibold text-yellow-500 mb-1">How it Started:</div>
                      <div className="text-muted-foreground text-xs">{incident.technique} was used to gain initial access</div>
                    </div>

                    <div>
                      <div className="font-semibold text-orange-500 mb-1">Impact:</div>
                      <div className="text-muted-foreground text-xs">{incident.impact}</div>
                    </div>

                    <div>
                      <div className="font-semibold text-cyan-500 mb-1">Detection Method:</div>
                      <div className="text-muted-foreground text-xs">{incident.detection}</div>
                    </div>

                    <div>
                      <div className="font-semibold text-blue-500 mb-1">Mitigation:</div>
                      <div className="text-muted-foreground text-xs">{incident.mitigation}</div>
                    </div>

                    <div>
                      <div className="font-semibold text-green-500 mb-1">Resolution:</div>
                      <div className="text-muted-foreground text-xs">{incident.resolution}</div>
                    </div>

                    <div>
                      <div className="font-semibold text-purple-500 mb-1">Lessons Learned:</div>
                      <div className="text-muted-foreground text-xs">{incident.lessons}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

      </div>
    </div>
  );
}
