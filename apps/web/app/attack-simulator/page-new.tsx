'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Input } from '@/components/ui/input';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import {
  Shield, Swords, Activity, Zap, Terminal, AlertTriangle,
  CheckCircle2, TrendingUp, Clock, Target, Play, Pause, RotateCcw, Users
} from 'lucide-react';

export default function LiveSimulationPage() {
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [attackData, setAttackData] = useState<any[]>([]);
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

  const [agents] = useState([
    { id: 1, name: 'APT29', type: 'Nation-State', status: 'active', attacks: 12, success: 8, team: 'red' },
    { id: 2, name: 'ScriptKid42', type: 'Script Kiddie', status: 'active', attacks: 5, success: 1, team: 'red' },
    { id: 3, name: 'RansomCrew', type: 'Ransomware', status: 'active', attacks: 8, success: 4, team: 'red' },
    { id: 4, name: 'InsiderX', type: 'Insider Threat', status: 'detected', attacks: 3, success: 2, team: 'red' },
    { id: 5, name: 'BlueDefender1', type: 'SOC Analyst', status: 'active', blocks: 15, team: 'blue' },
    { id: 6, name: 'HunterBot', type: 'Threat Hunter', status: 'active', blocks: 8, team: 'blue' },
    { id: 7, name: 'ResponseTeam', type: 'Incident Response', status: 'active', blocks: 12, team: 'blue' },
    { id: 8, name: 'WhiteHat01', type: 'Ethical Hacker', status: 'active', blocks: 6, team: 'blue' },
  ]);

  const [liveEvents, setLiveEvents] = useState<any[]>([]);

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
  };

  const addCommandOutput = (output: string) => {
    setCommandHistory(prev => [...prev, output]);
  };

  const handleCommandSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (currentCommand.trim()) {
      addCommandOutput(`> ${currentCommand}`);
      addCommandOutput(`Command executed: ${currentCommand}`);
      setCurrentCommand('');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/10 p-6">
      <div className="max-w-[1800px] mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-red-500 via-purple-500 to-blue-500 bg-clip-text text-transparent mb-2">
              Live Cyber Warfare Simulation
            </h1>
            <p className="text-muted-foreground text-lg">Real-time Attack & Defense Battle Arena</p>
          </div>
          <div className="flex items-center gap-3">
            <Button
              onClick={startSimulation}
              disabled={isSimulationRunning}
              className="gap-2 bg-green-500 hover:bg-green-600"
              size="lg"
            >
              <Play className="h-5 w-5" />
              Start
            </Button>
            <Button
              onClick={pauseSimulation}
              disabled={!isSimulationRunning}
              className="gap-2 bg-yellow-500 hover:bg-yellow-600"
              size="lg"
            >
              <Pause className="h-5 w-5" />
              Pause
            </Button>
            <Button
              onClick={resetSimulation}
              variant="outline"
              className="gap-2"
              size="lg"
            >
              <RotateCcw className="h-5 w-5" />
              Reset
            </Button>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card className="border-border/50 bg-card/50 backdrop-blur-xl shadow-lg hover:shadow-xl transition-all">
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <Activity className="h-8 w-8 text-blue-500" />
                <Badge className="bg-blue-500/10 text-blue-500 border-blue-500/20">Total</Badge>
              </div>
              <div className="text-4xl font-bold text-foreground mb-1">{stats.totalAttacks}</div>
              <div className="text-sm text-muted-foreground">Total Attacks</div>
            </CardContent>
          </Card>

          <Card className="border-border/50 bg-card/50 backdrop-blur-xl shadow-lg hover:shadow-xl transition-all">
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <Shield className="h-8 w-8 text-green-500" />
                <Badge className="bg-green-500/10 text-green-500 border-green-500/20">Blocked</Badge>
              </div>
              <div className="text-4xl font-bold text-foreground mb-1">{stats.blockedAttacks}</div>
              <div className="text-sm text-muted-foreground">Blocked Attacks</div>
            </CardContent>
          </Card>

          <Card className="border-border/50 bg-card/50 backdrop-blur-xl shadow-lg hover:shadow-xl transition-all">
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <AlertTriangle className="h-8 w-8 text-red-500" />
                <Badge className="bg-red-500/10 text-red-500 border-red-500/20">Success</Badge>
              </div>
              <div className="text-4xl font-bold text-foreground mb-1">{stats.successfulAttacks}</div>
              <div className="text-sm text-muted-foreground">Successful Attacks</div>
            </CardContent>
          </Card>

          <Card className="border-border/50 bg-card/50 backdrop-blur-xl shadow-lg hover:shadow-xl transition-all">
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <Target className="h-8 w-8 text-purple-500" />
                <Badge className="bg-purple-500/10 text-purple-500 border-purple-500/20">Health</Badge>
              </div>
              <div className="text-4xl font-bold text-foreground mb-1">{Math.round(networkHealth)}%</div>
              <div className="text-sm text-muted-foreground">Network Health</div>
              <div className="mt-2 w-full bg-muted rounded-full h-2">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-green-500 to-blue-500 transition-all"
                  style={{ width: `${networkHealth}%` }}
                />
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Chart */}
          <Card className="lg:col-span-2 border-border/50 bg-card/50 backdrop-blur-xl shadow-xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-foreground">
                <Activity className="h-5 w-5 text-blue-500" />
                Attack/Defense Timeline
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={attackData}>
                  <defs>
                    <linearGradient id="colorAttacks" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorBlocked" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                  <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" />
                  <YAxis stroke="hsl(var(--muted-foreground))" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      color: 'hsl(var(--foreground))'
                    }}
                  />
                  <Legend />
                  <Area type="monotone" dataKey="attacks" stroke="#ef4444" fillOpacity={1} fill="url(#colorAttacks)" />
                  <Area type="monotone" dataKey="blocked" stroke="#10b981" fillOpacity={1} fill="url(#colorBlocked)" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Agents */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-xl shadow-xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-foreground">
                <Users className="h-5 w-5 text-purple-500" />
                Active Agents
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px]">
                <div className="space-y-3">
                  {agents.map((agent) => (
                    <div
                      key={agent.id}
                      className={`p-3 rounded-xl border transition-all hover:scale-105 ${
                        agent.team === 'red'
                          ? 'bg-red-500/10 border-red-500/30'
                          : 'bg-blue-500/10 border-blue-500/30'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-foreground">{agent.name}</span>
                        {agent.team === 'red' ? (
                          <Swords className="h-4 w-4 text-red-500" />
                        ) : (
                          <Shield className="h-4 w-4 text-blue-500" />
                        )}
                      </div>
                      <div className="text-xs text-muted-foreground">{agent.type}</div>
                      <Badge className="mt-2 text-xs" variant="outline">
                        {agent.status}
                      </Badge>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Terminal */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-xl shadow-xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-foreground">
                <Terminal className="h-5 w-5 text-green-500" />
                Command Terminal
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div
                ref={terminalRef}
                className="h-[300px] bg-background/50 rounded-xl p-4 font-mono text-sm overflow-y-auto border border-border/50"
              >
                {commandHistory.map((cmd, i) => (
                  <div key={i} className="text-foreground/80 mb-1">
                    {cmd}
                  </div>
                ))}
              </div>
              <form onSubmit={handleCommandSubmit} className="mt-4">
                <Input
                  value={currentCommand}
                  onChange={(e) => setCurrentCommand(e.target.value)}
                  placeholder="Enter command..."
                  className="bg-background/50 border-border/50 text-foreground"
                />
              </form>
            </CardContent>
          </Card>

          {/* Live Events */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-xl shadow-xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-foreground">
                <Activity className="h-5 w-5 text-orange-500" />
                Live Event Feed
                {isSimulationRunning && (
                  <Badge className="ml-auto bg-red-500 text-white animate-pulse">LIVE</Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px]">
                <div className="space-y-2">
                  {liveEvents.map((event) => (
                    <div
                      key={event.id}
                      className={`p-3 rounded-xl border transition-all hover:bg-accent/50 ${
                        event.severity === 'high'
                          ? 'bg-red-500/10 border-red-500/30'
                          : 'bg-green-500/10 border-green-500/30'
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        {event.severity === 'high' ? (
                          <AlertTriangle className="h-4 w-4 text-red-500 flex-shrink-0 mt-0.5" />
                        ) : (
                          <CheckCircle2 className="h-4 w-4 text-green-500 flex-shrink-0 mt-0.5" />
                        )}
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium text-foreground truncate">
                            {event.action}
                          </div>
                          <div className="text-xs text-muted-foreground mt-1">
                            {event.attacker} vs {event.defender} • {event.time}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
