'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import {
  Shield,
  Swords,
  Activity,
  Target,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Play,
  Pause,
  BarChart3,
  Users,
  Network,
  Zap,
  Eye,
  Ban
} from 'lucide-react';

// Types
interface SimulationStatus {
  session_id: string;
  scenario: string;
  user_role: string;
  is_running: boolean;
  elapsed_minutes: number;
  score: number;
  stats: {
    total_attacks: number;
    attacks_detected: number;
    attacks_blocked: number;
    compromised_assets: number;
    defense_actions: number;
  };
  compromised_assets: Array<{
    id: string;
    name: string;
    type: string;
  }>;
}

interface SimulationEvent {
  type: string;
  timestamp: string;
  data: any;
}

interface Scenario {
  id: string;
  name: string;
  description: string;
  duration_minutes: number;
  difficulty: string;
  attacker_count: number;
  asset_count: number;
  objectives: string[];
}

export default function LiveSimulationPage() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<SimulationStatus | null>(null);
  const [events, setEvents] = useState<SimulationEvent[]>([]);
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);
  const [loading, setLoading] = useState(false);

  // Fetch available scenarios
  useEffect(() => {
    fetchScenarios();
  }, []);

  // WebSocket connection for live events
  useEffect(() => {
    if (sessionId && !websocket) {
      connectWebSocket();
    }

    return () => {
      if (websocket) {
        websocket.close();
      }
    };
  }, [sessionId]);

  // Poll for status updates
  useEffect(() => {
    if (sessionId) {
      const interval = setInterval(() => {
        fetchStatus();
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [sessionId]);

  const fetchScenarios = async () => {
    try {
      const response = await fetch('/api/simulation/scenarios');
      const data = await response.json();
      setScenarios(data.scenarios);
    } catch (error) {
      console.error('Error fetching scenarios:', error);
    }
  };

  const startSimulation = async () => {
    if (!selectedScenario) return;

    setLoading(true);
    try {
      const response = await fetch('/api/simulation/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: selectedScenario,
          user_role: 'soc_analyst'
        })
      });

      const data = await response.json();
      setSessionId(data.session_id);
    } catch (error) {
      console.error('Error starting simulation:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchStatus = async () => {
    if (!sessionId) return;

    try {
      const response = await fetch(`/api/simulation/${sessionId}/status`);
      const data = await response.json();
      setStatus(data);
    } catch (error) {
      console.error('Error fetching status:', error);
    }
  };

  const connectWebSocket = () => {
    if (!sessionId) return;

    const ws = new WebSocket(`ws://localhost:8002/api/simulation/${sessionId}/ws`);

    ws.onmessage = (event) => {
      const eventData = JSON.parse(event.data);
      setEvents((prev) => [...prev, eventData].slice(-100)); // Keep last 100 events

      if (eventData.type === 'status') {
        setStatus(eventData.data);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    setWebsocket(ws);
  };

  const takeDefenseAction = async (actionType: string, targetAssetId?: string) => {
    if (!sessionId) return;

    try {
      await fetch(`/api/simulation/${sessionId}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action_type: actionType,
          target_asset_id: targetAssetId
        })
      });

      // Refresh status
      fetchStatus();
    } catch (error) {
      console.error('Error taking defense action:', error);
    }
  };

  const getSeverityColor = (value: number) => {
    if (value >= 80) return 'text-green-500';
    if (value >= 60) return 'text-yellow-500';
    if (value >= 40) return 'text-orange-500';
    return 'text-red-500';
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'bg-green-500';
      case 'medium': return 'bg-yellow-500';
      case 'hard': return 'bg-orange-500';
      case 'expert': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 bg-clip-text text-transparent">
            Live Attack Simulation
          </h1>
          <p className="text-muted-foreground mt-2">
            Real-time cyber warfare training with multi-agent AI hackers
          </p>
        </div>
        {status && (
          <Badge className={`text-lg px-4 py-2 ${status.is_running ? 'bg-green-500' : 'bg-gray-500'}`}>
            {status.is_running ? 'ACTIVE' : 'COMPLETED'}
          </Badge>
        )}
      </div>

      {!sessionId ? (
        // Scenario Selection
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {scenarios.map((scenario) => (
            <Card
              key={scenario.id}
              className={`cursor-pointer transition-all hover:shadow-lg ${
                selectedScenario === scenario.id ? 'ring-2 ring-purple-500' : ''
              }`}
              onClick={() => setSelectedScenario(scenario.id)}
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>{scenario.name}</CardTitle>
                  <Badge className={getDifficultyColor(scenario.difficulty)}>
                    {scenario.difficulty}
                  </Badge>
                </div>
                <CardDescription>{scenario.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center gap-2">
                    <Swords className="h-4 w-4" />
                    <span>{scenario.attacker_count} Attackers</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Network className="h-4 w-4" />
                    <span>{scenario.asset_count} Assets</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Activity className="h-4 w-4" />
                    <span>{scenario.duration_minutes} minutes</span>
                  </div>
                </div>

                <div className="mt-4">
                  <p className="text-xs text-muted-foreground mb-2">Objectives:</p>
                  <ul className="text-xs space-y-1">
                    {scenario.objectives.map((obj, idx) => (
                      <li key={idx} className="flex items-start gap-1">
                        <Target className="h-3 w-3 mt-0.5 flex-shrink-0" />
                        <span>{obj}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        // Active Simulation
        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">
              <BarChart3 className="h-4 w-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="assets">
              <Network className="h-4 w-4 mr-2" />
              Assets
            </TabsTrigger>
            <TabsTrigger value="events">
              <Activity className="h-4 w-4 mr-2" />
              Live Events
            </TabsTrigger>
            <TabsTrigger value="actions">
              <Shield className="h-4 w-4 mr-2" />
              Defense Actions
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            {status && (
              <>
                {/* Stats Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm">Score</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className={`text-3xl font-bold ${getSeverityColor(status.score)}`}>
                        {status.score}
                      </div>
                      <p className="text-xs text-muted-foreground">out of 100</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm flex items-center gap-1">
                        <Swords className="h-4 w-4" />
                        Attacks
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold">{status.stats.total_attacks}</div>
                      <p className="text-xs text-muted-foreground">
                        {status.stats.attacks_detected} detected
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm flex items-center gap-1">
                        <Ban className="h-4 w-4" />
                        Blocked
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold text-green-500">
                        {status.stats.attacks_blocked}
                      </div>
                      <p className="text-xs text-muted-foreground">attacks stopped</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm flex items-center gap-1">
                        <AlertTriangle className="h-4 w-4" />
                        Compromised
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold text-red-500">
                        {status.stats.compromised_assets}
                      </div>
                      <p className="text-xs text-muted-foreground">assets breached</p>
                    </CardContent>
                  </Card>
                </div>

                {/* Detection Rate */}
                <Card>
                  <CardHeader>
                    <CardTitle>Detection Effectiveness</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm">Detection Rate</span>
                          <span className="text-sm font-bold">
                            {status.stats.total_attacks > 0
                              ? Math.round((status.stats.attacks_detected / status.stats.total_attacks) * 100)
                              : 0}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-500 h-2 rounded-full"
                            style={{
                              width: `${
                                status.stats.total_attacks > 0
                                  ? (status.stats.attacks_detected / status.stats.total_attacks) * 100
                                  : 0
                              }%`
                            }}
                          />
                        </div>
                      </div>

                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm">Block Rate</span>
                          <span className="text-sm font-bold">
                            {status.stats.attacks_detected > 0
                              ? Math.round((status.stats.attacks_blocked / status.stats.attacks_detected) * 100)
                              : 0}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-green-500 h-2 rounded-full"
                            style={{
                              width: `${
                                status.stats.attacks_detected > 0
                                  ? (status.stats.attacks_blocked / status.stats.attacks_detected) * 100
                                  : 0
                              }%`
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </TabsContent>

          {/* Assets Tab */}
          <TabsContent value="assets">
            <Card>
              <CardHeader>
                <CardTitle>Network Assets</CardTitle>
                <CardDescription>Current status of all network assets</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {status?.compromised_assets.map((asset) => (
                    <Alert key={asset.id} variant="destructive">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertTitle>{asset.name}</AlertTitle>
                      <AlertDescription>
                        Type: {asset.type} | Status: COMPROMISED
                      </AlertDescription>
                      <Button
                        size="sm"
                        className="mt-2"
                        onClick={() => takeDefenseAction('quarantine', asset.id)}
                      >
                        Quarantine Asset
                      </Button>
                    </Alert>
                  ))}
                  {status?.compromised_assets.length === 0 && (
                    <Alert>
                      <CheckCircle2 className="h-4 w-4" />
                      <AlertTitle>All Clear</AlertTitle>
                      <AlertDescription>
                        No assets are currently compromised
                      </AlertDescription>
                    </Alert>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Events Tab */}
          <TabsContent value="events">
            <Card>
              <CardHeader>
                <CardTitle>Live Event Stream</CardTitle>
                <CardDescription>Real-time simulation events</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 max-h-[600px] overflow-y-auto">
                  {events.slice().reverse().map((event, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded border border-border bg-card hover:bg-accent/50 transition-colors"
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-2">
                          {event.type === 'attack_action' && <Swords className="h-4 w-4 text-red-500 mt-0.5" />}
                          {event.type === 'detection' && <Eye className="h-4 w-4 text-yellow-500 mt-0.5" />}
                          {event.type === 'defense_action' && <Shield className="h-4 w-4 text-blue-500 mt-0.5" />}
                          <div>
                            <p className="text-sm font-medium">{event.type.replace('_', ' ').toUpperCase()}</p>
                            <p className="text-xs text-muted-foreground">
                              {new Date(event.timestamp).toLocaleTimeString()}
                            </p>
                          </div>
                        </div>
                        <Badge variant="outline">{event.type}</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Defense Actions Tab */}
          <TabsContent value="actions">
            <Card>
              <CardHeader>
                <CardTitle>Available Defense Actions</CardTitle>
                <CardDescription>Take action to defend your network</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <Button
                    onClick={() => takeDefenseAction('monitor')}
                    className="h-auto flex flex-col items-start p-4"
                  >
                    <Eye className="h-6 w-6 mb-2" />
                    <div>
                      <div className="font-bold">Monitor</div>
                      <div className="text-xs opacity-80">Increase surveillance</div>
                    </div>
                  </Button>
                  <Button
                    onClick={() => takeDefenseAction('block')}
                    className="h-auto flex flex-col items-start p-4"
                    variant="destructive"
                  >
                    <Ban className="h-6 w-6 mb-2" />
                    <div>
                      <div className="font-bold">Block</div>
                      <div className="text-xs opacity-80">Block detected threats</div>
                    </div>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}

      {/* Start Button */}
      {!sessionId && selectedScenario && (
        <div className="flex justify-center">
          <Button
            size="lg"
            onClick={startSimulation}
            disabled={loading}
            className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600"
          >
            <Play className="h-5 w-5 mr-2" />
            {loading ? 'Starting Simulation...' : 'Start Simulation'}
          </Button>
        </div>
      )}
    </div>
  );
}
