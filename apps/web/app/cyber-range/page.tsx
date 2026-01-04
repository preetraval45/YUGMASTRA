'use client';

import { useState, useEffect, useMemo } from 'react';
import { Server, Shield, Swords, Wifi, Database, Globe, Lock, AlertTriangle, CheckCircle, XCircle, Activity, Terminal, Network } from 'lucide-react';

interface SystemNode {
  id: string;
  name: string;
  type: 'web' | 'database' | 'api' | 'auth' | 'firewall';
  status: 'healthy' | 'compromised' | 'under-attack' | 'isolated';
  ip: string;
  ports: number[];
  vulnerabilities: string[];
}

interface SimulationEvent {
  id: string;
  timestamp: Date;
  type: 'attack' | 'defense' | 'system' | 'alert';
  severity: 'low' | 'medium' | 'high' | 'critical';
  source: string;
  target: string;
  description: string;
  success: boolean;
}

export default function CyberRangePage() {
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationTime, setSimulationTime] = useState(0);
  const [nodes, setNodes] = useState<SystemNode[]>([
    { id: 'web1', name: 'Web Server', type: 'web', status: 'healthy', ip: '10.0.1.10', ports: [80, 443], vulnerabilities: ['XSS', 'CSRF'] },
    { id: 'api1', name: 'API Gateway', type: 'api', status: 'healthy', ip: '10.0.1.20', ports: [8080, 8443], vulnerabilities: ['SQL Injection', 'Authentication Bypass'] },
    { id: 'db1', name: 'Database', type: 'database', status: 'healthy', ip: '10.0.1.30', ports: [5432], vulnerabilities: ['Weak Password', 'Unpatched CVE-2023-1234'] },
    { id: 'auth1', name: 'Auth Service', type: 'auth', status: 'healthy', ip: '10.0.1.40', ports: [3000], vulnerabilities: ['Session Fixation', 'Brute Force'] },
    { id: 'fw1', name: 'Firewall', type: 'firewall', status: 'healthy', ip: '10.0.1.1', ports: [], vulnerabilities: [] },
  ]);

  const [events, setEvents] = useState<SimulationEvent[]>([]);
  const [stats, setStats] = useState({
    totalAttacks: 0,
    blockedAttacks: 0,
    successfulAttacks: 0,
    nodesCompromised: 0,
  });

  const attackTypes = useMemo(() => [
    { name: 'SQL Injection', target: 'api1', severity: 'high' as const, successRate: 0.4 },
    { name: 'XSS Attack', target: 'web1', severity: 'medium' as const, successRate: 0.3 },
    { name: 'Brute Force', target: 'auth1', severity: 'high' as const, successRate: 0.25 },
    { name: 'Port Scan', target: 'web1', severity: 'low' as const, successRate: 0.9 },
    { name: 'DDoS', target: 'web1', severity: 'critical' as const, successRate: 0.5 },
    { name: 'Privilege Escalation', target: 'db1', severity: 'critical' as const, successRate: 0.35 },
  ], []);

  const defenseActions = useMemo(() => [
    'Rate limiting activated',
    'IP blocked',
    'WAF rule triggered',
    'Intrusion detected and blocked',
    'Session terminated',
    'Packet filtered',
  ], []);

  useEffect(() => {
    if (!isSimulating) return;

    const interval = setInterval(() => {
      setSimulationTime(prev => prev + 1);

      // Random attack every 2-4 seconds
      if (Math.random() > 0.6) {
        const attack = attackTypes[Math.floor(Math.random() * attackTypes.length)];
        const isBlocked = Math.random() > attack.successRate;

        const newEvent: SimulationEvent = {
          id: `evt-${Date.now()}-${Math.random()}`,
          timestamp: new Date(),
          type: isBlocked ? 'defense' : 'attack',
          severity: attack.severity,
          source: '203.0.113.' + Math.floor(Math.random() * 255),
          target: attack.target,
          description: isBlocked
            ? `${attack.name} blocked by ${defenseActions[Math.floor(Math.random() * defenseActions.length)]}`
            : `${attack.name} succeeded on ${nodes.find(n => n.id === attack.target)?.name}`,
          success: !isBlocked,
        };

        setEvents(prev => [newEvent, ...prev.slice(0, 49)]);

        setStats(prev => ({
          totalAttacks: prev.totalAttacks + 1,
          blockedAttacks: isBlocked ? prev.blockedAttacks + 1 : prev.blockedAttacks,
          successfulAttacks: !isBlocked ? prev.successfulAttacks + 1 : prev.successfulAttacks,
          nodesCompromised: prev.nodesCompromised,
        }));

        // Update node status
        if (!isBlocked && Math.random() > 0.7) {
          setNodes(prev => prev.map(node =>
            node.id === attack.target
              ? { ...node, status: 'compromised' as const }
              : node
          ));
          setStats(prev => ({ ...prev, nodesCompromised: prev.nodesCompromised + 1 }));
        } else {
          setNodes(prev => prev.map(node =>
            node.id === attack.target
              ? { ...node, status: 'under-attack' as const }
              : node.status === 'under-attack' ? { ...node, status: 'healthy' as const } : node
          ));
        }
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [isSimulating, attackTypes, defenseActions, nodes]);

  const getNodeIcon = (type: string) => {
    switch (type) {
      case 'web': return Globe;
      case 'database': return Database;
      case 'api': return Server;
      case 'auth': return Lock;
      case 'firewall': return Shield;
      default: return Server;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'border-green-500/50 bg-green-500/10';
      case 'under-attack': return 'border-yellow-500/50 bg-yellow-500/10 animate-pulse';
      case 'compromised': return 'border-red-500/50 bg-red-500/10';
      case 'isolated': return 'border-gray-500/50 bg-gray-500/10';
      default: return 'border-gray-500/50 bg-gray-500/10';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'low': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  const handleReset = () => {
    setIsSimulating(false);
    setSimulationTime(0);
    setEvents([]);
    setStats({
      totalAttacks: 0,
      blockedAttacks: 0,
      successfulAttacks: 0,
      nodesCompromised: 0,
    });
    setNodes(nodes.map(node => ({ ...node, status: 'healthy' as const })));
  };

  const handleIsolateNode = (nodeId: string) => {
    setNodes(prev => prev.map(node =>
      node.id === nodeId ? { ...node, status: 'isolated' as const } : node
    ));
  };

  const handleRestoreNode = (nodeId: string) => {
    setNodes(prev => prev.map(node =>
      node.id === nodeId ? { ...node, status: 'healthy' as const } : node
    ));
  };

  return (
    <div className="min-h-screen bg-background p-8 pt-32">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
              <Network className="w-8 h-8 text-blue-500" />
              Cyber Range Simulation
            </h1>
            <p className="text-muted-foreground">
              Real-time network attack/defense simulation - Preet Raval's Security Lab
            </p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={() => setIsSimulating(!isSimulating)}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                isSimulating
                  ? 'bg-yellow-600 hover:bg-yellow-700'
                  : 'bg-green-600 hover:bg-green-700'
              } text-white`}
            >
              {isSimulating ? '⏸ Pause Simulation' : '▶ Start Simulation'}
            </button>
            <button
              onClick={handleReset}
              className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold transition-all"
            >
              🔄 Reset
            </button>
          </div>
        </div>

        {/* Description Banner */}
        <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 flex items-start gap-3 mb-6">
          <Network className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              <strong className="text-foreground">What this page does:</strong> This Cyber Range is an OpenAI Gymnasium-compatible reinforcement learning environment that simulates realistic enterprise network attacks and defenses. It models 10 network nodes (web servers, databases, workstations, firewalls, routers) with realistic vulnerabilities and services. Red Team agents can execute 12 different attack actions (reconnaissance, port scanning, exploitation, privilege escalation, lateral movement, data exfiltration) while Blue Team agents deploy 9 defense actions (patching, firewall updates, node isolation, malware scanning). The simulation tracks compromise states, privilege levels, defense effectiveness, and stealth scores. Watch real-time attack progression, node status changes, and event logs as AI agents train to become better attackers and defenders through thousands of simulated battles.
            </p>
          </div>
        </div>

        {/* Simulation Time */}
        <div className="bg-card rounded-lg p-4 border">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Simulation Time</span>
            <span className="text-2xl font-bold font-mono">
              {Math.floor(simulationTime / 60)}:{(simulationTime % 60).toString().padStart(2, '0')}
            </span>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-card rounded-lg p-6 border border-blue-500/30">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-5 h-5 text-blue-500" />
            <h3 className="text-sm text-muted-foreground">Total Attacks</h3>
          </div>
          <p className="text-3xl font-bold">{stats.totalAttacks}</p>
        </div>

        <div className="bg-card rounded-lg p-6 border border-green-500/30">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <h3 className="text-sm text-muted-foreground">Blocked</h3>
          </div>
          <p className="text-3xl font-bold text-green-500">{stats.blockedAttacks}</p>
          <p className="text-xs text-muted-foreground mt-1">
            {stats.totalAttacks > 0 ? ((stats.blockedAttacks / stats.totalAttacks) * 100).toFixed(1) : 0}% success rate
          </p>
        </div>

        <div className="bg-card rounded-lg p-6 border border-red-500/30">
          <div className="flex items-center gap-2 mb-2">
            <XCircle className="w-5 h-5 text-red-500" />
            <h3 className="text-sm text-muted-foreground">Successful</h3>
          </div>
          <p className="text-3xl font-bold text-red-500">{stats.successfulAttacks}</p>
          <p className="text-xs text-muted-foreground mt-1">
            {stats.totalAttacks > 0 ? ((stats.successfulAttacks / stats.totalAttacks) * 100).toFixed(1) : 0}% breach rate
          </p>
        </div>

        <div className="bg-card rounded-lg p-6 border border-orange-500/30">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-5 h-5 text-orange-500" />
            <h3 className="text-sm text-muted-foreground">Compromised Nodes</h3>
          </div>
          <p className="text-3xl font-bold text-orange-500">{stats.nodesCompromised}</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Network Topology */}
        <div className="bg-card rounded-lg p-6 border">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Server className="w-6 h-6 text-blue-500" />
            Network Topology
          </h2>
          <div className="space-y-4">
            {nodes.map((node) => {
              const Icon = getNodeIcon(node.type);
              return (
                <div
                  key={node.id}
                  className={`p-4 rounded-lg border transition-all ${getStatusColor(node.status)}`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${
                        node.status === 'healthy' ? 'bg-green-500/20' :
                        node.status === 'under-attack' ? 'bg-yellow-500/20' :
                        node.status === 'compromised' ? 'bg-red-500/20' :
                        'bg-gray-500/20'
                      }`}>
                        <Icon className={`w-5 h-5 ${
                          node.status === 'healthy' ? 'text-green-500' :
                          node.status === 'under-attack' ? 'text-yellow-500' :
                          node.status === 'compromised' ? 'text-red-500' :
                          'text-gray-500'
                        }`} />
                      </div>
                      <div>
                        <h3 className="font-semibold">{node.name}</h3>
                        <p className="text-xs text-muted-foreground font-mono">{node.ip}</p>
                      </div>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded border ${
                      node.status === 'healthy' ? 'bg-green-500/20 text-green-400 border-green-500/30' :
                      node.status === 'under-attack' ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' :
                      node.status === 'compromised' ? 'bg-red-500/20 text-red-400 border-red-500/30' :
                      'bg-gray-500/20 text-gray-400 border-gray-500/30'
                    }`}>
                      {node.status.replace('-', ' ').toUpperCase()}
                    </span>
                  </div>

                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <Wifi className="w-4 h-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Ports:</span>
                      <span className="font-mono text-xs">
                        {node.ports.length > 0 ? node.ports.join(', ') : 'None'}
                      </span>
                    </div>

                    {node.vulnerabilities.length > 0 && (
                      <div className="flex items-start gap-2">
                        <AlertTriangle className="w-4 h-4 text-orange-500 mt-0.5" />
                        <div className="flex-1">
                          <span className="text-muted-foreground text-xs">Vulnerabilities:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {node.vulnerabilities.map((vuln, i) => (
                              <span key={i} className="text-xs bg-orange-500/20 text-orange-400 px-2 py-0.5 rounded">
                                {vuln}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="mt-3 flex gap-2">
                    {node.status === 'compromised' && (
                      <button
                        onClick={() => handleIsolateNode(node.id)}
                        className="text-xs px-3 py-1 bg-gray-600 hover:bg-gray-700 text-white rounded transition-all"
                      >
                        🔒 Isolate
                      </button>
                    )}
                    {node.status === 'isolated' && (
                      <button
                        onClick={() => handleRestoreNode(node.id)}
                        className="text-xs px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded transition-all"
                      >
                        ✓ Restore
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Event Log */}
        <div className="bg-card rounded-lg p-6 border">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Terminal className="w-6 h-6 text-green-500" />
            Event Log
          </h2>
          <div className="space-y-2 max-h-[700px] overflow-y-auto">
            {events.map((event) => (
              <div
                key={event.id}
                className={`p-3 rounded-lg border transition-all ${
                  event.type === 'defense'
                    ? 'bg-green-500/10 border-green-500/30'
                    : 'bg-red-500/10 border-red-500/30'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {event.type === 'defense' ? (
                      <Shield className="w-4 h-4 text-green-500" />
                    ) : (
                      <Swords className="w-4 h-4 text-red-500" />
                    )}
                    <span className="text-xs font-mono text-muted-foreground">
                      {event.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  <span className={`text-xs px-2 py-0.5 rounded border ${getSeverityColor(event.severity)}`}>
                    {event.severity.toUpperCase()}
                  </span>
                </div>
                <p className="text-sm mb-1">{event.description}</p>
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  <span>From: <code className="bg-muted px-1 rounded">{event.source}</code></span>
                  <span>To: <code className="bg-muted px-1 rounded">{event.target}</code></span>
                </div>
              </div>
            ))}
            {events.length === 0 && (
              <div className="text-center py-12 text-muted-foreground">
                <Terminal className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>No events yet. Start simulation to see network activity.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
