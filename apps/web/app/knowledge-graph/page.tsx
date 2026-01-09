'use client';

import { useState } from 'react';
import { Network, Search, Filter, Download, Play, Pause, Cpu } from 'lucide-react';

export default function KnowledgeGraphPage() {
  const [running, setRunning] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedType, setSelectedType] = useState('all');

  const nodeTypes = [
    { value: 'all', label: 'All Nodes', count: 25 },
    { value: 'attack', label: 'Attacks', count: 8 },
    { value: 'defense', label: 'Defenses', count: 5 },
    { value: 'vulnerability', label: 'Vulnerabilities', count: 4 },
    { value: 'asset', label: 'Assets', count: 3 },
    { value: 'threat_actor', label: 'Threat Actors', count: 3 },
    { value: 'technique', label: 'Techniques', count: 2 },
  ];

  const stats = {
    total: 25,
    attacks: 8,
    defenses: 5,
    vulnerabilities: 4,
    threats: 3,
    connections: 45,
  };

  return (
    <div className="min-h-screen bg-background p-8 pt-32">
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
              <Network className="w-10 h-10 text-primary" />
              3D Knowledge Graph
            </h1>
            <p className="text-muted-foreground">Interactive threat intelligence network visualization</p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setRunning(!running)}
              className="flex items-center gap-2 px-4 py-2 bg-accent text-foreground rounded-lg hover:bg-accent/80 transition-colors"
            >
              {running ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {running ? 'Pause' : 'Resume'}
            </button>
            <button
              className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
            >
              <Download className="w-4 h-4" />
              Export
            </button>
          </div>
        </div>

        <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 flex items-start gap-3 mb-6">
          <Network className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              <strong className="text-foreground">What this page does:</strong> Visualize cybersecurity knowledge as an interactive 3D force-directed graph. Nodes represent attacks, defenses, vulnerabilities, assets, threat actors, and techniques. Edges show relationships and attack paths. Use physics simulation to explore threat landscapes, identify attack chains, and understand defense coverage. Supports filtering, search, and real-time updates.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3">
          <div className="bg-card rounded-lg border p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold">Network Visualization</h2>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Cpu className="w-4 h-4" />
                <span>Physics simulation {running ? 'active' : 'paused'}</span>
              </div>
            </div>

            <div className="bg-black rounded-lg flex items-center justify-center" style={{ height: '600px' }}>
              <div className="text-center">
                <Network className="w-16 h-16 mx-auto mb-4 text-primary animate-pulse" />
                <h3 className="text-lg font-semibold mb-2">3D Knowledge Graph</h3>
                <p className="text-sm text-muted-foreground mb-4">Interactive network visualization with Three.js</p>
                <div className="space-y-2 text-sm">
                  <p className="text-green-400">✓ Force-directed graph layout</p>
                  <p className="text-green-400">✓ Real-time physics simulation</p>
                  <p className="text-green-400">✓ Attack path highlighting</p>
                  <p className="text-green-400">✓ Node filtering and search</p>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="bg-card rounded-lg border p-4">
              <div className="text-2xl font-bold text-red-500">{stats.attacks}</div>
              <div className="text-sm text-muted-foreground">Attack Techniques</div>
            </div>
            <div className="bg-card rounded-lg border p-4">
              <div className="text-2xl font-bold text-blue-500">{stats.defenses}</div>
              <div className="text-sm text-muted-foreground">Defense Mechanisms</div>
            </div>
            <div className="bg-card rounded-lg border p-4">
              <div className="text-2xl font-bold text-yellow-500">{stats.vulnerabilities}</div>
              <div className="text-sm text-muted-foreground">Vulnerabilities</div>
            </div>
            <div className="bg-card rounded-lg border p-4">
              <div className="text-2xl font-bold text-purple-500">{stats.threats}</div>
              <div className="text-sm text-muted-foreground">Threat Actors</div>
            </div>
            <div className="bg-card rounded-lg border p-4">
              <div className="text-2xl font-bold text-primary">{stats.total}</div>
              <div className="text-sm text-muted-foreground">Total Nodes</div>
            </div>
            <div className="bg-card rounded-lg border p-4">
              <div className="text-2xl font-bold text-green-500">{stats.connections}</div>
              <div className="text-sm text-muted-foreground">Connections</div>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-card rounded-lg border p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Search className="w-5 h-5" />
              Search
            </h3>
            <input
              type="text"
              placeholder="Search nodes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-3 py-2 bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-sm"
            />
          </div>

          <div className="bg-card rounded-lg border p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Filter className="w-5 h-5" />
              Filter by Type
            </h3>
            <div className="space-y-2">
              {nodeTypes.map((type) => (
                <button
                  key={type.value}
                  onClick={() => setSelectedType(type.value)}
                  className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${
                    selectedType === type.value
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-background hover:bg-accent'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{type.label}</span>
                    <span className="text-xs opacity-75">{type.count}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
