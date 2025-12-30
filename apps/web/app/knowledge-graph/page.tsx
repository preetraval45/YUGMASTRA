'use client';

import { useState, useEffect, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { Network, Search, Filter, Download, Play, Pause, Settings2, Cpu } from 'lucide-react';
import * as THREE from 'three';
import type { GraphNode } from '@/components/3d-knowledge-graph';

// Dynamically import 3D component with no SSR
const ThreeDKnowledgeGraph = dynamic(
  () => import('@/components/3d-knowledge-graph').then((mod) => mod.ThreeDKnowledgeGraph),
  {
    ssr: false,
    loading: () => (
      <div className="absolute inset-0 flex items-center justify-center bg-black">
        <div className="text-center">
          <Cpu className="w-16 h-16 mx-auto mb-4 text-primary animate-pulse" />
          <h3 className="text-lg font-semibold mb-2">Loading 3D Engine...</h3>
        </div>
      </div>
    ),
  }
);

const NODE_COLORS = {
  attack: '#ef4444',
  defense: '#3b82f6',
  vulnerability: '#eab308',
  asset: '#22c55e',
  threat_actor: '#a855f7',
  technique: '#ec4899',
};

export default function KnowledgeGraphPage() {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [running, setRunning] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedType, setSelectedType] = useState<string>('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const initialNodes: GraphNode[] = [
      {
        id: 'ta1',
        type: 'threat_actor',
        label: 'APT28',
        connections: ['t1', 't2', 't3'],
        position: new THREE.Vector3(-5, 0, 0),
        velocity: new THREE.Vector3(0, 0, 0),
        severity: 'critical',
        metadata: { description: 'Advanced Persistent Threat Group' }
      },
      {
        id: 'ta2',
        type: 'threat_actor',
        label: 'Lazarus Group',
        connections: ['t4', 't5', 'v2'],
        position: new THREE.Vector3(5, 0, 0),
        velocity: new THREE.Vector3(0, 0, 0),
        severity: 'critical',
        metadata: { description: 'Nation-state threat actor' }
      },
      {
        id: 't1',
        type: 'technique',
        label: 'Spearphishing',
        connections: ['a1', 'ta1'],
        position: new THREE.Vector3(-3, 2, 2),
        velocity: new THREE.Vector3(0, 0, 0),
        severity: 'high',
        metadata: { mitre: 'T1566.001' }
      },
      {
        id: 't2',
        type: 'technique',
        label: 'Credential Dumping',
        connections: ['a2', 'ta1', 'v1'],
        position: new THREE.Vector3(-3, -2, 2),
        velocity: new THREE.Vector3(0, 0, 0),
        severity: 'critical',
        metadata: { mitre: 'T1003' }
      },
      {
        id: 't3',
        type: 'technique',
        label: 'Lateral Movement',
        connections: ['a3', 'ta1'],
        position: new THREE.Vector3(-3, 0, -2),
        velocity: new THREE.Vector3(0, 0, 0),
        severity: 'high',
        metadata: { mitre: 'T1021' }
      },
      {
        id: 't4',
        type: 'technique',
        label: 'Supply Chain Attack',
        connections: ['a4', 'ta2', 'v3'],
        position: new THREE.Vector3(3, 2, 2),
        velocity: new THREE.Vector3(0, 0, 0),
        severity: 'critical',
        metadata: { mitre: 'T1195' }
      },
      {
        id: 't5',
        type: 'technique',
        label: 'Ransomware',
        connections: ['a5', 'ta2'],
        position: new THREE.Vector3(3, -2, 2),
        velocity: new THREE.Vector3(0, 0, 0),
        severity: 'critical',
        metadata: { mitre: 'T1486' }
      },
      {
        id: 'a1',
        type: 'attack',
        label: 'SQL Injection',
        connections: ['v1', 'd1', 'asset1'],
        position: new THREE.Vector3(0, 3, 0),
        velocity: new THREE.Vector3(0, 0, 0),
        severity: 'high',
        metadata: { description: 'Database injection attack' }
      },
      {
        id: 'a2',
        type: 'attack',
        label: 'XSS Attack',
        connections: ['v2', 'd2', 'asset2'],
        position: new THREE.Vector3(0, -3, 0),
        velocity: new THREE.Vector3(0, 0, 0),
        severity: 'medium',
        metadata: { description: 'Cross-site scripting' }
      },
      {
        id: 'a3',
        type: 'attack',
        label: 'RCE Exploit',
        connections: ['v3', 'd3', 'asset1'],
        position: new THREE.Vector3(2, 0, 3),
        velocity: new THREE.Vector3(0, 0, 0),
        severity: 'critical',
        metadata: { description: 'Remote code execution' }
      },
      {
        id: 'a4',
        type: 'attack',
        label: 'DDoS Attack',
        connections: ['d4', 'asset3'],
        position: new THREE.Vector3(-2, 0, 3),
        velocity: new THREE.Vector3(0, 0, 0),
        severity: 'high',
        metadata: { description: 'Distributed denial of service' }
      },
      {
        id: 'a5',
        type: 'attack',
        label: 'Privilege Escalation',
        connections: ['v1', 'd5', 'asset1'],
        position: new THREE.Vector3(0, 0, -3),
        velocity: new THREE.Vector3(0, 0, 0),
        severity: 'high',
        metadata: { description: 'Escalate user privileges' }
      },
      {
        id: 'v1',
        type: 'vulnerability',
        label: 'CVE-2024-1234',
        connections: ['a1', 'd1'],
        position: new THREE.Vector3(-2, 2, -1),
        velocity: new THREE.Vector3(0, 0, 0),
        severity: 'critical',
        metadata: { cve: 'CVE-2024-1234', score: 9.8 }
      },
      {
        id: 'v2',
        type: 'vulnerability',
        label: 'CVE-2024-5678',
        connections: ['a2', 'd2'],
        position: new THREE.Vector3(2, -2, -1),
        velocity: new THREE.Vector3(0, 0, 0),
        severity: 'high',
        metadata: { cve: 'CVE-2024-5678', score: 7.5 }
      },
      {
        id: 'v3',
        type: 'vulnerability',
        label: 'CVE-2024-9012',
        connections: ['a3', 'd3'],
        position: new THREE.Vector3(0, 2, 2),
        velocity: new THREE.Vector3(0, 0, 0),
        severity: 'critical',
        metadata: { cve: 'CVE-2024-9012', score: 10.0 }
      },
      {
        id: 'd1',
        type: 'defense',
        label: 'WAF Protection',
        connections: ['asset1', 'a1'],
        position: new THREE.Vector3(-1, 1, 1),
        velocity: new THREE.Vector3(0, 0, 0),
        metadata: { description: 'Web Application Firewall' }
      },
      {
        id: 'd2',
        type: 'defense',
        label: 'CSP Headers',
        connections: ['asset2', 'a2'],
        position: new THREE.Vector3(1, -1, 1),
        velocity: new THREE.Vector3(0, 0, 0),
        metadata: { description: 'Content Security Policy' }
      },
      {
        id: 'd3',
        type: 'defense',
        label: 'Sandboxing',
        connections: ['asset1', 'a3'],
        position: new THREE.Vector3(1, 1, -1),
        velocity: new THREE.Vector3(0, 0, 0),
        metadata: { description: 'Process isolation' }
      },
      {
        id: 'd4',
        type: 'defense',
        label: 'Rate Limiting',
        connections: ['asset3', 'a4'],
        position: new THREE.Vector3(-1, -1, -1),
        velocity: new THREE.Vector3(0, 0, 0),
        metadata: { description: 'Traffic throttling' }
      },
      {
        id: 'd5',
        type: 'defense',
        label: 'RBAC',
        connections: ['asset1', 'a5'],
        position: new THREE.Vector3(0, -1, 2),
        velocity: new THREE.Vector3(0, 0, 0),
        metadata: { description: 'Role-based access control' }
      },
      {
        id: 'asset1',
        type: 'asset',
        label: 'Web Server',
        connections: ['asset2', 'asset3'],
        position: new THREE.Vector3(0, 0, 0),
        velocity: new THREE.Vector3(0, 0, 0),
        metadata: { description: 'Primary web application server' }
      },
      {
        id: 'asset2',
        type: 'asset',
        label: 'Database',
        connections: ['asset1'],
        position: new THREE.Vector3(2, 0, 0),
        velocity: new THREE.Vector3(0, 0, 0),
        metadata: { description: 'PostgreSQL database server' }
      },
      {
        id: 'asset3',
        type: 'asset',
        label: 'API Gateway',
        connections: ['asset1'],
        position: new THREE.Vector3(-2, 0, 0),
        velocity: new THREE.Vector3(0, 0, 0),
        metadata: { description: 'REST API gateway' }
      },
    ];

    setNodes(initialNodes);
    setLoading(false);
  }, []);

  const filteredNodes = useMemo(() => {
    let filtered = selectedType === 'all' ? nodes : nodes.filter(n => n.type === selectedType);

    if (searchTerm) {
      filtered = filtered.filter(n =>
        n.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
        n.metadata.mitre?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        n.metadata.cve?.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    return filtered;
  }, [nodes, selectedType, searchTerm]);

  const selectedNodeData = useMemo(() =>
    selectedNode ? nodes.find(n => n.id === selectedNode) : null,
    [selectedNode, nodes]
  );

  const stats = useMemo(() => ({
    total: nodes.length,
    attacks: nodes.filter(n => n.type === 'attack').length,
    defenses: nodes.filter(n => n.type === 'defense').length,
    vulnerabilities: nodes.filter(n => n.type === 'vulnerability').length,
    threats: nodes.filter(n => n.type === 'threat_actor').length,
    connections: nodes.reduce((sum, n) => sum + n.connections.length, 0),
  }), [nodes]);

  const handleExport = () => {
    const data = {
      nodes: nodes.map(n => ({
        id: n.id,
        type: n.type,
        label: n.label,
        connections: n.connections,
        metadata: n.metadata,
      })),
      stats,
      exportedAt: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `yugmastra-knowledge-graph-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-8">
        <div className="mb-8 flex items-center justify-between">
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
              onClick={handleExport}
              className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
            >
              <Download className="w-4 h-4" />
              Export
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-6">
          <div className="bg-gradient-to-br from-blue-500/20 to-blue-600/20 border border-blue-500/30 rounded-lg p-4">
            <div className="text-2xl font-bold">{stats.total}</div>
            <div className="text-sm text-muted-foreground">Total Nodes</div>
          </div>
          <div className="bg-gradient-to-br from-red-500/20 to-red-600/20 border border-red-500/30 rounded-lg p-4">
            <div className="text-2xl font-bold">{stats.attacks}</div>
            <div className="text-sm text-muted-foreground">Attacks</div>
          </div>
          <div className="bg-gradient-to-br from-blue-500/20 to-blue-600/20 border border-blue-500/30 rounded-lg p-4">
            <div className="text-2xl font-bold">{stats.defenses}</div>
            <div className="text-sm text-muted-foreground">Defenses</div>
          </div>
          <div className="bg-gradient-to-br from-yellow-500/20 to-yellow-600/20 border border-yellow-500/30 rounded-lg p-4">
            <div className="text-2xl font-bold">{stats.vulnerabilities}</div>
            <div className="text-sm text-muted-foreground">CVEs</div>
          </div>
          <div className="bg-gradient-to-br from-purple-500/20 to-purple-600/20 border border-purple-500/30 rounded-lg p-4">
            <div className="text-2xl font-bold">{stats.threats}</div>
            <div className="text-sm text-muted-foreground">Threat Actors</div>
          </div>
          <div className="bg-gradient-to-br from-green-500/20 to-green-600/20 border border-green-500/30 rounded-lg p-4">
            <div className="text-2xl font-bold">{stats.connections}</div>
            <div className="text-sm text-muted-foreground">Connections</div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-card rounded-lg p-4 border">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Filter className="w-4 h-4" />
                Filters
              </h3>
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-muted-foreground mb-1 block">Node Type</label>
                  <select
                    value={selectedType}
                    onChange={(e) => setSelectedType(e.target.value)}
                    className="w-full px-3 py-2 bg-background border rounded-md text-sm"
                  >
                    <option value="all">All Types</option>
                    <option value="attack">Attacks</option>
                    <option value="defense">Defenses</option>
                    <option value="vulnerability">Vulnerabilities</option>
                    <option value="asset">Assets</option>
                    <option value="threat_actor">Threat Actors</option>
                    <option value="technique">Techniques</option>
                  </select>
                </div>
              </div>
            </div>

            {selectedNodeData && (
              <div className="bg-card rounded-lg p-4 border">
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  <Settings2 className="w-4 h-4" />
                  Node Details
                </h3>
                <div className="space-y-2">
                  <div>
                    <div className="text-xs text-muted-foreground">Label</div>
                    <div className="font-semibold">{selectedNodeData.label}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">Type</div>
                    <div className="capitalize">{selectedNodeData.type.replace('_', ' ')}</div>
                  </div>
                  {selectedNodeData.severity && (
                    <div>
                      <div className="text-xs text-muted-foreground">Severity</div>
                      <div className={`inline-block px-2 py-1 rounded text-xs font-semibold ${
                        selectedNodeData.severity === 'critical' ? 'bg-red-500/20 text-red-500' :
                        selectedNodeData.severity === 'high' ? 'bg-orange-500/20 text-orange-500' :
                        selectedNodeData.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-500' :
                        'bg-blue-500/20 text-blue-500'
                      }`}>
                        {selectedNodeData.severity.toUpperCase()}
                      </div>
                    </div>
                  )}
                  {selectedNodeData.metadata.mitre && (
                    <div>
                      <div className="text-xs text-muted-foreground">MITRE ATT&CK</div>
                      <div className="font-mono text-sm">{selectedNodeData.metadata.mitre}</div>
                    </div>
                  )}
                  {selectedNodeData.metadata.cve && (
                    <div>
                      <div className="text-xs text-muted-foreground">CVE ID</div>
                      <div className="font-mono text-sm">{selectedNodeData.metadata.cve}</div>
                    </div>
                  )}
                  {selectedNodeData.metadata.score && (
                    <div>
                      <div className="text-xs text-muted-foreground">CVSS Score</div>
                      <div className="font-semibold">{selectedNodeData.metadata.score}/10</div>
                    </div>
                  )}
                  <div>
                    <div className="text-xs text-muted-foreground">Connections</div>
                    <div>{selectedNodeData.connections.length}</div>
                  </div>
                </div>
              </div>
            )}

            <div className="bg-card rounded-lg p-4 border">
              <h3 className="font-semibold mb-3">Legend</h3>
              <div className="space-y-2 text-sm">
                {Object.entries(NODE_COLORS).map(([type, color]) => (
                  <div key={type} className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }}></div>
                    <span className="capitalize">{type.replace('_', ' ')}s</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="lg:col-span-3">
            <div className="bg-card rounded-lg p-6 border">
              <div className="mb-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Search by name, MITRE ID, or CVE..."
                    className="w-full pl-10 pr-4 py-2 bg-background border rounded-lg"
                  />
                </div>
              </div>

              <div className="bg-black rounded-lg h-[700px] relative overflow-hidden border-2 border-primary/20">
                {loading ? (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <Cpu className="w-16 h-16 mx-auto mb-4 text-primary animate-pulse" />
                      <h3 className="text-lg font-semibold mb-2">Initializing 3D Graph...</h3>
                    </div>
                  </div>
                ) : (
                  <ThreeDKnowledgeGraph
                    nodes={filteredNodes}
                    selectedNode={selectedNode}
                    onNodeClick={setSelectedNode}
                    running={running}
                  />
                )}
              </div>

              <div className="mt-4 flex items-center justify-between text-sm text-muted-foreground">
                <div>
                  Showing {filteredNodes.length} of {nodes.length} nodes
                </div>
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4" />
                  <span>Real-time physics simulation {running ? 'active' : 'paused'}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
