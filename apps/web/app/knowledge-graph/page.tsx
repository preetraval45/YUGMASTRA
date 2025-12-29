'use client';

import { useState, useEffect, useRef } from 'react';
import { Network, Search, Filter, Download, ZoomIn, ZoomOut, RefreshCw } from 'lucide-react';

interface Node {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  type: 'attack' | 'defense' | 'vulnerability' | 'asset';
  label: string;
  connections: string[];
}

export default function KnowledgeGraphPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [zoom, setZoom] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedType, setSelectedType] = useState<string>('all');

  // Initialize nodes
  useEffect(() => {
    const initialNodes: Node[] = [
      { id: '1', x: 400, y: 300, vx: 0, vy: 0, type: 'attack', label: 'SQL Injection', connections: ['2', '5'] },
      { id: '2', x: 250, y: 200, vx: 0, vy: 0, type: 'defense', label: 'WAF Rule', connections: ['1'] },
      { id: '3', x: 550, y: 250, vx: 0, vy: 0, type: 'attack', label: 'XSS', connections: ['4', '6'] },
      { id: '4', x: 300, y: 400, vx: 0, vy: 0, type: 'defense', label: 'CSP Header', connections: ['3'] },
      { id: '5', x: 500, y: 450, vx: 0, vy: 0, type: 'vulnerability', label: 'CVE-2024-1234', connections: ['1', '7'] },
      { id: '6', x: 650, y: 350, vx: 0, vy: 0, type: 'asset', label: 'Web Server', connections: ['3', '7'] },
      { id: '7', x: 450, y: 550, vx: 0, vy: 0, type: 'asset', label: 'Database', connections: ['5', '6'] },
    ];
    setNodes(initialNodes);
  }, []);

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Apply filter
      const filteredNodes = selectedType === 'all'
        ? nodes
        : nodes.filter(n => n.type === selectedType);

      // Draw connections
      ctx.strokeStyle = '#444';
      ctx.lineWidth = 1;
      filteredNodes.forEach(node => {
        node.connections.forEach(connId => {
          const target = nodes.find(n => n.id === connId);
          if (target && (selectedType === 'all' || filteredNodes.includes(target))) {
            ctx.beginPath();
            ctx.moveTo(node.x * zoom, node.y * zoom);
            ctx.lineTo(target.x * zoom, target.y * zoom);
            ctx.stroke();
          }
        });
      });

      // Draw nodes
      filteredNodes.forEach(node => {
        const colors = {
          attack: '#ef4444',
          defense: '#3b82f6',
          vulnerability: '#eab308',
          asset: '#22c55e'
        };

        ctx.fillStyle = colors[node.type];
        ctx.beginPath();
        ctx.arc(node.x * zoom, node.y * zoom, 20 * zoom, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#fff';
        ctx.font = `${12 * zoom}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillText(node.label, node.x * zoom, node.y * zoom - 30 * zoom);
      });

      requestAnimationFrame(animate);
    };

    animate();
  }, [nodes, zoom, selectedType]);

  const handleExport = () => {
    const data = JSON.stringify(nodes, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `knowledge-graph-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-8">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold mb-2">Knowledge Graph</h1>
            <p className="text-muted-foreground">Threat intelligence and attack path visualization</p>
          </div>
          <button
            onClick={handleExport}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
          >
            <Download className="w-4 h-4" />
            Export Graph
          </button>
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
                  </select>
                </div>
                <div>
                  <label className="text-sm text-muted-foreground mb-1 block">Time Range</label>
                  <select className="w-full px-3 py-2 bg-background border rounded-md text-sm">
                    <option>Last 24 hours</option>
                    <option>Last 7 days</option>
                    <option>Last 30 days</option>
                    <option>All time</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm text-muted-foreground mb-1 block">Severity</label>
                  <div className="flex gap-2">
                    <button className="flex-1 px-2 py-1 text-xs bg-red-500/20 text-red-500 rounded">High</button>
                    <button className="flex-1 px-2 py-1 text-xs bg-orange-500/20 text-orange-500 rounded">Med</button>
                    <button className="flex-1 px-2 py-1 text-xs bg-yellow-500/20 text-yellow-500 rounded">Low</button>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-card rounded-lg p-4 border">
              <h3 className="font-semibold mb-3">Graph Statistics</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Nodes</span>
                  <span className="font-semibold">1,247</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Edges</span>
                  <span className="font-semibold">3,891</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Attack Paths</span>
                  <span className="font-semibold">342</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Vulnerabilities</span>
                  <span className="font-semibold">89</span>
                </div>
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
                    placeholder="Search nodes..."
                    className="w-full pl-10 pr-4 py-2 bg-background border rounded-lg"
                  />
                </div>
              </div>

              <div className="bg-muted/20 rounded-lg h-[600px] flex items-center justify-center border-2 relative overflow-hidden">
                <canvas
                  ref={canvasRef}
                  width={800}
                  height={600}
                  className="absolute inset-0 w-full h-full"
                />
                {nodes.length === 0 && (
                  <div className="text-center z-10">
                    <Network className="w-16 h-16 mx-auto mb-4 text-muted-foreground animate-pulse" />
                    <h3 className="text-lg font-semibold mb-2">Loading Knowledge Graph...</h3>
                  </div>
                )}
              </div>

              <div className="mt-4 flex items-center justify-between">
                <div className="flex gap-2">
                  <button
                    onClick={() => setZoom(1)}
                    className="px-3 py-1 text-sm bg-background border rounded-md hover:bg-accent transition-colors flex items-center gap-1"
                  >
                    <RefreshCw className="w-3 h-3" />
                    Reset View
                  </button>
                  <button
                    onClick={() => setZoom(prev => Math.min(prev + 0.2, 3))}
                    className="px-3 py-1 text-sm bg-background border rounded-md hover:bg-accent transition-colors flex items-center gap-1"
                  >
                    <ZoomIn className="w-3 h-3" />
                    Zoom In
                  </button>
                  <button
                    onClick={() => setZoom(prev => Math.max(prev - 0.2, 0.5))}
                    className="px-3 py-1 text-sm bg-background border rounded-md hover:bg-accent transition-colors flex items-center gap-1"
                  >
                    <ZoomOut className="w-3 h-3" />
                    Zoom Out
                  </button>
                </div>
                <div className="flex items-center gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    <span>Attacks</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                    <span>Defenses</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    <span>Vulnerabilities</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    <span>Assets</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
