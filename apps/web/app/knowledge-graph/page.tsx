'use client';

import { Network, Search, Filter, Download } from 'lucide-react';

export default function KnowledgeGraphPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-8">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold mb-2">Knowledge Graph</h1>
            <p className="text-muted-foreground">Threat intelligence and attack path visualization</p>
          </div>
          <button className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90">
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
                  <select className="w-full px-3 py-2 bg-background border rounded-md text-sm">
                    <option>All Types</option>
                    <option>Attacks</option>
                    <option>Defenses</option>
                    <option>Vulnerabilities</option>
                    <option>Assets</option>
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
                    placeholder="Search nodes..."
                    className="w-full pl-10 pr-4 py-2 bg-background border rounded-lg"
                  />
                </div>
              </div>

              <div className="bg-muted/20 rounded-lg h-[600px] flex items-center justify-center border-2 border-dashed">
                <div className="text-center">
                  <Network className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-semibold mb-2">3D Knowledge Graph Visualization</h3>
                  <p className="text-sm text-muted-foreground mb-4">Interactive graph will render here</p>
                  <p className="text-xs text-muted-foreground max-w-md">
                    This will display an interactive 3D visualization using D3.js or Three.js<br/>
                    showing attack paths, vulnerabilities, and defense strategies
                  </p>
                </div>
              </div>

              <div className="mt-4 flex items-center justify-between">
                <div className="flex gap-2">
                  <button className="px-3 py-1 text-sm bg-background border rounded-md hover:bg-accent">Reset View</button>
                  <button className="px-3 py-1 text-sm bg-background border rounded-md hover:bg-accent">Zoom In</button>
                  <button className="px-3 py-1 text-sm bg-background border rounded-md hover:bg-accent">Zoom Out</button>
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
