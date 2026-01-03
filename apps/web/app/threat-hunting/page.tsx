"use client"

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Search, Target, TrendingUp, AlertCircle, CheckCircle, Brain, Download, Play } from 'lucide-react'

interface Hunt {
  id: string
  hypothesis: string
  status: 'active' | 'completed' | 'investigating'
  findings: number
  iocs: string[]
  techniques: string[]
  confidence: number
  created: string
  queries: string[]
}

export default function ThreatHuntingPage() {
  const [hypothesis, setHypothesis] = useState('')
  const [hunts, setHunts] = useState<Hunt[]>([])
  const [loading, setLoading] = useState(false)

  const generateHunt = async () => {
    if (!hypothesis) return
    setLoading(true)

    await new Promise(r => setTimeout(r, 2000))

    const newHunt: Hunt = {
      id: 'HUNT-' + Date.now(),
      hypothesis,
      status: 'active',
      findings: Math.floor(Math.random() * 20),
      iocs: ['192.168.1.100', 'malicious.com', 'hash:abc123'],
      techniques: ['T1059.001', 'T1071.001', 'T1055'],
      confidence: 70 + Math.floor(Math.random() * 25),
      created: new Date().toISOString(),
      queries: [
        'SELECT * FROM processes WHERE name LIKE "%powershell%" AND parent != "explorer.exe"',
        'SELECT * FROM network_connections WHERE remote_address NOT IN (whitelist) AND protocol = "HTTPS"'
      ]
    }

    setHunts([newHunt, ...hunts])
    setHypothesis('')
    setLoading(false)
  }

  const quickHypotheses = [
    'Detect lateral movement using Pass-the-Hash attacks',
    'Identify data exfiltration via DNS tunneling',
    'Hunt for living-off-the-land binaries (LOLBins) execution',
    'Find unauthorized PowerShell remoting sessions'
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
            AI Threat Hunting
          </h1>
          <p className="text-gray-400 mt-2">Proactive threat detection using AI-powered hypotheses</p>
        </div>
        <Button className="bg-gradient-to-r from-purple-500 to-pink-600">
          <Download className="mr-2 h-4 w-4" />
          Export Report
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="bg-black/40 border-purple-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Active Hunts</p>
              <p className="text-3xl font-bold text-white mt-1">{hunts.filter(h => h.status === 'active').length}</p>
            </div>
            <Target className="h-8 w-8 text-purple-400" />
          </div>
        </Card>
        <Card className="bg-black/40 border-green-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Total Findings</p>
              <p className="text-3xl font-bold text-green-400 mt-1">
                {hunts.reduce((acc, h) => acc + h.findings, 0)}
              </p>
            </div>
            <CheckCircle className="h-8 w-8 text-green-400" />
          </div>
        </Card>
        <Card className="bg-black/40 border-blue-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Avg Confidence</p>
              <p className="text-3xl font-bold text-blue-400 mt-1">
                {hunts.length > 0 ? Math.round(hunts.reduce((acc, h) => acc + h.confidence, 0) / hunts.length) : 0}%
              </p>
            </div>
            <TrendingUp className="h-8 w-8 text-blue-400" />
          </div>
        </Card>
        <Card className="bg-black/40 border-orange-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">IOCs Found</p>
              <p className="text-3xl font-bold text-orange-400 mt-1">
                {hunts.reduce((acc, h) => acc + h.iocs.length, 0)}
              </p>
            </div>
            <AlertCircle className="h-8 w-8 text-orange-400" />
          </div>
        </Card>
      </div>

      <Card className="bg-black/40 border-purple-500/20 p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Generate Hunt Hypothesis</h2>
        <div className="space-y-4">
          <div className="flex gap-4">
            <Input
              placeholder="Describe the threat behavior you want to hunt for..."
              value={hypothesis}
              onChange={(e) => setHypothesis(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && generateHunt()}
              className="flex-1 bg-black/40 border-purple-500/20"
            />
            <Button onClick={generateHunt} disabled={loading || !hypothesis} className="bg-purple-500 hover:bg-purple-600">
              <Brain className="mr-2 h-4 w-4" />
              {loading ? 'Generating...' : 'Start Hunt'}
            </Button>
          </div>
          <div>
            <p className="text-gray-400 text-sm mb-2">Quick Hypotheses:</p>
            <div className="flex flex-wrap gap-2">
              {quickHypotheses.map((hyp, idx) => (
                <button
                  key={idx}
                  onClick={() => setHypothesis(hyp)}
                  className="px-3 py-1 text-sm bg-purple-500/20 text-purple-400 border border-purple-500/50 rounded-lg hover:bg-purple-500/30 transition-all"
                >
                  {hyp}
                </button>
              ))}
            </div>
          </div>
        </div>
      </Card>

      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-white">Hunt Results ({hunts.length})</h2>
        {hunts.length === 0 ? (
          <Card className="bg-black/40 border-purple-500/20 p-12 text-center">
            <Search className="h-12 w-12 text-gray-500 mx-auto mb-4" />
            <p className="text-gray-400">No active hunts. Create a hypothesis above to start hunting!</p>
          </Card>
        ) : (
          hunts.map((hunt) => (
            <Card key={hunt.id} className="bg-black/40 border-purple-500/20 p-6 hover:border-purple-500/40 transition-all">
              <div className="space-y-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="text-lg font-semibold text-white">{hunt.hypothesis}</h3>
                      <Badge className={
                        hunt.status === 'active' ? 'bg-green-500/20 text-green-400 border-green-500/50' :
                        hunt.status === 'investigating' ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50' :
                        'bg-blue-500/20 text-blue-400 border-blue-500/50'
                      }>
                        {hunt.status}
                      </Badge>
                    </div>
                    <p className="text-gray-400 text-sm">{hunt.id} • Started {new Date(hunt.created).toLocaleString()}</p>
                  </div>
                  <Button size="sm" className="bg-purple-500/20 hover:bg-purple-500/30 text-purple-400">
                    <Play className="h-4 w-4" />
                  </Button>
                </div>

                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-purple-500/10 rounded-lg p-3">
                    <p className="text-gray-400 text-sm mb-1">Findings</p>
                    <p className="text-2xl font-bold text-white">{hunt.findings}</p>
                  </div>
                  <div className="bg-blue-500/10 rounded-lg p-3">
                    <p className="text-gray-400 text-sm mb-1">Confidence</p>
                    <p className="text-2xl font-bold text-blue-400">{hunt.confidence}%</p>
                  </div>
                  <div className="bg-green-500/10 rounded-lg p-3">
                    <p className="text-gray-400 text-sm mb-1">IOCs</p>
                    <p className="text-2xl font-bold text-green-400">{hunt.iocs.length}</p>
                  </div>
                </div>

                <div>
                  <p className="text-gray-400 text-sm mb-2">MITRE ATT&CK Techniques:</p>
                  <div className="flex flex-wrap gap-2">
                    {hunt.techniques.map((t) => (
                      <Badge key={t} className="bg-purple-500/20 text-purple-400 border-purple-500/50">
                        {t}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="text-gray-400 text-sm mb-2">Indicators of Compromise:</p>
                  <div className="flex flex-wrap gap-2">
                    {hunt.iocs.map((ioc, idx) => (
                      <Badge key={idx} className="bg-red-500/20 text-red-400 border-red-500/50 font-mono text-xs">
                        {ioc}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="text-gray-400 text-sm mb-2">AI-Generated Queries:</p>
                  <div className="space-y-2">
                    {hunt.queries.map((query, idx) => (
                      <div key={idx} className="bg-black/60 border border-gray-700 rounded-lg p-3">
                        <code className="text-sm text-cyan-400">{query}</code>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex gap-2">
                  <Button size="sm" className="bg-blue-500/20 hover:bg-blue-500/30 text-blue-400">
                    View Details
                  </Button>
                  <Button size="sm" className="bg-green-500/20 hover:bg-green-500/30 text-green-400">
                    Export IOCs
                  </Button>
                  <Button size="sm" className="bg-purple-500/20 hover:bg-purple-500/30 text-purple-400">
                    Create Alert
                  </Button>
                </div>
              </div>
            </Card>
          ))
        )}
      </div>
    </div>
  )
}
