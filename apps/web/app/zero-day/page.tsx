"use client"

import { useState, useEffect } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { AlertTriangle, Shield, TrendingUp, Search, Download, Filter } from 'lucide-react'

interface Vulnerability {
  id: string
  name: string
  cvss: number
  epss: number
  confidence: number
  category: string
  description: string
  discovered: string
  status: 'new' | 'analyzing' | 'confirmed' | 'mitigated'
  affectedSystems: string[]
  exploitAvailable: boolean
  severity: 'critical' | 'high' | 'medium' | 'low'
}

interface BehaviorAnomaly {
  id: string
  timestamp: string
  anomalyType: string
  confidence: number
  indicators: string[]
  riskScore: number
}

export default function ZeroDayPage() {
  const [vulnerabilities, setVulnerabilities] = useState<Vulnerability[]>([])
  const [anomalies, setAnomalies] = useState<BehaviorAnomaly[]>([])
  const [loading, setLoading] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedSeverity, setSelectedSeverity] = useState<string>('all')

  useEffect(() => {
    loadVulnerabilities()
    loadAnomalies()

    const interval = setInterval(() => {
      loadAnomalies()
    }, 30000)

    return () => clearInterval(interval)
  }, [])

  const loadVulnerabilities = async () => {
    setLoading(true)
    try {
      // Mock data for now - replace with actual API call
      const mockVulns: Vulnerability[] = [
        {
          id: 'ZD-2025-001',
          name: 'Remote Code Execution in Authentication Module',
          cvss: 9.8,
          epss: 0.92,
          confidence: 87,
          category: 'Authentication Bypass',
          description: 'AI-detected anomalous behavior in JWT validation allowing arbitrary code execution',
          discovered: new Date().toISOString(),
          status: 'analyzing',
          affectedSystems: ['auth-service', 'api-gateway'],
          exploitAvailable: false,
          severity: 'critical'
        },
        {
          id: 'ZD-2025-002',
          name: 'SQL Injection via API Parameter Pollution',
          cvss: 8.6,
          epss: 0.78,
          confidence: 93,
          category: 'Injection',
          description: 'Novel parameter pollution technique bypassing input validation',
          discovered: new Date(Date.now() - 3600000).toISOString(),
          status: 'confirmed',
          affectedSystems: ['database-api', 'admin-panel'],
          exploitAvailable: true,
          severity: 'high'
        },
        {
          id: 'ZD-2025-003',
          name: 'Privilege Escalation through Cache Poisoning',
          cvss: 7.2,
          epss: 0.45,
          confidence: 76,
          category: 'Privilege Escalation',
          description: 'Cache poisoning vulnerability allowing role elevation',
          discovered: new Date(Date.now() - 7200000).toISOString(),
          status: 'new',
          affectedSystems: ['redis-cache', 'session-manager'],
          exploitAvailable: false,
          severity: 'high'
        }
      ]
      setVulnerabilities(mockVulns)
    } catch (error) {
      console.error('Failed to load vulnerabilities:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadAnomalies = async () => {
    try {
      const mockAnomalies: BehaviorAnomaly[] = [
        {
          id: 'AN-' + Date.now(),
          timestamp: new Date().toISOString(),
          anomalyType: 'Unusual Memory Access Pattern',
          confidence: 84,
          indicators: ['heap_spray_detected', 'rop_chain_pattern', 'shellcode_signature'],
          riskScore: 91
        },
        {
          id: 'AN-' + (Date.now() - 1000),
          timestamp: new Date(Date.now() - 60000).toISOString(),
          anomalyType: 'Abnormal Network Behavior',
          confidence: 72,
          indicators: ['dns_tunneling', 'c2_beaconing', 'data_exfiltration'],
          riskScore: 78
        }
      ]
      setAnomalies(mockAnomalies)
    } catch (error) {
      console.error('Failed to load anomalies:', error)
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/50'
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/50'
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50'
      case 'low': return 'bg-blue-500/20 text-blue-400 border-blue-500/50'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'new': return 'bg-purple-500/20 text-purple-400 border-purple-500/50'
      case 'analyzing': return 'bg-blue-500/20 text-blue-400 border-blue-500/50'
      case 'confirmed': return 'bg-red-500/20 text-red-400 border-red-500/50'
      case 'mitigated': return 'bg-green-500/20 text-green-400 border-green-500/50'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50'
    }
  }

  const filteredVulnerabilities = vulnerabilities.filter(vuln => {
    const matchesSearch = vuln.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         vuln.description.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesSeverity = selectedSeverity === 'all' || vuln.severity === selectedSeverity
    return matchesSearch && matchesSeverity
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
            Zero-Day Discovery
          </h1>
          <p className="text-gray-400 mt-2">AI-powered vulnerability detection and behavior analysis</p>
        </div>
        <Button className="bg-gradient-to-r from-purple-500 to-pink-600">
          <Download className="mr-2 h-4 w-4" />
          Export Report
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="bg-black/40 border-purple-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Total Discovered</p>
              <p className="text-3xl font-bold text-white mt-1">{vulnerabilities.length}</p>
            </div>
            <AlertTriangle className="h-8 w-8 text-purple-400" />
          </div>
        </Card>

        <Card className="bg-black/40 border-red-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Critical</p>
              <p className="text-3xl font-bold text-red-400 mt-1">
                {vulnerabilities.filter(v => v.severity === 'critical').length}
              </p>
            </div>
            <Shield className="h-8 w-8 text-red-400" />
          </div>
        </Card>

        <Card className="bg-black/40 border-orange-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">High Risk</p>
              <p className="text-3xl font-bold text-orange-400 mt-1">
                {vulnerabilities.filter(v => v.severity === 'high').length}
              </p>
            </div>
            <TrendingUp className="h-8 w-8 text-orange-400" />
          </div>
        </Card>

        <Card className="bg-black/40 border-green-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Avg Confidence</p>
              <p className="text-3xl font-bold text-green-400 mt-1">
                {Math.round(vulnerabilities.reduce((acc, v) => acc + v.confidence, 0) / vulnerabilities.length || 0)}%
              </p>
            </div>
            <TrendingUp className="h-8 w-8 text-green-400" />
          </div>
        </Card>
      </div>

      {/* Search and Filters */}
      <div className="flex gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            placeholder="Search vulnerabilities..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 bg-black/40 border-purple-500/20"
          />
        </div>
        <select
          value={selectedSeverity}
          onChange={(e) => setSelectedSeverity(e.target.value)}
          className="px-4 py-2 bg-black/40 border border-purple-500/20 rounded-lg text-white"
        >
          <option value="all">All Severities</option>
          <option value="critical">Critical</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="vulnerabilities" className="space-y-6">
        <TabsList className="bg-black/40 border border-purple-500/20">
          <TabsTrigger value="vulnerabilities">Vulnerabilities</TabsTrigger>
          <TabsTrigger value="anomalies">Live Anomalies</TabsTrigger>
          <TabsTrigger value="analysis">AI Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="vulnerabilities" className="space-y-4">
          {filteredVulnerabilities.map((vuln) => (
            <Card key={vuln.id} className="bg-black/40 border-purple-500/20 p-6 hover:border-purple-500/40 transition-all">
              <div className="space-y-4">
                {/* Header */}
                <div className="flex items-start justify-between">
                  <div className="space-y-2">
                    <div className="flex items-center gap-3">
                      <h3 className="text-xl font-semibold text-white">{vuln.name}</h3>
                      <Badge className={getSeverityColor(vuln.severity)}>
                        {vuln.severity.toUpperCase()}
                      </Badge>
                      <Badge className={getStatusColor(vuln.status)}>
                        {vuln.status}
                      </Badge>
                    </div>
                    <p className="text-gray-400 text-sm">{vuln.id}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-2xl font-bold text-white">{vuln.cvss}</p>
                    <p className="text-gray-400 text-sm">CVSS Score</p>
                  </div>
                </div>

                {/* Description */}
                <p className="text-gray-300">{vuln.description}</p>

                {/* Metrics */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-purple-500/10 rounded-lg p-3">
                    <p className="text-gray-400 text-xs mb-1">EPSS Score</p>
                    <p className="text-lg font-semibold text-purple-400">{(vuln.epss * 100).toFixed(1)}%</p>
                  </div>
                  <div className="bg-blue-500/10 rounded-lg p-3">
                    <p className="text-gray-400 text-xs mb-1">AI Confidence</p>
                    <p className="text-lg font-semibold text-blue-400">{vuln.confidence}%</p>
                  </div>
                  <div className="bg-green-500/10 rounded-lg p-3">
                    <p className="text-gray-400 text-xs mb-1">Exploit Available</p>
                    <p className="text-lg font-semibold text-green-400">{vuln.exploitAvailable ? 'Yes' : 'No'}</p>
                  </div>
                </div>

                {/* Affected Systems */}
                <div>
                  <p className="text-gray-400 text-sm mb-2">Affected Systems:</p>
                  <div className="flex flex-wrap gap-2">
                    {vuln.affectedSystems.map((system) => (
                      <Badge key={system} className="bg-gray-500/20 text-gray-300 border-gray-500/50">
                        {system}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                  <Button size="sm" className="bg-purple-500/20 hover:bg-purple-500/30 text-purple-400">
                    Analyze
                  </Button>
                  <Button size="sm" className="bg-blue-500/20 hover:bg-blue-500/30 text-blue-400">
                    Generate Mitigation
                  </Button>
                  <Button size="sm" className="bg-green-500/20 hover:bg-green-500/30 text-green-400">
                    Export Details
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </TabsContent>

        <TabsContent value="anomalies" className="space-y-4">
          <div className="bg-black/40 border border-purple-500/20 rounded-lg p-4 mb-4">
            <p className="text-gray-400 text-sm">
              Live behavioral anomaly detection using AI. Updates every 30 seconds.
            </p>
          </div>
          {anomalies.map((anomaly) => (
            <Card key={anomaly.id} className="bg-black/40 border-orange-500/20 p-6">
              <div className="space-y-3">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-lg font-semibold text-white">{anomaly.anomalyType}</h3>
                    <p className="text-gray-400 text-sm">
                      {new Date(anomaly.timestamp).toLocaleString()}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xl font-bold text-orange-400">{anomaly.riskScore}</p>
                    <p className="text-gray-400 text-sm">Risk Score</p>
                  </div>
                </div>

                <div>
                  <p className="text-gray-400 text-sm mb-2">Indicators:</p>
                  <div className="flex flex-wrap gap-2">
                    {anomaly.indicators.map((indicator, idx) => (
                      <Badge key={idx} className="bg-red-500/20 text-red-400 border-red-500/50">
                        {indicator}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <div className="flex-1">
                    <p className="text-gray-400 text-sm mb-1">Confidence</p>
                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-orange-500 to-red-500"
                        style={{ width: `${anomaly.confidence}%` }}
                      />
                    </div>
                  </div>
                  <p className="text-white font-semibold">{anomaly.confidence}%</p>
                </div>
              </div>
            </Card>
          ))}
        </TabsContent>

        <TabsContent value="analysis" className="space-y-4">
          <Card className="bg-black/40 border-purple-500/20 p-6">
            <h3 className="text-xl font-semibold text-white mb-4">AI Analysis Pipeline</h3>
            <div className="space-y-4">
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center">
                  <div className="w-3 h-3 rounded-full bg-green-500" />
                </div>
                <div>
                  <h4 className="text-white font-semibold">Behavioral Analysis</h4>
                  <p className="text-gray-400 text-sm">Monitoring system calls, memory access, and network patterns</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center">
                  <div className="w-3 h-3 rounded-full bg-green-500" />
                </div>
                <div>
                  <h4 className="text-white font-semibold">Pattern Recognition</h4>
                  <p className="text-gray-400 text-sm">Identifying exploit patterns and zero-day signatures</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center">
                  <div className="w-3 h-3 rounded-full bg-blue-500 animate-pulse" />
                </div>
                <div>
                  <h4 className="text-white font-semibold">Threat Correlation</h4>
                  <p className="text-gray-400 text-sm">Cross-referencing with CVE databases and threat intelligence</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center">
                  <div className="w-3 h-3 rounded-full bg-purple-500" />
                </div>
                <div>
                  <h4 className="text-white font-semibold">Risk Assessment</h4>
                  <p className="text-gray-400 text-sm">Calculating CVSS, EPSS, and custom risk scores</p>
                </div>
              </div>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
