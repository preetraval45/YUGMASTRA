"use client"

import { useState, useEffect } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { AlertTriangle, Shield, Zap, CheckCircle, Clock, PlayCircle, StopCircle, Download, Terminal } from 'lucide-react'

interface Incident {
  id: string
  title: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  status: 'new' | 'analyzing' | 'containing' | 'eradicating' | 'recovering' | 'resolved'
  detectedAt: string
  affectedAssets: string[]
  indicators: string[]
  aiConfidence: number
}

interface PlaybookStep {
  id: string
  action: string
  status: 'pending' | 'executing' | 'completed' | 'failed'
  result?: string
  timestamp?: string
}

export default function IncidentResponsePage() {
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null)
  const [playbook, setPlaybook] = useState<PlaybookStep[]>([])
  const [autoMode, setAutoMode] = useState(false)

  useEffect(() => {
    // Load mock incidents
    const mockIncidents: Incident[] = [
      {
        id: 'INC-2025-001',
        title: 'Ransomware Detected - Encryption Activity',
        severity: 'critical',
        status: 'analyzing',
        detectedAt: new Date().toISOString(),
        affectedAssets: ['WKS-001', 'WKS-002', 'FILE-SRV-01'],
        indicators: ['suspicious_encryption', 'mass_file_modification', 'c2_beaconing'],
        aiConfidence: 94
      },
      {
        id: 'INC-2025-002',
        title: 'Lateral Movement - Pass-the-Hash Attack',
        severity: 'high',
        status: 'containing',
        detectedAt: new Date(Date.now() - 3600000).toISOString(),
        affectedAssets: ['DC-01', 'WKS-005'],
        indicators: ['abnormal_auth', 'ntlm_relay', 'privilege_escalation'],
        aiConfidence: 87
      },
      {
        id: 'INC-2025-003',
        title: 'Data Exfiltration Attempt via DNS Tunneling',
        severity: 'high',
        status: 'new',
        detectedAt: new Date(Date.now() - 7200000).toISOString(),
        affectedAssets: ['WEB-SRV-01'],
        indicators: ['dns_tunneling', 'large_dns_queries', 'data_exfiltration'],
        aiConfidence: 91
      }
    ]
    setIncidents(mockIncidents)
  }, [])

  const generatePlaybook = async (incident: Incident) => {
    setSelectedIncident(incident)

    // AI-generated playbook steps
    const steps: PlaybookStep[] = [
      { id: '1', action: 'Isolate affected hosts from network', status: 'pending' },
      { id: '2', action: 'Suspend user accounts with suspicious activity', status: 'pending' },
      { id: '3', action: 'Collect memory dumps from infected systems', status: 'pending' },
      { id: '4', action: 'Analyze malware samples and IOCs', status: 'pending' },
      { id: '5', action: 'Block malicious IPs at firewall', status: 'pending' },
      { id: '6', action: 'Deploy detection rules for similar attacks', status: 'pending' },
      { id: '7', action: 'Restore systems from clean backups', status: 'pending' },
      { id: '8', action: 'Conduct post-incident review', status: 'pending' }
    ]

    setPlaybook(steps)
  }

  const executePlaybook = async () => {
    if (!autoMode) return

    for (let i = 0; i < playbook.length; i++) {
      // Update step to executing
      setPlaybook(prev => prev.map((step, idx) =>
        idx === i ? { ...step, status: 'executing' } : step
      ))

      // Simulate execution
      await new Promise(resolve => setTimeout(resolve, 2000))

      // Mark as completed
      setPlaybook(prev => prev.map((step, idx) =>
        idx === i ? {
          ...step,
          status: 'completed',
          result: 'Success',
          timestamp: new Date().toISOString()
        } : step
      ))
    }

    // Update incident status
    if (selectedIncident) {
      setIncidents(prev => prev.map(inc =>
        inc.id === selectedIncident.id
          ? { ...inc, status: 'resolved' }
          : inc
      ))
    }
  }

  useEffect(() => {
    if (autoMode && playbook.length > 0) {
      executePlaybook()
    }
  }, [autoMode])

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
      case 'containing': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50'
      case 'eradicating': return 'bg-orange-500/20 text-orange-400 border-orange-500/50'
      case 'recovering': return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/50'
      case 'resolved': return 'bg-green-500/20 text-green-400 border-green-500/50'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-red-400 to-orange-600 bg-clip-text text-transparent">
            Automated Incident Response
          </h1>
          <p className="text-gray-400 mt-2">AI-powered incident detection, analysis, and orchestrated response</p>
        </div>
        <div className="flex gap-3">
          <Button
            onClick={() => setAutoMode(!autoMode)}
            className={autoMode ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'}
          >
            {autoMode ? <StopCircle className="mr-2 h-4 w-4" /> : <PlayCircle className="mr-2 h-4 w-4" />}
            {autoMode ? 'Auto Mode ON' : 'Manual Mode'}
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="bg-black/40 border-red-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Active Incidents</p>
              <p className="text-3xl font-bold text-white mt-1">
                {incidents.filter(i => i.status !== 'resolved').length}
              </p>
            </div>
            <AlertTriangle className="h-8 w-8 text-red-400" />
          </div>
        </Card>

        <Card className="bg-black/40 border-orange-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Critical</p>
              <p className="text-3xl font-bold text-orange-400 mt-1">
                {incidents.filter(i => i.severity === 'critical').length}
              </p>
            </div>
            <Zap className="h-8 w-8 text-orange-400" />
          </div>
        </Card>

        <Card className="bg-black/40 border-green-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Resolved</p>
              <p className="text-3xl font-bold text-green-400 mt-1">
                {incidents.filter(i => i.status === 'resolved').length}
              </p>
            </div>
            <CheckCircle className="h-8 w-8 text-green-400" />
          </div>
        </Card>

        <Card className="bg-black/40 border-blue-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Avg Response Time</p>
              <p className="text-3xl font-bold text-blue-400 mt-1">4.2m</p>
            </div>
            <Clock className="h-8 w-8 text-blue-400" />
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Incidents List */}
        <div className="space-y-4">
          <h2 className="text-xl font-semibold text-white">Active Incidents</h2>
          {incidents.map((incident) => (
            <Card
              key={incident.id}
              className={`bg-black/40 border-red-500/20 p-6 cursor-pointer hover:border-red-500/40 transition-all ${
                selectedIncident?.id === incident.id ? 'border-red-500 ring-2 ring-red-500/20' : ''
              }`}
              onClick={() => generatePlaybook(incident)}
            >
              <div className="space-y-3">
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <h3 className="text-lg font-semibold text-white">{incident.title}</h3>
                    </div>
                    <p className="text-gray-400 text-sm">{incident.id}</p>
                  </div>
                  <div className="flex flex-col gap-2 items-end">
                    <Badge className={getSeverityColor(incident.severity)}>
                      {incident.severity.toUpperCase()}
                    </Badge>
                    <Badge className={getStatusColor(incident.status)}>
                      {incident.status}
                    </Badge>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-400">AI Confidence</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-blue-500 to-cyan-500"
                          style={{ width: `${incident.aiConfidence}%` }}
                        />
                      </div>
                      <span className="text-white font-semibold">{incident.aiConfidence}%</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-gray-400">Detected</p>
                    <p className="text-white">{new Date(incident.detectedAt).toLocaleTimeString()}</p>
                  </div>
                </div>

                <div>
                  <p className="text-gray-400 text-sm mb-1">Affected Assets ({incident.affectedAssets.length})</p>
                  <div className="flex flex-wrap gap-1">
                    {incident.affectedAssets.map((asset) => (
                      <Badge key={asset} className="bg-gray-500/20 text-gray-300 border-gray-500/50 text-xs">
                        {asset}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="text-gray-400 text-sm mb-1">Indicators</p>
                  <div className="flex flex-wrap gap-1">
                    {incident.indicators.map((indicator) => (
                      <Badge key={indicator} className="bg-red-500/20 text-red-400 border-red-500/50 text-xs">
                        {indicator}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>

        {/* Response Playbook */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-white">AI-Generated Response Playbook</h2>
            {playbook.length > 0 && (
              <Button size="sm" className="bg-blue-500 hover:bg-blue-600">
                <Download className="mr-2 h-4 w-4" />
                Export
              </Button>
            )}
          </div>

          {!selectedIncident ? (
            <Card className="bg-black/40 border-gray-500/20 p-12 text-center">
              <Shield className="h-12 w-12 text-gray-500 mx-auto mb-4" />
              <p className="text-gray-400">Select an incident to generate response playbook</p>
            </Card>
          ) : (
            <Card className="bg-black/40 border-blue-500/20 p-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between pb-4 border-b border-gray-700">
                  <div>
                    <h3 className="text-lg font-semibold text-white">Response Plan</h3>
                    <p className="text-gray-400 text-sm">for {selectedIncident.id}</p>
                  </div>
                  <Button
                    onClick={() => setAutoMode(true)}
                    disabled={autoMode}
                    className="bg-green-500 hover:bg-green-600"
                  >
                    <PlayCircle className="mr-2 h-4 w-4" />
                    Execute Playbook
                  </Button>
                </div>

                <div className="space-y-3 max-h-[600px] overflow-y-auto">
                  {playbook.map((step, index) => (
                    <div
                      key={step.id}
                      className={`p-4 rounded-lg border ${
                        step.status === 'completed'
                          ? 'bg-green-500/10 border-green-500/50'
                          : step.status === 'executing'
                          ? 'bg-blue-500/10 border-blue-500/50 animate-pulse'
                          : step.status === 'failed'
                          ? 'bg-red-500/10 border-red-500/50'
                          : 'bg-gray-500/10 border-gray-500/50'
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center">
                          <span className="text-blue-400 font-semibold text-sm">{index + 1}</span>
                        </div>
                        <div className="flex-1">
                          <div className="flex items-start justify-between">
                            <p className="text-white font-medium">{step.action}</p>
                            <div className="flex items-center gap-2">
                              {step.status === 'completed' && (
                                <CheckCircle className="h-5 w-5 text-green-400" />
                              )}
                              {step.status === 'executing' && (
                                <Terminal className="h-5 w-5 text-blue-400 animate-pulse" />
                              )}
                              {step.status === 'pending' && (
                                <Clock className="h-5 w-5 text-gray-400" />
                              )}
                            </div>
                          </div>
                          {step.result && (
                            <p className="text-green-400 text-sm mt-1">✓ {step.result}</p>
                          )}
                          {step.timestamp && (
                            <p className="text-gray-500 text-xs mt-1">
                              {new Date(step.timestamp).toLocaleTimeString()}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
