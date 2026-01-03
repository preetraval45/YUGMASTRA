"use client"

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { FileCode, Download, Sparkles, Copy, CheckCircle, AlertTriangle } from 'lucide-react'
import { toast } from '@/components/ui/toast'

interface GeneratedRule {
  id: string
  name: string
  description: string
  format: 'sigma' | 'splunk' | 'elastic' | 'suricata' | 'snort'
  severity: 'critical' | 'high' | 'medium' | 'low'
  rule: string
  tags: string[]
  mitre: string[]
  generated: string
}

export default function SIEMRulesPage() {
  const [threatDescription, setThreatDescription] = useState('')
  const [selectedFormat, setSelectedFormat] = useState<string>('sigma')
  const [loading, setLoading] = useState(false)
  const [generatedRules, setGeneratedRules] = useState<GeneratedRule[]>([])
  const [copiedId, setCopiedId] = useState<string | null>(null)

  const handleGenerateRule = async () => {
    if (!threatDescription.trim()) {
      toast({
        title: "Error",
        description: "Please describe the threat you want to detect",
        variant: "destructive"
      })
      return
    }

    setLoading(true)
    try {
      // Mock generation - replace with actual API call
      await new Promise(resolve => setTimeout(resolve, 2000))

      const mockRule: GeneratedRule = {
        id: 'RULE-' + Date.now(),
        name: `Detection for ${threatDescription.substring(0, 50)}`,
        description: threatDescription,
        format: selectedFormat as any,
        severity: 'high',
        rule: generateMockRule(selectedFormat, threatDescription),
        tags: ['detection', 'ai-generated', selectedFormat],
        mitre: ['T1059', 'T1071', 'T1055'],
        generated: new Date().toISOString()
      }

      setGeneratedRules([mockRule, ...generatedRules])
      setThreatDescription('')

      toast({
        title: "Success",
        description: "SIEM rule generated successfully",
      })
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to generate rule",
        variant: "destructive"
      })
    } finally {
      setLoading(false)
    }
  }

  const generateMockRule = (format: string, description: string): string => {
    switch (format) {
      case 'sigma':
        return `title: ${description}
id: ${Math.random().toString(36).substring(7)}
status: experimental
description: AI-generated detection rule for ${description}
author: YUGMASTRA AI
date: ${new Date().toISOString().split('T')[0]}
tags:
    - attack.execution
    - attack.t1059
logsource:
    category: process_creation
    product: windows
detection:
    selection:
        EventID: 1
        CommandLine|contains:
            - 'powershell'
            - 'cmd.exe'
            - 'wscript'
    condition: selection
falsepositives:
    - Legitimate administrative activity
level: high`

      case 'splunk':
        return `index=* sourcetype=*
| eval threat_desc="${description}"
| search (CommandLine="*powershell*" OR CommandLine="*cmd.exe*")
| stats count by host, user, CommandLine, _time
| where count > 5
| eval severity="high"
| eval mitre_technique="T1059"
| table _time, host, user, CommandLine, severity, mitre_technique`

      case 'elastic':
        return `{
  "query": {
    "bool": {
      "must": [
        {
          "query_string": {
            "query": "process.command_line:(*powershell* OR *cmd.exe*)",
            "analyze_wildcard": true
          }
        }
      ],
      "filter": [
        {
          "range": {
            "@timestamp": {
              "gte": "now-1h"
            }
          }
        }
      ]
    }
  },
  "aggs": {
    "hosts": {
      "terms": {
        "field": "host.name",
        "size": 100
      }
    }
  }
}`

      case 'suricata':
        return `alert tcp any any -> any any (msg:"${description}";
  flow:established,to_server;
  content:"|${Math.random().toString(16).substring(2, 10)}|";
  nocase;
  classtype:trojan-activity;
  sid:${Math.floor(Math.random() * 1000000)};
  rev:1;
  metadata:created_at ${new Date().toISOString().split('T')[0]};)`

      case 'snort':
        return `alert tcp $EXTERNAL_NET any -> $HOME_NET any (
  msg:"${description}";
  flow:established,to_server;
  content:"${description.substring(0, 20)}";
  nocase;
  classtype:attempted-recon;
  sid:${Math.floor(Math.random() * 1000000)};
  rev:1;
  metadata:policy balanced-ips drop, policy security-ips drop;
)`

      default:
        return '# Rule generation failed'
    }
  }

  const copyToClipboard = (rule: GeneratedRule) => {
    navigator.clipboard.writeText(rule.rule)
    setCopiedId(rule.id)
    setTimeout(() => setCopiedId(null), 2000)

    toast({
      title: "Copied!",
      description: "Rule copied to clipboard",
    })
  }

  const downloadRule = (rule: GeneratedRule) => {
    const fileExtensions: Record<string, string> = {
      'sigma': 'yml',
      'splunk': 'spl',
      'elastic': 'json',
      'suricata': 'rules',
      'snort': 'rules'
    }

    const blob = new Blob([rule.rule], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${rule.name.replace(/\s+/g, '_')}.${fileExtensions[rule.format]}`
    a.click()
    URL.revokeObjectURL(url)

    toast({
      title: "Downloaded!",
      description: `Rule saved as ${a.download}`,
    })
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent">
          SIEM Rule Generator
        </h1>
        <p className="text-gray-400 mt-2">AI-powered detection rule generation for multiple SIEM platforms</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="bg-black/40 border-cyan-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Total Rules</p>
              <p className="text-3xl font-bold text-white mt-1">{generatedRules.length}</p>
            </div>
            <FileCode className="h-8 w-8 text-cyan-400" />
          </div>
        </Card>

        <Card className="bg-black/40 border-blue-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Formats Supported</p>
              <p className="text-3xl font-bold text-blue-400 mt-1">5</p>
            </div>
            <Sparkles className="h-8 w-8 text-blue-400" />
          </div>
        </Card>

        <Card className="bg-black/40 border-green-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Success Rate</p>
              <p className="text-3xl font-bold text-green-400 mt-1">98%</p>
            </div>
            <CheckCircle className="h-8 w-8 text-green-400" />
          </div>
        </Card>

        <Card className="bg-black/40 border-purple-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">MITRE Coverage</p>
              <p className="text-3xl font-bold text-purple-400 mt-1">180+</p>
            </div>
            <AlertTriangle className="h-8 w-8 text-purple-400" />
          </div>
        </Card>
      </div>

      {/* Generator */}
      <Card className="bg-black/40 border-cyan-500/20 p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Generate Detection Rule</h2>
        <div className="space-y-4">
          <div>
            <label className="text-gray-400 text-sm mb-2 block">Threat Description</label>
            <Input
              placeholder="Describe the threat you want to detect (e.g., 'PowerShell download cradle executing encoded commands')"
              value={threatDescription}
              onChange={(e) => setThreatDescription(e.target.value)}
              className="bg-black/40 border-cyan-500/20"
            />
          </div>

          <div>
            <label className="text-gray-400 text-sm mb-2 block">Output Format</label>
            <div className="grid grid-cols-5 gap-2">
              {['sigma', 'splunk', 'elastic', 'suricata', 'snort'].map((format) => (
                <button
                  key={format}
                  onClick={() => setSelectedFormat(format)}
                  className={`p-3 rounded-lg border transition-all ${
                    selectedFormat === format
                      ? 'bg-cyan-500/20 border-cyan-500 text-cyan-400'
                      : 'bg-black/40 border-gray-500/20 text-gray-400 hover:border-cyan-500/50'
                  }`}
                >
                  {format.charAt(0).toUpperCase() + format.slice(1)}
                </button>
              ))}
            </div>
          </div>

          <Button
            onClick={handleGenerateRule}
            disabled={loading}
            className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700"
          >
            {loading ? (
              <>
                <Sparkles className="mr-2 h-4 w-4 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-4 w-4" />
                Generate Rule
              </>
            )}
          </Button>
        </div>
      </Card>

      {/* Generated Rules */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-white">Generated Rules ({generatedRules.length})</h2>

        {generatedRules.length === 0 ? (
          <Card className="bg-black/40 border-cyan-500/20 p-12 text-center">
            <FileCode className="h-12 w-12 text-gray-500 mx-auto mb-4" />
            <p className="text-gray-400">No rules generated yet. Create your first detection rule above!</p>
          </Card>
        ) : (
          generatedRules.map((rule) => (
            <Card key={rule.id} className="bg-black/40 border-cyan-500/20 p-6">
              <div className="space-y-4">
                {/* Header */}
                <div className="flex items-start justify-between">
                  <div className="space-y-2">
                    <div className="flex items-center gap-3">
                      <h3 className="text-lg font-semibold text-white">{rule.name}</h3>
                      <Badge className={getSeverityColor(rule.severity)}>
                        {rule.severity.toUpperCase()}
                      </Badge>
                      <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500/50">
                        {rule.format.toUpperCase()}
                      </Badge>
                    </div>
                    <p className="text-gray-400 text-sm">{rule.description}</p>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => copyToClipboard(rule)}
                      className="bg-black/40 border-cyan-500/20 hover:bg-cyan-500/10"
                    >
                      {copiedId === rule.id ? (
                        <>
                          <CheckCircle className="h-4 w-4 text-green-400" />
                        </>
                      ) : (
                        <>
                          <Copy className="h-4 w-4" />
                        </>
                      )}
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => downloadRule(rule)}
                      className="bg-black/40 border-cyan-500/20 hover:bg-cyan-500/10"
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                </div>

                {/* MITRE ATT&CK Tags */}
                <div>
                  <p className="text-gray-400 text-sm mb-2">MITRE ATT&CK Techniques:</p>
                  <div className="flex flex-wrap gap-2">
                    {rule.mitre.map((technique) => (
                      <Badge key={technique} className="bg-purple-500/20 text-purple-400 border-purple-500/50">
                        {technique}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* Rule Code */}
                <div>
                  <p className="text-gray-400 text-sm mb-2">Rule:</p>
                  <pre className="bg-black/60 border border-gray-700 rounded-lg p-4 overflow-x-auto">
                    <code className="text-sm text-gray-300">{rule.rule}</code>
                  </pre>
                </div>

                {/* Tags */}
                <div className="flex flex-wrap gap-2">
                  {rule.tags.map((tag, idx) => (
                    <Badge key={idx} className="bg-gray-500/20 text-gray-300 border-gray-500/50">
                      {tag}
                    </Badge>
                  ))}
                </div>

                {/* Timestamp */}
                <p className="text-gray-500 text-xs">
                  Generated: {new Date(rule.generated).toLocaleString()}
                </p>
              </div>
            </Card>
          ))
        )}
      </div>

      {/* Quick Examples */}
      <Card className="bg-black/40 border-cyan-500/20 p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Quick Examples</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <button
            onClick={() => setThreatDescription('PowerShell executing base64 encoded commands')}
            className="p-4 bg-black/40 border border-gray-700 rounded-lg text-left hover:border-cyan-500/50 transition-all"
          >
            <p className="text-white font-semibold mb-1">PowerShell Obfuscation</p>
            <p className="text-gray-400 text-sm">Detect encoded PowerShell execution</p>
          </button>
          <button
            onClick={() => setThreatDescription('Suspicious network connections to known C2 servers')}
            className="p-4 bg-black/40 border border-gray-700 rounded-lg text-left hover:border-cyan-500/50 transition-all"
          >
            <p className="text-white font-semibold mb-1">C2 Communication</p>
            <p className="text-gray-400 text-sm">Detect command and control traffic</p>
          </button>
          <button
            onClick={() => setThreatDescription('Credential dumping using LSASS memory access')}
            className="p-4 bg-black/40 border border-gray-700 rounded-lg text-left hover:border-cyan-500/50 transition-all"
          >
            <p className="text-white font-semibold mb-1">Credential Theft</p>
            <p className="text-gray-400 text-sm">Detect LSASS memory dumping</p>
          </button>
          <button
            onClick={() => setThreatDescription('Lateral movement using PsExec or WMI')}
            className="p-4 bg-black/40 border border-gray-700 rounded-lg text-left hover:border-cyan-500/50 transition-all"
          >
            <p className="text-white font-semibold mb-1">Lateral Movement</p>
            <p className="text-gray-400 text-sm">Detect PsExec/WMI execution</p>
          </button>
        </div>
      </Card>
    </div>
  )
}
