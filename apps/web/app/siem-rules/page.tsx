"use client"

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { FileCode, Download, Sparkles, Copy, CheckCircle, AlertTriangle, BookOpen, Lightbulb, Code2, Shield } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'

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

interface RuleExample {
  title: string
  format: string
  description: string
  technicalDetails: string
  rule: string
  mitre: string[]
  useCase: string
  detectionLogic: string
  falsePositives: string[]
}

const RULE_EXAMPLES: RuleExample[] = [
  {
    title: 'PowerShell Download Cradle Detection',
    format: 'sigma',
    description: 'Detects PowerShell executing download cradles commonly used by attackers to download and execute malicious payloads.',
    technicalDetails: 'This Sigma rule monitors Windows process creation events (Sysmon Event ID 1) for PowerShell commands containing patterns like "IEX", "Invoke-Expression", "DownloadString", and "DownloadFile". These are common techniques used in fileless malware and living-off-the-land attacks.',
    rule: `title: PowerShell Download Cradle Detection
id: 85b0b087-eddf-4a2b-b033-af5fa235bbb7
status: stable
description: Detects PowerShell download cradle patterns used for malware delivery
author: YUGMASTRA Security Team
date: 2025-01-03
modified: 2025-01-03
tags:
    - attack.execution
    - attack.t1059.001
    - attack.defense_evasion
    - attack.t1027
logsource:
    category: process_creation
    product: windows
detection:
    selection_process:
        Image|endswith:
            - '\\\\powershell.exe'
            - '\\\\pwsh.exe'
    selection_commands:
        CommandLine|contains:
            - 'IEX'
            - 'Invoke-Expression'
            - 'DownloadString'
            - 'DownloadFile'
            - 'Net.WebClient'
            - 'Start-BitsTransfer'
            - 'Invoke-WebRequest'
            - 'curl'
            - 'wget'
    condition: all of selection_*
falsepositives:
    - Legitimate software deployment scripts
    - Administrative PowerShell scripts
    - Windows Update processes
level: high
fields:
    - CommandLine
    - User
    - Image
    - ParentImage`,
    mitre: ['T1059.001', 'T1027', 'T1105'],
    useCase: 'Detecting initial access attempts where attackers use PowerShell to download second-stage payloads from remote servers.',
    detectionLogic: 'Correlates PowerShell execution with network download commands. Matches specific cmdlets and .NET methods commonly used in attack chains.',
    falsePositives: ['Software deployment tools', 'System administration scripts', 'Automated update mechanisms']
  },
  {
    title: 'Credential Dumping via LSASS',
    format: 'splunk',
    description: 'Detects attempts to dump credentials from LSASS (Local Security Authority Subsystem Service) memory, a critical Windows process.',
    technicalDetails: 'This Splunk query identifies suspicious access to lsass.exe process memory. Attackers use tools like Mimikatz, ProcDump, or Comsvcs.dll to extract plaintext passwords, NTLM hashes, and Kerberos tickets from LSASS memory.',
    rule: `index=windows sourcetype="WinEventLog:Security" OR sourcetype="WinEventLog:Sysmon"
| eval threat_name="LSASS Credential Dumping"
| search (
    (EventCode=10 TargetImage="*lsass.exe" (GrantedAccess=0x1010 OR GrantedAccess=0x1410 OR GrantedAccess=0x1438))
    OR (EventCode=4656 ObjectName="*lsass.exe" AccessMask IN ("0x1010", "0x1410", "0x1438", "0x1fffff"))
    OR (CommandLine="*lsass*" AND (CommandLine="*procdump*" OR CommandLine="*comsvcs.dll*" OR CommandLine="*MiniDump*"))
    OR (Image="*\\\\mimikatz.exe" OR CommandLine="*sekurlsa::*")
)
| eval severity="critical"
| eval mitre_technique="T1003.001"
| eval mitre_tactic="Credential Access"
| stats count min(_time) as first_seen max(_time) as last_seen values(User) as users values(Image) as processes by host, CommandLine
| eval duration=last_seen-first_seen
| where count > 0
| eval alert_priority=case(
    count > 5, "IMMEDIATE",
    count > 2, "HIGH",
    1=1, "MEDIUM"
)
| table _time, host, users, processes, CommandLine, count, duration, alert_priority, severity, mitre_technique
| sort -_time`,
    mitre: ['T1003.001', 'T1003.002', 'T1558.003'],
    useCase: 'Critical detection for post-exploitation credential theft. Often used in APT campaigns and ransomware attacks for lateral movement.',
    detectionLogic: 'Monitors Sysmon Event ID 10 (ProcessAccess) for LSASS access, Security Event 4656 for object access, and command-line execution of dumping tools. Correlates multiple indicators for higher confidence.',
    falsePositives: ['Legitimate security tools', 'Antivirus scanners', 'EDR solutions', 'Process monitoring software']
  },
  {
    title: 'SQL Injection Attack Detection',
    format: 'elastic',
    description: 'Detects SQL injection attempts in web application logs by identifying malicious SQL syntax patterns in HTTP requests.',
    technicalDetails: 'This Elasticsearch query analyzes HTTP request parameters for SQL injection indicators including UNION SELECT, OR 1=1, DROP TABLE, and encoded variants. Uses regex patterns and field analysis to identify suspicious payloads.',
    rule: `{
  "query": {
    "bool": {
      "must": [
        {
          "query_string": {
            "query": "(http.request.body.content:(*UNION*SELECT*) OR http.request.body.content:(*OR*1=1*) OR http.request.body.content:(*DROP*TABLE*) OR http.request.body.content:(*;*xp_cmdshell*) OR http.request.body.content:(*EXEC*sp_*) OR url.query:(*'*OR*'*=*'*) OR url.query:(*--*) OR url.query:(*;*--*) OR url.query:(*%27*OR*%27*) OR url.query:(*1'*ORDER*BY*) OR url.query:(*UNION*ALL*SELECT*NULL*))",
            "analyze_wildcard": true,
            "default_field": "*"
          }
        }
      ],
      "filter": [
        {
          "range": {
            "@timestamp": {
              "gte": "now-15m"
            }
          }
        }
      ],
      "should": [
        {
          "match": {
            "http.response.status_code": {
              "query": "500 400",
              "operator": "or"
            }
          }
        }
      ]
    }
  },
  "aggs": {
    "attack_sources": {
      "terms": {
        "field": "source.ip",
        "size": 100
      },
      "aggs": {
        "targeted_endpoints": {
          "terms": {
            "field": "url.path",
            "size": 50
          }
        },
        "attack_volume": {
          "value_count": {
            "field": "source.ip"
          }
        }
      }
    },
    "time_series": {
      "date_histogram": {
        "field": "@timestamp",
        "fixed_interval": "1m"
      }
    }
  },
  "size": 100,
  "sort": [
    {
      "@timestamp": {
        "order": "desc"
      }
    }
  ]
}`,
    mitre: ['T1190', 'T1059.007'],
    useCase: 'Web application security monitoring. Protects against SQL injection attacks that could lead to data breaches, authentication bypass, or database compromise.',
    detectionLogic: 'Pattern matching on HTTP parameters for SQL syntax. Correlates with HTTP error codes (500, 400) which often indicate successful SQL injection attempts. Aggregates by source IP to identify attack patterns.',
    falsePositives: ['Legitimate database management tools', 'Security scanning tools', 'Penetration testing activities', 'Web application firewalls testing']
  },
  {
    title: 'C2 Beaconing Detection',
    format: 'suricata',
    description: 'Network-based detection of Command and Control (C2) beaconing patterns indicating compromised systems communicating with attacker infrastructure.',
    technicalDetails: 'This Suricata rule identifies periodic network connections to external IPs with specific timing patterns, DNS queries to known C2 domains, and HTTP User-Agent strings associated with common malware families like Cobalt Strike, Metasploit, and Empire.',
    rule: `# Cobalt Strike HTTP Beacon Detection
alert http $HOME_NET any -> $EXTERNAL_NET any (
  msg:"YUGMASTRA - Cobalt Strike HTTP Beacon Detected";
  flow:established,to_server;
  content:"User-Agent|3a 20|Mozilla/";
  http_header;
  content:!"Accept"; http_header;
  content:!"Referer"; http_header;
  pcre:"/^(GET|POST)\\s+\\/[a-z]{3,4}\\s+HTTP\\/1\\.1/i";
  threshold:type both, track by_src, count 3, seconds 60;
  classtype:trojan-activity;
  sid:2025001;
  rev:1;
  metadata:created_at 2025_01_03, updated_at 2025_01_03, mitre_technique T1071.001, severity critical;
  reference:url,attack.mitre.org/techniques/T1071/001;
)

# DNS Tunneling for C2 Communication
alert dns $HOME_NET any -> any 53 (
  msg:"YUGMASTRA - Potential DNS Tunneling for C2";
  dns_query;
  content:"|00 00 10 00 01|";
  pcre:"/^[a-f0-9]{32,}\\..*$/i";
  byte_test:2,>,128,0,relative;
  threshold:type threshold, track by_src, count 10, seconds 60;
  classtype:policy-violation;
  sid:2025002;
  rev:1;
  metadata:created_at 2025_01_03, mitre_technique T1071.004, severity high;
)

# HTTPS Anomalous SNI (Server Name Indication)
alert tls $HOME_NET any -> $EXTERNAL_NET 443 (
  msg:"YUGMASTRA - Suspicious TLS SNI for C2";
  flow:established,to_server;
  tls_sni;
  content:!"cloudflare"; nocase;
  content:!"microsoft"; nocase;
  content:!"google"; nocase;
  pcre:"/^[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}$/";
  threshold:type both, track by_src, count 5, seconds 300;
  classtype:trojan-activity;
  sid:2025003;
  rev:1;
  metadata:created_at 2025_01_03, mitre_technique T1573, severity high;
)`,
    mitre: ['T1071.001', 'T1071.004', 'T1573', 'T1568.002'],
    useCase: 'Network perimeter defense and internal network monitoring. Essential for detecting compromised hosts that have established C2 channels with external attackers.',
    detectionLogic: 'Analyzes HTTP headers for malware-specific patterns, monitors DNS query lengths and entropy for tunneling, and identifies TLS connections to IP addresses instead of domains. Uses threshold detection to reduce false positives.',
    falsePositives: ['CDN services', 'Cloud infrastructure', 'Legitimate automation tools', 'Mobile applications', 'IoT devices']
  }
]

export default function SIEMRulesPage() {
  const { toast } = useToast()
  const [activeTab, setActiveTab] = useState('generate')
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

  const copyToClipboard = (text: string, id?: string) => {
    navigator.clipboard.writeText(text)
    if (id) {
      setCopiedId(id)
      setTimeout(() => setCopiedId(null), 2000)
    }

    toast({
      title: "Copied!",
      description: "Content copied to clipboard",
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
    <div className="min-h-screen bg-background p-8 pt-32">
      {/* Header */}
      <div className="mb-8">
        <div className="mb-4">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent">
            SIEM Rule Generator
          </h1>
          <p className="text-muted-foreground mt-2">AI-powered detection rule generation for multiple SIEM platforms</p>
        </div>

        {/* Description Banner */}
        <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-4 flex items-start gap-3">
          <FileCode className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              <strong className="text-foreground">What this page does:</strong> This AI-powered SIEM rule generator creates detection rules for 5 major security platforms: Sigma (universal format), Splunk SPL, Elasticsearch Query DSL, Suricata, and Snort. Simply describe a threat in plain English and the AI generates optimized detection logic with proper syntax, MITRE ATT&CK mappings, and false positive considerations. The educational section teaches you about each SIEM format, detection engineering best practices, and includes real-world examples like PowerShell download cradle detection, LSASS credential dumping, SQL injection patterns, and C2 beaconing signatures. Perfect for SOC analysts and security engineers building their detection capabilities.
            </p>
          </div>
        </div>
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

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3 h-auto p-1 bg-muted">
          <TabsTrigger value="generate" className="data-[state=active]:bg-primary">
            <Code2 className="mr-2 h-4 w-4" />
            Generate
          </TabsTrigger>
          <TabsTrigger value="learn" className="data-[state=active]:bg-primary">
            <BookOpen className="mr-2 h-4 w-4" />
            Learn
          </TabsTrigger>
          <TabsTrigger value="examples" className="data-[state=active]:bg-primary">
            <Lightbulb className="mr-2 h-4 w-4" />
            Examples
          </TabsTrigger>
        </TabsList>

        {/* Generate Tab */}
        <TabsContent value="generate" className="space-y-6">
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
                          onClick={() => copyToClipboard(rule.rule, rule.id)}
                          className="bg-black/40 border-cyan-500/20 hover:bg-cyan-500/10"
                        >
                          {copiedId === rule.id ? (
                            <CheckCircle className="h-4 w-4 text-green-400" />
                          ) : (
                            <Copy className="h-4 w-4" />
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
        </TabsContent>

        {/* Learn Tab */}
        <TabsContent value="learn" className="space-y-6">
          <Card className="bg-black/40 border-cyan-500/20 p-8">
            <div className="space-y-8">
              <div>
                <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                  <Shield className="h-7 w-7 text-cyan-400" />
                  Detection Engineering: Building SIEM Rules
                </h2>
                <p className="text-gray-300 text-lg leading-relaxed">
                  Detection engineering is the practice of creating rules and queries that identify malicious activity in your security logs and network traffic.
                  Effective detection rules are the foundation of any Security Operations Center (SOC) and are critical for identifying threats before they cause damage.
                </p>
              </div>

              <div className="border-t border-gray-700 pt-8">
                <h3 className="text-xl font-semibold text-white mb-4">Understanding SIEM Rule Formats</h3>
                <div className="space-y-6">
                  <div className="bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/30 rounded-lg p-6">
                    <h4 className="text-lg font-semibold text-cyan-400 mb-3 flex items-center gap-2">
                      <FileCode className="h-5 w-5" />
                      Sigma Rules
                    </h4>
                    <p className="text-gray-300 mb-3">
                      Sigma is a generic signature format for SIEM systems, similar to how Snort/Suricata rules work for network IDS.
                      Sigma rules are written in YAML and can be converted to multiple SIEM query languages.
                    </p>
                    <div className="bg-black/40 rounded-lg p-4 border border-gray-700">
                      <p className="text-sm text-gray-400 mb-2"><strong>Use Cases:</strong></p>
                      <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
                        <li>Platform-agnostic detection logic</li>
                        <li>Sharing threat intelligence across different SIEM platforms</li>
                        <li>Windows event log analysis (Sysmon, Security, PowerShell logs)</li>
                        <li>Process creation, network connections, file access monitoring</li>
                      </ul>
                    </div>
                    <div className="mt-3 p-3 bg-blue-500/10 border border-blue-500/30 rounded">
                      <p className="text-sm text-blue-300">
                        <strong>Technical Detail:</strong> Sigma rules use a detection section with selection criteria and conditions.
                        They support field modifiers like |contains, |endswith, |startswith for flexible matching.
                      </p>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-orange-500/10 to-red-500/10 border border-orange-500/30 rounded-lg p-6">
                    <h4 className="text-lg font-semibold text-orange-400 mb-3 flex items-center gap-2">
                      <FileCode className="h-5 w-5" />
                      Splunk SPL (Search Processing Language)
                    </h4>
                    <p className="text-gray-300 mb-3">
                      SPL is Splunk's powerful query language for searching, analyzing, and visualizing machine data.
                      It uses a pipeline approach where data flows through commands connected by pipes (|).
                    </p>
                    <div className="bg-black/40 rounded-lg p-4 border border-gray-700">
                      <p className="text-sm text-gray-400 mb-2"><strong>Key Commands:</strong></p>
                      <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
                        <li><code className="text-cyan-400">search</code> - Filter events based on criteria</li>
                        <li><code className="text-cyan-400">stats</code> - Aggregate data (count, sum, avg, etc.)</li>
                        <li><code className="text-cyan-400">eval</code> - Create calculated fields</li>
                        <li><code className="text-cyan-400">where</code> - Filter results based on complex logic</li>
                        <li><code className="text-cyan-400">table</code> - Format output into table view</li>
                      </ul>
                    </div>
                    <div className="mt-3 p-3 bg-orange-500/10 border border-orange-500/30 rounded">
                      <p className="text-sm text-orange-300">
                        <strong>Technical Detail:</strong> SPL is left-to-right pipelined. Each command transforms the data stream.
                        Use subsearches with [search ...] for complex correlations across different data sources.
                      </p>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-lg p-6">
                    <h4 className="text-lg font-semibold text-green-400 mb-3 flex items-center gap-2">
                      <FileCode className="h-5 w-5" />
                      Elasticsearch Query DSL
                    </h4>
                    <p className="text-gray-300 mb-3">
                      Elasticsearch uses JSON-based Query DSL (Domain Specific Language) for searching and aggregating data.
                      It's the backend for Elastic SIEM and Kibana dashboards.
                    </p>
                    <div className="bg-black/40 rounded-lg p-4 border border-gray-700">
                      <p className="text-sm text-gray-400 mb-2"><strong>Query Types:</strong></p>
                      <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
                        <li><code className="text-cyan-400">bool</code> - Combine multiple queries (must, should, must_not, filter)</li>
                        <li><code className="text-cyan-400">match</code> - Full-text search with relevance scoring</li>
                        <li><code className="text-cyan-400">term</code> - Exact value matching</li>
                        <li><code className="text-cyan-400">range</code> - Numeric/date range queries</li>
                        <li><code className="text-cyan-400">query_string</code> - Lucene query syntax support</li>
                      </ul>
                    </div>
                    <div className="mt-3 p-3 bg-green-500/10 border border-green-500/30 rounded">
                      <p className="text-sm text-green-300">
                        <strong>Technical Detail:</strong> Use aggregations (aggs) for statistical analysis.
                        Terms aggregations group by field values, date_histogram creates time-series buckets, and metrics calculate statistics.
                      </p>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/30 rounded-lg p-6">
                    <h4 className="text-lg font-semibold text-purple-400 mb-3 flex items-center gap-2">
                      <FileCode className="h-5 w-5" />
                      Suricata & Snort Rules
                    </h4>
                    <p className="text-gray-300 mb-3">
                      Network-based Intrusion Detection System (IDS) rules that analyze packet data in real-time.
                      Suricata is the modern evolution of Snort with multi-threading and protocol analysis.
                    </p>
                    <div className="bg-black/40 rounded-lg p-4 border border-gray-700">
                      <p className="text-sm text-gray-400 mb-2"><strong>Rule Structure:</strong></p>
                      <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
                        <li><strong>Action:</strong> alert, drop, reject, pass</li>
                        <li><strong>Protocol:</strong> tcp, udp, icmp, http, dns, tls</li>
                        <li><strong>Source/Dest:</strong> IP addresses and ports</li>
                        <li><strong>Direction:</strong> → (to server) or &lt;&gt; (bidirectional)</li>
                        <li><strong>Options:</strong> msg, content, pcre, flow, threshold, etc.</li>
                      </ul>
                    </div>
                    <div className="mt-3 p-3 bg-purple-500/10 border border-purple-500/30 rounded">
                      <p className="text-sm text-purple-300">
                        <strong>Technical Detail:</strong> Use content matching with modifiers like nocase, offset, depth.
                        PCRE (Perl Compatible Regular Expressions) allows complex pattern matching. Flow tracking ensures stateful inspection.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="border-t border-gray-700 pt-8">
                <h3 className="text-xl font-semibold text-white mb-4">Detection Engineering Best Practices</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-5">
                    <h4 className="font-semibold text-blue-400 mb-3 flex items-center gap-2">
                      <CheckCircle className="h-5 w-5" />
                      Rule Quality
                    </h4>
                    <ul className="space-y-2 text-gray-300 text-sm">
                      <li className="flex gap-2">
                        <span className="text-blue-400">•</span>
                        <span><strong>Specificity:</strong> Target specific behaviors, not broad patterns that cause noise</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-blue-400">•</span>
                        <span><strong>Performance:</strong> Optimize queries to reduce SIEM load and search times</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-blue-400">•</span>
                        <span><strong>Maintainability:</strong> Document why the rule exists and what it detects</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-blue-400">•</span>
                        <span><strong>Testing:</strong> Validate rules against known good and bad data before production</span>
                      </li>
                    </ul>
                  </div>

                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-5">
                    <h4 className="font-semibold text-purple-400 mb-3 flex items-center gap-2">
                      <AlertTriangle className="h-5 w-5" />
                      Managing False Positives
                    </h4>
                    <ul className="space-y-2 text-gray-300 text-sm">
                      <li className="flex gap-2">
                        <span className="text-purple-400">•</span>
                        <span><strong>Baseline Normal:</strong> Understand your environment's typical behavior first</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-purple-400">•</span>
                        <span><strong>Whitelisting:</strong> Exclude known-good processes, IPs, or users from alerts</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-purple-400">•</span>
                        <span><strong>Threshold Tuning:</strong> Adjust count thresholds to reduce noisy alerts</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-purple-400">•</span>
                        <span><strong>Correlation:</strong> Combine multiple weak signals for high-confidence detection</span>
                      </li>
                    </ul>
                  </div>

                  <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-5">
                    <h4 className="font-semibold text-green-400 mb-3 flex items-center gap-2">
                      <Shield className="h-5 w-5" />
                      MITRE ATT&CK Mapping
                    </h4>
                    <ul className="space-y-2 text-gray-300 text-sm">
                      <li className="flex gap-2">
                        <span className="text-green-400">•</span>
                        <span><strong>Coverage Analysis:</strong> Map rules to ATT&CK techniques for gap identification</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-green-400">•</span>
                        <span><strong>Tactic Focus:</strong> Ensure detection across all attack lifecycle stages</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-green-400">•</span>
                        <span><strong>Sub-Techniques:</strong> Drill down to specific attack variations (T1059.001 vs T1059.003)</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-green-400">•</span>
                        <span><strong>Threat Intel:</strong> Prioritize techniques used by relevant threat actors</span>
                      </li>
                    </ul>
                  </div>

                  <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-5">
                    <h4 className="font-semibold text-orange-400 mb-3 flex items-center gap-2">
                      <Sparkles className="h-5 w-5" />
                      AI-Enhanced Detection
                    </h4>
                    <ul className="space-y-2 text-gray-300 text-sm">
                      <li className="flex gap-2">
                        <span className="text-orange-400">•</span>
                        <span><strong>Behavioral Baselining:</strong> ML models learn normal patterns for anomaly detection</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-orange-400">•</span>
                        <span><strong>Natural Language:</strong> Describe threats in plain English, generate optimized rules</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-orange-400">•</span>
                        <span><strong>Auto-Tuning:</strong> AI adjusts thresholds based on alert feedback and outcomes</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-orange-400">•</span>
                        <span><strong>Threat Hunting:</strong> Generate hypotheses and detection logic from CTI reports</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-cyan-500/10 via-blue-500/10 to-purple-500/10 border border-cyan-500/30 rounded-lg p-6">
                <h3 className="text-xl font-semibold text-white mb-3">Detection Maturity Model</h3>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="bg-red-500 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 text-white font-bold">1</div>
                    <div>
                      <p className="text-white font-semibold">Initial - Ad-hoc Rules</p>
                      <p className="text-gray-400 text-sm">Basic signature-based detection, high false positive rate, reactive approach</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-orange-500 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 text-white font-bold">2</div>
                    <div>
                      <p className="text-white font-semibold">Managed - Standardized Detection</p>
                      <p className="text-gray-400 text-sm">Documented rules, some tuning, basic MITRE ATT&CK mapping</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-yellow-500 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 text-white font-bold">3</div>
                    <div>
                      <p className="text-white font-semibold">Defined - Detection Engineering Process</p>
                      <p className="text-gray-400 text-sm">Systematic rule development, testing pipeline, version control, metrics tracking</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-green-500 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 text-white font-bold">4</div>
                    <div>
                      <p className="text-white font-semibold">Quantitatively Managed - Data-Driven Optimization</p>
                      <p className="text-gray-400 text-sm">Continuous tuning based on metrics, automated testing, coverage analysis, threat intel integration</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-blue-500 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 text-white font-bold">5</div>
                    <div>
                      <p className="text-white font-semibold">Optimizing - AI-Enhanced Detection</p>
                      <p className="text-gray-400 text-sm">ML/AI-powered anomaly detection, auto-generated rules, predictive threat hunting, continuous evolution</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </TabsContent>

        {/* Examples Tab */}
        <TabsContent value="examples" className="space-y-6">
          <div className="grid grid-cols-1 gap-6">
            {RULE_EXAMPLES.map((example, index) => (
              <Card key={index} className="bg-black/40 border-cyan-500/20 p-8">
                <div className="space-y-6">
                  {/* Header */}
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="text-2xl font-bold text-white">{example.title}</h3>
                        <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500/50 text-sm">
                          {example.format.toUpperCase()}
                        </Badge>
                      </div>
                      <p className="text-gray-300 text-lg">{example.description}</p>
                    </div>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => copyToClipboard(example.rule)}
                      className="bg-black/40 border-cyan-500/20 hover:bg-cyan-500/10"
                    >
                      <Copy className="mr-2 h-4 w-4" />
                      Copy Rule
                    </Button>
                  </div>

                  {/* Technical Details */}
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-5">
                    <h4 className="text-lg font-semibold text-blue-400 mb-3 flex items-center gap-2">
                      <Code2 className="h-5 w-5" />
                      Technical Details
                    </h4>
                    <p className="text-gray-300 leading-relaxed">{example.technicalDetails}</p>
                  </div>

                  {/* Use Case */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-5">
                      <h4 className="text-lg font-semibold text-green-400 mb-3">Use Case</h4>
                      <p className="text-gray-300">{example.useCase}</p>
                    </div>

                    <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-5">
                      <h4 className="text-lg font-semibold text-purple-400 mb-3">Detection Logic</h4>
                      <p className="text-gray-300">{example.detectionLogic}</p>
                    </div>
                  </div>

                  {/* MITRE ATT&CK */}
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-3">MITRE ATT&CK Techniques</h4>
                    <div className="flex flex-wrap gap-2">
                      {example.mitre.map((technique) => (
                        <Badge key={technique} className="bg-purple-500/20 text-purple-400 border-purple-500/50">
                          {technique}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Rule Code */}
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-3">Detection Rule</h4>
                    <pre className="bg-black/60 border border-gray-700 rounded-lg p-6 overflow-x-auto">
                      <code className="text-sm text-gray-300 leading-relaxed">{example.rule}</code>
                    </pre>
                  </div>

                  {/* False Positives */}
                  <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-5">
                    <h4 className="text-lg font-semibold text-yellow-400 mb-3 flex items-center gap-2">
                      <AlertTriangle className="h-5 w-5" />
                      Potential False Positives
                    </h4>
                    <ul className="space-y-2">
                      {example.falsePositives.map((fp, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-gray-300">
                          <span className="text-yellow-400 mt-1">•</span>
                          <span>{fp}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
